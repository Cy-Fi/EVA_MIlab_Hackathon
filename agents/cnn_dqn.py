import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
import os
import random
import csv
from utils.memory import ReplayMemory
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt

# Ensure logs and models directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class CNN_DQN(nn.Module):
    """CNN Model for Deep Q-Learning"""
    def __init__(self, input_shape):
        super(CNN_DQN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(2000, 100)
        self.fc2_steering = nn.Linear(100, 1)
        self.fc2_gas = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.unsqueeze(torch.mean(x, dim=-1), 1) #RGB to Gray by taking the mean -> shape [batch_size, 1, 96, 96]
        # print(f"Model in shape: {x.shape}")
        x = self.pool1(torch.relu(self.conv1(x)))

        x = self.pool2(torch.relu(self.conv2(x)))

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))

        action = torch.tanh(self.fc2_steering(x))
        gas = torch.sigmoid(self.fc2_gas(x))
        
        out = torch.cat([action.unsqueeze(0), gas.unsqueeze(0), (1 - gas).unsqueeze(0)], dim=-1).squeeze() # return shape [batch_size, 3]
        # print(f"Model out shape: {out.shape}")
        return out

class CNN_DQN_Agent:
    """Deep Q-Learning Agent with CNN"""
    def __init__(self, 
          input_shape, 
          action_space, 
          lr=0.001, 
          gamma=0.99,
          tau = 0.005,
          epsilon=0.5,
          epsilon_end=0.0,
          epsilon_decay_steps=1000,
          memory_size=10000, 
          batch_size=2,
          steps_per_target_net_update = 100,
          replay_buffer_size = 100
          ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = CNN_DQN(input_shape).to(self.device)
        self.target_net = CNN_DQN(input_shape).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        self.memory = ReplayMemory(capacity = replay_buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_schedule = torch.logspace(torch.log10(torch.tensor(self.epsilon)), torch.log10(torch.tensor(self.epsilon_end + 1e-9)), steps = self.epsilon_decay_steps)
        self.batch_size = batch_size
        self.steps_per_target_net_update = steps_per_target_net_update
        self.steps = 0
        self.episode_durations = []
        self.action_space = action_space

        self.checkpoint_path = "checkpoints/cnn_dqn.pth"
        self.reward_log_file = "logs/rewards.csv"
    
    def select_action(self, state, explore=True):
        """Selects an action using epsilon-greedy strategy"""
        epsilon = self.epsilon
        if self.steps >= self.epsilon_decay_steps:
            epsilon = self.epsilon_end
        else:
            epsilon = self.epsilon_schedule[self.steps]

        if explore and np.random.rand() < epsilon:
            # Random action
            steering = torch.tanh(torch.randn(1)).item()  # Random steering in [-1, 1]
            gas = torch.sigmoid(torch.randn(1)).item()  # Random gas in [0, 1]

            # Construct the action array
            return torch.tensor([steering, gas, 1 - gas], dtype=torch.float32)
        
        with torch.no_grad():
            return self.policy_net(state)

    def train_step(self):
        """Trains the policy_net using replay memory"""
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.stack(batch.action, 0)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)

        # Compute V(s_{t+1})= max_a Q(s_{t+1}, a) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # Compute next best actions using the policy network (actor)
            next_best_actions = self.policy_net(non_final_next_states)  # Expected shape: [batch_size, 3]

            # Compute Q-values for these best actions using the target network (critic)
            next_q_values = self.target_net(non_final_next_states)  # Expected shape: [batch_size, 3]

            # Compute max Q-value across the three action dimensions
            max_q_values = next_q_values.max(dim=1).values  # Expected shape: [batch_size]

            # Ensure that max_q_values[non_final_mask] is indexed correctly
            next_state_values[non_final_mask] = max_q_values[non_final_mask]

        # Compute the expected state-action values (Q values)
        #   Q(s_t,a_t) = r  + [GAMMA * max_a Q(s_{t+1}, a)]
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

        self.steps += 1

    def save_checkpoint(self, episode):
        """Saves the model checkpoint"""
        torch.save(self.policy_net.state_dict(), self.checkpoint_path)
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self):
        """Loads the model checkpoint"""
        if os.path.exists(self.checkpoint_path):
            self.policy_net.load_state_dict(torch.load(self.checkpoint_path))
            print("Checkpoint loaded.")

    def log_reward(self, episode, reward):
        """Logs episode rewards to CSV"""
        with open(self.reward_log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([episode, reward])


    def log_loss(self, loss):
        with open(self.loss_log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.steps, loss])
