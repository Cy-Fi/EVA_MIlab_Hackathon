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
from pathlib import Path

# Ensure logs and models directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class CNN_DQN(nn.Module):
    """Improved CNN Model for Deep Q-Learning"""
    def __init__(self, input_shape, img_stack, DISCRETE_ACTIONS):
        super(CNN_DQN, self).__init__()

        self.DISCRETE_ACTIONS = DISCRETE_ACTIONS

        self.conv1 = nn.Conv2d(img_stack, 16, kernel_size=3, stride=2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1) 
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1)  
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1) 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.flatten = nn.Flatten()

        # Calculate the output shape dynamically after convolutions
        dummy_input = torch.zeros(1, img_stack, 96, 96)
        with torch.no_grad():
            conv_out_size = self._get_conv_output(dummy_input).shape[1]

        self.fc1 = nn.Linear(conv_out_size, 256) 
        self.fc2 = nn.Linear(256, len(self.DISCRETE_ACTIONS))  # Output Q-values for each discrete action

    def _get_conv_output(self, x):
        """Pass a dummy tensor to get output shape dynamically"""
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)

        x = torch.relu(self.conv5(x))  
        x = torch.relu(self.conv6(x))  # New convolutional layer
        x = self.pool3(x)  # New pooling layer
        
        x = self.flatten(x)
        
        return x

    def forward(self, x):
        x = self._get_conv_output(x)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw Q-values
        
        # print(f"out: {x.shape}")
        return x

class CNN_DQN_Agent:
    """Deep Q-Learning Agent with CNN"""
    def __init__(self, 
          input_shape,
          DISCRETE_ACTIONS,
          run_name,
          img_stack, 
          learning_rate, 
          gamma,
          tau,
          epsilon_start,
          epsilon_end,
          epsilon_decay_steps,
          batch_size,
          replay_buffer_size
          ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_stack = img_stack
        self.policy_net = CNN_DQN(input_shape, img_stack, DISCRETE_ACTIONS).to(self.device)
        self.target_net = CNN_DQN(input_shape, img_stack, DISCRETE_ACTIONS).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.run_name = run_name
        self.memory = ReplayMemory(capacity = replay_buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.steps = 0
        self.episode_durations = []
        self.DISCRETE_ACTIONS = DISCRETE_ACTIONS

        self.checkpoint_path = "checkpoints/cnn_dqn"
        self.reward_log_file = "logs/cnn_dqn"
    
    def select_action(self, state, explore=True):
        epsilon = self.epsilon if self.steps < self.epsilon_decay_steps else self.epsilon_end
        if explore and np.random.rand() < epsilon:
            # rand_action = torch.tensor(random.randint(0, len(self.DISCRETE_ACTIONS) - 1), device = self.device)
            rand_action = torch.rand((state.shape[0], len(self.DISCRETE_ACTIONS) - 1)).argmax(dim=1).to(self.device).view(-1, 1)
            # print(f"rand_action: {rand_action.shape}")
            return rand_action
        
        with torch.no_grad():
            # print(f"state: {state.shape}")
            out = self.policy_net(state.to(self.device))
            # print(f"out: {out.shape}, greed action: {out.argmax(-1).view(-1, 1)}")
            # Convert Q-values to discrete actions
            return out.argmax(-1).view(-1, 1)  # Get index of max Q-value

    def get_action_from_action_index(self, action_indices):
        discrete_actions = torch.tensor(self.DISCRETE_ACTIONS, device=self.device)
        return discrete_actions[action_indices]

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
        action_batch = torch.stack(batch.action).view(-1, 1)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1})= max_a Q(s_{t+1}, a) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            
        # Compute the expected state-action values (Q values)
        #   Q(s_t,a_t) = r  + [GAMMA * max_a Q(s_{t+1}, a)]
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
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
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{self.checkpoint_path}/{self.run_name}_episode_{episode}.pth")
        print(f"Checkpoint saved at episode {episode}")


    def load_checkpoint(self, file):
        """Loads the model checkpoint"""
        if os.path.exists(self.checkpoint_path):
            self.policy_net.load_state_dict(torch.load(f"{self.checkpoint_path}/{file}"))
            print("Checkpoint loaded.")


    def log_reward(self, episode, reward):
        """Logs episode rewards to CSV"""
        Path(self.reward_log_file).mkdir(parents=True, exist_ok=True)
        with open(f"{self.reward_log_file}/{self.run_name}.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([episode, reward])

