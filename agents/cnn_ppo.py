import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import rgb_to_grayscale
from torch.distributions import Normal
import numpy as np
import os
import random
import csv
from pathlib import Path

class CNN_PPO(nn.Module):
    """CNN Model for PPO"""
    def __init__(self, input_shape, num_actions):
        super(CNN_PPO, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1) 
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Calculate the output shape dynamically after convolutions
        dummy_input = torch.zeros(1, 1, 96, 96)
        with torch.no_grad():
            conv_out_size = self._get_conv_output(dummy_input)

        self.fc1 = nn.Linear(conv_out_size, 256) 
        self.fc2_steering = nn.Linear(256, 1)  # Steering output
        self.fc2_gas = nn.Linear(256, 1)       # Gas output
        self.value_head = nn.Linear(256, 1)    # Value function for critic

        self.log_std = nn.Parameter(torch.zeros(num_actions))  # Log standard deviation for policy

    def _get_conv_output(self, x):
        """Pass a dummy tensor to get output shape dynamically"""
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))  
        return self.flatten(x).shape[1]

    def forward(self, x):
        x = x.float()
        if x.ndim == 5:  # If extra dimension exists, remove it
            x = x.squeeze(1)

        x = torch.mean(x, dim=1, keepdim=True)  # Convert RGB to grayscale

        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))

        # Compute actions
        steering = torch.tanh(self.fc2_steering(x))  # Steering: [-1, 1]
        gas = torch.sigmoid(self.fc2_gas(x))  # Gas: [0, 1]
        brake = 1 - gas  # Brake is complementary to gas

        # Compute value estimate
        value = self.value_head(x)

        # Compute standard deviation (for sampling actions)
        std = self.log_std.exp().expand_as(torch.cat([steering, gas, brake], dim=-1))

        return torch.cat([steering, gas, brake], dim=-1), value, std


class CNN_PPO_Agent:
    """PPO Agent with CNN"""
    def __init__(self, 
                 input_shape, 
                 run_name,
                 num_actions=3, 
                 gamma=0.99, 
                 lam=0.95, 
                 clip_epsilon=0.2
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = run_name
        
        self.gamma = gamma  # Discount factor
        self.lam = lam  # GAE Lambda
        self.clip_epsilon = clip_epsilon  # PPO Clipping epsilon
        self.epochs = 10  # PPO update epochs
        self.batch_size = 32  # Minibatch size
        self.training_step = 0

        #print("State Before CNN:", input_shape)
        self.net = CNN_PPO(input_shape, num_actions).to(self.device).float()  # Force float32
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)

        self.memory = []  # Store trajectories for training

        self.checkpoint_path = "checkpoints/cnn_ppo.pth"
        self.reward_log_file = "logs/rewards.csv"


    def select_action(self, state):
        """Selects action using PPO policy for continuous action space"""
        
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        # Ensure correct shape (batch, channels, height, width)
        if state.ndim == 3:
            state = state.permute(2, 0, 1).unsqueeze(0)  # Convert (H, W, C) → (1, C, H, W)

        with torch.no_grad():
            action_mean, value, std = self.net(state)
            action_mean = action_mean.squeeze(0)
            value = value.squeeze(0)
            std = std.squeeze(0)

        # Create Gaussian distribution and sample action
        dist = Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        # Convert action tensor to NumPy array
        action = action.cpu().numpy().flatten()

        action[0] = np.clip(action[0], -1, 1)   # Steering: [-1, 1]
        action[1] = np.clip(action[1], 0, 1)    # Gas: [0, 1]
        action[2] = np.clip(action[2], 0, 1)    # Brake: [0, 1]

        return action, log_prob.cpu().numpy(), value.cpu().numpy()



    def store_transition(self, transition):
        """Stores (state, action, reward, next_state, log_prob, value, done)"""
        self.memory.append(transition)

    def compute_advantages(self, rewards, values, next_values, dones):
        """Computes GAE (Generalized Advantage Estimation)"""
        if len(rewards) == 0:
            return np.array([]), np.array([])

        advantages = []
        advantage = 0

        rewards = np.array(rewards)
        values = np.array(values)
        next_values = np.array(next_values)
        dones = np.array(dones)

        if next_values.ndim == 0:
            next_values = np.expand_dims(next_values, axis=0)

        if values.ndim == 0:
            values = np.expand_dims(values, axis=0)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.lam * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)

        returns = np.array(advantages) + values
        return np.array(advantages), returns


    def update(self):
        """Performs PPO update"""
        if len(self.memory) == 0:
            return

        # Convert memory to tensors
        states, actions, rewards, next_states, log_probs, values, dones = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32, device=self.device)
        values = torch.tensor(np.array(values), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, next_values, _ = self.net(torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device))
            next_values = next_values.squeeze()

        next_values = next_values.cpu().numpy()
        if next_values.ndim == 0:  # If it's a scalar, expand it
            next_values = np.expand_dims(next_values, axis=0)

        advantages, returns = self.compute_advantages(rewards, values.cpu().numpy(), next_values, dones.cpu().numpy())

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        for _ in range(self.epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Forward pass
                mean, values, std = self.net(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                # PPO Clipped Objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                # Value loss
                value_loss = (values.squeeze() - batch_returns).pow(2).mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear memory after training
        self.memory = []


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

    def _log_reward(self, episode, reward):
        """Logs episode rewards to CSV"""
        Path(self.reward_log_file).mkdir(parents=True, exist_ok=True)
        with open(f"{self.reward_log_file}/{self.run_name}.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([episode, reward])


    def _log_loss(self, loss):
        with open(self.loss_log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.steps, loss])