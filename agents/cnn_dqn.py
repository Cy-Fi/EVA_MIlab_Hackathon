import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
import os
import random
import csv
from ..utils.memory import ReplayBuffer

# Ensure logs and models directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

class CNN_DQN(nn.Module):
    """CNN Model for Deep Q-Learning"""
    def __init__(self, input_shape, num_actions):
        super(CNN_DQN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(2000, 100)
        self.fc2 = nn.Linear(100, num_actions)

    def forward(self, x):
        x = rgb_to_grayscale(x)

        x = self.pool1(torch.relu(self.conv1(x)))

        x = self.pool2(torch.relu(self.conv2(x)))

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        
        return self.fc2(x)

class CNN_DQN_Agent:
    """Deep Q-Learning Agent with CNN"""
    def __init__(self, input_shape, num_actions, lr=0.001, gamma=0.99, epsilon=0.2, memory_size=10000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = CNN_DQN(input_shape, num_actions).to(self.device)
        self.target_net = CNN_DQN(input_shape, num_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        self.memory = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = 0.0
        self.epsilon_decay_steps = 1000
        self.epsilon_schedule = torch.logspace(torch.log10(torch.tensor(self.epsilon)), torch.log10(torch.tensor(self.epsilon_end)), steps = self.epsilon_decay_steps)
        self.batch_size = batch_size
        self.steps = 0
        self.num_actions

        self.checkpoint_path = "checkpoints/cnn_dqn.pth"
        self.reward_log_file = "logs/rewards.csv"

    def select_action(self, state, explore=True):
        """Selects an action using epsilon-greedy strategy"""
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.num_actions))  # Random action
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Stores experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """Trains the model using replay memory"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            targets = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % 1000 == 0:
            self.target_model.load_state_dict(self.model.state_dict())  # Sync target model

    def save_checkpoint(self, episode):
        """Saves the model checkpoint"""
        torch.save(self.model.state_dict(), self.checkpoint_path)
        print(f"Checkpoint saved at episode {episode}")

    def load_checkpoint(self):
        """Loads the model checkpoint"""
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            print("Checkpoint loaded.")

    def _log_reward(self, episode, reward):
        """Logs episode rewards to CSV"""
        with open(self.reward_log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([episode, reward])


    def _log_loss(self, loss):
        with open(self.loss_log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.steps, loss])
