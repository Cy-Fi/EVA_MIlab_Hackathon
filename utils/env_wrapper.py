import gymnasium as gym
import torch
import numpy as np

class Env():
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, env_name, random_seed, img_stack: int, action_repeat, render_mode=None):
        self.env = gym.make(env_name, continuous=True, render_mode = render_mode)
        self.action_space = self.env.action_space
        self.reward_threshold = self.env.spec.reward_threshold
        self.img_stack = img_stack
        self.action_repeat = action_repeat

        # Define discrete action space
        self.STEERING_VALUES = np.linspace(-1.0, 1.0, 10)  # 10 discrete steering values
        self.GAS_VALUES = np.linspace(0.0, 1.0, 5)  # 5 discrete gas values

        self.DISCRETE_ACTIONS = [
            [s, g, 1 - g] for s in self.STEERING_VALUES for g in self.GAS_VALUES
        ]

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb, info = self.env.reset()
        #         print(img_rgb)
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [np.expand_dims(img_gray, axis=0)] * self.img_stack  # four frames per state
        return torch.FloatTensor(self.stack).permute(1, 0, 2, 3), info

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, die, truncated, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(np.expand_dims(img_gray, axis=0))
        assert len(self.stack) == self.img_stack
        
        return torch.FloatTensor(self.stack).squeeze(), total_reward, done, die, truncated

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory