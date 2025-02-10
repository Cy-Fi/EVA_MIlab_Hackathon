import gymnasium as gym
import torch
from agents.cnn_dqn import DQNAgent

def evaluate(model_path="dqn_model.pth"):
    env = gym.make("CarRacing-v2")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.shape[0])
    agent.model.load_state_dict(torch.load(model_path))
    
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.act(state, epsilon=0)
        state, _, done, _, _ = env.step(action)
        env.render()

if __name__ == "__main__":
    evaluate()
