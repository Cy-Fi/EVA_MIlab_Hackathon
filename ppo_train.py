from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from time import gmtime, strftime
import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from agents.cnn_ppo import CNN_PPO_Agent

def plot_durations(episode_durations, show_result=False, save_path=None):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')

    plt.xlabel('Episode')
    plt.ylabel('Duration')

    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")


def train_agent(episodes, run_name):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    env._max_episode_steps = 1000
    buffer = []
    update_every = 2048  # Update PPO after collecting 2048 transitions

    agent = CNN_PPO_Agent(input_shape=env.env.observation_space.shape, run_name=run_name)

    for episode in range(episodes):
        state, info = env.reset()

        state = torch.tensor(state, dtype=torch.float32, device=agent.device)
        state = state.permute(2, 0, 1).unsqueeze(0)  # Ensure shape (1, C, H, W)

        total_reward = 0
        done = False
        t = 0

        while not done:
            action, log_prob, value = agent.select_action(state)
            action = np.array(action, dtype=np.float32)

            next_state, reward, done, _, _ = env.step(action)

            next_state = torch.tensor(next_state, dtype=torch.float32, device=agent.device)
            next_state = next_state.permute(2, 0, 1).unsqueeze(0)  # Ensure shape (1, 3, 96, 96)

            buffer.append((state, action, reward, next_state, log_prob, value, done))

            if len(buffer) >= update_every:
                for transition in buffer:
                    agent.store_transition(transition)
                agent.update()
                buffer = []  # Clear buffer after update

            state = next_state
            total_reward += reward
            t += 1

            # print(f"Step {t}: Reward={reward:.2f}, Done={done}")

            if t >= 1000:  # Force episode termination after 1000 steps
                print(f"Episode {episode} reached max steps (1000), terminating.")
                done = True

            if done:
                agent.episode_durations.append(episode + 1)
                plot_durations(agent.episode_durations)

                if episode == episodes - 1:
                    plot_durations(agent.episode_durations, show_result=True, save_path=f"plots/{run_name}_training_plot.png")
                break

        agent._log_reward(episode, total_reward)

        if episode % 50 == 0:
            agent.save_checkpoint(episode)

        print(f"Episode {episode}: Total Reward: {total_reward}")

    env.close()

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")


RUN_NAME = f"CNN_PPO_{strftime('%Y%m%d%H%M%S', gmtime())}"
train_agent(episodes = 2, run_name = RUN_NAME)