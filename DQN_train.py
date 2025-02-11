
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from time import gmtime, strftime
import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt



from agents.cnn_dqn import CNN_DQN_Agent


def plot_durations(episode_durations, show_result=False):
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
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")


def train_agent(episodes, run_name, hyperparameters, load_from_checkpoint = ''):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    
    agent = CNN_DQN_Agent(
        input_shape=env.observation_space.shape, 
        action_space=env.action_space,
        run_name = run_name,
        **hyperparameters
        )

    if load_from_checkpoint != '':
        agent.load_checkpoint(load_from_checkpoint)


    for episode in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        done = False

        for t in count():
            action = agent.select_action(state)

            observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent.memory.push(state.to(agent.device), action.to(agent.device), next_state.to(agent.device), reward.to(agent.device))
            agent.train_step()
            state = next_state
            total_reward += reward

            if done:
              agent.episode_durations.append(t + 1)
              plot_durations(agent.episode_durations)

              # Save plot at the end of training
              if episode == episodes - 1:  # Last episode
                  plot_durations(agent.episode_durations, show_result=True, save_path=f"{run_name}_training_plot.png")
              break

        agent.log_reward(episode, total_reward)

        if episode % 50 == 0 and episode > 0:
            agent.save_checkpoint(episode)

        

        print(f"Episode {episode}: Total Reward: {total_reward.cpu().item()}")

    env.close()

if __name__ == '__main__':
    


  is_ipython = 'inline' in matplotlib.get_backend()
  if is_ipython:
      from IPython import display
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device {device}")

  hyperparameters = {
      "batch_size": 64,
      "gamma": 0.95,
      "epsilon_start": 1.0,
      "epsilon_end": 0.1,
      "tau": 0.005,
      "epsilon_decay_steps": 200,
      "learning_rate": 0.001,
      "replay_buffer_size": 64,
      "steps_per_target_net_update": 512
  }


  RUN_NAME = f"CNN_DQN_{strftime('%Y%m%d%H%M%S', gmtime())}"
  train_agent(episodes = 300, run_name = RUN_NAME, hyperparameters=hyperparameters)

