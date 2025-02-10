import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics():
    os.makedirs("plots", exist_ok=True)

    # Plot Rewards
    rewards_df = pd.read_csv("logs/rewards.csv", names=["Episode", "Reward"])
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_df["Episode"], rewards_df["Reward"], label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Time")
    plt.legend()
    plt.savefig("plots/reward_plot.png")
    plt.show()

    # Plot Losses
    losses_df = pd.read_csv("logs/losses.csv", names=["Step", "Loss"])
    plt.figure(figsize=(10, 5))
    plt.plot(losses_df["Step"], losses_df["Loss"], label="Loss", color="red")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.savefig("plots/loss_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_metrics()
