"""
Solve BipedalWalker-v3 with TD3.

BipedalWalker is a challenging continuous control benchmark with 24-dim state
and 4-dim action space. TD3's off-policy nature and twin critics provide
advantages over on-policy PPO for this environment.

Author: RL Speedrun
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '22_td3'))
from td3 import TD3Agent, train, evaluate

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    print("=" * 70)
    print("BipedalWalker-v3: Twin Delayed DDPG (TD3)")
    print("=" * 70)

    print("\n" + "-" * 40)
    print("Environment Info")
    print("-" * 40)
    print("  State dim: 24 (hull angle/vel, joint angles/speeds, lidar)")
    print("  Action dim: 4 (continuous torques in [-1, 1])")
    print("  Reward: +300 total for walking to end, -100 for falling")
    print("  Solved: avg return >= 300 over 100 episodes")

    print("\n" + "-" * 40)
    print("TD3 Configuration")
    print("-" * 40)
    print("  Network: 256 -> 256 (ReLU)")
    print("  Actor lr: 3e-4, Critic lr: 3e-4")
    print("  Policy delay: 2, Target noise: 0.2, Noise clip: 0.5")
    print("  Exploration noise: 0.1, Warmup: 10000 steps")

    print("\n" + "=" * 70)
    print("Training TD3 (this may take a while...)")
    print("=" * 70)
    print("-" * 50)

    agent, returns, lengths = train(
        "BipedalWalker-v3",
        n_episodes=1000,
        seed=42,
        actor_lr=3e-4,
        critic_lr=3e-4,
        tau=0.005,
        batch_size=256,
        buffer_size=1000000,
        noise_sigma=0.1,
        policy_delay=2,
        target_noise=0.2,
        noise_clip=0.5,
        warmup_steps=10000,
        hidden_dim=256,
    )

    print("-" * 50)

    mean, std = evaluate(agent, "BipedalWalker-v3")
    print(f"\nFinal evaluation (100 episodes): {mean:.1f} +/- {std:.1f}")
    if mean >= 300:
        print("SOLVED!")
    else:
        print(f"Not solved (need >= 300, got {mean:.1f})")

    # 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    w = 20
    if len(returns) > w:
        smoothed = np.convolve(returns, np.ones(w)/w, mode='valid')
        axes[0, 0].plot(smoothed, color='blue', linewidth=1.5)
    axes[0, 0].axhline(y=300, color='green', linestyle='--', label='Solved (300)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('TD3 Learning Curve (BipedalWalker)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(returns[-50:], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Return')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Return Distribution (Last 50)')

    if agent.training_stats:
        step = max(1, len(agent.training_stats) // 500)
        q1s = [s['q1_mean'] for s in agent.training_stats[::step]]
        q2s = [s['q2_mean'] for s in agent.training_stats[::step]]
        axes[1, 0].plot(q1s, color='blue', linewidth=1, alpha=0.8, label='Q1')
        axes[1, 0].plot(q2s, color='red', linewidth=1, alpha=0.8, label='Q2')
        axes[1, 0].legend()
    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Mean Q-Value')
    axes[1, 0].set_title('Twin Q-Values')
    axes[1, 0].grid(True, alpha=0.3)

    if agent.training_stats:
        c_losses = [s['critic_loss'] for s in agent.training_stats[::step]]
        axes[1, 1].plot(c_losses, color='red', linewidth=1)
    axes[1, 1].set_xlabel('Update Step')
    axes[1, 1].set_ylabel('Critic Loss')
    axes[1, 1].set_title('Critic Loss')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'bipedal_walker_td3.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    main()
