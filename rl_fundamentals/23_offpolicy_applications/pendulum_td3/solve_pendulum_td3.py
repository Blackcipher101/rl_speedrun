"""
Solve Pendulum-v1 with TD3.

Comparing TD3's improvements over DDPG on the same environment.
TD3 should be more stable and potentially faster to converge.

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
    print("Pendulum-v1: Twin Delayed DDPG (TD3)")
    print("=" * 70)

    print("\n" + "-" * 40)
    print("Environment Info")
    print("-" * 40)
    print("  State dim: 3 (cos θ, sin θ, θ̇)")
    print("  Action dim: 1 (torque in [-2, 2])")
    print("  Reward: -(θ² + 0.1·θ̇² + 0.001·u²)")
    print("  Solved: avg return >= -200 over 100 episodes")

    print("\n" + "-" * 40)
    print("TD3 Configuration")
    print("-" * 40)
    print("  Network: 256 -> 256 (ReLU)")
    print("  Actor lr: 1e-4, Critic lr: 1e-3")
    print("  Policy delay: 2, Target noise: 0.2, Noise clip: 0.5")
    print("  Exploration noise: 0.1 (Gaussian)")

    print("\n" + "=" * 70)
    print("Training TD3")
    print("=" * 70)
    print("-" * 50)

    agent, returns, lengths = train(
        "Pendulum-v1",
        n_episodes=200,
        seed=42,
        actor_lr=1e-4,
        critic_lr=1e-3,
        tau=0.005,
        batch_size=64,
        buffer_size=100000,
        noise_sigma=0.1,
        policy_delay=2,
        target_noise=0.2,
        noise_clip=0.5,
        warmup_steps=1000,
        hidden_dim=256,
    )

    print("-" * 50)

    mean, std = evaluate(agent, "Pendulum-v1")
    print(f"\nFinal evaluation (100 episodes): {mean:.1f} +/- {std:.1f}")
    if mean >= -200:
        print("SOLVED!")
    else:
        print(f"Not solved (need >= -200, got {mean:.1f})")

    window = 50
    for i in range(len(returns) - window):
        if np.mean(returns[i:i + window]) >= -200:
            print(f"First solved at episode {i + window}")
            break

    # 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    w = 10
    smoothed = np.convolve(returns, np.ones(w)/w, mode='valid')
    axes[0, 0].plot(smoothed, color='blue', linewidth=1.5)
    axes[0, 0].axhline(y=-200, color='green', linestyle='--', label='Solved (-200)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('TD3 Learning Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(returns[-50:], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=-200, color='green', linestyle='--', label='Solved')
    axes[0, 1].set_xlabel('Return')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Return Distribution (Last 50)')
    axes[0, 1].legend()

    if agent.training_stats:
        step = max(1, len(agent.training_stats) // 500)
        q1s = [s['q1_mean'] for s in agent.training_stats[::step]]
        q2s = [s['q2_mean'] for s in agent.training_stats[::step]]
        axes[1, 0].plot(q1s, color='blue', linewidth=1, alpha=0.8, label='Q1')
        axes[1, 0].plot(q2s, color='red', linewidth=1, alpha=0.8, label='Q2')
        axes[1, 0].legend()
    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Mean Q-Value')
    axes[1, 0].set_title('Twin Q-Values (Q1 vs Q2)')
    axes[1, 0].grid(True, alpha=0.3)

    if agent.training_stats:
        c_losses = [s['critic_loss'] for s in agent.training_stats[::step]]
        axes[1, 1].plot(c_losses, color='red', linewidth=1)
    axes[1, 1].set_xlabel('Update Step')
    axes[1, 1].set_ylabel('Critic Loss')
    axes[1, 1].set_title('Critic Loss')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'pendulum_td3.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    main()
