"""
BipedalWalker Solver using PPO

Demonstrates PPO on BipedalWalker-v3 (continuous action space).
This is a significantly harder environment with 24-dim state and 4-dim
continuous actions (torques for hip and knee joints on two legs).

Uses a Gaussian policy with learnable standard deviation.

Solved: Average return >= 300 over 100 consecutive episodes.

Author: RL Speedrun
"""

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '18_ppo'))
from ppo import PPOAgent, train, evaluate


def main():
    print("=" * 70)
    print("BipedalWalker: Proximal Policy Optimization (PPO)")
    print("=" * 70)

    print("\n" + "-" * 40)
    print("Environment Info")
    print("-" * 40)
    print("  State dim: 24 (hull angle/vel, joint angles/speeds, lidar)")
    print("  Action dim: 4 (continuous torques in [-1, 1])")
    print("  Reward: +300 total for walking to end, -100 for falling")
    print("  Solved: avg return >= 300 over 100 episodes")

    print("\n" + "-" * 40)
    print("PPO Configuration")
    print("-" * 40)
    print("  Network: 64 -> 64 separate actor/critic (Tanh)")
    print("  Policy: Gaussian (learnable log_std)")
    print("  Learning rate: 3e-4")
    print("  Clip range: 0.2")
    print("  Epochs per update: 10")
    print("  Steps per rollout: 2048")
    print("  Batch size: 64")

    print("\n" + "=" * 70)
    print("Training PPO (this may take a while...)")
    print("=" * 70)
    print("-" * 50)

    agent, returns, lengths = train(
        "BipedalWalker-v3",
        n_episodes=3000,
        seed=42,
        lr=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.0,
        value_coef=0.5,
        max_grad_norm=0.5,
        hidden_dim=64,
    )

    print("-" * 50)

    mean, std = evaluate(agent, "BipedalWalker-v3")
    print(f"\nFinal evaluation (100 episodes): {mean:.1f} +/- {std:.1f}")
    if mean >= 300:
        print("SOLVED!")
    else:
        print(f"Not solved (need >= 300, got {mean:.1f})")

    # Check when solved
    window = 100
    for i in range(len(returns) - window):
        if np.mean(returns[i:i+window]) >= 300:
            print(f"First solved at episode {i + window}")
            break

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    window_plot = 50

    # Learning curve
    if len(returns) > window_plot:
        smoothed_returns = np.convolve(returns, np.ones(window_plot)/window_plot, mode='valid')
        axes[0, 0].plot(smoothed_returns, color='blue', linewidth=1.5)
    else:
        axes[0, 0].plot(returns, color='blue', linewidth=1.5)
    axes[0, 0].axhline(y=300, color='green', linestyle='--', label='Solved (300)')
    axes[0, 0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Return')
    axes[0, 0].set_title('PPO Learning Curve on BipedalWalker')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Return distribution
    last_n = min(200, len(returns))
    axes[0, 1].hist(returns[-last_n:], bins=30, color='steelblue',
                    edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=np.mean(returns[-last_n:]), color='red', linestyle='--',
                       label=f'Mean: {np.mean(returns[-last_n:]):.1f}')
    axes[0, 1].axvline(x=300, color='green', linestyle='--', alpha=0.7,
                       label='Solved threshold')
    axes[0, 1].set_xlabel('Episode Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Return Distribution (Last {last_n} Episodes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Entropy
    if agent.training_stats:
        entropies = [s['entropy'] for s in agent.training_stats]
        w = min(100, max(1, len(entropies) // 5))
        if w > 1:
            smoothed_ent = np.convolve(entropies, np.ones(w)/w, mode='valid')
            axes[1, 0].plot(smoothed_ent, color='purple', linewidth=1.5)
        else:
            axes[1, 0].plot(entropies, color='purple', linewidth=1.5)

    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Policy Entropy')
    axes[1, 0].set_title('Policy Entropy (Gaussian)')
    axes[1, 0].grid(True, alpha=0.3)

    # Clip fraction
    if agent.training_stats:
        clip_fracs = [s['clip_fraction'] for s in agent.training_stats]
        w = min(100, max(1, len(clip_fracs) // 5))
        if w > 1:
            smoothed_clip = np.convolve(clip_fracs, np.ones(w)/w, mode='valid')
            axes[1, 1].plot(smoothed_clip, color='orange', linewidth=1.5)
        else:
            axes[1, 1].plot(clip_fracs, color='orange', linewidth=1.5)

    axes[1, 1].set_xlabel('Update Step')
    axes[1, 1].set_ylabel('Clip Fraction')
    axes[1, 1].set_title('PPO Clip Fraction')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'bipedal_walker_ppo.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    main()
