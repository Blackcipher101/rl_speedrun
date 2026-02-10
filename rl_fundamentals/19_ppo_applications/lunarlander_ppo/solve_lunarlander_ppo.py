"""
LunarLander Solver using PPO

Demonstrates PPO on LunarLander-v3 (discrete action space, 8-dim state).
The clipped surrogate objective keeps learning stable on this harder task.

Solved: Average return >= 200 over 100 consecutive episodes.

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
    print("LunarLander: Proximal Policy Optimization (PPO)")
    print("=" * 70)

    print("\n" + "-" * 40)
    print("Environment Info")
    print("-" * 40)
    print("  State: [x, y, vx, vy, angle, angular_vel, leg1, leg2]")
    print("  Actions: 0=nothing, 1=left, 2=main, 3=right")
    print("  Reward: -100 (crash) to +100 (perfect landing)")
    print("  Solved: avg return >= 200 over 100 episodes")

    print("\n" + "-" * 40)
    print("PPO Configuration")
    print("-" * 40)
    print("  Network: 128 -> 128 shared backbone (Tanh)")
    print("  Learning rate: 5e-4")
    print("  Clip range: 0.2")
    print("  Epochs per update: 10")
    print("  Steps per rollout: 512")
    print("  Batch size: 64")

    print("\n" + "=" * 70)
    print("Training PPO")
    print("=" * 70)
    print("-" * 50)

    agent, returns, lengths = train(
        "LunarLander-v3",
        n_episodes=3000,
        seed=42,
        lr=5e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        hidden_dim=128,
    )

    print("-" * 50)

    mean, std = evaluate(agent, "LunarLander-v3")
    print(f"\nFinal evaluation (100 episodes): {mean:.1f} +/- {std:.1f}")
    if mean >= 200:
        print("SOLVED!")
    else:
        print(f"Not solved (need >= 200)")

    # Check when solved
    window = 100
    for i in range(len(returns) - window):
        if np.mean(returns[i:i+window]) >= 200:
            print(f"First solved at episode {i + window}")
            break

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    window_plot = 50

    # Learning curve
    smoothed_returns = np.convolve(returns, np.ones(window_plot)/window_plot, mode='valid')
    axes[0, 0].plot(smoothed_returns, color='blue', linewidth=1.5)
    axes[0, 0].axhline(y=200, color='green', linestyle='--', label='Solved (200)')
    axes[0, 0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Return')
    axes[0, 0].set_title('PPO Learning Curve on LunarLander')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Return distribution
    last_n = min(200, len(returns))
    axes[0, 1].hist(returns[-last_n:], bins=30, color='steelblue',
                    edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=np.mean(returns[-last_n:]), color='red', linestyle='--',
                       label=f'Mean: {np.mean(returns[-last_n:]):.1f}')
    axes[0, 1].axvline(x=200, color='green', linestyle='--', alpha=0.7,
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

    axes[1, 0].axhline(y=np.log(4), color='gray', linestyle='--',
                       alpha=0.5, label='Max entropy (log 4)')
    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Policy Entropy')
    axes[1, 0].set_title('Policy Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Clip fraction
    if agent.training_stats:
        clip_fracs = [s['clip_fraction'] for s in agent.training_stats]
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

    save_path = os.path.join(os.path.dirname(__file__), 'lunarlander_ppo.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    main()
