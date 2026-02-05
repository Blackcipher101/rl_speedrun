"""
LunarLander Solver using A2C

Demonstrates A2C on a more challenging control task.
LunarLander has 8-dimensional state and 4 discrete actions.

Solved: Average return >= 200 over 100 consecutive episodes.

Author: RL Speedrun
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Tuple, List
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '16_actor_critic'))

from a2c import A2CAgent, A2CConfig, train_a2c, evaluate_agent


def solve_lunarlander_a2c(
    n_episodes: int = 1000,
    seed: int = 42
) -> Tuple[A2CAgent, List[float], List[int], bool]:
    """
    Solve LunarLander using A2C.

    Args:
        n_episodes: Maximum training episodes
        seed: Random seed

    Returns:
        agent: Trained agent
        returns: Episode returns
        lengths: Episode lengths
        solved: Whether environment was solved
    """
    env = gym.make("LunarLander-v3")

    # Optimized hyperparameters for LunarLander
    config = A2CConfig(
        state_dim=8,
        action_dim=4,
        hidden_dims=[128, 128],  # Larger for harder problem
        n_episodes=n_episodes,
        max_steps=1000,
        learning_rate=0.0007,  # Slightly lower than CartPole
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_freq=8,  # Slightly longer rollouts
        seed=seed
    )

    # Train
    agent, returns, lengths = train_a2c(env, config)

    # Evaluate
    mean_return, std_return = evaluate_agent(env, agent, n_episodes=100, seed=seed + 1000)

    solved = mean_return >= 200
    env.close()

    return agent, returns, lengths, solved, mean_return, std_return


def main():
    print("=" * 70)
    print("LunarLander: Advantage Actor-Critic (A2C)")
    print("=" * 70)

    print("\n" + "-" * 40)
    print("Environment Info")
    print("-" * 40)
    print("  State: [x, y, vx, vy, angle, angular_vel, leg1, leg2]")
    print("  Actions: 0=nothing, 1=left, 2=main, 3=right")
    print("  Reward: -100 (crash) to +100 (perfect landing)")
    print("  Solved: avg return >= 200 over 100 episodes")

    print("\n" + "-" * 40)
    print("A2C Configuration")
    print("-" * 40)
    print("  Network: 128 → 128 shared backbone")
    print("  Learning rate: 0.0007")
    print("  GAE lambda: 0.95")
    print("  Update freq: Every 8 steps")

    print("\n" + "=" * 70)
    print("Training A2C")
    print("=" * 70)
    print("-" * 50)

    agent, returns, lengths, solved, mean, std = solve_lunarlander_a2c(
        n_episodes=1000, seed=42
    )

    print("-" * 50)
    print(f"\nFinal evaluation (100 episodes): {mean:.1f} +/- {std:.1f}")
    if solved:
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
    axes[0, 0].set_title('A2C Learning Curve on LunarLander')
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
        window_stats = min(100, len(entropies) // 5)
        if window_stats > 1:
            smoothed_ent = np.convolve(entropies, np.ones(window_stats)/window_stats, mode='valid')
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

    # Value loss
    if agent.training_stats:
        vlosses = [s['value_loss'] for s in agent.training_stats]
        if window_stats > 1:
            smoothed_vloss = np.convolve(vlosses, np.ones(window_stats)/window_stats, mode='valid')
            axes[1, 1].plot(smoothed_vloss, color='red', linewidth=1.5)
        else:
            axes[1, 1].plot(vlosses, color='red', linewidth=1.5)

    axes[1, 1].set_xlabel('Update Step')
    axes[1, 1].set_ylabel('Value Loss')
    axes[1, 1].set_title('Critic Loss')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'lunarlander_a2c.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
