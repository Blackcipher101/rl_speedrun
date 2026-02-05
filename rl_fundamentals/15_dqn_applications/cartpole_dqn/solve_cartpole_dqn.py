"""
CartPole Solver using DQN and Double DQN

Demonstrates DQN on CartPole-v1, the classic control benchmark.
Shows both standard DQN and Double DQN for comparison.

Solved: Average episode length >= 195 over 100 consecutive episodes.

Author: RL Speedrun
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
import sys

# Add parent directories for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '13_dqn_fundamentals'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '14_dqn_improvements'))

from dqn import DQNAgent, DQNConfig, train_dqn, evaluate_agent
from double_dqn import DoubleDQNAgent, DoubleDQNConfig, train_double_dqn


def solve_cartpole_dqn(use_double_dqn: bool = True,
                        n_episodes: int = 500,
                        seed: int = 42) -> Tuple[object, List[float], List[int], bool]:
    """
    Solve CartPole using DQN or Double DQN.

    Args:
        use_double_dqn: If True, use Double DQN; otherwise standard DQN
        n_episodes: Maximum training episodes
        seed: Random seed

    Returns:
        agent: Trained agent
        returns: Episode returns
        lengths: Episode lengths
        solved: Whether environment was solved
    """
    env = gym.make("CartPole-v1")

    # Optimized hyperparameters for CartPole
    config_class = DoubleDQNConfig if use_double_dqn else DQNConfig
    config = config_class(
        state_dim=4,
        action_dim=2,
        hidden_dims=[128, 128],  # Slightly larger for reliability
        n_episodes=n_episodes,
        max_steps=500,
        batch_size=64,
        learning_rate=0.001,
        gamma=0.99,
        buffer_size=50000,
        min_buffer_size=1000,
        target_update_freq=100,
        use_soft_update=False,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        seed=seed
    )

    # Train
    train_fn = train_double_dqn if use_double_dqn else train_dqn
    agent, returns, lengths = train_fn(env, config)

    # Evaluate
    mean_length, std_length = evaluate_agent(env, agent, n_episodes=100, seed=seed + 1000)

    solved = mean_length >= 195
    env.close()

    return agent, returns, lengths, solved, mean_length, std_length


def compare_algorithms(n_runs: int = 3, n_episodes: int = 400) -> Dict:
    """
    Compare DQN vs Double DQN on CartPole.

    Args:
        n_runs: Number of runs per algorithm
        n_episodes: Episodes per run

    Returns:
        results: Dictionary with comparison data
    """
    results = {
        'dqn': {'lengths': [], 'final_means': [], 'solved': []},
        'double_dqn': {'lengths': [], 'final_means': [], 'solved': []}
    }

    for run in range(n_runs):
        seed = 42 + run * 100
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{n_runs}")
        print(f"{'='*60}")

        # Standard DQN
        print("\n--- Standard DQN ---")
        _, _, lengths, solved, mean, std = solve_cartpole_dqn(
            use_double_dqn=False, n_episodes=n_episodes, seed=seed
        )
        results['dqn']['lengths'].append(lengths)
        results['dqn']['final_means'].append(mean)
        results['dqn']['solved'].append(solved)
        print(f"Final: {mean:.1f} +/- {std:.1f} | Solved: {solved}")

        # Double DQN
        print("\n--- Double DQN ---")
        _, _, lengths, solved, mean, std = solve_cartpole_dqn(
            use_double_dqn=True, n_episodes=n_episodes, seed=seed
        )
        results['double_dqn']['lengths'].append(lengths)
        results['double_dqn']['final_means'].append(mean)
        results['double_dqn']['solved'].append(solved)
        print(f"Final: {mean:.1f} +/- {std:.1f} | Solved: {solved}")

    return results


def visualize_comparison(results: Dict, save_path: str = None):
    """Create comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves (average across runs)
    colors = {'dqn': 'blue', 'double_dqn': 'green'}
    labels = {'dqn': 'Standard DQN', 'double_dqn': 'Double DQN'}

    for algo in ['dqn', 'double_dqn']:
        all_lengths = results[algo]['lengths']
        min_len = min(len(l) for l in all_lengths)
        truncated = [l[:min_len] for l in all_lengths]
        mean_lengths = np.mean(truncated, axis=0)
        std_lengths = np.std(truncated, axis=0)

        # Smooth
        window = 20
        smoothed_mean = np.convolve(mean_lengths, np.ones(window)/window, mode='valid')
        smoothed_std = np.convolve(std_lengths, np.ones(window)/window, mode='valid')
        x = np.arange(len(smoothed_mean))

        axes[0].plot(x, smoothed_mean, color=colors[algo], label=labels[algo], linewidth=2)
        axes[0].fill_between(x, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                             color=colors[algo], alpha=0.2)

    axes[0].axhline(y=195, color='red', linestyle='--', label='Solved (195)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Length')
    axes[0].set_title('CartPole: DQN vs Double DQN Learning Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Final performance bar chart
    dqn_means = results['dqn']['final_means']
    ddqn_means = results['double_dqn']['final_means']

    x = np.arange(2)
    means = [np.mean(dqn_means), np.mean(ddqn_means)]
    stds = [np.std(dqn_means), np.std(ddqn_means)]

    bars = axes[1].bar(x, means, yerr=stds, capsize=5,
                       color=[colors['dqn'], colors['double_dqn']], alpha=0.8)
    axes[1].axhline(y=195, color='red', linestyle='--', label='Solved threshold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Standard DQN', 'Double DQN'])
    axes[1].set_ylabel('Mean Episode Length')
    axes[1].set_title('Final Performance Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add solve rate text
    dqn_solve_rate = sum(results['dqn']['solved']) / len(results['dqn']['solved']) * 100
    ddqn_solve_rate = sum(results['double_dqn']['solved']) / len(results['double_dqn']['solved']) * 100
    axes[1].text(0, means[0] + stds[0] + 20, f'{dqn_solve_rate:.0f}% solved',
                 ha='center', fontsize=10)
    axes[1].text(1, means[1] + stds[1] + 20, f'{ddqn_solve_rate:.0f}% solved',
                 ha='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.show()


def main():
    print("=" * 70)
    print("CartPole: DQN and Double DQN")
    print("=" * 70)

    print("\n" + "-" * 40)
    print("Environment Info")
    print("-" * 40)
    print("  State: [cart_pos, cart_vel, pole_angle, pole_angular_vel]")
    print("  Actions: 0=push left, 1=push right")
    print("  Reward: +1 per timestep")
    print("  Solved: avg length >= 195 over 100 episodes")
    print("  Max episode length: 500")

    # Single run demonstration
    print("\n" + "=" * 70)
    print("Training Double DQN (Single Run)")
    print("=" * 70)

    agent, returns, lengths, solved, mean, std = solve_cartpole_dqn(
        use_double_dqn=True, n_episodes=400, seed=42
    )

    print(f"\nFinal evaluation: {mean:.1f} +/- {std:.1f}")
    if solved:
        print("SOLVED!")
    else:
        print(f"Not solved (need >= 195)")

    # Quick visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 20
    smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='green', linewidth=1.5, label='Double DQN')
    ax.axhline(y=195, color='red', linestyle='--', label='Solved (195)')
    ax.axhline(y=500, color='gray', linestyle=':', alpha=0.5, label='Max (500)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length')
    ax.set_title('CartPole: Double DQN Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'cartpole_dqn.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nLearning curve saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
