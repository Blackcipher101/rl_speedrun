"""
CartPole Solver using A2C

Demonstrates Advantage Actor-Critic on CartPole-v1.
Compares learning characteristics with DQN and REINFORCE.

Solved: Average episode length >= 195 over 100 consecutive episodes.

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


def solve_cartpole_a2c(
    n_episodes: int = 500,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[A2CAgent, List[float], List[int], bool, float, float]:
    """
    Solve CartPole using A2C.

    Args:
        n_episodes: Maximum training episodes
        seed: Random seed
        verbose: Print progress

    Returns:
        agent: Trained agent
        returns: Episode returns
        lengths: Episode lengths
        solved: Whether environment was solved
        mean_return: Final evaluation mean
        std_return: Final evaluation std
    """
    env = gym.make("CartPole-v1")

    # Optimized hyperparameters for CartPole
    config = A2CConfig(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64, 64],
        n_episodes=n_episodes,
        max_steps=500,
        learning_rate=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_freq=5,  # Update every 5 steps
        seed=seed
    )

    # Train
    agent, returns, lengths = train_a2c(env, config)

    # Evaluate
    mean_return, std_return = evaluate_agent(env, agent, n_episodes=100, seed=seed + 1000)

    solved = mean_return >= 195
    env.close()

    return agent, returns, lengths, solved, mean_return, std_return


def compare_a2c_configurations(n_runs: int = 3) -> dict:
    """
    Compare different A2C configurations.

    Tests:
    - Different GAE lambda values
    - Different entropy coefficients
    - Different update frequencies
    """
    results = {}

    configurations = {
        'baseline': {'gae_lambda': 0.95, 'entropy_coef': 0.01, 'update_freq': 5},
        'low_lambda': {'gae_lambda': 0.5, 'entropy_coef': 0.01, 'update_freq': 5},
        'high_entropy': {'gae_lambda': 0.95, 'entropy_coef': 0.05, 'update_freq': 5},
        'frequent_update': {'gae_lambda': 0.95, 'entropy_coef': 0.01, 'update_freq': 1},
    }

    env = gym.make("CartPole-v1")

    for name, params in configurations.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {name}")
        print(f"Params: {params}")
        print(f"{'='*60}")

        config_results = {'lengths': [], 'final_means': []}

        for run in range(n_runs):
            seed = 42 + run * 100
            print(f"\n--- Run {run + 1}/{n_runs} ---")

            config = A2CConfig(
                state_dim=4,
                action_dim=2,
                hidden_dims=[64, 64],
                n_episodes=400,
                seed=seed,
                **params
            )

            agent, _, lengths = train_a2c(env, config)
            mean, std = evaluate_agent(env, agent, n_episodes=50, seed=seed + 1000)

            config_results['lengths'].append(lengths)
            config_results['final_means'].append(mean)

            print(f"Final: {mean:.1f} +/- {std:.1f}")

        results[name] = config_results
        print(f"\n{name} average: {np.mean(config_results['final_means']):.1f}")

    env.close()
    return results


def visualize_training(agent: A2CAgent, lengths: List[int], returns: List[float],
                       save_path: str = None):
    """Create comprehensive training visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    window = 20

    # Learning curve (episode length)
    smoothed_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(smoothed_lengths, color='blue', linewidth=1.5)
    axes[0, 0].axhline(y=195, color='green', linestyle='--', label='Solved (195)')
    axes[0, 0].axhline(y=500, color='red', linestyle=':', alpha=0.5, label='Max (500)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Length')
    axes[0, 0].set_title('A2C Learning Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Episode returns
    smoothed_returns = np.convolve(returns, np.ones(window)/window, mode='valid')
    axes[0, 1].plot(smoothed_returns, color='green', linewidth=1.5)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Return')
    axes[0, 1].set_title('Episode Returns')
    axes[0, 1].grid(True, alpha=0.3)

    # Entropy over training
    if agent.training_stats:
        entropies = [s['entropy'] for s in agent.training_stats]
        window_stats = min(100, len(entropies) // 5)

        if window_stats > 1:
            smoothed_ent = np.convolve(entropies, np.ones(window_stats)/window_stats, mode='valid')
            axes[1, 0].plot(smoothed_ent, color='purple', linewidth=1.5)
        else:
            axes[1, 0].plot(entropies, color='purple', linewidth=1.5)

        axes[1, 0].axhline(y=np.log(2), color='gray', linestyle='--',
                           alpha=0.5, label='Max entropy (log 2)')

    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Policy Entropy')
    axes[1, 0].set_title('Policy Entropy (Should Decrease)')
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
    axes[1, 1].set_title('Critic Loss (Should Decrease)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.show()


def main():
    print("=" * 70)
    print("CartPole: Advantage Actor-Critic (A2C)")
    print("=" * 70)

    print("\n" + "-" * 40)
    print("Environment Info")
    print("-" * 40)
    print("  State: [cart_pos, cart_vel, pole_angle, pole_angular_vel]")
    print("  Actions: 0=push left, 1=push right")
    print("  Reward: +1 per timestep")
    print("  Solved: avg length >= 195 over 100 episodes")

    print("\n" + "-" * 40)
    print("A2C Configuration")
    print("-" * 40)
    print("  Actor-Critic: Shared 64→64 backbone")
    print("  GAE lambda: 0.95")
    print("  Entropy coef: 0.01")
    print("  Value coef: 0.5")
    print("  Update freq: Every 5 steps")

    print("\n" + "=" * 70)
    print("Training A2C")
    print("=" * 70)
    print("-" * 50)

    agent, returns, lengths, solved, mean, std = solve_cartpole_a2c(
        n_episodes=500, seed=42
    )

    print("-" * 50)
    print(f"\nFinal evaluation (100 episodes): {mean:.1f} +/- {std:.1f}")
    if solved:
        print("SOLVED!")
    else:
        print(f"Not solved (need >= 195)")

    # Find episode where it first solved
    window = 100
    for i in range(len(lengths) - window):
        if np.mean(lengths[i:i+window]) >= 195:
            print(f"First solved at episode {i + window}")
            break

    # Visualize
    save_path = os.path.join(os.path.dirname(__file__), 'cartpole_a2c.png')
    visualize_training(agent, lengths, returns, save_path)


if __name__ == "__main__":
    main()
