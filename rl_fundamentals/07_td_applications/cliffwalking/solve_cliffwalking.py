"""
CliffWalking: Q-Learning vs SARSA Comparison

This script demonstrates the key difference between off-policy (Q-Learning)
and on-policy (SARSA) temporal difference learning.

Q-Learning: Learns optimal path (risky, along cliff)
SARSA: Learns safe path (accounts for exploration mistakes)

Author: RL Speedrun
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def epsilon_greedy_action(
    Q: np.ndarray,
    state: int,
    epsilon: float,
    n_actions: int,
    rng: np.random.Generator
) -> int:
    """Select action using ε-greedy policy."""
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    else:
        q_values = Q[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return rng.choice(best_actions)


def q_learning(
    env,
    n_episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, List[float]]:
    """
    Q-Learning (off-policy TD control).

    Uses max_a Q(S', a) regardless of actual next action.
    """
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    episode_returns = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        truncated = False
        episode_return = 0.0

        while not done and not truncated:
            action = epsilon_greedy_action(Q, state, epsilon, n_actions, rng)
            next_state, reward, done, truncated, _ = env.step(action)

            # Q-Learning: use max over next state
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            episode_return += reward

        episode_returns.append(episode_return)

    return Q, episode_returns


def sarsa(
    env,
    n_episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, List[float]]:
    """
    SARSA (on-policy TD control).

    Uses Q(S', A') where A' is the actual next action from ε-greedy.
    """
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    episode_returns = []

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        truncated = False
        episode_return = 0.0
        action = epsilon_greedy_action(Q, state, epsilon, n_actions, rng)

        while not done and not truncated:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = epsilon_greedy_action(Q, next_state, epsilon, n_actions, rng)

            # SARSA: use actual next action
            td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            action = next_action
            episode_return += reward

        episode_returns.append(episode_return)

    return Q, episode_returns


def extract_policy(Q: np.ndarray) -> np.ndarray:
    """Extract greedy policy from Q-table."""
    return np.argmax(Q, axis=1)


def visualize_policy_grid(Q: np.ndarray, title: str, ax: plt.Axes):
    """Visualize policy on 4x12 CliffWalking grid."""
    policy = extract_policy(Q)
    V = np.max(Q, axis=1).reshape(4, 12)

    # Action symbols: 0=up, 1=right, 2=down, 3=left
    action_symbols = ['↑', '→', '↓', '←']

    # Plot value function
    im = ax.imshow(V, cmap='RdYlGn', aspect='auto')

    # Add policy arrows and labels
    for row in range(4):
        for col in range(12):
            state = row * 12 + col
            if row == 3 and col == 0:  # Start
                ax.text(col, row, 'S', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='blue')
            elif row == 3 and col == 11:  # Goal
                ax.text(col, row, 'G', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='green')
            elif row == 3 and 1 <= col <= 10:  # Cliff
                ax.text(col, row, 'C', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='red')
            else:
                ax.text(col, row, action_symbols[policy[state]],
                       ha='center', va='center', fontsize=14, color='black')

    ax.set_title(title)
    ax.set_xticks(range(12))
    ax.set_yticks(range(4))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    return im


def smooth_curve(data: List[float], window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    return np.convolve(data, np.ones(window) / window, mode='valid')


def main():
    print("=" * 70)
    print("CliffWalking: Q-Learning vs SARSA Comparison")
    print("=" * 70)

    # Create environment
    env = gym.make("CliffWalking-v0")

    # Hyperparameters (from Sutton & Barto)
    config = {
        "n_episodes": 500,
        "alpha": 0.5,
        "gamma": 1.0,  # No discounting for this episodic task
        "epsilon": 0.1,
        "seed": 42
    }

    print("\nEnvironment: 4x12 grid with cliff")
    print("  Start: bottom-left (state 36)")
    print("  Goal: bottom-right (state 47)")
    print("  Cliff: bottom row, states 37-46")
    print("\nHyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Train both algorithms
    print("\nTraining Q-Learning...")
    Q_qlearning, returns_qlearning = q_learning(env, **config)

    print("Training SARSA...")
    Q_sarsa, returns_sarsa = sarsa(env, **config)

    # Evaluate with greedy policy (epsilon=0)
    def evaluate(Q, name, n_eval=100):
        returns = []
        for ep in range(n_eval):
            state, _ = env.reset(seed=1000 + ep)
            done = False
            truncated = False
            ret = 0
            while not done and not truncated:
                action = np.argmax(Q[state])
                state, reward, done, truncated, _ = env.step(action)
                ret += reward
            returns.append(ret)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        print(f"  {name:12s} - Greedy evaluation: {mean_ret:.1f} ± {std_ret:.1f}")
        return mean_ret, std_ret

    print("\nEvaluation (greedy policy, ε=0):")
    evaluate(Q_qlearning, "Q-Learning")
    evaluate(Q_sarsa, "SARSA")

    # Compute training statistics
    print("\nTraining statistics (with ε=0.1 exploration):")
    print(f"  Q-Learning - Mean return: {np.mean(returns_qlearning):.1f} ± {np.std(returns_qlearning):.1f}")
    print(f"  SARSA      - Mean return: {np.mean(returns_sarsa):.1f} ± {np.std(returns_sarsa):.1f}")

    # Create visualization
    fig = plt.figure(figsize=(14, 10))

    # Learning curves
    ax1 = plt.subplot(2, 2, 1)
    window = 10
    smooth_q = smooth_curve(returns_qlearning, window)
    smooth_s = smooth_curve(returns_sarsa, window)

    ax1.plot(smooth_q, label='Q-Learning', color='blue', alpha=0.8)
    ax1.plot(smooth_s, label='SARSA', color='green', alpha=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return (smoothed)')
    ax1.set_title('Learning Curves: Q-Learning vs SARSA')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-150, 0])

    # Sum of rewards per episode (cumulative performance)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(np.cumsum(returns_qlearning), label='Q-Learning', color='blue')
    ax2.plot(np.cumsum(returns_sarsa), label='SARSA', color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Return')
    ax2.set_title('Cumulative Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Q-Learning policy
    ax3 = plt.subplot(2, 2, 3)
    im = visualize_policy_grid(Q_qlearning, 'Q-Learning Policy (Risky Optimal Path)', ax3)
    plt.colorbar(im, ax=ax3, label='V(s)')

    # SARSA policy
    ax4 = plt.subplot(2, 2, 4)
    im = visualize_policy_grid(Q_sarsa, 'SARSA Policy (Safe Path)', ax4)
    plt.colorbar(im, ax=ax4, label='V(s)')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'cliffwalking_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()

    # Print policy analysis
    print("\n" + "=" * 70)
    print("Policy Analysis")
    print("=" * 70)

    policy_q = extract_policy(Q_qlearning)
    policy_s = extract_policy(Q_sarsa)

    action_names = ['up', 'right', 'down', 'left']

    print("\nQ-Learning path from start (state 36):")
    state = 36
    path = [state]
    for _ in range(20):
        action = policy_q[state]
        row, col = state // 12, state % 12
        if action == 0: row = max(0, row - 1)
        elif action == 1: col = min(11, col + 1)
        elif action == 2: row = min(3, row + 1)
        elif action == 3: col = max(0, col - 1)
        state = row * 12 + col
        path.append(state)
        if state == 47:
            break
    print(f"  Path: {path[:15]}...")
    print(f"  Length: {len(path) - 1} steps")

    print("\nSARSA path from start (state 36):")
    state = 36
    path = [state]
    for _ in range(30):
        action = policy_s[state]
        row, col = state // 12, state % 12
        if action == 0: row = max(0, row - 1)
        elif action == 1: col = min(11, col + 1)
        elif action == 2: row = min(3, row + 1)
        elif action == 3: col = max(0, col - 1)
        state = row * 12 + col
        path.append(state)
        if state == 47:
            break
    print(f"  Path: {path[:20]}...")
    print(f"  Length: {len(path) - 1} steps")

    print("\nKey Insight:")
    print("  Q-Learning: Takes the optimal but risky path along the cliff edge")
    print("  SARSA: Takes a longer but safer path away from the cliff")
    print("  This difference arises because SARSA accounts for exploration mistakes!")

    env.close()


if __name__ == "__main__":
    main()
