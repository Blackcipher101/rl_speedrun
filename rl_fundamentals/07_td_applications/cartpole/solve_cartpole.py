"""
CartPole: Q-Learning with State Discretization

Demonstrates how to apply tabular Q-Learning to continuous state spaces
by discretizing the state space into bins.

State space: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
Action space: {0: push left, 1: push right}

Solved when average episode length >= 195 over 100 episodes.

Author: RL Speedrun
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os


class StateDiscretizer:
    """
    Discretizes continuous state space into bins.

    For CartPole:
        - Cart position: [-2.4, 2.4]
        - Cart velocity: [-3.0, 3.0] (clipped)
        - Pole angle: [-0.25, 0.25] radians
        - Pole angular velocity: [-3.0, 3.0] (clipped)
    """

    def __init__(
        self,
        n_bins: Tuple[int, int, int, int] = (6, 6, 12, 12),
        bounds: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Initialize discretizer.

        Args:
            n_bins: Number of bins for each state dimension
            bounds: (min, max) bounds for each dimension
        """
        self.n_bins = n_bins

        # Default bounds for CartPole (practical ranges)
        if bounds is None:
            bounds = [
                (-2.4, 2.4),      # Cart position (actual termination bounds)
                (-3.0, 3.0),      # Cart velocity (practical range)
                (-0.25, 0.25),    # Pole angle in radians (~14 degrees)
                (-3.5, 3.5)       # Pole angular velocity (practical range)
            ]
        self.bounds = bounds

        # Create bin edges for each dimension
        self.bins = []
        for i, (low, high) in enumerate(bounds):
            edges = np.linspace(low, high, n_bins[i] + 1)
            self.bins.append(edges)

        # Total number of discrete states
        self.n_states = np.prod(n_bins)

    def discretize(self, observation: np.ndarray) -> int:
        """
        Convert continuous observation to discrete state index.

        Args:
            observation: Continuous state [x, x_dot, theta, theta_dot]

        Returns:
            Discrete state index (integer)
        """
        indices = []
        for i, (obs_val, bin_edges) in enumerate(zip(observation, self.bins)):
            # Clip to bounds
            clipped = np.clip(obs_val, bin_edges[0], bin_edges[-1])
            # Find bin index
            idx = np.digitize(clipped, bin_edges) - 1
            # Ensure valid index
            idx = np.clip(idx, 0, self.n_bins[i] - 1)
            indices.append(idx)

        # Convert multi-index to flat index
        state = 0
        multiplier = 1
        for idx, n_bin in zip(reversed(indices), reversed(self.n_bins)):
            state += idx * multiplier
            multiplier *= n_bin

        return int(state)


def q_learning_cartpole(
    env,
    discretizer: StateDiscretizer,
    n_episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    seed: int = 42
) -> Tuple[np.ndarray, List[int], List[float]]:
    """
    Q-Learning with state discretization for CartPole.

    Args:
        env: Gymnasium CartPole environment
        discretizer: StateDiscretizer instance
        n_episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Epsilon decay per episode
        seed: Random seed

    Returns:
        Q: Learned Q-table
        episode_lengths: Steps per episode
        epsilons: Epsilon values over training
    """
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n

    # Initialize Q-table
    Q = np.zeros((discretizer.n_states, n_actions))

    # Tracking
    episode_lengths = []
    epsilons = []
    epsilon = epsilon_start

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = discretizer.discretize(obs)
        done = False
        truncated = False
        steps = 0

        while not done and not truncated:
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = rng.integers(n_actions)
            else:
                action = np.argmax(Q[state])

            # Take action
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = discretizer.discretize(next_obs)

            # Q-Learning update
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            steps += 1

        episode_lengths.append(steps)
        epsilons.append(epsilon)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Progress logging
        if (episode + 1) % 100 == 0:
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode + 1:5d}: Avg length (last 100) = {avg_length:.1f}, ε = {epsilon:.3f}")

    return Q, episode_lengths, epsilons


def evaluate_policy(
    env,
    Q: np.ndarray,
    discretizer: StateDiscretizer,
    n_episodes: int = 100,
    seed: int = 0
) -> Tuple[float, float]:
    """
    Evaluate learned policy (greedy, no exploration).

    Returns:
        mean_length: Average episode length
        std_length: Standard deviation of lengths
    """
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = discretizer.discretize(obs)
        done = False
        truncated = False
        steps = 0

        while not done and not truncated:
            action = np.argmax(Q[state])
            obs, _, done, truncated, _ = env.step(action)
            state = discretizer.discretize(obs)
            steps += 1

        lengths.append(steps)

    return np.mean(lengths), np.std(lengths)


def main():
    print("=" * 70)
    print("CartPole: Q-Learning with State Discretization")
    print("=" * 70)

    # Create environment
    env = gym.make("CartPole-v1")

    # Create discretizer
    n_bins = (6, 6, 12, 12)
    discretizer = StateDiscretizer(n_bins=n_bins)

    print(f"\nState discretization:")
    print(f"  Bins per dimension: {n_bins}")
    print(f"  Total discrete states: {discretizer.n_states:,}")
    print(f"  Actions: 2 (left, right)")
    print(f"  Q-table size: {discretizer.n_states} × 2 = {discretizer.n_states * 2:,} entries")

    # Hyperparameters
    config = {
        "n_episodes": 2000,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "seed": 42
    }

    print("\nHyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Train
    print("\nTraining...")
    print("-" * 50)
    Q, lengths, epsilons = q_learning_cartpole(env, discretizer, **config)

    # Evaluate
    print("-" * 50)
    print("\nEvaluating learned policy (greedy, ε=0)...")
    mean_len, std_len = evaluate_policy(env, Q, discretizer, n_episodes=100, seed=0)
    print(f"  Mean episode length: {mean_len:.1f} ± {std_len:.1f}")

    # Check if solved
    solved = mean_len >= 195
    if solved:
        print("\n✓ SOLVED! Average length >= 195")
    else:
        print(f"\n✗ Not yet solved (need >= 195, got {mean_len:.1f})")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Learning curve
    ax1 = axes[0, 0]
    window = 50
    smoothed = np.convolve(lengths, np.ones(window) / window, mode='valid')
    ax1.plot(smoothed, color='blue', linewidth=1)
    ax1.axhline(y=195, color='green', linestyle='--', label='Solved threshold (195)')
    ax1.axhline(y=500, color='red', linestyle=':', alpha=0.5, label='Max length (500)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length (smoothed)')
    ax1.set_title('Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Epsilon decay
    ax2 = axes[0, 1]
    ax2.plot(epsilons, color='orange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate Decay')
    ax2.grid(True, alpha=0.3)

    # 3. Episode length histogram
    ax3 = axes[1, 0]
    ax3.hist(lengths[-500:], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=np.mean(lengths[-500:]), color='red', linestyle='--',
                label=f'Mean: {np.mean(lengths[-500:]):.1f}')
    ax3.set_xlabel('Episode Length')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Episode Length Distribution (Last 500 Episodes)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Rolling average
    ax4 = axes[1, 1]
    rolling_100 = [np.mean(lengths[max(0, i-99):i+1]) for i in range(len(lengths))]
    ax4.plot(rolling_100, color='purple', linewidth=1)
    ax4.axhline(y=195, color='green', linestyle='--', label='Solved threshold')
    ax4.fill_between(range(len(rolling_100)),
                     [max(0, r-20) for r in rolling_100],
                     [min(500, r+20) for r in rolling_100],
                     alpha=0.2, color='purple')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Length (100 episodes)')
    ax4.set_title('Rolling Average (Window=100)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'cartpole_learning.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()

    # Print some Q-value statistics
    print("\nQ-table statistics:")
    print(f"  Min Q-value: {Q.min():.2f}")
    print(f"  Max Q-value: {Q.max():.2f}")
    print(f"  Mean Q-value: {Q.mean():.2f}")
    print(f"  Non-zero entries: {np.count_nonzero(Q)} / {Q.size} ({100*np.count_nonzero(Q)/Q.size:.1f}%)")

    env.close()


if __name__ == "__main__":
    main()
