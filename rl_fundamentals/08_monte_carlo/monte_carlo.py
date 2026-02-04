"""
Monte Carlo Methods for Reinforcement Learning

Monte Carlo methods learn from complete episodes without bootstrapping.
They use actual returns rather than estimated returns.

Methods implemented:
- First-Visit MC Prediction
- Every-Visit MC Prediction
- MC Control with ε-greedy (On-Policy)
- MC Control with Exploring Starts

Author: RL Speedrun
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from collections import defaultdict


def generate_episode(
    env,
    policy: Callable[[int], int],
    max_steps: int = 1000
) -> List[Tuple[int, int, float]]:
    """
    Generate a complete episode following a policy.

    Args:
        env: Gymnasium environment
        policy: Function that maps state to action
        max_steps: Maximum steps to prevent infinite episodes

    Returns:
        episode: List of (state, action, reward) tuples
    """
    episode = []
    state, _ = env.reset()

    for _ in range(max_steps):
        action = policy(state)
        next_state, reward, done, truncated, _ = env.step(action)
        episode.append((state, action, reward))

        if done or truncated:
            break
        state = next_state

    return episode


def first_visit_mc_prediction(
    env,
    policy: Callable[[int], int],
    n_episodes: int = 10000,
    gamma: float = 1.0,
    alpha: Optional[float] = None
) -> Tuple[np.ndarray, List[float]]:
    """
    First-Visit Monte Carlo Prediction.

    Estimates V(s) using only the first visit to each state in an episode.

    Args:
        env: Gymnasium environment
        policy: Policy function (state -> action)
        n_episodes: Number of episodes to sample
        gamma: Discount factor
        alpha: Learning rate (None for exact averaging)

    Returns:
        V: Estimated value function
        episode_returns: Return of each episode
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    N = np.zeros(n_states)  # Visit counts
    episode_returns = []

    for _ in range(n_episodes):
        episode = generate_episode(env, policy)

        # Calculate returns backwards
        G = 0
        visited_states = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            # First-visit check
            if state not in visited_states:
                visited_states.add(state)
                N[state] += 1

                if alpha is None:
                    # Exact incremental mean
                    V[state] += (G - V[state]) / N[state]
                else:
                    # Constant learning rate
                    V[state] += alpha * (G - V[state])

        episode_returns.append(G)

    return V, episode_returns


def every_visit_mc_prediction(
    env,
    policy: Callable[[int], int],
    n_episodes: int = 10000,
    gamma: float = 1.0,
    alpha: Optional[float] = None
) -> Tuple[np.ndarray, List[float]]:
    """
    Every-Visit Monte Carlo Prediction.

    Estimates V(s) using all visits to each state in an episode.

    Args:
        env: Gymnasium environment
        policy: Policy function (state -> action)
        n_episodes: Number of episodes to sample
        gamma: Discount factor
        alpha: Learning rate (None for exact averaging)

    Returns:
        V: Estimated value function
        episode_returns: Return of each episode
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    N = np.zeros(n_states)  # Visit counts
    episode_returns = []

    for _ in range(n_episodes):
        episode = generate_episode(env, policy)

        # Calculate returns backwards
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            # Every visit: no check needed
            N[state] += 1

            if alpha is None:
                V[state] += (G - V[state]) / N[state]
            else:
                V[state] += alpha * (G - V[state])

        episode_returns.append(G)

    return V, episode_returns


def mc_control_epsilon_greedy(
    env,
    n_episodes: int = 10000,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    epsilon_decay: float = 1.0,
    epsilon_min: float = 0.01,
    alpha: Optional[float] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    On-Policy First-Visit MC Control with ε-greedy policy.

    Args:
        env: Gymnasium environment
        n_episodes: Number of episodes
        gamma: Discount factor
        epsilon: Exploration rate
        epsilon_decay: Decay factor for epsilon
        epsilon_min: Minimum epsilon
        alpha: Learning rate (None for exact averaging)
        seed: Random seed

    Returns:
        Q: Action-value function
        policy: Greedy policy derived from Q
        episode_returns: Return of each episode
    """
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions))  # Visit counts
    episode_returns = []

    def epsilon_greedy_policy(state: int) -> int:
        if rng.random() < epsilon:
            return rng.integers(n_actions)
        return np.argmax(Q[state])

    current_epsilon = epsilon

    for ep in range(n_episodes):
        # Generate episode with current ε-greedy policy
        episode = []
        state, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False

        while not done and not truncated:
            if rng.random() < current_epsilon:
                action = rng.integers(n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Calculate returns and update Q
        G = 0
        visited_pairs = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            # First-visit check
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                N[state, action] += 1

                if alpha is None:
                    Q[state, action] += (G - Q[state, action]) / N[state, action]
                else:
                    Q[state, action] += alpha * (G - Q[state, action])

        episode_returns.append(G)

        # Decay epsilon
        current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)

    # Extract greedy policy
    policy = np.argmax(Q, axis=1)

    return Q, policy, episode_returns


def mc_control_exploring_starts(
    env,
    n_episodes: int = 10000,
    gamma: float = 1.0,
    alpha: Optional[float] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Monte Carlo Control with Exploring Starts.

    Assumes episodes can start from any state-action pair.
    Note: This requires environment support for setting initial state.

    Args:
        env: Gymnasium environment (must support reset with options)
        n_episodes: Number of episodes
        gamma: Discount factor
        alpha: Learning rate (None for exact averaging)
        seed: Random seed

    Returns:
        Q: Action-value function
        policy: Greedy policy derived from Q
        episode_returns: Return of each episode
    """
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions))
    policy = rng.integers(n_actions, size=n_states)  # Random initial policy
    episode_returns = []

    for ep in range(n_episodes):
        # Exploring starts: random initial state and action
        # Note: Most gym envs don't support this, so we use normal reset
        # and take a random first action
        state, _ = env.reset(seed=seed + ep)
        first_action = rng.integers(n_actions)

        # Generate episode
        episode = []
        next_state, reward, done, truncated, _ = env.step(first_action)
        episode.append((state, first_action, reward))
        state = next_state

        while not done and not truncated:
            action = policy[state]  # Follow greedy policy after first action
            next_state, reward, done, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Calculate returns and update Q
        G = 0
        visited_pairs = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                N[state, action] += 1

                if alpha is None:
                    Q[state, action] += (G - Q[state, action]) / N[state, action]
                else:
                    Q[state, action] += alpha * (G - Q[state, action])

                # Policy improvement
                policy[state] = np.argmax(Q[state])

        episode_returns.append(G)

    return Q, policy, episode_returns


def evaluate_policy(
    env,
    policy: np.ndarray,
    n_episodes: int = 100,
    seed: int = 0
) -> Tuple[float, float]:
    """
    Evaluate a deterministic policy.

    Args:
        env: Gymnasium environment
        policy: Array mapping states to actions
        n_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        mean_return: Average return
        std_return: Standard deviation of returns
    """
    returns = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        episode_return = 0.0

        while not done and not truncated:
            action = policy[state]
            state, reward, done, truncated, _ = env.step(action)
            episode_return += reward

        returns.append(episode_return)

    return np.mean(returns), np.std(returns)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    import gymnasium as gym
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Monte Carlo Methods Demo: FrozenLake-v1")
    print("=" * 60)

    # Create environment
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    # Train with MC Control
    print("\nTraining MC Control (ε-greedy)...")
    Q, policy, returns = mc_control_epsilon_greedy(
        env,
        n_episodes=50000,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9999,
        epsilon_min=0.01,
        seed=42
    )

    # Evaluate
    print("\nEvaluating learned policy...")
    mean_ret, std_ret = evaluate_policy(env, policy, n_episodes=1000, seed=0)
    print(f"  Mean return: {mean_ret:.3f} ± {std_ret:.3f}")

    # Display policy
    action_symbols = ['←', '↓', '→', '↑']
    print("\nLearned Policy:")
    print("-" * 20)
    for row in range(4):
        row_str = ""
        for col in range(4):
            state = row * 4 + col
            if state in [5, 7, 11, 12]:
                row_str += " H "
            elif state == 15:
                row_str += " G "
            else:
                row_str += f" {action_symbols[policy[state]]} "
        print(row_str)
    print("-" * 20)

    # Plot learning curve
    window = 1000
    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 4))
    plt.plot(smoothed, color='purple', linewidth=1)
    plt.xlabel('Episode')
    plt.ylabel('Return (smoothed)')
    plt.title('MC Control Learning Curve (FrozenLake)')
    plt.axhline(y=mean_ret, color='red', linestyle='--', label=f'Final: {mean_ret:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mc_frozenlake.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to: mc_frozenlake.png")
    plt.show()

    env.close()
