"""
SARSA: On-Policy Temporal Difference Control

SARSA (State-Action-Reward-State-Action) learns the action-value function
for the policy being followed, including exploration behavior.

Update Rule:
    Q(S_t, A_t) <- Q(S_t, A_t) + α[R_{t+1} + γ·Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

Key Property: On-policy
    - Behavior policy = Target policy (both ε-greedy)
    - Uses actual next action A' (not max)
    - Converges to Q^π (value of policy with exploration)

Name origin: Uses the quintuple (S, A, R, S', A')

Author: RL Speedrun
"""

import numpy as np
from typing import Tuple, List, Optional


def epsilon_greedy_action(
    Q: np.ndarray,
    state: int,
    epsilon: float,
    n_actions: int,
    rng: np.random.Generator
) -> int:
    """
    Select action using ε-greedy policy.

    With probability ε: random action (exploration)
    With probability 1-ε: greedy action (exploitation)

    Args:
        Q: Q-table of shape (n_states, n_actions)
        state: Current state index
        epsilon: Exploration probability
        n_actions: Number of possible actions
        rng: NumPy random generator for reproducibility

    Returns:
        Selected action index
    """
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    else:
        # Greedy: break ties randomly
        q_values = Q[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return rng.choice(best_actions)


def sarsa(
    env,
    n_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, List[float], List[int]]:
    """
    Tabular SARSA algorithm (on-policy TD control).

    Algorithm:
        1. Initialize Q(s, a) = 0 for all s, a
        2. For each episode:
            - Initialize S
            - Choose A from S using ε-greedy
            - For each step until terminal:
                - Take action A, observe R, S'
                - Choose A' from S' using ε-greedy
                - Q(S,A) <- Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
                - S <- S', A <- A'
            - Decay ε

    Key difference from Q-Learning:
        SARSA uses Q(S', A') where A' is the actual next action chosen
        Q-Learning uses max_a Q(S', a) regardless of what action is taken

    Args:
        env: Gymnasium environment with discrete state/action spaces
        n_episodes: Number of training episodes
        alpha: Learning rate (step size), α ∈ (0, 1]
        gamma: Discount factor, γ ∈ [0, 1]
        epsilon_start: Initial exploration rate
        epsilon_end: Minimum exploration rate
        epsilon_decay: Multiplicative decay factor per episode
        seed: Random seed for reproducibility

    Returns:
        Q: Learned Q-table of shape (n_states, n_actions)
        episode_returns: Total return per episode
        episode_lengths: Number of steps per episode
    """
    # Initialize
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Q-table initialized to zeros
    Q = np.zeros((n_states, n_actions))

    # Tracking
    episode_returns = []
    episode_lengths = []
    epsilon = epsilon_start

    for episode in range(n_episodes):
        # Reset environment
        state, _ = env.reset(seed=seed + episode if seed else None)
        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0

        # Choose initial action using ε-greedy
        action = epsilon_greedy_action(Q, state, epsilon, n_actions, rng)

        while not done and not truncated:
            # Take action, observe next state and reward
            next_state, reward, done, truncated, _ = env.step(action)

            # Choose next action using ε-greedy (SARSA is on-policy)
            next_action = epsilon_greedy_action(Q, next_state, epsilon, n_actions, rng)

            # SARSA update: use actual next action A'
            # This is the key difference from Q-Learning
            td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            # Transition to next state and action
            state = next_state
            action = next_action
            episode_return += reward
            episode_length += 1

        # Track episode statistics
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return Q, episode_returns, episode_lengths


def extract_policy(Q: np.ndarray) -> np.ndarray:
    """
    Extract greedy policy from Q-table.

    π(s) = argmax_a Q(s, a)

    Args:
        Q: Q-table of shape (n_states, n_actions)

    Returns:
        policy: Array of shape (n_states,) with optimal action per state
    """
    return np.argmax(Q, axis=1)


def evaluate_policy(
    env,
    Q: np.ndarray,
    n_episodes: int = 100,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Evaluate learned Q-table using greedy policy.

    Args:
        env: Gymnasium environment
        Q: Learned Q-table
        n_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        mean_return: Average return over episodes
        std_return: Standard deviation of returns
    """
    returns = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep if seed else None)
        done = False
        truncated = False
        episode_return = 0.0

        while not done and not truncated:
            # Greedy action selection
            action = np.argmax(Q[state])
            state, reward, done, truncated, _ = env.step(action)
            episode_return += reward

        returns.append(episode_return)

    return np.mean(returns), np.std(returns)


# =============================================================================
# Demo: SARSA on FrozenLake
# =============================================================================

if __name__ == "__main__":
    import gymnasium as gym
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("SARSA Demo: FrozenLake-v1")
    print("=" * 60)

    # Create environment
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    # Hyperparameters
    config = {
        "n_episodes": 10000,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.9995,
        "seed": 42
    }

    print("\nHyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Train
    print("\nTraining SARSA...")
    Q, returns, lengths = sarsa(env, **config)

    # Evaluate
    print("\nEvaluating learned policy...")
    mean_ret, std_ret = evaluate_policy(env, Q, n_episodes=100, seed=0)
    print(f"  Mean return: {mean_ret:.3f} ± {std_ret:.3f}")

    # Extract and display policy
    policy = extract_policy(Q)
    action_symbols = ['←', '↓', '→', '↑']

    print("\nLearned Policy (4x4 grid):")
    print("-" * 20)
    for row in range(4):
        row_str = ""
        for col in range(4):
            state = row * 4 + col
            if state in [5, 7, 11, 12]:  # Holes
                row_str += " H "
            elif state == 15:  # Goal
                row_str += " G "
            else:
                row_str += f" {action_symbols[policy[state]]} "
        print(row_str)
    print("-" * 20)

    # Display Q-values for state 0
    print("\nQ-values for state 0 (top-left):")
    for a, symbol in enumerate(action_symbols):
        print(f"  {symbol}: {Q[0, a]:.4f}")

    # Plot learning curve
    window_size = 100
    smoothed_returns = np.convolve(
        returns,
        np.ones(window_size) / window_size,
        mode='valid'
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Learning curve
    axes[0].plot(smoothed_returns, color='green', linewidth=1)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return (smoothed)')
    axes[0].set_title('SARSA on FrozenLake-v1')
    axes[0].axhline(y=mean_ret, color='red', linestyle='--',
                    label=f'Final: {mean_ret:.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q-value heatmap
    V = np.max(Q, axis=1).reshape(4, 4)
    im = axes[1].imshow(V, cmap='viridis')
    axes[1].set_title('State Values V(s) = max_a Q(s,a)')
    plt.colorbar(im, ax=axes[1])

    # Add policy arrows
    for row in range(4):
        for col in range(4):
            state = row * 4 + col
            if state not in [5, 7, 11, 12, 15]:  # Not holes or goal
                axes[1].text(col, row, action_symbols[policy[state]],
                           ha='center', va='center', fontsize=16, color='white')
            elif state == 15:
                axes[1].text(col, row, 'G', ha='center', va='center',
                           fontsize=14, color='white', fontweight='bold')
            else:
                axes[1].text(col, row, 'H', ha='center', va='center',
                           fontsize=14, color='white', fontweight='bold')

    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))

    plt.tight_layout()
    plt.savefig('sarsa_frozenlake.png', dpi=150, bbox_inches='tight')
    print("\nSaved learning curve to: sarsa_frozenlake.png")

    plt.show()

    env.close()
