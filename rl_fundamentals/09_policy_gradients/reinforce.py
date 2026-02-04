"""
REINFORCE: Policy Gradient Algorithm

REINFORCE is a Monte Carlo policy gradient method that directly optimizes
the policy by following the gradient of expected return.

Update Rule:
    θ ← θ + α · γᵗ · Gₜ · ∇_θ log π_θ(aₜ|sₜ)

This implementation includes:
- Basic REINFORCE
- REINFORCE with baseline (variance reduction)
- Return normalization

Author: RL Speedrun
"""

import numpy as np
from typing import Tuple, List, Optional
import warnings


class SoftmaxPolicy:
    """
    Tabular softmax policy for discrete state and action spaces.

    π(a|s) = exp(θ[s,a]) / Σ_a' exp(θ[s,a'])
    """

    def __init__(self, n_states: int, n_actions: int, seed: int = 42):
        """
        Initialize softmax policy.

        Args:
            n_states: Number of states
            n_actions: Number of actions
            seed: Random seed
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        # Policy parameters (preferences)
        self.theta = np.zeros((n_states, n_actions))

    def get_probs(self, state: int) -> np.ndarray:
        """Get action probabilities for a state."""
        # Softmax with numerical stability
        preferences = self.theta[state]
        exp_prefs = np.exp(preferences - np.max(preferences))
        return exp_prefs / np.sum(exp_prefs)

    def sample_action(self, state: int) -> int:
        """Sample action from policy."""
        probs = self.get_probs(state)
        return self.rng.choice(self.n_actions, p=probs)

    def log_prob(self, state: int, action: int) -> float:
        """Compute log probability of action given state."""
        probs = self.get_probs(state)
        return np.log(probs[action] + 1e-10)

    def grad_log_prob(self, state: int, action: int) -> np.ndarray:
        """
        Compute gradient of log π(a|s) with respect to θ.

        ∇_θ log π(a|s) = φ(s,a) - Σ_a' π(a'|s) φ(s,a')

        For tabular case:
        - ∇_θ[s,a] log π(a|s) = 1 - π(a|s) for the taken action
        - ∇_θ[s,a'] log π(a|s) = -π(a'|s) for other actions
        """
        grad = np.zeros_like(self.theta)
        probs = self.get_probs(state)

        for a in range(self.n_actions):
            if a == action:
                grad[state, a] = 1 - probs[a]
            else:
                grad[state, a] = -probs[a]

        return grad

    def get_greedy_policy(self) -> np.ndarray:
        """Extract greedy policy from parameters."""
        return np.argmax(self.theta, axis=1)


def reinforce(
    env,
    n_episodes: int = 1000,
    gamma: float = 0.99,
    alpha: float = 0.01,
    normalize_returns: bool = True,
    seed: int = 42
) -> Tuple[SoftmaxPolicy, List[float]]:
    """
    Basic REINFORCE algorithm.

    Args:
        env: Gymnasium environment
        n_episodes: Number of training episodes
        gamma: Discount factor
        alpha: Learning rate
        normalize_returns: Whether to normalize returns
        seed: Random seed

    Returns:
        policy: Trained softmax policy
        episode_returns: Return of each episode
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy = SoftmaxPolicy(n_states, n_actions, seed)
    episode_returns = []

    for ep in range(n_episodes):
        # Generate episode
        trajectory = []
        state, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False

        while not done and not truncated:
            action = policy.sample_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state

        # Compute returns
        T = len(trajectory)
        returns = np.zeros(T)
        G = 0
        for t in range(T - 1, -1, -1):
            G = gamma * G + trajectory[t][2]
            returns[t] = G

        episode_returns.append(returns[0] if T > 0 else 0)

        # Normalize returns (variance reduction)
        if normalize_returns and len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Policy gradient update
        for t in range(T):
            state, action, _ = trajectory[t]
            grad = policy.grad_log_prob(state, action)
            policy.theta += alpha * (gamma ** t) * returns[t] * grad

    return policy, episode_returns


def reinforce_with_baseline(
    env,
    n_episodes: int = 1000,
    gamma: float = 0.99,
    alpha_policy: float = 0.01,
    alpha_value: float = 0.1,
    seed: int = 42
) -> Tuple[SoftmaxPolicy, np.ndarray, List[float]]:
    """
    REINFORCE with learned value function baseline.

    The baseline V(s) reduces variance without introducing bias.

    Args:
        env: Gymnasium environment
        n_episodes: Number of training episodes
        gamma: Discount factor
        alpha_policy: Learning rate for policy
        alpha_value: Learning rate for value function
        seed: Random seed

    Returns:
        policy: Trained softmax policy
        V: Learned value function
        episode_returns: Return of each episode
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy = SoftmaxPolicy(n_states, n_actions, seed)
    V = np.zeros(n_states)  # Value function baseline
    episode_returns = []

    for ep in range(n_episodes):
        # Generate episode
        trajectory = []
        state, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False

        while not done and not truncated:
            action = policy.sample_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state

        # Compute returns
        T = len(trajectory)
        returns = np.zeros(T)
        G = 0
        for t in range(T - 1, -1, -1):
            G = gamma * G + trajectory[t][2]
            returns[t] = G

        episode_returns.append(returns[0] if T > 0 else 0)

        # Update policy and value function
        for t in range(T):
            state, action, _ = trajectory[t]

            # Advantage: return minus baseline
            advantage = returns[t] - V[state]

            # Update value function (baseline)
            V[state] += alpha_value * advantage

            # Policy gradient update with baseline
            grad = policy.grad_log_prob(state, action)
            policy.theta += alpha_policy * (gamma ** t) * advantage * grad

    return policy, V, episode_returns


def evaluate_policy(
    env,
    policy: SoftmaxPolicy,
    n_episodes: int = 100,
    seed: int = 0,
    deterministic: bool = True
) -> Tuple[float, float]:
    """
    Evaluate a policy.

    Args:
        env: Gymnasium environment
        policy: Trained policy
        n_episodes: Number of evaluation episodes
        seed: Random seed
        deterministic: Use greedy policy

    Returns:
        mean_return: Average return
        std_return: Standard deviation
    """
    returns = []
    greedy_policy = policy.get_greedy_policy()

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        episode_return = 0

        while not done and not truncated:
            if deterministic:
                action = greedy_policy[state]
            else:
                action = policy.sample_action(state)

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
    print("REINFORCE Demo: FrozenLake-v1")
    print("=" * 60)

    # Create environment
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    # Hyperparameters
    config = {
        "n_episodes": 50000,
        "gamma": 0.99,
        "alpha_policy": 0.01,
        "alpha_value": 0.1,
        "seed": 42
    }

    print("\nHyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Train REINFORCE with baseline
    print("\nTraining REINFORCE with baseline...")
    policy, V, returns = reinforce_with_baseline(
        env,
        n_episodes=config["n_episodes"],
        gamma=config["gamma"],
        alpha_policy=config["alpha_policy"],
        alpha_value=config["alpha_value"],
        seed=config["seed"]
    )

    # Evaluate
    print("\nEvaluating learned policy...")
    mean_ret, std_ret = evaluate_policy(env, policy, n_episodes=1000, seed=0)
    print(f"  Mean return: {mean_ret:.3f} ± {std_ret:.3f}")

    # Display policy
    greedy = policy.get_greedy_policy()
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
                row_str += f" {action_symbols[greedy[state]]} "
        print(row_str)
    print("-" * 20)

    # Display value function
    print("\nLearned Value Function:")
    for row in range(4):
        row_str = ""
        for col in range(4):
            state = row * 4 + col
            row_str += f" {V[state]:5.2f}"
        print(row_str)

    # Plot learning curve
    window = 1000
    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Learning curve
    axes[0].plot(smoothed, color='orange', linewidth=1)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return (smoothed)')
    axes[0].set_title('REINFORCE Learning Curve')
    axes[0].axhline(y=mean_ret, color='red', linestyle='--',
                    label=f'Final: {mean_ret:.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Value function heatmap
    V_grid = V.reshape(4, 4)
    im = axes[1].imshow(V_grid, cmap='viridis')
    axes[1].set_title('Learned Value Function V(s)')
    plt.colorbar(im, ax=axes[1])
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            if state in [5, 7, 11, 12]:
                axes[1].text(j, i, 'H', ha='center', va='center',
                           fontsize=12, color='white', fontweight='bold')
            elif state == 15:
                axes[1].text(j, i, 'G', ha='center', va='center',
                           fontsize=12, color='white', fontweight='bold')
            else:
                axes[1].text(j, i, f'{V[state]:.2f}', ha='center', va='center',
                           fontsize=10, color='white')

    plt.tight_layout()
    plt.savefig('reinforce_frozenlake.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot to: reinforce_frozenlake.png")
    plt.show()

    env.close()
