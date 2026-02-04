"""
CartPole Solver using REINFORCE Policy Gradient

This implements REINFORCE with a simple neural network policy for CartPole.
Pure NumPy implementation with careful numerical stability.

Author: RL Speedrun
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Tuple, List
import os


class SimplePolicy:
    """
    Simple 2-layer neural network policy.
    Uses careful numerical stability for softmax.
    """

    def __init__(self, state_dim: int = 4, hidden_dim: int = 32,
                 action_dim: int = 2, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Initialize with small weights
        scale = 0.1
        self.W1 = self.rng.standard_normal((state_dim, hidden_dim)) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.standard_normal((hidden_dim, action_dim)) * scale
        self.b2 = np.zeros(action_dim)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass, returns action probabilities."""
        # Hidden layer with ReLU
        h = np.maximum(0, state @ self.W1 + self.b1)
        # Output with stable softmax
        logits = h @ self.W2 + self.b2
        logits = logits - np.max(logits)  # Numerical stability
        exp_logits = np.exp(np.clip(logits, -20, 20))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        return probs, h

    def sample_action(self, state: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Sample action and return log probability."""
        probs, h = self.forward(state)
        action = self.rng.choice(self.action_dim, p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        return action, log_prob, h

    def update(self, states, actions, hidden_activations, advantages, lr: float):
        """
        Update policy parameters using policy gradient.

        ∇θ J(θ) ≈ Σ_t advantage_t * ∇θ log π(a_t|s_t)
        """
        # Accumulate gradients
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)

        for state, action, h, adv in zip(states, actions, hidden_activations, advantages):
            # Forward pass for gradients
            logits = h @ self.W2 + self.b2
            logits = logits - np.max(logits)
            exp_logits = np.exp(np.clip(logits, -20, 20))
            probs = exp_logits / (np.sum(exp_logits) + 1e-10)

            # Gradient of log softmax: e_a - π
            dlogits = -probs.copy()
            dlogits[action] += 1
            dlogits *= adv  # Scale by advantage

            # Clip gradient for stability
            dlogits = np.clip(dlogits, -1, 1)

            # Backprop to W2, b2
            dW2 += np.outer(h, dlogits)
            db2 += dlogits

            # Backprop through ReLU to W1, b1
            dh = dlogits @ self.W2.T
            dh = dh * (h > 0)  # ReLU gradient
            dh = np.clip(dh, -1, 1)

            dW1 += np.outer(state, dh)
            db1 += dh

        # Apply updates with gradient clipping
        n = len(states)
        self.W2 += lr * np.clip(dW2 / n, -0.5, 0.5)
        self.b2 += lr * np.clip(db2 / n, -0.5, 0.5)
        self.W1 += lr * np.clip(dW1 / n, -0.5, 0.5)
        self.b1 += lr * np.clip(db1 / n, -0.5, 0.5)


def reinforce_cartpole(
    env,
    n_episodes: int = 3000,
    gamma: float = 0.99,
    lr: float = 0.01,
    seed: int = 42
) -> Tuple[SimplePolicy, List[int]]:
    """
    REINFORCE algorithm for CartPole.

    Args:
        env: CartPole environment
        n_episodes: Training episodes
        gamma: Discount factor
        lr: Learning rate
        seed: Random seed

    Returns:
        policy: Trained policy
        episode_lengths: Length of each episode
    """
    policy = SimplePolicy(seed=seed)
    episode_lengths = []

    for ep in range(n_episodes):
        # Collect trajectory
        states = []
        actions = []
        rewards = []
        hidden_acts = []

        state, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False

        while not done and not truncated:
            action, log_prob, h = policy.sample_action(state)

            states.append(state.copy())
            actions.append(action)
            hidden_acts.append(h.copy())

            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)

        episode_lengths.append(len(rewards))

        # Compute returns
        T = len(rewards)
        returns = np.zeros(T)
        G = 0
        for t in range(T - 1, -1, -1):
            G = gamma * G + rewards[t]
            returns[t] = G

        # Normalize returns for stability (acts as baseline)
        if T > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        else:
            returns = np.array([0.0])

        # Update policy
        policy.update(states, actions, hidden_acts, returns, lr)

        # Progress
        if (ep + 1) % 100 == 0:
            avg_len = np.mean(episode_lengths[-100:])
            print(f"Episode {ep + 1:5d}: Avg length = {avg_len:.1f}")

            if avg_len >= 195:
                print(f"\nSOLVED at episode {ep + 1}!")
                break

    return policy, episode_lengths


def evaluate_policy(env, policy: SimplePolicy, n_episodes: int = 100,
                    seed: int = 0) -> Tuple[float, float]:
    """Evaluate policy using greedy actions."""
    lengths = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        length = 0

        while not done and not truncated:
            probs, _ = policy.forward(state)
            action = np.argmax(probs)
            state, _, done, truncated, _ = env.step(action)
            length += 1

        lengths.append(length)

    return np.mean(lengths), np.std(lengths)


def main():
    print("=" * 70)
    print("CartPole: REINFORCE Policy Gradient (NumPy)")
    print("=" * 70)

    env = gym.make("CartPole-v1")

    config = {
        "n_episodes": 3000,
        "gamma": 0.99,
        "lr": 0.01,
        "seed": 42
    }

    print("\nHyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\nTraining REINFORCE...")
    print("-" * 50)

    policy, lengths = reinforce_cartpole(env, **config)

    print("-" * 50)
    print("\nEvaluating learned policy...")
    mean_len, std_len = evaluate_policy(env, policy, n_episodes=100, seed=0)
    print(f"  Mean episode length: {mean_len:.1f} ± {std_len:.1f}")

    solved = mean_len >= 195
    if solved:
        print("\n✓ SOLVED! Average length >= 195")
    else:
        print(f"\n✗ Not solved (need >= 195, got {mean_len:.1f})")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Learning curve
    window = 50
    if len(lengths) > window:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, color='blue', linewidth=1)
    else:
        axes[0].plot(lengths, color='blue', linewidth=1)

    axes[0].axhline(y=195, color='green', linestyle='--', label='Solved (195)')
    axes[0].axhline(y=500, color='red', linestyle=':', alpha=0.5, label='Max (500)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Length')
    axes[0].set_title('REINFORCE Learning Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Episode length distribution (last 200)
    last_n = min(200, len(lengths))
    axes[1].hist(lengths[-last_n:], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=np.mean(lengths[-last_n:]), color='red', linestyle='--',
                    label=f'Mean: {np.mean(lengths[-last_n:]):.1f}')
    axes[1].set_xlabel('Episode Length')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Episode Length Distribution (Last {last_n})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'cartpole_reinforce.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")

    plt.show()
    env.close()


if __name__ == "__main__":
    main()
