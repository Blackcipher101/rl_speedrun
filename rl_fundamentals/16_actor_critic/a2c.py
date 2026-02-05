"""
Advantage Actor-Critic (A2C) Implementation

A2C combines policy gradient (actor) with value function learning (critic):
- Actor: π(a|s) - learns which actions to take
- Critic: V(s) - learns state values for variance reduction

Key benefits over REINFORCE:
1. Lower variance via baseline (critic provides it)
2. Online learning (no need to wait for episode end)
3. Better credit assignment via advantage function

This is a synchronous version. A3C adds parallel actors (not implemented).

Author: RL Speedrun
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import gymnasium as gym

from advantage import compute_gae, normalize_advantages
from entropy import compute_entropy, compute_batch_entropy


@dataclass
class A2CConfig:
    """Configuration for A2C training."""
    # Environment
    state_dim: int = 4
    action_dim: int = 2

    # Network architecture
    hidden_dims: List[int] = None

    # Training
    n_episodes: int = 1000
    max_steps: int = 500
    learning_rate: float = 0.001
    gamma: float = 0.99

    # Advantage estimation
    gae_lambda: float = 0.95
    normalize_advantages: bool = True

    # Loss weights
    value_coef: float = 0.5      # Weight for value loss
    entropy_coef: float = 0.01   # Weight for entropy bonus

    # Gradient clipping
    max_grad_norm: float = 0.5

    # Update frequency
    update_freq: int = 5  # Steps between updates (n-step)

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


class ActorCriticNetwork:
    """
    Combined Actor-Critic network with shared feature layers.

    Architecture:
        state → shared_layers → actor_head → π(a|s)
                              → critic_head → V(s)

    Sharing features is more parameter-efficient and can help
    both heads learn useful representations.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 learning_rate: float = 0.001,
                 seed: int = 42):
        """
        Initialize actor-critic network.

        Args:
            state_dim: Input dimension
            action_dim: Number of actions
            hidden_dims: Hidden layer sizes (shared)
            learning_rate: Learning rate
            seed: Random seed
        """
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        self.params = {}
        self._init_weights()

        # Adam optimizer state
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def _init_weights(self) -> None:
        """Initialize weights using scaled He initialization for stability."""
        # Shared layers - smaller initialization for stability
        dims = [self.state_dim] + self.hidden_dims
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i]) * 0.1  # 0.1x He init
            self.params[f'shared_W{i}'] = self.rng.standard_normal((dims[i], dims[i+1])) * scale
            self.params[f'shared_b{i}'] = np.zeros(dims[i+1])

        # Actor head (policy) - small init important for policy
        last_hidden = self.hidden_dims[-1]
        scale = np.sqrt(2.0 / last_hidden) * 0.01
        self.params['actor_W'] = self.rng.standard_normal((last_hidden, self.action_dim)) * scale
        self.params['actor_b'] = np.zeros(self.action_dim)

        # Critic head (value) - smaller init
        scale = np.sqrt(2.0 / last_hidden) * 0.1
        self.params['critic_W'] = self.rng.standard_normal((last_hidden, 1)) * scale
        self.params['critic_b'] = np.zeros(1)

    def forward(self, state: np.ndarray, return_cache: bool = False):
        """
        Forward pass through actor-critic network.

        Args:
            state: Input state(s)
            return_cache: Whether to return intermediate activations

        Returns:
            action_probs: π(a|s) for all actions
            value: V(s) estimate
            cache: (optional) intermediate activations for backprop
        """
        single_input = state.ndim == 1
        if single_input:
            state = state.reshape(1, -1)

        cache = {'input': state}

        # Shared layers
        x = state
        n_shared = len(self.hidden_dims)
        for i in range(n_shared):
            W = self.params[f'shared_W{i}']
            b = self.params[f'shared_b{i}']
            z = x @ W + b
            z = np.clip(z, -50, 50)  # Prevent overflow
            cache[f'shared_z{i}'] = z
            x = np.maximum(0, z)  # ReLU
            cache[f'shared_a{i}'] = x

        shared_features = x

        # Actor head (policy)
        actor_z = shared_features @ self.params['actor_W'] + self.params['actor_b']
        actor_z = np.clip(actor_z, -50, 50)  # Prevent overflow
        cache['actor_z'] = actor_z

        # Stable softmax
        actor_z_stable = actor_z - np.max(actor_z, axis=1, keepdims=True)
        exp_z = np.exp(np.clip(actor_z_stable, -20, 20))
        action_probs = exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-10)
        cache['action_probs'] = action_probs

        # Critic head (value)
        value = shared_features @ self.params['critic_W'] + self.params['critic_b']
        value = np.clip(value, -1000, 1000)  # Reasonable value range
        value = value.squeeze(-1)
        cache['value'] = value

        if single_input:
            action_probs = action_probs.squeeze(0)
            value = value.item() if value.size == 1 else value.squeeze(0)

        if return_cache:
            return action_probs, value, cache
        return action_probs, value

    def backward(self, states: np.ndarray, actions: np.ndarray,
                 advantages: np.ndarray, returns: np.ndarray,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5) -> Dict[str, float]:
        """
        Backward pass and parameter update.

        Total loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        Args:
            states: Batch of states
            actions: Batch of actions taken
            advantages: Advantage estimates
            returns: Value function targets
            value_coef: Weight for value loss
            entropy_coef: Weight for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            Dictionary of loss components for logging
        """
        batch_size = states.shape[0]

        # Forward pass with cache
        action_probs, values, cache = self.forward(states, return_cache=True)

        # Ensure 2D
        if action_probs.ndim == 1:
            action_probs = action_probs.reshape(1, -1)

        # Get probabilities of taken actions
        action_indices = np.arange(batch_size)
        taken_probs = action_probs[action_indices, actions]
        log_probs = np.log(np.clip(taken_probs, 1e-10, 1.0))

        # === Policy (Actor) Loss ===
        # L_policy = -E[log π(a|s) * A(s,a)]
        policy_loss = -np.mean(log_probs * advantages)

        # === Value (Critic) Loss ===
        # L_value = E[(V(s) - G)^2]
        value_errors = values - returns
        value_loss = 0.5 * np.mean(value_errors ** 2)

        # === Entropy Bonus ===
        entropies = compute_batch_entropy(action_probs)
        entropy = np.mean(entropies)

        # === Total Loss ===
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        # === Backpropagation ===
        gradients = {}

        # Gradient of policy loss w.r.t. logits (before softmax)
        # d(loss)/d(logits) = π(a|s) - one_hot(a) (weighted by advantage)
        d_logits = action_probs.copy()
        d_logits[action_indices, actions] -= 1.0  # Subtract 1 for taken action
        d_logits = d_logits * (-advantages.reshape(-1, 1)) / batch_size

        # Add entropy gradient (encourage higher entropy)
        # d(entropy)/d(logits) = -π(a|s) * (1 + log π(a|s))
        d_entropy = -action_probs * (1 + np.log(np.clip(action_probs, 1e-10, 1.0)))
        d_logits -= entropy_coef * d_entropy / batch_size

        # Clip policy gradients
        d_logits = np.clip(d_logits, -1, 1)

        # Gradient through actor head
        shared_features = cache[f'shared_a{len(self.hidden_dims)-1}']
        gradients['actor_W'] = shared_features.T @ d_logits
        gradients['actor_b'] = np.sum(d_logits, axis=0)

        # Gradient of value loss w.r.t. value output
        d_value = value_coef * value_errors.reshape(-1, 1) / batch_size

        # Gradient through critic head
        gradients['critic_W'] = shared_features.T @ d_value
        gradients['critic_b'] = np.sum(d_value, axis=0)

        # Combined gradient flowing back to shared layers
        d_shared = d_logits @ self.params['actor_W'].T + d_value @ self.params['critic_W'].T

        # Backprop through shared layers
        for i in range(len(self.hidden_dims) - 1, -1, -1):
            # ReLU gradient
            d_shared = d_shared * (cache[f'shared_z{i}'] > 0)
            d_shared = np.clip(d_shared, -10, 10)  # Clip intermediate gradients

            # Gradient w.r.t. weights and biases
            if i == 0:
                prev_activation = cache['input']
            else:
                prev_activation = cache[f'shared_a{i-1}']

            gradients[f'shared_W{i}'] = prev_activation.T @ d_shared
            gradients[f'shared_b{i}'] = np.sum(d_shared, axis=0)

            # Clip individual gradients
            gradients[f'shared_W{i}'] = np.clip(gradients[f'shared_W{i}'], -10, 10)
            gradients[f'shared_b{i}'] = np.clip(gradients[f'shared_b{i}'], -10, 10)

            # Gradient for next layer
            if i > 0:
                d_shared = d_shared @ self.params[f'shared_W{i}'].T
                d_shared = np.clip(d_shared, -10, 10)

        # Handle NaN in gradients
        for key in gradients:
            if np.any(~np.isfinite(gradients[key])):
                gradients[key] = np.nan_to_num(gradients[key], nan=0.0, posinf=1.0, neginf=-1.0)

        # Gradient clipping by global norm
        total_norm = 0
        for g in gradients.values():
            total_norm += np.sum(g ** 2)
        total_norm = np.sqrt(total_norm)

        if total_norm > max_grad_norm:
            scale = max_grad_norm / (total_norm + 1e-6)
            for key in gradients:
                gradients[key] *= scale

        # Adam update
        self._adam_update(gradients)

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }

    def _adam_update(self, gradients: Dict[str, np.ndarray],
                     beta1: float = 0.9, beta2: float = 0.999,
                     epsilon: float = 1e-8) -> None:
        """Update parameters using Adam optimizer."""
        self.t += 1

        for key in self.params:
            if key not in gradients:
                continue

            g = gradients[key]

            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)

            self.params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


class A2CAgent:
    """
    Advantage Actor-Critic agent.

    Collects experience for n steps, then updates both actor and critic
    using GAE for advantage estimation.
    """

    def __init__(self, config: A2CConfig):
        """Initialize A2C agent."""
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.network = ActorCriticNetwork(
            config.state_dim, config.action_dim,
            config.hidden_dims, config.learning_rate, config.seed
        )

        # Experience buffer (for n-step updates)
        self.reset_buffer()

        # Logging
        self.training_stats = []

    def reset_buffer(self):
        """Reset experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """
        Select action from policy.

        Args:
            state: Current state
            training: If True, sample; if False, take argmax

        Returns:
            action: Selected action
            value: Value estimate V(s)
        """
        action_probs, value = self.network.forward(state)

        if training:
            action = self.rng.choice(self.config.action_dim, p=action_probs)
        else:
            action = np.argmax(action_probs)

        return int(action), value

    def store_transition(self, state: np.ndarray, action: int,
                        reward: float, value: float, done: bool):
        """Store transition in buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def update(self, next_value: float = 0.0) -> Optional[Dict[str, float]]:
        """
        Update actor and critic using collected experience.

        Args:
            next_value: V(s_T+1) for bootstrapping (0 if episode ended)

        Returns:
            Training statistics or None if buffer empty
        """
        if len(self.states) == 0:
            return None

        # Convert to arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        # Compute next values (shift values and add bootstrap)
        next_values = np.append(values[1:], next_value)

        # Compute advantages using GAE
        advantages, returns = compute_gae(
            rewards, values, next_values, dones,
            self.config.gamma, self.config.gae_lambda
        )

        if self.config.normalize_advantages:
            advantages = normalize_advantages(advantages)

        # Update network
        stats = self.network.backward(
            states, actions, advantages, returns,
            self.config.value_coef, self.config.entropy_coef,
            self.config.max_grad_norm
        )

        self.training_stats.append(stats)
        self.reset_buffer()

        return stats


def train_a2c(env, config: A2CConfig) -> Tuple[A2CAgent, List[float], List[int]]:
    """
    Train A2C agent on environment.

    Args:
        env: Gymnasium environment
        config: A2C configuration

    Returns:
        agent: Trained A2C agent
        episode_returns: Return of each episode
        episode_lengths: Length of each episode
    """
    agent = A2CAgent(config)
    episode_returns = []
    episode_lengths = []

    for episode in range(config.n_episodes):
        state, _ = env.reset(seed=config.seed + episode)
        episode_return = 0
        episode_length = 0
        steps_since_update = 0

        done = False
        while not done:
            # Select action
            action, value = agent.select_action(state, training=True)

            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, value, done)
            steps_since_update += 1

            # Update every n steps or at episode end
            if steps_since_update >= config.update_freq or done:
                if done:
                    next_value = 0.0
                else:
                    _, next_value = agent.select_action(next_state, training=False)
                agent.update(next_value)
                steps_since_update = 0

            episode_return += reward
            episode_length += 1
            state = next_state

            if episode_length >= config.max_steps:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # Progress reporting
        if (episode + 1) % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            avg_length = np.mean(episode_lengths[-50:])

            # Get recent training stats
            recent_stats = agent.training_stats[-50:] if agent.training_stats else []
            if recent_stats:
                avg_entropy = np.mean([s['entropy'] for s in recent_stats])
                avg_vloss = np.mean([s['value_loss'] for s in recent_stats])
            else:
                avg_entropy = 0
                avg_vloss = 0

            print(f"Episode {episode + 1:4d} | "
                  f"Avg Return: {avg_return:7.2f} | "
                  f"Avg Length: {avg_length:6.1f} | "
                  f"Entropy: {avg_entropy:.3f} | "
                  f"V_loss: {avg_vloss:.3f}")

    return agent, episode_returns, episode_lengths


def evaluate_agent(env, agent: A2CAgent, n_episodes: int = 100,
                   seed: int = 0) -> Tuple[float, float]:
    """Evaluate trained agent."""
    returns = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        episode_return = 0
        done = False

        while not done:
            action, _ = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        returns.append(episode_return)

    return np.mean(returns), np.std(returns)


# Demonstration
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    print("=" * 70)
    print("A2C Training Demo on CartPole-v1")
    print("=" * 70)

    env = gym.make("CartPole-v1")

    config = A2CConfig(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64, 64],
        n_episodes=1000,
        max_steps=500,
        learning_rate=0.0003,  # Lower for stability
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_freq=10,  # Longer rollouts
        seed=42
    )

    print("\nConfiguration:")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  GAE lambda: {config.gae_lambda}")
    print(f"  Value coef: {config.value_coef}")
    print(f"  Entropy coef: {config.entropy_coef}")
    print(f"  Update freq: {config.update_freq} steps")

    print("\nTraining A2C...")
    print("-" * 50)

    agent, returns, lengths = train_a2c(env, config)

    print("-" * 50)
    print("\nEvaluating trained agent...")

    mean_return, std_return = evaluate_agent(env, agent, n_episodes=100, seed=0)
    print(f"  Mean return: {mean_return:.2f} +/- {std_return:.2f}")

    solved = mean_return >= 195
    if solved:
        print("\n  SOLVED! (Mean return >= 195)")
    else:
        print(f"\n  Not solved (need >= 195, got {mean_return:.1f})")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Learning curve
    window = 20
    smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(smoothed, color='blue', linewidth=1.5)
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

    # Entropy over time
    if agent.training_stats:
        entropies = [s['entropy'] for s in agent.training_stats]
        axes[1, 0].plot(entropies, color='purple', alpha=0.5, linewidth=0.5)
        window_ent = min(100, len(entropies) // 5)
        if window_ent > 1:
            smoothed_ent = np.convolve(entropies, np.ones(window_ent)/window_ent, mode='valid')
            axes[1, 0].plot(smoothed_ent, color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Policy Entropy')
    axes[1, 0].set_title('Policy Entropy Over Training')
    axes[1, 0].grid(True, alpha=0.3)

    # Value loss over time
    if agent.training_stats:
        vlosses = [s['value_loss'] for s in agent.training_stats]
        axes[1, 1].plot(vlosses, color='red', alpha=0.5, linewidth=0.5)
        if window_ent > 1:
            smoothed_vloss = np.convolve(vlosses, np.ones(window_ent)/window_ent, mode='valid')
            axes[1, 1].plot(smoothed_vloss, color='red', linewidth=2)
    axes[1, 1].set_xlabel('Update Step')
    axes[1, 1].set_ylabel('Value Loss')
    axes[1, 1].set_title('Value Loss Over Training')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'a2c_cartpole.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")

    plt.show()
    env.close()
