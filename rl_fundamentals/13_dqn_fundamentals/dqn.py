"""
Deep Q-Network (DQN) Implementation

Pure NumPy implementation of DQN with:
- Experience replay for breaking sample correlation
- Target network for stable learning targets
- Epsilon-greedy exploration with decay
- Huber loss for robust TD error handling

This implements the algorithm from:
Mnih et al., 2015 - "Human-level control through deep reinforcement learning"

Author: RL Speedrun
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import gymnasium as gym

from replay_buffer import ReplayBuffer
from target_network import NumpyNetwork, TargetNetworkManager, compute_td_targets


@dataclass
class DQNConfig:
    """Configuration for DQN training."""
    # Environment
    state_dim: int = 4
    action_dim: int = 2

    # Network architecture
    hidden_dims: List[int] = None  # Default: [64, 64]

    # Training
    n_episodes: int = 500
    max_steps: int = 500
    batch_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99

    # Replay buffer
    buffer_size: int = 10000
    min_buffer_size: int = 1000  # Start training after this many samples

    # Target network
    target_update_freq: int = 100  # Steps between hard updates
    use_soft_update: bool = False
    tau: float = 0.005  # Soft update coefficient

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


class DQNetwork:
    """
    Deep Q-Network with backpropagation.

    Architecture: state -> hidden1 (ReLU) -> hidden2 (ReLU) -> Q-values
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 learning_rate: float = 0.001,
                 seed: int = 42):
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
        """Xavier uniform initialization - stable for both ReLU and linear."""
        dims = [self.state_dim] + self.hidden_dims + [self.action_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.params[f'W{i}'] = self.rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float64)
            self.params[f'b{i}'] = np.zeros(fan_out, dtype=np.float64)

    def forward(self, state: np.ndarray, return_cache: bool = False):
        """Forward pass through the network."""
        single_input = state.ndim == 1
        if single_input:
            state = state.reshape(1, -1)

        state = state.astype(np.float64)
        cache = {'input': state}
        x = state
        n_layers = len(self.hidden_dims) + 1

        for i in range(n_layers):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']

            z = x @ W + b
            z = np.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)
            z = np.clip(z, -50, 50)
            cache[f'z{i}'] = z

            if i < n_layers - 1:  # ReLU for hidden layers
                x = np.maximum(0, z)
                cache[f'a{i}'] = x
            else:  # Linear output
                x = z

        if single_input:
            x = x.squeeze(0)

        if return_cache:
            return x, cache
        return x

    def backward(self, states: np.ndarray, actions: np.ndarray,
                 targets: np.ndarray) -> float:
        """
        Backward pass using Huber loss for stability.

        Huber loss clips the gradient for large errors (|e|>1) to ±1,
        while still giving correct gradients for small errors.
        """
        batch_size = states.shape[0]
        n_layers = len(self.hidden_dims) + 1

        # Forward pass with cache
        q_values, cache = self.forward(states, return_cache=True)

        # Get Q-values for taken actions
        q_taken = q_values[np.arange(batch_size), actions]

        # Huber loss (smooth L1) - robust to large TD errors
        td_errors = q_taken - targets
        abs_errors = np.abs(td_errors)
        huber_mask = abs_errors <= 1.0
        loss = np.mean(np.where(huber_mask, 0.5 * td_errors**2, abs_errors - 0.5))

        # Huber loss gradient: clip td_error to [-1, 1]
        grad_td = np.clip(td_errors, -1.0, 1.0) / batch_size

        # Gradient of loss w.r.t. Q-values
        dq = np.zeros_like(q_values)
        dq[np.arange(batch_size), actions] = grad_td

        # Backpropagate through layers
        gradients = {}
        dx = dq

        for i in range(n_layers - 1, -1, -1):
            if i == 0:
                a_prev = cache['input']
            else:
                a_prev = cache[f'a{i-1}']

            gradients[f'W{i}'] = a_prev.T @ dx
            gradients[f'b{i}'] = np.sum(dx, axis=0)

            if i > 0:
                dx = dx @ self.params[f'W{i}'].T
                dx = dx * (cache[f'z{i-1}'] > 0)  # ReLU gradient

        # Replace NaN/Inf gradients with zero
        for key in gradients:
            gradients[key] = np.nan_to_num(gradients[key], nan=0.0, posinf=0.0, neginf=0.0)

        # Global gradient norm clipping
        total_norm = sum(np.sum(g**2) for g in gradients.values())
        total_norm = np.sqrt(total_norm)
        max_norm = 10.0
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            gradients = {k: v * scale for k, v in gradients.items()}

        self._adam_update(gradients)
        return loss

    def _adam_update(self, gradients: Dict[str, np.ndarray],
                     beta1: float = 0.9, beta2: float = 0.999,
                     epsilon: float = 1e-8,
                     weight_decay: float = 1e-4) -> None:
        """Adam optimizer update with weight decay and NaN safety."""
        self.t += 1

        for key in self.params:
            g = gradients[key]

            # AdamW-style weight decay (applied to weights only, not biases)
            if 'W' in key:
                g = g + weight_decay * self.params[key]

            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)

            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            # NaN safety: skip update if it produces NaN
            new_param = self.params[key] - update
            if np.all(np.isfinite(new_param)):
                self.params[key] = new_param

        # Clip weights to prevent unbounded growth
        for key in self.params:
            if 'W' in key:
                np.clip(self.params[key], -10.0, 10.0, out=self.params[key])

    def get_params(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            self.params[k] = v.copy()

    def copy(self) -> 'DQNetwork':
        new_net = DQNetwork(
            self.state_dim, self.action_dim,
            self.hidden_dims, self.learning_rate, seed=0
        )
        new_net.set_params(self.get_params())
        return new_net


class DQNAgent:
    """DQN Agent with experience replay and target network."""

    def __init__(self, config: DQNConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.q_network = DQNetwork(
            config.state_dim, config.action_dim,
            config.hidden_dims, config.learning_rate, config.seed
        )
        self.target_network = self.q_network.copy()

        self.buffer = ReplayBuffer(
            config.buffer_size, config.state_dim, config.seed
        )

        self.epsilon = config.epsilon_start
        self.total_steps = 0
        self.training_losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(self.config.action_dim)
        else:
            q_values = self.q_network.forward(state)
            return int(np.argmax(q_values))

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        if not self.buffer.is_ready(self.config.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size
        )

        # Compute TD targets using target network
        next_q_values = self.target_network.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + self.config.gamma * max_next_q * (1 - dones.astype(np.float64))

        loss = self.q_network.backward(states, actions, targets)
        self.training_losses.append(loss)

        self.total_steps += 1
        if self.config.use_soft_update:
            self._soft_update_target()
        else:
            if self.total_steps % self.config.target_update_freq == 0:
                self._hard_update_target()

        return loss

    def _hard_update_target(self):
        self.target_network.set_params(self.q_network.get_params())

    def _soft_update_target(self):
        tau = self.config.tau
        online_params = self.q_network.get_params()
        target_params = self.target_network.get_params()
        for key in online_params:
            target_params[key] = tau * online_params[key] + (1 - tau) * target_params[key]
        self.target_network.set_params(target_params)

    def decay_epsilon(self):
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)


def train_dqn(env, config: DQNConfig) -> Tuple[DQNAgent, List[float], List[int]]:
    """Train DQN agent on environment."""
    agent = DQNAgent(config)
    episode_returns = []
    episode_lengths = []

    for episode in range(config.n_episodes):
        state, _ = env.reset(seed=config.seed + episode)
        episode_return = 0
        episode_length = 0

        for step in range(config.max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            episode_return += reward
            episode_length += 1
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        if (episode + 1) % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            print(f"Episode {episode + 1:4d} | "
                  f"Avg Return: {avg_return:7.2f} | "
                  f"Avg Length: {avg_length:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.buffer)}")

    return agent, episode_returns, episode_lengths


def evaluate_agent(env, agent: DQNAgent, n_episodes: int = 100,
                   seed: int = 0) -> Tuple[float, float]:
    """Evaluate trained agent."""
    returns = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        episode_return = 0
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
    return np.mean(returns), np.std(returns)


# Demonstration
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    print("=" * 70)
    print("DQN Training Demo on CartPole-v1 (NumPy)")
    print("=" * 70)

    env = gym.make("CartPole-v1")

    config = DQNConfig(
        state_dim=4,
        action_dim=2,
        hidden_dims=[128, 128],
        n_episodes=500,
        max_steps=500,
        batch_size=128,
        learning_rate=0.001,
        gamma=0.99,
        buffer_size=50000,
        min_buffer_size=1000,
        target_update_freq=200,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99,
        seed=42
    )

    print(f"\nConfig: lr={config.learning_rate}, hidden={config.hidden_dims}, "
          f"target_update={config.target_update_freq}")

    print("\nTraining...")
    print("-" * 50)
    agent, returns, lengths = train_dqn(env, config)
    print("-" * 50)

    mean_return, std_return = evaluate_agent(env, agent, n_episodes=100, seed=0)
    print(f"\nEval: {mean_return:.2f} +/- {std_return:.2f}")
    print("SOLVED!" if mean_return >= 195 else f"Not solved (need >= 195)")

    fig, ax = plt.subplots(figsize=(10, 5))
    window = 20
    smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='blue', linewidth=1.5)
    ax.axhline(y=195, color='green', linestyle='--', label='Solved (195)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Episode Length')
    ax.set_title('DQN Learning Curve (NumPy)'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'dqn_cartpole.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    env.close()
