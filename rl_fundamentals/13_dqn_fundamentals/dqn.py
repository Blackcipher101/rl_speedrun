"""
Deep Q-Network (DQN) Implementation

Pure NumPy implementation of DQN with:
- Experience replay for breaking sample correlation
- Target network for stable learning targets
- Epsilon-greedy exploration with decay

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
        """
        Initialize DQN.

        Args:
            state_dim: Input dimension
            action_dim: Output dimension (number of actions)
            hidden_dims: Hidden layer sizes
            learning_rate: Learning rate for gradient descent
            seed: Random seed
        """
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        # Initialize weights
        self.params = {}
        self._init_weights()

        # For Adam optimizer
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0  # Time step for bias correction

    def _init_weights(self) -> None:
        """Initialize weights using He initialization."""
        dims = [self.state_dim] + self.hidden_dims + [self.action_dim]

        for i in range(len(dims) - 1):
            fan_in = dims[i]

            # He initialization for ReLU
            scale = np.sqrt(2.0 / fan_in)

            self.params[f'W{i}'] = self.rng.standard_normal((dims[i], dims[i+1])) * scale
            self.params[f'b{i}'] = np.zeros(dims[i+1])

    def forward(self, state: np.ndarray, return_cache: bool = False):
        """
        Forward pass through the network.

        Args:
            state: Input state(s)
            return_cache: If True, return intermediate activations for backprop

        Returns:
            Q-values, and optionally the cache of activations
        """
        single_input = state.ndim == 1
        if single_input:
            state = state.reshape(1, -1)

        cache = {'input': state}
        x = state
        n_layers = len(self.hidden_dims) + 1

        for i in range(n_layers):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']

            z = x @ W + b
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
        Backward pass and parameter update.

        Computes gradients using backpropagation and updates weights.

        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions taken (batch_size,)
            targets: TD targets y = r + gamma * max Q'(s', a')

        Returns:
            Loss value (MSE)
        """
        batch_size = states.shape[0]
        n_layers = len(self.hidden_dims) + 1

        # Forward pass with cache
        q_values, cache = self.forward(states, return_cache=True)

        # Get Q-values for taken actions
        q_taken = q_values[np.arange(batch_size), actions]

        # Loss: MSE between Q(s,a) and target
        td_errors = q_taken - targets
        loss = np.mean(td_errors ** 2)

        # Gradient of loss w.r.t. Q-values
        # d(MSE)/d(Q) = 2 * (Q - target) / batch_size
        dq = np.zeros_like(q_values)
        dq[np.arange(batch_size), actions] = 2 * td_errors / batch_size

        # Clip gradients for stability
        dq = np.clip(dq, -1, 1)

        # Backpropagate through layers
        gradients = {}
        dx = dq

        for i in range(n_layers - 1, -1, -1):
            # Get input to this layer
            if i == 0:
                a_prev = cache['input']
            else:
                a_prev = cache[f'a{i-1}']

            # Gradient w.r.t. weights and biases
            gradients[f'W{i}'] = a_prev.T @ dx
            gradients[f'b{i}'] = np.sum(dx, axis=0)

            # Gradient w.r.t. previous layer output
            if i > 0:
                dx = dx @ self.params[f'W{i}'].T
                # ReLU gradient
                dx = dx * (cache[f'z{i-1}'] > 0)

        # Update parameters using Adam optimizer
        self._adam_update(gradients)

        return loss

    def _adam_update(self, gradients: Dict[str, np.ndarray],
                     beta1: float = 0.9, beta2: float = 0.999,
                     epsilon: float = 1e-8) -> None:
        """
        Update parameters using Adam optimizer.

        Adam maintains exponential moving averages of gradients (m) and
        squared gradients (v), with bias correction.
        """
        self.t += 1

        for key in self.params:
            g = gradients[key]

            # Clip gradient
            g = np.clip(g, -1, 1)

            # Update biased first moment estimate
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g

            # Update biased second raw moment estimate
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (g ** 2)

            # Compute bias-corrected estimates
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)

            # Update parameters
            self.params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return a copy of network parameters."""
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set network parameters."""
        for k, v in params.items():
            self.params[k] = v.copy()

    def copy(self) -> 'DQNetwork':
        """Create a copy of this network."""
        new_net = DQNetwork(
            self.state_dim, self.action_dim,
            self.hidden_dims, self.learning_rate, seed=0
        )
        new_net.set_params(self.get_params())
        return new_net


class DQNAgent:
    """
    DQN Agent with experience replay and target network.

    The full DQN algorithm:
    1. Select action using epsilon-greedy from Q-network
    2. Execute action, observe reward and next state
    3. Store transition in replay buffer
    4. Sample random minibatch from buffer
    5. Compute TD targets using target network
    6. Update Q-network via gradient descent
    7. Periodically update target network
    """

    def __init__(self, config: DQNConfig):
        """
        Initialize DQN agent.

        Args:
            config: DQN configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Q-network
        self.q_network = DQNetwork(
            config.state_dim, config.action_dim,
            config.hidden_dims, config.learning_rate, config.seed
        )

        # Target network (copy of Q-network)
        self.target_network = self.q_network.copy()

        # Replay buffer
        self.buffer = ReplayBuffer(
            config.buffer_size, config.state_dim, config.seed
        )

        # Exploration
        self.epsilon = config.epsilon_start

        # Tracking
        self.total_steps = 0
        self.training_losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action
        """
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(self.config.action_dim)
        else:
            q_values = self.q_network.forward(state)
            return int(np.argmax(q_values))

    def store_transition(self, state: np.ndarray, action: int,
                        reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value, or None if buffer not ready
        """
        if not self.buffer.is_ready(self.config.batch_size):
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size
        )

        # Compute TD targets using target network
        # y = r + gamma * max_a' Q_target(s', a') * (1 - done)
        next_q_values = self.target_network.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + self.config.gamma * max_next_q * (1 - dones.astype(np.float32))

        # Update Q-network
        loss = self.q_network.backward(states, actions, targets)
        self.training_losses.append(loss)

        # Update target network
        self.total_steps += 1
        if self.config.use_soft_update:
            # Soft update
            self._soft_update_target()
        else:
            # Hard update every N steps
            if self.total_steps % self.config.target_update_freq == 0:
                self._hard_update_target()

        return loss

    def _hard_update_target(self) -> None:
        """Copy Q-network weights to target network."""
        self.target_network.set_params(self.q_network.get_params())

    def _soft_update_target(self) -> None:
        """Soft update: target = tau * online + (1-tau) * target."""
        tau = self.config.tau
        online_params = self.q_network.get_params()
        target_params = self.target_network.get_params()

        for key in online_params:
            target_params[key] = tau * online_params[key] + (1 - tau) * target_params[key]

        self.target_network.set_params(target_params)

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )


def train_dqn(env, config: DQNConfig) -> Tuple[DQNAgent, List[float], List[int]]:
    """
    Train DQN agent on environment.

    Args:
        env: Gymnasium environment
        config: DQN configuration

    Returns:
        agent: Trained DQN agent
        episode_returns: Return of each episode
        episode_lengths: Length of each episode
    """
    agent = DQNAgent(config)
    episode_returns = []
    episode_lengths = []

    for episode in range(config.n_episodes):
        state, _ = env.reset(seed=config.seed + episode)
        episode_return = 0
        episode_length = 0

        for step in range(config.max_steps):
            # Select action
            action = agent.select_action(state, training=True)

            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train
            agent.train_step()

            episode_return += reward
            episode_length += 1
            state = next_state

            if done:
                break

        # Decay epsilon after each episode
        agent.decay_epsilon()

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # Progress reporting
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
    """
    Evaluate trained agent.

    Args:
        env: Gymnasium environment
        agent: Trained DQN agent
        n_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        mean_return: Average episode return
        std_return: Standard deviation of returns
    """
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
    import matplotlib.pyplot as plt
    import os

    print("=" * 70)
    print("DQN Training Demo on CartPole-v1")
    print("=" * 70)

    # Create environment
    env = gym.make("CartPole-v1")

    # Configuration
    config = DQNConfig(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64, 64],
        n_episodes=300,
        max_steps=500,
        batch_size=64,
        learning_rate=0.001,
        gamma=0.99,
        buffer_size=10000,
        min_buffer_size=1000,
        target_update_freq=100,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        seed=42
    )

    print("\nConfiguration:")
    print(f"  Episodes: {config.n_episodes}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Buffer size: {config.buffer_size}")
    print(f"  Target update freq: {config.target_update_freq}")

    print("\nTraining DQN...")
    print("-" * 50)

    agent, returns, lengths = train_dqn(env, config)

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curve
    window = 20
    if len(lengths) > window:
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, color='blue', linewidth=1.5)
    else:
        axes[0].plot(lengths, color='blue', linewidth=1.5)

    axes[0].axhline(y=195, color='green', linestyle='--', label='Solved (195)')
    axes[0].axhline(y=500, color='red', linestyle=':', alpha=0.5, label='Max (500)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Length')
    axes[0].set_title('DQN Learning Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Training loss
    if agent.training_losses:
        window_loss = min(100, len(agent.training_losses) // 10)
        if window_loss > 1:
            smoothed_loss = np.convolve(
                agent.training_losses,
                np.ones(window_loss)/window_loss,
                mode='valid'
            )
            axes[1].plot(smoothed_loss, color='red', linewidth=1)
        else:
            axes[1].plot(agent.training_losses, color='red', linewidth=1)

    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('TD Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'dqn_cartpole.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")

    plt.show()
    env.close()
