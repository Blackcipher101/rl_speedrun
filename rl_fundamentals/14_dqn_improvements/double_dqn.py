"""
Double DQN Implementation

Double DQN addresses the overestimation bias in standard DQN by decoupling
action selection from action evaluation.

Standard DQN: y = r + γ * max_a' Q_target(s', a')
              └─ Target network both SELECTS and EVALUATES

Double DQN:   y = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
              └─ Online network SELECTS, target network EVALUATES

Reference: van Hasselt et al., 2016 - "Deep Reinforcement Learning with Double Q-learning"

Author: RL Speedrun
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import gymnasium as gym
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '13_dqn_fundamentals'))

from replay_buffer import ReplayBuffer
from dqn import DQNetwork, DQNConfig


@dataclass
class DoubleDQNConfig(DQNConfig):
    """Configuration for Double DQN (inherits from DQN config)."""
    pass  # Same hyperparameters, just different target computation


class DoubleDQNAgent:
    """
    Double DQN Agent.

    Key difference from DQN:
    - Standard DQN: y = r + γ * max_a' Q_target(s', a')
    - Double DQN:   y = r + γ * Q_target(s', argmax_a' Q_online(s', a'))

    This reduces overestimation by using:
    - Online network to SELECT the best action
    - Target network to EVALUATE that action's value
    """

    def __init__(self, config: DoubleDQNConfig):
        """
        Initialize Double DQN agent.

        Args:
            config: Double DQN configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Q-network (online network)
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
        """Select action using epsilon-greedy policy."""
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(self.config.action_dim)
        else:
            q_values = self.q_network.forward(state)
            return int(np.argmax(q_values))

    def store_transition(self, state: np.ndarray, action: int,
                        reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def compute_double_dqn_targets(
        self,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> np.ndarray:
        """
        Compute Double DQN targets.

        Standard DQN:
            y = r + γ * max_a' Q_target(s', a')

        Double DQN:
            a* = argmax_a' Q_online(s', a')   # Online selects
            y = r + γ * Q_target(s', a*)       # Target evaluates

        This decoupling reduces overestimation bias.
        """
        # Online network selects best actions
        online_next_q = self.q_network.forward(next_states)
        best_actions = np.argmax(online_next_q, axis=1)

        # Target network evaluates those actions
        target_next_q = self.target_network.forward(next_states)
        double_q_values = target_next_q[np.arange(len(best_actions)), best_actions]

        # TD targets
        targets = rewards + self.config.gamma * double_q_values * (1 - dones.astype(np.float32))

        return targets

    def train_step(self) -> Optional[float]:
        """
        Perform one training step with Double DQN target computation.

        Returns:
            Loss value, or None if buffer not ready
        """
        if not self.buffer.is_ready(self.config.batch_size):
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size
        )

        # Compute Double DQN targets
        targets = self.compute_double_dqn_targets(rewards, next_states, dones)

        # Update Q-network
        loss = self.q_network.backward(states, actions, targets)
        self.training_losses.append(loss)

        # Update target network
        self.total_steps += 1
        if self.config.use_soft_update:
            self._soft_update_target()
        else:
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


def train_double_dqn(env, config: DoubleDQNConfig) -> Tuple[DoubleDQNAgent, List[float], List[int]]:
    """
    Train Double DQN agent on environment.

    Args:
        env: Gymnasium environment
        config: Double DQN configuration

    Returns:
        agent: Trained Double DQN agent
        episode_returns: Return of each episode
        episode_lengths: Length of each episode
    """
    agent = DoubleDQNAgent(config)
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
                  f"Epsilon: {agent.epsilon:.3f}")

    return agent, episode_returns, episode_lengths


def evaluate_agent(env, agent: DoubleDQNAgent, n_episodes: int = 100,
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


def compare_dqn_vs_double_dqn(env_name: str = "CartPole-v1",
                              n_episodes: int = 300,
                              n_runs: int = 3) -> Dict:
    """
    Compare standard DQN vs Double DQN.

    Runs both algorithms multiple times and compares:
    - Learning curves
    - Final performance
    - Q-value estimates (to show overestimation reduction)
    """
    from dqn import DQNAgent, train_dqn

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    results = {
        'dqn': {'returns': [], 'lengths': [], 'final_returns': []},
        'double_dqn': {'returns': [], 'lengths': [], 'final_returns': []}
    }

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{n_runs}")
        print(f"{'='*60}")

        seed = 42 + run * 100

        # Standard DQN
        print(f"\n--- Standard DQN ---")
        dqn_config = DQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            n_episodes=n_episodes,
            seed=seed
        )
        dqn_agent, dqn_returns, dqn_lengths = train_dqn(env, dqn_config)
        dqn_mean, dqn_std = evaluate_agent(env, dqn_agent, n_episodes=50, seed=seed)
        print(f"Final evaluation: {dqn_mean:.1f} +/- {dqn_std:.1f}")

        results['dqn']['returns'].append(dqn_returns)
        results['dqn']['lengths'].append(dqn_lengths)
        results['dqn']['final_returns'].append(dqn_mean)

        # Double DQN
        print(f"\n--- Double DQN ---")
        ddqn_config = DoubleDQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            n_episodes=n_episodes,
            seed=seed
        )
        ddqn_agent, ddqn_returns, ddqn_lengths = train_double_dqn(env, ddqn_config)
        ddqn_mean, ddqn_std = evaluate_agent(env, ddqn_agent, n_episodes=50, seed=seed)
        print(f"Final evaluation: {ddqn_mean:.1f} +/- {ddqn_std:.1f}")

        results['double_dqn']['returns'].append(ddqn_returns)
        results['double_dqn']['lengths'].append(ddqn_lengths)
        results['double_dqn']['final_returns'].append(ddqn_mean)

    env.close()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"DQN:        {np.mean(results['dqn']['final_returns']):.1f} +/- "
          f"{np.std(results['dqn']['final_returns']):.1f}")
    print(f"Double DQN: {np.mean(results['double_dqn']['final_returns']):.1f} +/- "
          f"{np.std(results['double_dqn']['final_returns']):.1f}")

    return results


# Demonstration
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Double DQN Training Demo on CartPole-v1")
    print("=" * 70)

    # Create environment
    env = gym.make("CartPole-v1")

    # Configuration
    config = DoubleDQNConfig(
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
    print(f"  Algorithm: Double DQN")

    print("\nTraining Double DQN...")
    print("-" * 50)

    agent, returns, lengths = train_double_dqn(env, config)

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
        axes[0].plot(smoothed, color='green', linewidth=1.5, label='Double DQN')
    else:
        axes[0].plot(lengths, color='green', linewidth=1.5, label='Double DQN')

    axes[0].axhline(y=195, color='gray', linestyle='--', label='Solved (195)')
    axes[0].axhline(y=500, color='red', linestyle=':', alpha=0.5, label='Max (500)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Length')
    axes[0].set_title('Double DQN Learning Curve')
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

    save_path = os.path.join(os.path.dirname(__file__), 'double_dqn_cartpole.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")

    plt.show()
    env.close()
