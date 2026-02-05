"""
LunarLander Solver using Double DQN

Demonstrates DQN on a more challenging control task.
LunarLander has 8-dimensional state and 4 discrete actions.

Solved: Average return >= 200 over 100 consecutive episodes.

Author: RL Speedrun
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Tuple, List
import os
import sys

# Add parent directories for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '13_dqn_fundamentals'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '14_dqn_improvements'))

from dqn import DQNetwork
from replay_buffer import ReplayBuffer
from double_dqn import DoubleDQNConfig


class LunarLanderDQN:
    """
    DQN agent specifically tuned for LunarLander.

    LunarLander is harder than CartPole:
    - 8-dimensional state (vs 4)
    - 4 actions (vs 2)
    - Sparse rewards
    - Longer episodes needed
    """

    def __init__(self, config: DoubleDQNConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Networks
        self.q_network = DQNetwork(
            config.state_dim, config.action_dim,
            config.hidden_dims, config.learning_rate, config.seed
        )
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
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(self.config.action_dim)
        q_values = self.q_network.forward(state)
        return int(np.argmax(q_values))

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if not self.buffer.is_ready(self.config.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size
        )

        # Double DQN targets
        online_next_q = self.q_network.forward(next_states)
        best_actions = np.argmax(online_next_q, axis=1)

        target_next_q = self.target_network.forward(next_states)
        selected_q = target_next_q[np.arange(len(best_actions)), best_actions]

        targets = rewards + self.config.gamma * selected_q * (1 - dones.astype(np.float32))

        loss = self.q_network.backward(states, actions, targets)
        self.training_losses.append(loss)

        self.total_steps += 1
        if self.total_steps % self.config.target_update_freq == 0:
            self.target_network.set_params(self.q_network.get_params())

        return loss

    def decay_epsilon(self):
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )


def train_lunarlander(
    n_episodes: int = 1000,
    seed: int = 42
) -> Tuple[LunarLanderDQN, List[float], List[int]]:
    """
    Train DQN on LunarLander-v2.

    Args:
        n_episodes: Training episodes
        seed: Random seed

    Returns:
        agent: Trained agent
        returns: Episode returns
        lengths: Episode lengths
    """
    env = gym.make("LunarLander-v3")

    # LunarLander-tuned hyperparameters
    config = DoubleDQNConfig(
        state_dim=8,
        action_dim=4,
        hidden_dims=[256, 256],  # Larger network for harder problem
        n_episodes=n_episodes,
        max_steps=1000,
        batch_size=64,
        learning_rate=0.0005,  # Slightly lower for stability
        gamma=0.99,
        buffer_size=100000,  # Larger buffer
        min_buffer_size=5000,  # More warmup
        target_update_freq=200,  # Less frequent updates
        use_soft_update=False,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,  # Slower decay for harder exploration
        seed=seed
    )

    agent = LunarLanderDQN(config)
    episode_returns = []
    episode_lengths = []

    best_avg_return = -float('inf')

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_return = 0
        episode_length = 0

        done = False
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            episode_return += reward
            episode_length += 1
            state = next_state

        agent.decay_epsilon()
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # Progress
        if (episode + 1) % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            if avg_return > best_avg_return:
                best_avg_return = avg_return

            print(f"Episode {episode + 1:4d} | "
                  f"Avg Return: {avg_return:8.2f} | "
                  f"Best: {best_avg_return:8.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

            # Check if solved
            if len(episode_returns) >= 100:
                recent_avg = np.mean(episode_returns[-100:])
                if recent_avg >= 200:
                    print(f"\nSOLVED at episode {episode + 1}!")
                    print(f"Average return over last 100 episodes: {recent_avg:.2f}")
                    break

    env.close()
    return agent, episode_returns, episode_lengths


def evaluate_lunarlander(agent: LunarLanderDQN, n_episodes: int = 100,
                         seed: int = 0, render: bool = False) -> Tuple[float, float]:
    """Evaluate trained agent."""
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
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

    env.close()
    return np.mean(returns), np.std(returns)


def main():
    print("=" * 70)
    print("LunarLander: Double DQN")
    print("=" * 70)

    print("\n" + "-" * 40)
    print("Environment Info")
    print("-" * 40)
    print("  State: [x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact]")
    print("  Actions: 0=nothing, 1=left engine, 2=main engine, 3=right engine")
    print("  Reward: -100 to +100 (crash to perfect landing)")
    print("  Solved: avg return >= 200 over 100 episodes")

    print("\n" + "=" * 70)
    print("Training Double DQN")
    print("=" * 70)
    print("-" * 50)

    agent, returns, lengths = train_lunarlander(n_episodes=1000, seed=42)

    print("-" * 50)
    print("\nEvaluating trained agent...")

    mean_return, std_return = evaluate_lunarlander(agent, n_episodes=100, seed=0)
    print(f"\nFinal evaluation (100 episodes): {mean_return:.2f} +/- {std_return:.2f}")

    solved = mean_return >= 200
    if solved:
        print("SOLVED!")
    else:
        print(f"Not solved (need >= 200)")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curve
    window = 50
    if len(returns) > window:
        smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, color='blue', linewidth=1.5)
    else:
        axes[0].plot(returns, color='blue', linewidth=1.5)

    axes[0].axhline(y=200, color='green', linestyle='--', label='Solved (200)')
    axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Return')
    axes[0].set_title('LunarLander: Double DQN Learning Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Return distribution (last 200 episodes)
    last_n = min(200, len(returns))
    axes[1].hist(returns[-last_n:], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=np.mean(returns[-last_n:]), color='red', linestyle='--',
                    label=f'Mean: {np.mean(returns[-last_n:]):.1f}')
    axes[1].axvline(x=200, color='green', linestyle='--', alpha=0.7, label='Solved threshold')
    axes[1].set_xlabel('Episode Return')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Return Distribution (Last {last_n} Episodes)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'lunarlander_dqn.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
