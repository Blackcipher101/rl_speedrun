"""
Deep Q-Network (DQN) - PyTorch Implementation

Clean PyTorch DQN with experience replay and target network.
This is the production-quality version that reliably solves CartPole.

Author: RL Speedrun
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random
from typing import Tuple, List


class QNetwork(nn.Module):
    """Q-Network: maps states to Q-values for each action."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Simple replay buffer using deque."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with experience replay and target network."""

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 5e-4, gamma: float = 0.99,
                 buffer_size: int = 50000, batch_size: int = 128,
                 tau: float = 0.005, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01, epsilon_decay: float = 0.99,
                 seed: int = 42):

        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cpu")

        # Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.steps = 0
        self.losses = []

        # Track best model for evaluation
        self.best_avg_length = 0
        self.best_state_dict = None

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for taken actions
        q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN style)
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(dim=1)
            next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        # Huber loss
        loss = nn.SmoothL1Loss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Soft update target network: θ' = τθ + (1-τ)θ'
        self.steps += 1
        for tp, op in zip(self.target_network.parameters(), self.q_network.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_if_best(self, avg_length: float):
        """Save model if it's the best so far."""
        if avg_length > self.best_avg_length:
            self.best_avg_length = avg_length
            self.best_state_dict = {k: v.clone() for k, v in self.q_network.state_dict().items()}

    def load_best(self):
        """Load the best saved model."""
        if self.best_state_dict is not None:
            self.q_network.load_state_dict(self.best_state_dict)


def train(env_name: str = "CartPole-v1", n_episodes: int = 500,
          seed: int = 42) -> Tuple[DQNAgent, List[float], List[int]]:
    """Train DQN on environment."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, seed=seed)

    episode_returns = []
    episode_lengths = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_return = 0
        ep_length = 0

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, done)
            agent.train_step()

            ep_return += reward
            ep_length += 1
            state = next_state

        agent.decay_epsilon()
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)

        # Save best model based on rolling average
        if ep >= 50:
            avg = np.mean(episode_lengths[-50:])
            agent.save_if_best(avg)

        if (ep + 1) % 50 == 0:
            avg = np.mean(episode_lengths[-50:])
            print(f"Episode {ep+1:4d} | Avg Length: {avg:6.1f} | Epsilon: {agent.epsilon:.3f}")

    # Restore best model for evaluation
    agent.load_best()

    env.close()
    return agent, episode_returns, episode_lengths


def evaluate(agent: DQNAgent, env_name: str = "CartPole-v1",
             n_episodes: int = 100, seed: int = 0) -> Tuple[float, float]:
    """Evaluate trained agent."""
    env = gym.make(env_name)
    returns = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_return = 0
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return np.mean(returns), np.std(returns)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    print("=" * 70)
    print("DQN Training on CartPole-v1 (PyTorch)")
    print("=" * 70)

    agent, returns, lengths = train("CartPole-v1", n_episodes=500, seed=42)

    mean, std = evaluate(agent)
    print(f"\nEvaluation (100 episodes): {mean:.1f} +/- {std:.1f}")
    print("SOLVED!" if mean >= 195 else f"Not solved (need >= 195)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 20
    smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='blue', linewidth=1.5, label='DQN (PyTorch)')
    ax.axhline(y=195, color='green', linestyle='--', label='Solved (195)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Episode Length')
    ax.set_title('DQN Learning Curve (PyTorch)'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'dqn_torch_cartpole.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
