"""
Deep Deterministic Policy Gradient (DDPG) - PyTorch Implementation

Off-policy actor-critic for continuous action spaces.
Unlike PPO's stochastic Gaussian policy, DDPG uses a deterministic policy
μ(s) that directly outputs actions, with exploration via additive noise.

Key components:
- Deterministic actor μ(s) with target network
- Q-function critic Q(s,a) with target network (takes state+action as input)
- Experience replay buffer (off-policy)
- Soft target updates (Polyak averaging)
- Ornstein-Uhlenbeck or Gaussian exploration noise

∇_θ J ≈ E[∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ(s)]

Reference: Lillicrap et al., 2015 - "Continuous control with deep reinforcement learning"

Author: RL Speedrun
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
from typing import Tuple, List


# ==============================================================================
# Network Architectures
# ==============================================================================

class DeterministicActor(nn.Module):
    """Deterministic policy μ(s) → action.

    Unlike PPO's Gaussian actor that samples from a distribution,
    this outputs a single action vector directly. Tanh squashes to [-1, 1],
    then scaled to the environment's action range.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_high: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.action_high = action_high

        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        # Small init for output layer (smoother initial policy)
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-2].bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.action_high


class DDPGCritic(nn.Module):
    """Q(s, a) network — takes state AND action as input.

    CRITICAL difference from DQN:
    - DQN: state → Q-values for ALL discrete actions
    - DDPG: (state, action) → single Q-value scalar

    State and action are concatenated as input.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ==============================================================================
# Exploration Noise
# ==============================================================================

class OrnsteinUhlenbeckNoise:
    """Temporally correlated noise for smooth exploration.

    dx = θ(μ - x)dt + σ√(dt) N(0,1)

    The mean-reverting property produces smoother trajectories than
    pure Gaussian noise, which can be beneficial for physical control.
    """

    def __init__(self, action_dim: int, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.2):
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def __call__(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state.copy()

    def reset(self):
        self.state = self.mu.copy()


class GaussianNoise:
    """Simple Gaussian exploration noise. Often works as well as OU."""

    def __init__(self, action_dim: int, sigma: float = 0.1):
        self.action_dim = action_dim
        self.sigma = sigma

    def __call__(self) -> np.ndarray:
        return np.random.randn(self.action_dim) * self.sigma

    def reset(self):
        pass


# ==============================================================================
# Replay Buffer
# ==============================================================================

class ReplayBuffer:
    """Replay buffer for off-policy learning with continuous actions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)),
                torch.FloatTensor(np.array(actions)),
                torch.FloatTensor(np.array(rewards, dtype=np.float32)),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(np.array(dones, dtype=np.float32)))

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# DDPG Agent
# ==============================================================================

class DDPGAgent:
    """Deep Deterministic Policy Gradient agent.

    Off-policy actor-critic for continuous control.
    Uses deterministic policy + replay buffer + target networks.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 action_high: float = 1.0,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 noise_type: str = 'ou',
                 noise_sigma: float = 0.2,
                 warmup_steps: int = 1000,
                 hidden_dim: int = 256,
                 seed: int = 42):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.action_high = action_high
        self.action_dim = action_dim

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Actor and target actor
        self.actor = DeterministicActor(state_dim, action_dim, hidden_dim, action_high)
        self.actor_target = DeterministicActor(state_dim, action_dim, hidden_dim, action_high)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic and target critic
        self.critic = DDPGCritic(state_dim, action_dim, hidden_dim)
        self.critic_target = DDPGCritic(state_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Separate optimizers (actor learns slower than critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(buffer_size)

        # Noise for exploration
        if noise_type == 'ou':
            self.noise = OrnsteinUhlenbeckNoise(action_dim, sigma=noise_sigma)
        else:
            self.noise = GaussianNoise(action_dim, sigma=noise_sigma)

        self.steps = 0
        self.training_stats = []
        self.best_avg_return = -float('inf')
        self.best_state_dict = None

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action with optional exploration noise."""
        if training and self.steps < self.warmup_steps:
            return np.random.uniform(-self.action_high, self.action_high,
                                     size=self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state_t).squeeze(0).numpy()

        if training:
            action = action + self.noise() * self.action_high
            action = np.clip(action, -self.action_high, self.action_high)

        return action

    def update(self) -> dict:
        """One DDPG training step.

        1. Sample batch from replay buffer
        2. Update critic: minimize (Q(s,a) - y)^2 where y = r + γQ'(s', μ'(s'))
        3. Update actor: maximize Q(s, μ(s))
        4. Soft update target networks
        """
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # --- Critic update ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions).squeeze(-1)
            y = rewards + self.gamma * target_q * (1 - dones)

        current_q = self.critic(states, actions).squeeze(-1)
        critic_loss = F.mse_loss(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        # Differentiate Q(s, μ(s)) through both critic and actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft update targets ---
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        stats = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_mean': current_q.mean().item(),
        }
        self.training_stats.append(stats)
        return stats

    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Polyak averaging: θ_target = τ·θ_source + (1-τ)·θ_target."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save_if_best(self, avg_return: float):
        if avg_return > self.best_avg_return:
            self.best_avg_return = avg_return
            self.best_state_dict = {
                'actor': {k: v.clone() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.clone() for k, v in self.critic.state_dict().items()},
            }

    def load_best(self):
        if self.best_state_dict is not None:
            self.actor.load_state_dict(self.best_state_dict['actor'])
            self.critic.load_state_dict(self.best_state_dict['critic'])


# ==============================================================================
# Training & Evaluation
# ==============================================================================

def train(env_name: str = "Pendulum-v1", n_episodes: int = 200,
          seed: int = 42, **kwargs) -> Tuple[DDPGAgent, List[float], List[int]]:
    """Train DDPG on continuous control environment.

    Unlike PPO's n_steps rollout, DDPG updates every step (off-policy).
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, action_high=action_high,
                      seed=seed, **kwargs)

    episode_returns = []
    episode_lengths = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        agent.noise.reset()
        ep_return = 0
        ep_length = 0
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            agent.steps += 1

            ep_return += reward
            ep_length += 1
            state = next_state

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)

        if ep >= 50:
            avg = np.mean(episode_returns[-50:])
            agent.save_if_best(avg)

        if (ep + 1) % 50 == 0:
            avg = np.mean(episode_returns[-50:])
            q_mean = agent.training_stats[-1].get('q_mean', 0) if agent.training_stats else 0
            print(f"Episode {ep+1:4d} | Avg Return: {avg:8.1f} | Q mean: {q_mean:7.2f}")

    agent.load_best()
    env.close()
    return agent, episode_returns, episode_lengths


def evaluate(agent: DDPGAgent, env_name: str = "Pendulum-v1",
             n_episodes: int = 100, seed: int = 0) -> Tuple[float, float]:
    """Evaluate trained agent deterministically."""
    env = gym.make(env_name)
    action_high = float(env.action_space.high[0])
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


# ==============================================================================
# Demo
# ==============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    print("=" * 70)
    print("DDPG Training on Pendulum-v1")
    print("=" * 70)
    print()
    print("Environment: Pendulum-v1")
    print("  State dim: 3 (cos θ, sin θ, θ̇)")
    print("  Action dim: 1 (torque in [-2, 2])")
    print("  Reward: -(θ² + 0.1·θ̇² + 0.001·u²)")
    print("  Goal: keep pendulum upright (return >= -200)")
    print()
    print("DDPG Configuration:")
    print("  Network: 256 -> 256 (ReLU)")
    print("  Actor lr: 1e-4, Critic lr: 1e-3")
    print("  Tau: 0.005, Batch: 64, Buffer: 100k")
    print("  Noise: Ornstein-Uhlenbeck (sigma=0.2)")
    print("-" * 50)

    agent, returns, lengths = train(
        "Pendulum-v1",
        n_episodes=200,
        seed=42,
        actor_lr=1e-4,
        critic_lr=1e-3,
        tau=0.005,
        batch_size=64,
        buffer_size=100000,
        noise_type='ou',
        noise_sigma=0.2,
        warmup_steps=1000,
        hidden_dim=256,
    )

    mean, std = evaluate(agent)
    print(f"\nEvaluation (100 episodes): {mean:.1f} +/- {std:.1f}")
    if mean >= -200:
        print("SOLVED!")
    else:
        print(f"Not solved (need >= -200, got {mean:.1f})")

    # 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Learning curve
    window = 10
    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(smoothed, color='blue', linewidth=1.5)
    axes[0, 0].axhline(y=-200, color='green', linestyle='--', label='Solved (-200)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('DDPG Learning Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Return distribution (last 50 episodes)
    axes[0, 1].hist(returns[-50:], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=-200, color='green', linestyle='--', label='Solved')
    axes[0, 1].set_xlabel('Return')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Return Distribution (Last 50)')
    axes[0, 1].legend()

    # Q-value evolution
    if agent.training_stats:
        q_means = [s['q_mean'] for s in agent.training_stats[::50]]
        axes[1, 0].plot(q_means, color='orange', linewidth=1.5)
    axes[1, 0].set_xlabel('Update Step (x50)')
    axes[1, 0].set_ylabel('Mean Q-Value')
    axes[1, 0].set_title('Q-Value Evolution')
    axes[1, 0].grid(True, alpha=0.3)

    # Critic loss
    if agent.training_stats:
        c_losses = [s['critic_loss'] for s in agent.training_stats[::50]]
        axes[1, 1].plot(c_losses, color='red', linewidth=1.5)
    axes[1, 1].set_xlabel('Update Step (x50)')
    axes[1, 1].set_ylabel('Critic Loss')
    axes[1, 1].set_title('Critic Loss')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'ddpg_pendulum.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
