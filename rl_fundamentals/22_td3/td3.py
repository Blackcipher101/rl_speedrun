"""
Twin Delayed DDPG (TD3) - PyTorch Implementation

TD3 addresses three failure modes of DDPG:
1. Overestimation bias → Twin Q-networks (take min)
2. Noisy critic gradients → Delayed policy updates
3. Brittle Q-function → Target policy smoothing

These three simple fixes dramatically improve DDPG's stability.

Reference: Fujimoto et al., 2018 - "Addressing Function Approximation Error
           in Actor-Critic Methods"

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

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '21_ddpg'))
from ddpg import DeterministicActor, ReplayBuffer, GaussianNoise


# ==============================================================================
# Network Architectures
# ==============================================================================

class TD3Critic(nn.Module):
    """Twin Q-networks: two independent Q(s,a) estimators.

    Using min(Q1, Q2) for the target reduces overestimation bias.
    Both networks have identical architecture but different initializations.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 (independent network)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        for net in [self.q1, self.q2]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.uniform_(net[-1].weight, -3e-3, 3e-3)
            nn.init.uniform_(net[-1].bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (Q1(s,a), Q2(s,a))."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Q1 only — used for actor gradient (avoids computing Q2)."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


# ==============================================================================
# TD3 Agent
# ==============================================================================

class TD3Agent:
    """Twin Delayed DDPG agent.

    Three improvements over DDPG:
    1. Twin critics: min(Q1, Q2) reduces overestimation
    2. Delayed actor: update actor every policy_delay critic updates
    3. Target smoothing: add noise to target actions as regularizer
    """

    def __init__(self, state_dim: int, action_dim: int,
                 action_high: float = 1.0,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 noise_sigma: float = 0.1,
                 policy_delay: int = 2,
                 target_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 warmup_steps: int = 1000,
                 hidden_dim: int = 256,
                 seed: int = 42):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.action_high = action_high
        self.action_dim = action_dim
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Actor + target
        self.actor = DeterministicActor(state_dim, action_dim, hidden_dim, action_high)
        self.actor_target = DeterministicActor(state_dim, action_dim, hidden_dim, action_high)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin critic + target
        self.critic = TD3Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = TD3Critic(state_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(buffer_size)
        self.noise = GaussianNoise(action_dim, sigma=noise_sigma)

        self.steps = 0
        self.total_updates = 0
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
        """TD3 update with twin critics and delayed actor.

        Every call: update both critics
        Every policy_delay calls: update actor + soft update targets
        """
        if len(self.buffer) < self.batch_size:
            return {}

        self.total_updates += 1
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # --- Target computation with smoothing ---
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = torch.randn_like(actions) * self.target_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_actions = self.actor_target(next_states) + noise
            next_actions = next_actions.clamp(-self.action_high, self.action_high)

            # Twin Q targets: take the minimum (reduces overestimation)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2).squeeze(-1)
            y = rewards + self.gamma * target_q * (1 - dones)

        # --- Critic update (both Q1 and Q2) ---
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1.squeeze(-1), y) + \
                      F.mse_loss(current_q2.squeeze(-1), y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        stats = {
            'critic_loss': critic_loss.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item(),
        }

        # --- Delayed actor update ---
        if self.total_updates % self.policy_delay == 0:
            actor_actions = self.actor(states)
            actor_loss = -self.critic.q1_forward(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update targets (only when actor updates)
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)

            stats['actor_loss'] = actor_loss.item()

        self.training_stats.append(stats)
        return stats

    def _soft_update(self, target: nn.Module, source: nn.Module):
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
          seed: int = 42, **kwargs) -> Tuple[TD3Agent, List[float], List[int]]:
    """Train TD3 on continuous control environment."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = float(env.action_space.high[0])

    agent = TD3Agent(state_dim, action_dim, action_high=action_high,
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
            q1 = agent.training_stats[-1].get('q1_mean', 0) if agent.training_stats else 0
            q2 = agent.training_stats[-1].get('q2_mean', 0) if agent.training_stats else 0
            print(f"Episode {ep+1:4d} | Avg Return: {avg:8.1f} | Q1: {q1:7.2f} | Q2: {q2:7.2f}")

    agent.load_best()
    env.close()
    return agent, episode_returns, episode_lengths


def evaluate(agent: TD3Agent, env_name: str = "Pendulum-v1",
             n_episodes: int = 100, seed: int = 0) -> Tuple[float, float]:
    """Evaluate trained agent deterministically."""
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


# ==============================================================================
# Demo
# ==============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("TD3 Training on Pendulum-v1")
    print("=" * 70)
    print()
    print("Environment: Pendulum-v1")
    print("  State dim: 3 (cos θ, sin θ, θ̇)")
    print("  Action dim: 1 (torque in [-2, 2])")
    print("  Goal: keep pendulum upright (return >= -200)")
    print()
    print("TD3 Improvements over DDPG:")
    print("  1. Twin Q-networks (min reduces overestimation)")
    print("  2. Delayed policy updates (every 2 critic updates)")
    print("  3. Target policy smoothing (noise on target actions)")
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
        noise_sigma=0.1,
        policy_delay=2,
        target_noise=0.2,
        noise_clip=0.5,
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
    axes[0, 0].set_title('TD3 Learning Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Return distribution
    axes[0, 1].hist(returns[-50:], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=-200, color='green', linestyle='--', label='Solved')
    axes[0, 1].set_xlabel('Return')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Return Distribution (Last 50)')
    axes[0, 1].legend()

    # Twin Q-values (show they're similar but not identical)
    if agent.training_stats:
        step = max(1, len(agent.training_stats) // 500)
        q1s = [s['q1_mean'] for s in agent.training_stats[::step]]
        q2s = [s['q2_mean'] for s in agent.training_stats[::step]]
        axes[1, 0].plot(q1s, color='blue', linewidth=1, alpha=0.8, label='Q1')
        axes[1, 0].plot(q2s, color='red', linewidth=1, alpha=0.8, label='Q2')
        axes[1, 0].legend()
    axes[1, 0].set_xlabel(f'Update Step (x{step})')
    axes[1, 0].set_ylabel('Mean Q-Value')
    axes[1, 0].set_title('Twin Q-Values (Q1 vs Q2)')
    axes[1, 0].grid(True, alpha=0.3)

    # Critic loss
    if agent.training_stats:
        c_losses = [s['critic_loss'] for s in agent.training_stats[::step]]
        axes[1, 1].plot(c_losses, color='red', linewidth=1)
    axes[1, 1].set_xlabel(f'Update Step (x{step})')
    axes[1, 1].set_ylabel('Critic Loss')
    axes[1, 1].set_title('Critic Loss')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'td3_pendulum.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
