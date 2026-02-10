"""
Advantage Actor-Critic (A2C) - PyTorch Implementation

Clean PyTorch A2C with GAE and entropy regularization.
This is the production-quality version that reliably solves CartPole.

Author: RL Speedrun
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from typing import Tuple, List


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Shared feature layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal initialization (standard for policy gradient methods)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Smaller init for output heads
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        dist = Categorical(logits=logits)
        return dist, value

    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action, return (action, log_prob, value)."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        dist, value = self.forward(state_t)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for given states."""
        dist, values = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


def compute_gae(rewards, values, next_values, dones,
                gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


class A2CAgent:
    """A2C Agent with GAE and entropy regularization."""

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-3, gamma: float = 0.99,
                 gae_lambda: float = 0.95, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 n_steps: int = 64, seed: int = 42):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Rollout storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        self.training_stats = []

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Select action from policy."""
        if training:
            action, log_prob, value = self.network.get_action(state)
            self.log_probs.append(log_prob)
            self.values.append(value)
            return action, value
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                dist, _ = self.network(state_t)
                return dist.probs.argmax(dim=1).item(), 0.0

    def store(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self, next_value: float = 0.0):
        """Update networks using collected rollout."""
        if len(self.states) == 0:
            return

        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones, dtype=np.float32)
        next_values = np.append(values[1:], next_value)

        # GAE
        advantages, returns = compute_gae(
            rewards, values, next_values, dones,
            self.gamma, self.gae_lambda
        )

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(self.states))
        actions_t = torch.LongTensor(self.actions)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        # Evaluate actions
        log_probs, values_pred, entropy = self.network.evaluate(states_t, actions_t)

        # Policy loss
        policy_loss = -(log_probs * advantages_t).mean()

        # Value loss
        value_loss = F.mse_loss(values_pred, returns_t)

        # Entropy bonus
        entropy_bonus = entropy.mean()

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_bonus.item(),
            'total_loss': loss.item()
        }
        self.training_stats.append(stats)

        # Clear rollout
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        return stats


def train(env_name: str = "CartPole-v1", n_episodes: int = 500,
          seed: int = 42) -> Tuple[A2CAgent, List[float], List[int]]:
    """Train A2C on environment."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim, action_dim, seed=seed)

    episode_returns = []
    episode_lengths = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_return = 0
        ep_length = 0
        steps_since_update = 0

        done = False
        while not done:
            action, value = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, done)
            steps_since_update += 1

            if steps_since_update >= agent.n_steps or done:
                if done:
                    next_val = 0.0
                else:
                    with torch.no_grad():
                        ns_t = torch.FloatTensor(next_state).unsqueeze(0)
                        _, next_val = agent.network(ns_t)
                        next_val = next_val.item()
                agent.update(next_val)
                steps_since_update = 0

            ep_return += reward
            ep_length += 1
            state = next_state

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)

        if (ep + 1) % 50 == 0:
            avg = np.mean(episode_lengths[-50:])
            ent = agent.training_stats[-1]['entropy'] if agent.training_stats else 0
            print(f"Episode {ep+1:4d} | Avg Length: {avg:6.1f} | Entropy: {ent:.3f}")

    env.close()
    return agent, episode_returns, episode_lengths


def evaluate(agent: A2CAgent, env_name: str = "CartPole-v1",
             n_episodes: int = 100, seed: int = 0) -> Tuple[float, float]:
    """Evaluate trained agent."""
    env = gym.make(env_name)
    returns = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_return = 0
        done = False
        while not done:
            action, _ = agent.select_action(state, training=False)
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
    print("A2C Training on CartPole-v1 (PyTorch)")
    print("=" * 70)

    agent, returns, lengths = train("CartPole-v1", n_episodes=1500, seed=42)

    mean, std = evaluate(agent)
    print(f"\nEvaluation (100 episodes): {mean:.1f} +/- {std:.1f}")
    print("SOLVED!" if mean >= 195 else f"Not solved (need >= 195)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 20
    smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
    axes[0].plot(smoothed, color='blue', linewidth=1.5)
    axes[0].axhline(y=195, color='green', linestyle='--', label='Solved (195)')
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Episode Length')
    axes[0].set_title('A2C Learning Curve (PyTorch)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if agent.training_stats:
        entropies = [s['entropy'] for s in agent.training_stats]
        w = min(100, len(entropies) // 5)
        if w > 1:
            sm = np.convolve(entropies, np.ones(w)/w, mode='valid')
            axes[1].plot(sm, color='purple', linewidth=1.5)
        else:
            axes[1].plot(entropies, color='purple', linewidth=1.5)
    axes[1].set_xlabel('Update Step'); axes[1].set_ylabel('Entropy')
    axes[1].set_title('Policy Entropy'); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'a2c_torch_cartpole.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
