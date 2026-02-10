"""
Proximal Policy Optimization (PPO) - PyTorch Implementation

PPO with clipped surrogate objective, GAE, and multi-epoch mini-batch training.
Supports both discrete (Categorical) and continuous (Gaussian) action spaces.

The key insight: constrain policy updates so the new policy stays close to the old one,
preventing the catastrophic performance collapses common in vanilla policy gradients.

L^CLIP(θ) = E_t[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

Reference: Schulman et al., 2017 - "Proximal Policy Optimization Algorithms"

Author: RL Speedrun
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import gymnasium as gym
from typing import Tuple, List, Optional, Union


# ==============================================================================
# Network Architectures
# ==============================================================================

class DiscreteActorCritic(nn.Module):
    """Actor-Critic for discrete action spaces (e.g., CartPole, LunarLander)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal initialization (standard for policy gradient methods)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
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

    def get_deterministic_action(self, state: np.ndarray) -> int:
        """Greedy action for evaluation."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            dist, _ = self.forward(state_t)
            return dist.probs.argmax(dim=1).item()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO ratio computation."""
        dist, values = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


class ContinuousActorCritic(nn.Module):
    """Actor-Critic for continuous action spaces (e.g., BipedalWalker).

    Uses separate actor and critic networks to prevent value loss gradients
    from corrupting policy features. This is the SB3 default for continuous
    environments and is essential for stable PPO training.

    Actor outputs a Gaussian policy with learnable state-independent log_std.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Separate actor network (policy gradients don't affect critic features)
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Separate critic network (value gradients don't affect actor features)
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Orthogonal initialization
        for module in [self.actor_net, self.critic_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        # Actor path
        actor_features = self.actor_net(x)
        mean = self.actor_mean(actor_features)
        # Clamp log_std to prevent numerical issues (std in [0.007, 7.4])
        log_std_clamped = torch.clamp(self.actor_log_std, -5.0, 2.0)
        std = log_std_clamped.exp().expand_as(mean)
        dist = Normal(mean, std)
        # Critic path (separate computation graph)
        critic_features = self.critic_net(x)
        value = self.critic_head(critic_features).squeeze(-1)
        return dist, value

    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action, return (action, log_prob, value).
        Log prob is summed across action dimensions (independent Gaussians).
        Log prob is computed on the CLAMPED action to match evaluate().
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        dist, value = self.forward(state_t)
        action = dist.sample()
        # Clamp to action bounds FIRST, then compute log_prob on clamped action
        # This ensures consistency with evaluate() which also uses clamped actions
        action_clamped = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action_clamped).sum(dim=-1)
        return action_clamped.squeeze(0).numpy(), log_prob.item(), value.item()

    def get_deterministic_action(self, state: np.ndarray) -> np.ndarray:
        """Mean action for evaluation."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            actor_features = self.actor_net(state_t)
            mean = self.actor_mean(actor_features)
            return torch.clamp(mean, -1.0, 1.0).squeeze(0).numpy()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO ratio computation."""
        dist, values = self.forward(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values, entropy


# ==============================================================================
# Rollout Buffer
# ==============================================================================

class RolloutBuffer:
    """Stores rollout data and supports mini-batch iteration for PPO.

    Unlike A2C which uses data once, PPO reuses rollout data across
    multiple epochs with shuffled mini-batches.
    """

    def __init__(self, n_steps: int, state_dim: int, action_dim: int,
                 is_continuous: bool = False):
        self.n_steps = n_steps
        self.is_continuous = is_continuous

        self.states = np.zeros((n_steps, state_dim), dtype=np.float32)
        if is_continuous:
            self.actions = np.zeros((n_steps, action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(n_steps, dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)
        self.ptr = 0

    def store(self, state, action, reward, value, log_prob, done):
        """Store a single transition."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1

    def compute_gae(self, next_value: float, gamma: float, lam: float):
        """Compute GAE advantages and returns in-place.

        GAE(γ,λ): A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_val = next_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_val = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_batches(self, batch_size: int):
        """Yield shuffled mini-batches as tensors for one epoch."""
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)

        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            batch_idx = indices[start:end]

            states = torch.FloatTensor(self.states[batch_idx])
            if self.is_continuous:
                actions = torch.FloatTensor(self.actions[batch_idx])
            else:
                actions = torch.LongTensor(self.actions[batch_idx])
            old_log_probs = torch.FloatTensor(self.log_probs[batch_idx])
            advantages = torch.FloatTensor(self.advantages[batch_idx])
            returns = torch.FloatTensor(self.returns[batch_idx])

            yield states, actions, old_log_probs, advantages, returns

    @property
    def is_full(self):
        return self.ptr >= self.n_steps

    def reset(self):
        self.ptr = 0


# ==============================================================================
# PPO Agent
# ==============================================================================

class PPOAgent:
    """PPO Agent with clipped surrogate objective.

    Supports both discrete and continuous action spaces.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 is_continuous: bool = False,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 n_steps: int = 2048,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 hidden_dim: int = 64,
                 seed: int = 42):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.is_continuous = is_continuous

        torch.manual_seed(seed)
        np.random.seed(seed)

        if is_continuous:
            self.network = ContinuousActorCritic(state_dim, action_dim, hidden_dim)
        else:
            self.network = DiscreteActorCritic(state_dim, action_dim, hidden_dim)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer(n_steps, state_dim, action_dim, is_continuous)

        self.training_stats = []
        self.best_avg_return = -float('inf')
        self.best_state_dict = None

    def select_action(self, state: np.ndarray, training: bool = True):
        """Select action from policy.

        Returns:
            training=True: (action, log_prob, value)
            training=False: action (deterministic)
        """
        if training:
            return self.network.get_action(state)
        else:
            return self.network.get_deterministic_action(state)

    def store(self, state, action, reward, value, log_prob, done):
        """Store transition in rollout buffer."""
        self.buffer.store(state, action, reward, value, log_prob, done)

    def update(self, next_value: float = 0.0) -> dict:
        """Core PPO update with clipped surrogate objective.

        1. Compute GAE advantages from collected rollout
        2. Normalize advantages
        3. Train for n_epochs with shuffled mini-batches
        4. Reset buffer
        """
        if self.buffer.ptr == 0:
            return {}

        # Compute GAE
        self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)

        # Normalize advantages across full buffer
        advs = self.buffer.advantages[:self.buffer.ptr]
        self.buffer.advantages[:self.buffer.ptr] = \
            (advs - advs.mean()) / (advs.std() + 1e-8)

        # Multi-epoch mini-batch training
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_approx_kls = []
        all_clip_fracs = []

        for epoch in range(self.n_epochs):
            for states, actions, old_log_probs, advantages, returns in \
                    self.buffer.get_batches(self.batch_size):

                # Evaluate current policy on stored transitions
                log_probs, values, entropy = self.network.evaluate(states, actions)

                # Probability ratio: π_θ(a|s) / π_θ_old(a|s)
                # Clamp log-ratio to prevent overflow in exp()
                log_ratio = log_probs - old_log_probs
                log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
                ratio = torch.exp(log_ratio)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range,
                                    1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns)

                # Entropy bonus (encourages exploration)
                entropy_bonus = entropy.mean()

                # Combined loss
                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy_bonus)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl = (old_log_probs - log_probs).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_range) \
                        .float().mean().item()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy_bonus.item())
                all_approx_kls.append(approx_kl)
                all_clip_fracs.append(clip_frac)

        # Reset buffer for next rollout
        self.buffer.reset()

        stats = {
            'policy_loss': np.mean(all_policy_losses),
            'value_loss': np.mean(all_value_losses),
            'entropy': np.mean(all_entropies),
            'approx_kl': np.mean(all_approx_kls),
            'clip_fraction': np.mean(all_clip_fracs),
        }
        self.training_stats.append(stats)
        return stats

    def save_if_best(self, avg_return: float):
        """Save model if it's the best so far."""
        if avg_return > self.best_avg_return:
            self.best_avg_return = avg_return
            self.best_state_dict = {
                k: v.clone() for k, v in self.network.state_dict().items()
            }

    def load_best(self):
        """Load the best saved model."""
        if self.best_state_dict is not None:
            self.network.load_state_dict(self.best_state_dict)


# ==============================================================================
# Training & Evaluation
# ==============================================================================

def train(env_name: str = "CartPole-v1", n_episodes: int = 500,
          seed: int = 42, **kwargs) -> Tuple['PPOAgent', List[float], List[int]]:
    """Train PPO on environment.

    Auto-detects discrete vs continuous action space.
    Collects n_steps transitions before each PPO update.
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]

    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    if is_continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim,
                     is_continuous=is_continuous, seed=seed, **kwargs)

    episode_returns = []
    episode_lengths = []

    state, _ = env.reset(seed=seed)
    ep_return = 0
    ep_length = 0
    ep_count = 0

    while ep_count < n_episodes:
        # Collect n_steps transitions
        for _ in range(agent.n_steps):
            action, log_prob, value = agent.select_action(state, training=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, value, log_prob, done)

            ep_return += reward
            ep_length += 1
            state = next_state

            if done:
                episode_returns.append(ep_return)
                episode_lengths.append(ep_length)
                ep_count += 1

                # Save best model
                if ep_count >= 50:
                    avg = np.mean(episode_returns[-50:])
                    agent.save_if_best(avg)

                if ep_count % 50 == 0:
                    avg = np.mean(episode_returns[-50:])
                    stats = agent.training_stats[-1] if agent.training_stats else {}
                    clip_f = stats.get('clip_fraction', 0)
                    ent = stats.get('entropy', 0)
                    print(f"Episode {ep_count:4d} | Avg Return: {avg:7.1f} | "
                          f"Entropy: {ent:.3f} | Clip: {clip_f:.3f}")

                ep_return = 0
                ep_length = 0
                state, _ = env.reset(seed=seed + ep_count)

                if ep_count >= n_episodes:
                    break

        # Bootstrap value for incomplete episode
        if agent.buffer.ptr > 0:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                _, next_val = agent.network(state_t)
                next_val = next_val.item() if not done else 0.0

            agent.update(next_val)

    # Restore best model for evaluation
    agent.load_best()
    env.close()
    return agent, episode_returns, episode_lengths


def evaluate(agent: 'PPOAgent', env_name: str = "CartPole-v1",
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
    import os

    print("=" * 70)
    print("PPO Training on CartPole-v1 (PyTorch)")
    print("=" * 70)

    agent, returns, lengths = train(
        "CartPole-v1",
        n_episodes=500,
        seed=42,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        lr=3e-4,
        clip_range=0.2,
        entropy_coef=0.01,
    )

    mean, std = evaluate(agent)
    print(f"\nEvaluation (100 episodes): {mean:.1f} +/- {std:.1f}")
    print("SOLVED!" if mean >= 195 else f"Not solved (need >= 195)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Learning curve
    window = 20
    smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
    axes[0].plot(smoothed, color='blue', linewidth=1.5)
    axes[0].axhline(y=195, color='green', linestyle='--', label='Solved (195)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Length')
    axes[0].set_title('PPO Learning Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Entropy
    if agent.training_stats:
        entropies = [s['entropy'] for s in agent.training_stats]
        axes[1].plot(entropies, color='purple', linewidth=1.5)
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Policy Entropy')
    axes[1].grid(True, alpha=0.3)

    # Clip fraction
    if agent.training_stats:
        clip_fracs = [s['clip_fraction'] for s in agent.training_stats]
        axes[2].plot(clip_fracs, color='orange', linewidth=1.5)
    axes[2].set_xlabel('Update Step')
    axes[2].set_ylabel('Clip Fraction')
    axes[2].set_title('PPO Clip Fraction')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'ppo_cartpole.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
