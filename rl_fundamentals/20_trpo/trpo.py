"""
Trust Region Policy Optimization (TRPO) - PyTorch Implementation

PPO's predecessor. Instead of clipping the surrogate objective, TRPO uses
a hard KL divergence constraint enforced via conjugate gradient and line search.

The natural gradient update:
    θ_new = θ_old + sqrt(2δ / g^T F^{-1} g) · F^{-1} g

where:
    g = ∇_θ L(θ)  (policy gradient)
    F = Fisher Information Matrix (second-order curvature)
    δ = max KL divergence (trust region size)

We never form F explicitly. Instead, conjugate gradient solves F^{-1}g
using only Fisher-vector products computed via automatic differentiation.

Reference: Schulman et al., 2015 - "Trust Region Policy Optimization"

Author: RL Speedrun
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import gymnasium as gym
from typing import Tuple, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '18_ppo'))
from ppo import RolloutBuffer


# ==============================================================================
# Network Architectures
# ==============================================================================

class TRPODiscreteActor(nn.Module):
    """Discrete policy network for TRPO."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(x))


class TRPOCritic(nn.Module):
    """Value network V(s) for TRPO. Updated with standard Adam."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ==============================================================================
# TRPO Utilities: Conjugate Gradient, Fisher-Vector Product, Line Search
# ==============================================================================

def flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten all parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat_vector: torch.Tensor):
    """Set model parameters from a flat vector."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_vector[offset:offset + numel].view_as(p))
        offset += numel


def flat_grad(output, params, retain_graph=False, create_graph=False):
    """Compute gradient of output w.r.t. params and flatten."""
    grads = torch.autograd.grad(output, params,
                                 retain_graph=retain_graph,
                                 create_graph=create_graph,
                                 allow_unused=True)
    return torch.cat([g.contiguous().view(-1) if g is not None
                      else torch.zeros(p.numel())
                      for g, p in zip(grads, params)])


def conjugate_gradient(Avp_fn, b, n_steps=10, residual_tol=1e-10):
    """Solve Ax = b via conjugate gradient, where A is the Fisher matrix.

    We never form A. Instead, Avp_fn computes A @ v for any vector v.
    This finds the natural gradient direction F^{-1} g in O(n_steps) Avp calls.

    Args:
        Avp_fn: Function computing Fisher-vector product F @ v
        b: Policy gradient vector g (right-hand side)
        n_steps: Max CG iterations (10 is standard)
        residual_tol: Early stopping tolerance

    Returns:
        x: Approximate solution to Fx = g
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)

    for _ in range(n_steps):
        Ap = Avp_fn(p)
        pAp = p.dot(Ap)
        if pAp <= 0:
            # Negative curvature — break to avoid divergence
            break
        alpha = rdotr / (pAp + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr

    return x


def compute_fvp(actor, states, vector, damping=0.1):
    """Compute Fisher-vector product F @ v without forming F.

    The Fisher Information Matrix is the Hessian of KL divergence.
    We compute Fv via two automatic differentiation passes:
    1. KL = KL(π_θ_detached || π_θ)  (gradient flows only through new policy)
    2. g_kl = ∂KL/∂θ  (with create_graph=True)
    3. g_kl^T v  (dot product)
    4. ∂(g_kl^T v)/∂θ = Fv

    Args:
        actor: Policy network
        states: Batch of states
        vector: The vector v to multiply with F
        damping: Regularization for numerical stability

    Returns:
        Fv + damping * v
    """
    params = list(actor.parameters())

    # Get current distribution
    dist = actor(states)

    # KL divergence between detached (old) and current (new) distribution
    # For categorical: KL(p_old || p_new) = sum(p_old * log(p_old / p_new))
    if isinstance(dist, Categorical):
        old_probs = dist.probs.detach()
        kl = (old_probs * (old_probs.log() - dist.probs.log())).sum(dim=-1).mean()
    else:
        # Gaussian
        old_mean = dist.loc.detach()
        old_std = dist.scale.detach()
        kl = (torch.log(dist.scale / old_std) +
              (old_std.pow(2) + (old_mean - dist.loc).pow(2)) /
              (2 * dist.scale.pow(2)) - 0.5).sum(dim=-1).mean()

    # First gradient: g_kl = ∂KL/∂θ (with computation graph preserved)
    g_kl = flat_grad(kl, params, retain_graph=True, create_graph=True)

    # Dot product with vector
    gvp = g_kl.dot(vector)

    # Second gradient: ∂(g_kl^T v)/∂θ = Fv
    fvp = flat_grad(gvp, params, retain_graph=True)

    return fvp + damping * vector


def line_search(actor, loss_fn, kl_fn, old_params, fullstep,
                max_kl, max_steps=10):
    """Backtracking line search to satisfy KL constraint.

    Try step sizes: fullstep, fullstep/2, fullstep/4, ...
    Accept first step where:
    1. KL divergence < max_kl
    2. Loss improves (surrogate loss decreases)

    Returns:
        (success, final_params)
    """
    old_loss = loss_fn().item()

    for step_frac in [0.5 ** i for i in range(max_steps)]:
        new_params = old_params + step_frac * fullstep
        set_flat_params(actor, new_params)

        new_loss = loss_fn().item()
        kl = kl_fn().item()

        # Accept if loss improves and KL constraint satisfied
        if kl < max_kl and new_loss < old_loss:
            return True

    # Revert if no valid step found
    set_flat_params(actor, old_params)
    return False


# ==============================================================================
# TRPO Agent
# ==============================================================================

class TRPOAgent:
    """TRPO Agent with conjugate gradient and line search.

    Key differences from PPO:
    1. No clip_range — uses KL divergence constraint instead
    2. No optimizer for actor — uses CG + line search
    3. Separate Adam optimizer for critic only
    4. Single pass through data for actor (no multi-epoch)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 is_continuous: bool = False,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 max_kl: float = 0.01,
                 cg_iterations: int = 10,
                 line_search_steps: int = 10,
                 damping: float = 0.1,
                 n_critic_updates: int = 5,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 hidden_dim: int = 64,
                 seed: int = 42):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl
        self.cg_iterations = cg_iterations
        self.line_search_steps = line_search_steps
        self.damping = damping
        self.n_critic_updates = n_critic_updates
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.is_continuous = is_continuous

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Separate actor and critic (TRPO updates them independently)
        self.actor = TRPODiscreteActor(state_dim, action_dim, hidden_dim)
        self.critic = TRPOCritic(state_dim, hidden_dim)

        # Only critic uses an optimizer; actor uses CG + line search
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = RolloutBuffer(n_steps, state_dim, action_dim, is_continuous)

        self.training_stats = []
        self.best_avg_return = -float('inf')
        self.best_state_dict = None

    def select_action(self, state: np.ndarray, training: bool = True):
        """Select action from policy."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            dist = self.actor(state_t)
            value = self.critic(state_t)

            if training:
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob.item(), value.item()
            else:
                return dist.probs.argmax(dim=1).item()

    def store(self, state, action, reward, value, log_prob, done):
        self.buffer.store(state, action, reward, value, log_prob, done)

    def update(self, next_value: float = 0.0) -> dict:
        """Core TRPO update.

        1. Compute GAE advantages
        2. Update actor via CG + line search (natural gradient)
        3. Update critic via standard Adam (multiple epochs)
        """
        if self.buffer.ptr == 0:
            return {}

        # Compute GAE advantages
        self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)

        # Get all data as tensors
        n = self.buffer.ptr
        states = torch.FloatTensor(self.buffer.states[:n])
        actions = torch.LongTensor(self.buffer.actions[:n])
        old_log_probs = torch.FloatTensor(self.buffer.log_probs[:n])
        advantages = torch.FloatTensor(self.buffer.advantages[:n])
        returns = torch.FloatTensor(self.buffer.returns[:n])

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Actor update (TRPO: CG + line search) ---
        actor_stats = self._update_actor(states, actions, old_log_probs, advantages)

        # --- Critic update (standard gradient descent) ---
        critic_loss = self._update_critic(states, returns)

        self.buffer.reset()

        stats = {**actor_stats, 'value_loss': critic_loss}
        self.training_stats.append(stats)
        return stats

    def _update_actor(self, states, actions, old_log_probs, advantages) -> dict:
        """TRPO actor update using conjugate gradient + line search."""
        actor_params = list(self.actor.parameters())

        # 1. Compute surrogate loss and its gradient
        def surrogate_loss():
            dist = self.actor(states)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            return -(ratio * advantages).mean()

        loss = surrogate_loss()
        # Gradient of negative surrogate: points toward INCREASING negative loss
        g = flat_grad(loss, actor_params, retain_graph=True)

        # If gradient is near zero, skip update
        if g.norm() < 1e-8:
            return {'policy_loss': loss.item(), 'kl': 0.0, 'ls_success': 1.0}

        # 2. Negate gradient: we want to DECREASE the negative loss (maximize surrogate)
        #    CG solves F^{-1} (-g) = natural gradient descent direction
        def fvp_fn(v):
            return compute_fvp(self.actor, states, v, self.damping)

        step_dir = conjugate_gradient(fvp_fn, -g, self.cg_iterations)

        # 3. Compute step size: β = sqrt(2δ / x^T F x)
        shs = 0.5 * step_dir.dot(fvp_fn(step_dir))
        if shs <= 0:
            return {'policy_loss': loss.item(), 'kl': 0.0, 'ls_success': 0.0}

        lm = torch.sqrt(shs / self.max_kl)
        fullstep = step_dir / (lm + 1e-8)

        # 4. Line search with KL constraint
        old_params = flat_params(self.actor)

        def kl_fn():
            dist = self.actor(states)
            if isinstance(dist, Categorical):
                old_probs = Categorical(logits=self.actor.net(states).detach()).probs
                return (old_probs * (old_probs.log() - dist.probs.log())).sum(-1).mean()
            return torch.tensor(0.0)

        success = line_search(
            self.actor, surrogate_loss, kl_fn, old_params,
            fullstep, self.max_kl, self.line_search_steps
        )

        # Compute final KL
        with torch.no_grad():
            final_kl = kl_fn().item()

        return {
            'policy_loss': surrogate_loss().item(),
            'kl': final_kl,
            'ls_success': float(success),
        }

    def _update_critic(self, states, returns) -> float:
        """Standard gradient descent for critic (multiple epochs)."""
        total_loss = 0
        for _ in range(self.n_critic_updates):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_returns = returns[batch_idx]

                values = self.critic(batch_states)
                loss = F.mse_loss(values, batch_returns)

                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()

                total_loss += loss.item()

        return total_loss / (self.n_critic_updates * max(1, len(states) // self.batch_size))

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

def train(env_name: str = "CartPole-v1", n_episodes: int = 500,
          seed: int = 42, **kwargs) -> Tuple[TRPOAgent, List[float], List[int]]:
    """Train TRPO on environment. Same rollout structure as PPO."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]

    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    if is_continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    agent = TRPOAgent(state_dim, action_dim,
                      is_continuous=is_continuous, seed=seed, **kwargs)

    episode_returns = []
    episode_lengths = []

    state, _ = env.reset(seed=seed)
    ep_return = 0
    ep_length = 0
    ep_count = 0

    while ep_count < n_episodes:
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

                if ep_count >= 50:
                    avg = np.mean(episode_returns[-50:])
                    agent.save_if_best(avg)

                if ep_count % 50 == 0:
                    avg = np.mean(episode_returns[-50:])
                    stats = agent.training_stats[-1] if agent.training_stats else {}
                    kl = stats.get('kl', 0)
                    ls = stats.get('ls_success', 0)
                    print(f"Episode {ep_count:4d} | Avg Return: {avg:7.1f} | "
                          f"KL: {kl:.4f} | LS Success: {ls:.1f}")

                ep_return = 0
                ep_length = 0
                state, _ = env.reset(seed=seed + ep_count)

                if ep_count >= n_episodes:
                    break

        # Bootstrap and update
        if agent.buffer.ptr > 0:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                next_val = agent.critic(state_t).item() if not done else 0.0
            agent.update(next_val)

    agent.load_best()
    env.close()
    return agent, episode_returns, episode_lengths


def evaluate(agent: TRPOAgent, env_name: str = "CartPole-v1",
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
    print("TRPO Training on CartPole-v1")
    print("=" * 70)
    print()
    print("Environment: CartPole-v1")
    print("  State dim: 4 (cart position, velocity, pole angle, angular velocity)")
    print("  Action dim: 2 (push left/right)")
    print("  Solved: avg length >= 195 over 100 episodes")
    print()
    print("TRPO Configuration:")
    print("  Network: 64 -> 64 (Tanh)")
    print("  Max KL: 0.01")
    print("  CG iterations: 10")
    print("  Line search steps: 10")
    print("  Damping: 0.1")
    print("  Critic lr: 1e-3 (Adam)")
    print("-" * 50)

    agent, returns, lengths = train(
        "CartPole-v1",
        n_episodes=500,
        seed=42,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        max_kl=0.01,
        cg_iterations=10,
        line_search_steps=10,
        damping=0.1,
        n_critic_updates=5,
        n_steps=2048,
        hidden_dim=64,
    )

    mean, std = evaluate(agent)
    print(f"\nEvaluation (100 episodes): {mean:.1f} +/- {std:.1f}")
    print("SOLVED!" if mean >= 195 else f"Not solved (need >= 195)")

    # 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Learning curve
    window = 20
    smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
    axes[0].plot(smoothed, color='blue', linewidth=1.5)
    axes[0].axhline(y=195, color='green', linestyle='--', label='Solved (195)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Length')
    axes[0].set_title('TRPO Learning Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # KL divergence
    if agent.training_stats:
        kls = [s['kl'] for s in agent.training_stats]
        axes[1].plot(kls, color='purple', linewidth=1.5)
        axes[1].axhline(y=0.01, color='red', linestyle='--', label='Max KL (0.01)')
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('KL Divergence')
    axes[1].set_title('KL per Update (Trust Region)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Value loss
    if agent.training_stats:
        vlosses = [s['value_loss'] for s in agent.training_stats]
        axes[2].plot(vlosses, color='red', linewidth=1.5)
    axes[2].set_xlabel('Update Step')
    axes[2].set_ylabel('Value Loss')
    axes[2].set_title('Critic Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'trpo_cartpole.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
