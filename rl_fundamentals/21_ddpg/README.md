# Deep Deterministic Policy Gradient (DDPG)

Continuous control with deterministic policies. DDPG combines DQN's ideas (replay buffer, target networks) with an actor-critic architecture that works for continuous action spaces.

---

## The Problem: Continuous Actions

DQN maps states to Q-values for **all** discrete actions. But with continuous actions (e.g., torque = 0.73), we can't enumerate all possible actions:

```
DQN (discrete):  state → [Q(s, left), Q(s, right)]     ← finite actions
DDPG (continuous): state → action ∈ [-1, 1]              ← infinite actions
                   (state, action) → Q(s, a)              ← single Q-value
```

---

## Deterministic Policy Gradient

Instead of sampling from a distribution like PPO, DDPG uses a **deterministic** policy:

$$a = \mu_\theta(s)$$

The policy gradient theorem for deterministic policies:

$$\nabla_\theta J \approx \mathbb{E}\left[\nabla_a Q(s,a)\big|_{a=\mu(s)} \cdot \nabla_\theta \mu_\theta(s)\right]$$

The gradient flows **through the critic** to update the actor:

```
┌─────────────────────────────────────────┐
│           Gradient Flow in DDPG          │
├─────────────────────────────────────────┤
│                                          │
│   state ──→ [Actor μ(s)] ──→ action      │
│                                  │       │
│                                  ▼       │
│   state ──→ [Critic Q(s,a)] ──→ Q-value  │
│                                  │       │
│                  ∂Q/∂a · ∂μ/∂θ  │       │
│                                  ▼       │
│              Update actor weights θ      │
│                                          │
└─────────────────────────────────────────┘
```

---

## DDPG = DQN + Actor-Critic

| Component | DQN | DDPG |
|-----------|-----|------|
| Policy | ε-greedy on Q-values | Deterministic μ(s) + noise |
| Q-Network | Q(s) → values for all actions | Q(s, a) → single value |
| Replay Buffer | Yes | Yes |
| Target Networks | Yes (Q only) | Yes (actor + critic) |
| Action Space | Discrete | Continuous |
| Exploration | ε-greedy | Additive noise |

---

## Exploration: OU Noise vs Gaussian

Since the policy is deterministic, we need **external** noise for exploration:

**Ornstein-Uhlenbeck Process** (temporally correlated):
$$dx = \theta(\mu - x)dt + \sigma\sqrt{dt}\,\mathcal{N}(0,1)$$

- Mean-reverting → smoother exploration trajectories
- Used in the original DDPG paper

**Gaussian Noise** (i.i.d.):
$$a = \mu(s) + \mathcal{N}(0, \sigma)$$

- Simpler, works just as well in practice
- TD3 paper uses Gaussian exclusively

---

## Algorithm

```
Algorithm: DDPG
──────────────────────────────────────────────
Initialize actor μ_θ, critic Q_φ
Initialize targets μ_θ', Q_φ' (copy weights)
Initialize replay buffer B

for each episode:
    Reset noise process

    for each step:
        a = μ_θ(s) + noise          # Deterministic + exploration
        s', r, done = env.step(a)
        B.push(s, a, r, s', done)

        # Sample mini-batch from B
        y = r + γ · Q_φ'(s', μ_θ'(s'))     # Target
        Update Q_φ: minimize (Q_φ(s,a) - y)²  # Critic
        Update μ_θ: maximize Q_φ(s, μ_θ(s))    # Actor

        # Soft update targets
        θ' ← τθ + (1-τ)θ'
        φ' ← τφ + (1-τ)φ'
```

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Actor lr | 1e-4 | Slower than critic for stability |
| Critic lr | 1e-3 | 10x faster than actor |
| γ (discount) | 0.99 | Standard |
| τ (soft update) | 0.005 | Small for stability |
| Buffer size | 100,000 | Off-policy replay |
| Batch size | 64 | Per-step mini-batch |
| Noise σ | 0.2 | OU noise scale |
| Warmup steps | 1,000 | Random actions before training |
| Hidden dim | 256 | ReLU activations |

---

## Known Issues (Motivation for TD3)

1. **Overestimation bias**: Q-function systematically overestimates values
2. **Sensitivity**: Small hyperparameter changes can cause divergence
3. **Exploration**: OU noise is an ad-hoc solution

These issues are addressed by TD3 (see `22_td3/`).

---

## Module Structure

```
21_ddpg/
├── README.md    # This file
└── ddpg.py      # DDPG implementation
```

---

## Running

```bash
python rl_fundamentals/21_ddpg/ddpg.py
```

---

## References

1. Lillicrap, T.P., et al. (2015). Continuous control with deep reinforcement learning. *arXiv:1509.02971*.
2. Silver, D., et al. (2014). Deterministic Policy Gradient Algorithms. *ICML*.
