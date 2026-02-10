# Proximal Policy Optimization (PPO)

The algorithm that made deep RL actually practical. PPO takes A2C's actor-critic framework and adds a simple but powerful idea: **clip the policy update so it can't change too much in one step**.

---

## From A2C to PPO: The Problem

A2C updates the policy using the gradient:
$$\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot A(s,a)]$$

The issue? **No constraint on how far the policy moves.** A single bad update can destroy a good policy.

```
A2C Policy Update (no guardrails):

Performance │    ╱╲
            │   ╱  ╲         ← Good update
            │  ╱    ╲
            │ ╱      ╲
            │╱        ╲      ← Catastrophic collapse
            └──────────────── Steps

PPO Policy Update (clipped):

Performance │         ╱─────
            │        ╱       ← Steady improvement
            │       ╱
            │      ╱
            │─────╱          ← Can't fall off a cliff
            └──────────────── Steps
```

---

## The Core Idea: Trust Regions via Clipping

### Probability Ratio

Instead of working with log-probabilities, PPO uses the **probability ratio**:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

This tells us how much the policy has changed:
- $r_t = 1.0$: Policy hasn't changed for this action
- $r_t = 1.5$: Action is now 50% more likely
- $r_t = 0.5$: Action is now 50% less likely

### The Clipped Surrogate Objective

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where $\epsilon = 0.2$ typically.

### Why This Works: Two Cases

**Case 1: Advantage is positive** ($\hat{A}_t > 0$, action was good)

We want to increase $r_t$ (make this action more likely). But the clip caps it at $1 + \epsilon$:

```
Objective │     ╱─────────  ← Clipped: can't increase past (1+ε)·A
           │    ╱
           │   ╱
           │  ╱
           │ ╱
           └────────────────
           0   1-ε  1  1+ε    ratio r_t
```

**Case 2: Advantage is negative** ($\hat{A}_t < 0$, action was bad)

We want to decrease $r_t$ (make this action less likely). But the clip caps it at $1 - \epsilon$:

```
           ────────────────
                    ╲      │
                     ╲     │
                      ╲    │
           ───────────╲   │  ← Clipped: can't decrease past (1-ε)·A
           0   1-ε  1  1+ε    ratio r_t
```

The `min` in the objective is **pessimistic**: it picks whichever of the clipped and unclipped objective is worse. This ensures the policy never moves too far in either direction.

---

## Multi-Epoch Training

This is PPO's key practical advantage over A2C:

| Feature | A2C | PPO |
|---------|-----|-----|
| Data usage | One gradient step per rollout | Multiple epochs per rollout |
| Update constraint | None (hope for the best) | Clipped ratio |
| Sample efficiency | Low | Higher |
| Stability | Fragile | Robust |

Because the clipping prevents the policy from diverging too far, PPO can safely reuse the same batch of data for **multiple gradient steps** (typically 10 epochs with shuffled mini-batches).

---

## Discrete vs Continuous Actions

### Discrete (e.g., LunarLander)

Uses `Categorical` distribution over actions:
$$\pi(a|s) = \text{softmax}(f_\theta(s))_a$$

Entropy: $H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$

### Continuous (e.g., BipedalWalker)

Uses Gaussian policy with learnable standard deviation:
$$\pi(a|s) = \mathcal{N}(\mu_\theta(s), \sigma^2)$$

where $\mu$ is state-dependent (network output) and $\sigma$ is a learnable parameter.

Log probability (summed across independent action dimensions):
$$\log \pi(\mathbf{a}|s) = \sum_{i=1}^{d} \log \mathcal{N}(a_i | \mu_i(s), \sigma_i^2)$$

Entropy: $H = \frac{d}{2} \log(2\pi e) + \sum_{i=1}^{d} \log \sigma_i$

---

## Full Algorithm

```
Algorithm: PPO with Clipped Surrogate
────────────────────────────────────────────────────────
Initialize policy π_θ and value function V_φ

for iteration = 1, 2, ...:

    # 1. Collect rollout
    for t = 1 to n_steps:
        a_t ~ π_θ_old(·|s_t)
        Store (s_t, a_t, r_t, log π_θ_old(a_t|s_t), V(s_t), done_t)

    # 2. Compute advantages using GAE
    for t = T-1 down to 0:
        δ_t = r_t + γ·V(s_{t+1})·(1 - done_t) - V(s_t)
        A_t = δ_t + γ·λ·(1 - done_t)·A_{t+1}
    Returns = Advantages + Values
    Normalize advantages

    # 3. Optimize (K epochs, mini-batches)
    for epoch = 1 to K:
        for mini-batch in shuffle(rollout):
            # Evaluate current policy
            log π_θ(a|s), V_θ(s), H(π_θ)

            # Probability ratio
            r = exp(log π_θ - log π_θ_old)

            # Clipped surrogate
            L_clip = min(r·A, clip(r, 1-ε, 1+ε)·A)

            # Combined loss
            L = -L_clip + c₁·MSE(V, Returns) - c₂·H(π)

            # Gradient step
            θ ← θ - α·∇L
```

---

## Combined Loss

$$L = L^{CLIP} + c_1 \cdot L^{VF} - c_2 \cdot S[\pi_\theta]$$

| Component | Formula | Typical Coefficient |
|-----------|---------|-------------------|
| Policy (clipped) | $-\min(r \hat{A}, \text{clip}(r) \hat{A})$ | 1.0 |
| Value | $\text{MSE}(V_\theta(s), G_t)$ | $c_1 = 0.5$ |
| Entropy | $-H(\pi_\theta)$ | $c_2 = 0.01$ |

---

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Learning rate | $3 \times 10^{-4}$ | Adam optimizer |
| $\gamma$ (discount) | 0.99 | Standard |
| $\lambda$ (GAE) | 0.95 | Good bias-variance balance |
| $\epsilon$ (clip range) | 0.2 | Standard, rarely needs tuning |
| K (epochs) | 10 | More epochs = more data reuse |
| Mini-batch size | 64 | Smaller = more gradient steps |
| n_steps | 2048 | Transitions per rollout |
| Value coef $c_1$ | 0.5 | Weight for value loss |
| Entropy coef $c_2$ | 0.01 | 0.0 often fine for continuous |
| Max grad norm | 0.5 | Gradient clipping |

---

## Key Diagnostics

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| approx_kl | 0.01 - 0.03 | > 0.05 means policy moving too fast |
| clip_fraction | 0.05 - 0.20 | 0 = no clipping (too conservative), > 0.3 = too aggressive |
| entropy | Gradually decreasing | Sudden collapse = premature convergence |
| value_loss | Decreasing | Increasing = critic failing |

---

## PPO vs Other Methods

| Method | On/Off Policy | Update Constraint | Sample Efficiency | Handles Continuous |
|--------|---------------|-------------------|-------------------|--------------------|
| REINFORCE | On | None | Very poor | Yes |
| A2C | On | None | Poor | Yes |
| **PPO** | **On** | **Clipping** | **Good** | **Yes** |
| TRPO | On | KL constraint | Good | Yes |
| DQN | Off | Target network | Good | No |
| SAC | Off | Max entropy | Very good | Yes |

---

## Module Structure

```
18_ppo/
├── README.md    # This file
└── ppo.py       # PPO implementation (discrete + continuous)
```

---

## Running the Code

```bash
# Train PPO on CartPole (quick demo)
python rl_fundamentals/18_ppo/ppo.py

# LunarLander (discrete actions)
python rl_fundamentals/19_ppo_applications/lunarlander_ppo/solve_lunarlander_ppo.py

# BipedalWalker (continuous actions)
python rl_fundamentals/19_ppo_applications/bipedal_walker_ppo/solve_bipedal_walker_ppo.py
```

---

## References

1. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

2. Schulman, J., et al. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*.

3. Schulman, J., et al. (2015). Trust Region Policy Optimization. *ICML*.
