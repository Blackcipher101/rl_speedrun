# Trust Region Policy Optimization (TRPO)

PPO's predecessor. TRPO enforces a **hard KL divergence constraint** on policy updates using second-order optimization (conjugate gradient + line search). More theoretically principled but more complex than PPO's simple clipping.

---

## From TRPO to PPO

TRPO came first (2015). PPO (2017) simplified TRPO's approach:

| Aspect | TRPO | PPO |
|--------|------|-----|
| Constraint | Hard KL ≤ δ | Soft clipping |
| Optimization | Conjugate gradient + line search | Standard SGD |
| Data reuse | Single pass | Multiple epochs |
| Complexity | High (2nd order) | Low (1st order) |
| Performance | Strong | Similar or better |

PPO achieves comparable results with much simpler code, which is why it's more popular in practice.

---

## The Trust Region Idea

Standard policy gradient has no constraint on how far the policy moves:

```
Vanilla PG:     θ_new = θ - α · ∇J(θ)    ← step size is ad-hoc

TRPO:           maximize  L(θ)             ← surrogate loss
                subject to  KL(π_old || π_new) ≤ δ   ← trust region
```

The KL constraint ensures the new policy stays "close" to the old one in distribution space, preventing catastrophic updates.

---

## Mathematical Foundation

### Surrogate Loss

$$L(\theta) = \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \hat{A}(s,a)\right]$$

### Constrained Optimization

$$\max_\theta L(\theta) \quad \text{s.t.} \quad D_{KL}(\pi_{\theta_{old}} \| \pi_\theta) \leq \delta$$

### Natural Gradient Solution

Using the Lagrangian, the update direction is the **natural gradient**:

$$\theta_{new} = \theta_{old} + \sqrt{\frac{2\delta}{g^T F^{-1} g}} \cdot F^{-1} g$$

where:
- $g = \nabla_\theta L(\theta)$ is the policy gradient
- $F$ is the Fisher Information Matrix
- $\delta$ is the max KL divergence

---

## Conjugate Gradient

The Fisher matrix F is huge (params × params). We never form it explicitly. Instead, **conjugate gradient** solves $F^{-1}g$ using only matrix-vector products $Fv$:

```
Conjugate Gradient (solving Fx = g):
──────────────────────────────────────
x = 0, r = g, p = g

for i = 1 to n_steps:
    Fp = fisher_vector_product(p)    ← only need F·v, not F itself
    α = (r^T r) / (p^T Fp)
    x = x + α · p
    r = r - α · Fp
    if ||r|| < tol: break
    β = (r_new^T r_new) / (r_old^T r_old)
    p = r + β · p

return x ≈ F^{-1} g
```

---

## Fisher-Vector Product

Computing $Fv$ without forming $F$, using two passes of automatic differentiation:

```
1. KL = KL(π_θ_detached || π_θ)     ← gradient flows only through new policy
2. g_kl = ∂KL/∂θ                     ← first derivatives (create_graph=True)
3. dot = g_kl^T · v                   ← scalar dot product
4. Fv = ∂(dot)/∂θ                    ← second derivatives
5. return Fv + damping · v            ← regularization
```

This computes $Fv$ in $O(\text{params})$ time and space.

---

## Line Search

After computing the natural gradient direction, a **backtracking line search** finds a step size that satisfies the KL constraint:

```
fullstep = sqrt(2δ / x^T F x) · x    ← full natural gradient step

for k = 0, 1, 2, ...:
    step = fullstep · 0.5^k            ← shrink by half each time
    θ_new = θ_old + step
    if KL(θ_old, θ_new) < δ AND loss improved:
        accept!
        break
```

---

## Full Algorithm

```
Algorithm: TRPO
────────────────────────────────────────────────────────
Initialize actor π_θ and value function V_φ

for iteration = 1, 2, ...:

    # 1. Collect rollout (same as PPO)
    for t = 1 to n_steps:
        a_t ~ π_θ(·|s_t)
        Store (s_t, a_t, r_t, log π(a_t|s_t), V(s_t), done_t)

    # 2. Compute GAE advantages (same as PPO)
    A_t = Σ (γλ)^l δ_{t+l}

    # 3. Actor update (TRPO-specific)
    g = ∇_θ L(θ)                               ← policy gradient
    x = CG(F, g, 10 steps)                      ← natural gradient
    β = sqrt(2δ / (x^T F x))                    ← step size
    Line search: θ ← θ + β · 0.5^k · x         ← until KL < δ

    # 4. Critic update (standard Adam)
    for epoch = 1 to K:
        V_φ ← V_φ - α · ∇_φ MSE(V_φ(s), Returns)
```

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max KL (δ) | 0.01 | Trust region size |
| CG iterations | 10 | Conjugate gradient steps |
| Line search steps | 10 | Backtracking attempts |
| Damping | 0.1 | Fisher matrix regularization |
| Critic lr | 1e-3 | Adam for critic only |
| Critic updates | 5 | Epochs per TRPO update |
| γ (discount) | 0.99 | Standard |
| λ (GAE) | 0.95 | Same as PPO |
| n_steps | 2048 | Rollout length |

---

## Key Diagnostics

| Metric | Healthy | Warning |
|--------|---------|---------|
| KL divergence | < 0.01 (≈ max_kl) | >> 0.01 means trust region violated |
| Line search success | > 80% | < 50% means updates are too aggressive |
| Value loss | Decreasing | Increasing means critic is failing |

---

## Module Structure

```
20_trpo/
├── README.md    # This file
└── trpo.py      # TRPO implementation
```

---

## Running

```bash
python rl_fundamentals/20_trpo/trpo.py
```

---

## References

1. Schulman, J., et al. (2015). Trust Region Policy Optimization. *ICML*.
2. Schulman, J., et al. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*.
