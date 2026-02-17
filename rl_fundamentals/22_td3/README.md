# Twin Delayed DDPG (TD3)

Three targeted fixes for DDPG's failure modes. TD3 is to DDPG what PPO is to TRPO: a simpler, more robust version.

---

## DDPG's Three Problems

| Problem | Cause | TD3 Fix |
|---------|-------|---------|
| Overestimation bias | Single Q-network overestimates | **Twin critics** (take min) |
| Noisy policy gradients | Critic errors propagate to actor | **Delayed updates** (less frequent) |
| Brittle Q-function | Policy exploits narrow Q-peaks | **Target smoothing** (regularize Q) |

---

## Fix 1: Twin Q-Networks

Train **two** independent Q-functions. Use `min(Q1, Q2)` for the target:

$$y = r + \gamma \min(Q_1'(s', a'), Q_2'(s', a'))$$

Both networks have the same architecture but different initializations, leading to different error distributions. Taking the minimum makes simultaneous overestimation unlikely.

```
DDPG:  target = r + γ · Q'(s', μ'(s'))           ← can overestimate

TD3:   target = r + γ · min(Q1'(s', a'), Q2'(s', a'))  ← conservative
```

---

## Fix 2: Delayed Policy Updates

Update the actor only every **d** critic updates (typically d=2):

```
Step 1: Update Q1, Q2 (critic)
Step 2: Update Q1, Q2 (critic) + Update μ (actor) + Soft update targets
Step 3: Update Q1, Q2 (critic)
Step 4: Update Q1, Q2 (critic) + Update μ (actor) + Soft update targets
...
```

This ensures the critic is more accurate before its gradient is used to update the actor.

---

## Fix 3: Target Policy Smoothing

Add clipped noise to the target actor's actions:

$$a' = \mu'(s') + \text{clip}(\mathcal{N}(0, \sigma), -c, c)$$

This smooths the Q-function over similar actions, preventing the policy from exploiting narrow peaks in Q(s, a) that are due to estimation error.

---

## Full Algorithm

```
Algorithm: TD3
──────────────────────────────────────────────
Initialize actor μ_θ, twin critic Q_φ = (Q1, Q2)
Initialize targets μ_θ', Q_φ'
Initialize replay buffer B, update counter t = 0

for each step:
    a = μ_θ(s) + N(0, σ)            # Exploration noise
    s', r, done = env.step(a)
    B.push(s, a, r, s', done)
    t += 1

    # Critic update (every step)
    ã = μ_θ'(s') + clip(N(0, σ̃), -c, c)  # Target smoothing
    y = r + γ · min(Q1'(s', ã), Q2'(s', ã))
    Update Q1, Q2: minimize Σ(Qi(s,a) - y)²

    # Actor update (every d steps)
    if t mod d == 0:
        Update μ_θ: maximize Q1(s, μ_θ(s))
        Soft update θ', φ'
```

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Actor lr | 1e-4 | Same as DDPG |
| Critic lr | 1e-3 | Same as DDPG |
| γ (discount) | 0.99 | Standard |
| τ (soft update) | 0.005 | Standard |
| **Policy delay** | **2** | Update actor every 2 critic updates |
| **Target noise σ̃** | **0.2** | Smoothing noise scale |
| **Noise clip c** | **0.5** | Smoothing noise bounds |
| Exploration σ | 0.1 | Lower than DDPG's 0.2 |
| Buffer size | 100,000 | Same as DDPG |
| Batch size | 64 | Same as DDPG |

---

## TD3 vs DDPG

| Aspect | DDPG | TD3 |
|--------|------|-----|
| Critics | 1 Q-network | 2 Q-networks (twin) |
| Target Q | Q'(s', μ'(s')) | min(Q1', Q2') |
| Actor updates | Every step | Every d steps |
| Target actions | Clean | Noised + clipped |
| Overestimation | High | Low |
| Stability | Fragile | Robust |

---

## Module Structure

```
22_td3/
├── README.md    # This file
└── td3.py       # TD3 implementation
```

---

## Running

```bash
python rl_fundamentals/22_td3/td3.py
```

---

## References

1. Fujimoto, S., et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *ICML*.
2. Lillicrap, T.P., et al. (2015). Continuous control with deep reinforcement learning. *arXiv:1509.02971*.
