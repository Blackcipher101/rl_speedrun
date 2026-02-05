# Actor-Critic Methods: A2C

This module implements Advantage Actor-Critic (A2C), which combines the best of policy gradients and value-based methods for lower variance, more efficient learning.

---

## The Actor-Critic Idea

### Two Components Working Together

```
                        ┌─────────────────┐
                        │  Environment    │
                        └────────┬────────┘
                                 │
                        state s, reward r
                                 │
                                 ▼
┌────────────────────────────────────────────────────────┐
│                    AGENT                               │
│  ┌──────────────┐              ┌──────────────┐       │
│  │    ACTOR     │              │    CRITIC    │       │
│  │   π(a|s)     │              │    V(s)      │       │
│  │              │              │              │       │
│  │ "What to do" │              │ "How good    │       │
│  │              │              │  is this?"   │       │
│  └──────────────┘              └──────────────┘       │
│         │                             │               │
│         │    ◄─── Advantage ───►      │               │
│         │     A(s,a) = Q(s,a) - V(s)  │               │
│         ▼                             ▼               │
│      action a                   baseline for          │
│                                variance reduction     │
└────────────────────────────────────────────────────────┘
```

### Why Combine Them?

| Component | Strength | Weakness |
|-----------|----------|----------|
| **Policy Gradient** | Directly optimizes policy, handles continuous actions | High variance |
| **Value Function** | Low variance estimates | Can't directly output actions |
| **Actor-Critic** | Best of both! | More complex |

---

## The Advantage Function

### Definition

$$A(s,a) = Q(s,a) - V(s)$$

**Interpretation**: How much better is action $a$ compared to the average action in state $s$?

- $A > 0$: Action is better than average → increase probability
- $A < 0$: Action is worse than average → decrease probability
- $A = 0$: Action is exactly average

### Why Use Advantages?

**REINFORCE** uses raw returns:
$$\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot G_t]$$

**Actor-Critic** uses advantages:
$$\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot A(s,a)]$$

Benefits of advantages:
1. **Variance reduction**: Subtracting baseline $V(s)$ reduces variance
2. **Centered signal**: Positive/negative indicates good/bad actions
3. **Faster learning**: Clearer gradient signal

---

## Generalized Advantage Estimation (GAE)

### The Bias-Variance Tradeoff

| Method | Bias | Variance | Formula |
|--------|------|----------|---------|
| TD (n=1) | High | Low | $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ |
| MC (n=∞) | Low | High | $G_t - V(s_t)$ |
| GAE | Tunable | Tunable | $\sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$ |

### GAE Formula

$$A^{GAE}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

**Lambda controls the tradeoff:**
- $\lambda = 0$: Pure TD (low variance, high bias)
- $\lambda = 1$: Pure MC (high variance, low bias)
- $\lambda = 0.95$: Good balance (commonly used)

### Efficient Computation

```python
def compute_gae(rewards, values, next_values, dones, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T)

    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * gae * (1-dones[t])
        advantages[t] = gae

    return advantages
```

---

## Entropy Regularization

### Problem: Premature Convergence

Without regularization, the policy can become deterministic too quickly:
- Gets stuck in local optima
- Stops exploring
- Misses better strategies

### Solution: Entropy Bonus

Add entropy term to the objective:
$$J = \mathbb{E}[R] + \beta H(\pi)$$

where $H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$

**Effect**: Encourages the policy to maintain some randomness.

```
Low entropy (deterministic):  [0.99, 0.005, 0.005] → H ≈ 0.05
High entropy (exploratory):   [0.4, 0.3, 0.3]      → H ≈ 1.06
```

### Typical Values

- $\beta = 0.01$ to $0.1$ (start higher, decay over time)
- Too high: Policy stays too random, never converges
- Too low: Premature convergence, poor exploration

---

## A2C Algorithm

### Full Algorithm

```
Algorithm: Advantage Actor-Critic (A2C)
────────────────────────────────────────────────────────
Initialize actor π_θ and critic V_φ (often shared backbone)

for episode = 1 to M:
    s ← env.reset()

    for t = 1 to T:
        # Collect n steps of experience
        for step = 1 to n_steps:
            a ~ π(·|s)
            s', r, done ← env.step(a)
            store (s, a, r, V(s), done)
            s ← s'

        # Compute advantages using GAE
        if not done:
            bootstrap = V(s)
        else:
            bootstrap = 0

        advantages, returns = GAE(stored_transitions, bootstrap)
        advantages = normalize(advantages)

        # Update networks
        L_policy = -E[log π(a|s) · A(s,a)]
        L_value = E[(V(s) - returns)²]
        L_entropy = -E[H(π(·|s))]

        L_total = L_policy + c₁·L_value - c₂·L_entropy

        θ, φ ← θ, φ - α∇L_total
```

### Loss Function Components

$$L_{total} = L_{policy} + c_1 \cdot L_{value} - c_2 \cdot L_{entropy}$$

| Component | Formula | Purpose |
|-----------|---------|---------|
| Policy loss | $-\mathbb{E}[\log \pi(a\|s) \cdot A]$ | Improve action selection |
| Value loss | $\mathbb{E}[(V(s) - G)^2]$ | Improve value estimates |
| Entropy bonus | $-\mathbb{E}[H(\pi)]$ | Maintain exploration |

Typical coefficients: $c_1 = 0.5$, $c_2 = 0.01$

---

## Network Architecture

### Shared Features (Parameter Efficient)

```
State s
    │
    ▼
┌───────────────┐
│ Shared Layers │  64 → 64 (ReLU)
│   (features)  │
└───────┬───────┘
        │
    ┌───┴───┐
    │       │
    ▼       ▼
┌───────┐ ┌───────┐
│ Actor │ │Critic │
│ Head  │ │ Head  │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
  π(a|s)    V(s)
```

**Benefits of sharing:**
- Fewer parameters
- Features useful for both tasks
- More stable training

---

## Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Learning rate | 0.001 | Same for actor and critic |
| γ (discount) | 0.99 | Standard |
| λ (GAE) | 0.95 | Good bias-variance balance |
| n_steps | 5-20 | Steps between updates |
| Value coef | 0.5 | Weight for value loss |
| Entropy coef | 0.01 | Weight for entropy bonus |
| Max grad norm | 0.5 | Gradient clipping |

---

## A2C vs Other Methods

### Comparison

| Method | Variance | Bias | Sample Efficiency | Stability |
|--------|----------|------|-------------------|-----------|
| REINFORCE | High | Low | Poor | Unstable |
| DQN | Low | Medium | Good | Stable |
| **A2C** | Medium | Low | Medium | Stable |
| PPO | Low | Low | Good | Very stable |

### When to Use A2C

**Good for:**
- Continuous or discrete actions
- When you want on-policy learning
- Moderate-complexity problems
- Learning from scratch

**Consider alternatives if:**
- Need maximum sample efficiency → PPO, SAC
- Have replay buffer → DDPG, TD3
- Simple discrete problem → DQN

---

## Module Structure

```
16_actor_critic/
├── README.md        # This file
├── advantage.py     # Advantage estimation (GAE, n-step, etc.)
├── entropy.py       # Entropy regularization
└── a2c.py           # Complete A2C implementation
```

---

## Running the Code

```bash
cd rl_fundamentals/16_actor_critic

# Test advantage estimation
python advantage.py

# Test entropy utilities
python entropy.py

# Train A2C on CartPole
python a2c.py
```

### Expected Output

```
Episode   50 | Avg Return:   24.50 | Entropy: 0.685 | V_loss: 2.341
Episode  100 | Avg Return:   58.30 | Entropy: 0.612 | V_loss: 1.856
Episode  150 | Avg Return:  112.40 | Entropy: 0.534 | V_loss: 1.245
Episode  200 | Avg Return:  198.60 | Entropy: 0.456 | V_loss: 0.892
Episode  250 | Avg Return:  345.80 | Entropy: 0.389 | V_loss: 0.534
Episode  300 | Avg Return:  478.20 | Entropy: 0.312 | V_loss: 0.245

Evaluation: Mean return = 485.5 +/- 22.3
SOLVED! (Mean return >= 195)
```

---

## Key Insights

1. **Advantages are key**: Subtracting baseline dramatically reduces variance
2. **GAE is powerful**: Tunable bias-variance via λ parameter
3. **Entropy matters**: Prevents premature convergence
4. **Shared features help**: More parameter-efficient, often more stable
5. **n_steps tradeoff**: More steps = lower bias, higher variance

---

## A3C Note

A3C (Asynchronous A2C) uses multiple parallel actors:
- Each actor collects experience independently
- Asynchronously updates shared parameters
- Not implemented here (requires multiprocessing)
- A2C is synchronous and often performs similarly with simpler code

---

## References

1. Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML*.

2. Schulman, J., et al. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*.

3. Williams, R. J., & Peng, J. (1991). Function Optimization Using Connectionist Reinforcement Learning Algorithms. *Connection Science*.

4. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. Chapter 13.
