# Monte Carlo Methods

Monte Carlo (MC) methods learn directly from complete episodes of experience without requiring a model of the environment. Unlike TD methods, MC waits until the end of an episode to update value estimates.

---

## Prerequisites

- MDP definition and Bellman equations
- Value functions V(s) and Q(s,a)
- Basic probability (expected values, sampling)

---

## 1. The Key Insight: Learning from Complete Returns

### 1.1 Comparison with Other Methods

| Method | Requires Model? | Updates When? | Uses Estimates? |
|--------|----------------|---------------|-----------------|
| **Dynamic Programming** | Yes | After full sweep | Yes (bootstraps) |
| **Monte Carlo** | No | After episode ends | No (uses actual returns) |
| **Temporal Difference** | No | After each step | Yes (bootstraps) |

**Monte Carlo's approach:**
- Sample complete episodes
- Use actual returns (not estimates)
- Average returns to estimate value functions

### 1.2 The Return

For an episode that visits state $s$ at time $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots + \gamma^{T-t-1} R_T$$

where $T$ is the final timestep of the episode.

**Key property**: $G_t$ is an **unbiased** estimate of $V^\pi(s)$:

$$\mathbb{E}_\pi[G_t | S_t = s] = V^\pi(s)$$

---

## 2. MC Prediction: Estimating V(s)

### 2.1 First-Visit MC

**Idea**: Estimate $V(s)$ as the average of returns following the **first** visit to $s$ in each episode.

**Algorithm**:
```
Initialize:
    V(s) ← arbitrary, for all s ∈ S
    Returns(s) ← empty list, for all s ∈ S

For each episode:
    Generate episode: S₀, A₀, R₁, S₁, A₁, R₂, ..., Sₜ₋₁, Aₜ₋₁, Rₜ
    G ← 0
    For t = T-1, T-2, ..., 0:
        G ← γG + R_{t+1}
        If S_t not in {S₀, S₁, ..., S_{t-1}}:  # First visit check
            Append G to Returns(S_t)
            V(S_t) ← average(Returns(S_t))
```

### 2.2 Every-Visit MC

**Idea**: Estimate $V(s)$ as the average of returns following **all** visits to $s$.

**Algorithm**:
```
Initialize:
    V(s) ← arbitrary, for all s ∈ S
    Returns(s) ← empty list, for all s ∈ S

For each episode:
    Generate episode: S₀, A₀, R₁, S₁, A₁, R₂, ..., Sₜ₋₁, Aₜ₋₁, Rₜ
    G ← 0
    For t = T-1, T-2, ..., 0:
        G ← γG + R_{t+1}
        Append G to Returns(S_t)  # No first-visit check
        V(S_t) ← average(Returns(S_t))
```

### 2.3 First-Visit vs Every-Visit

| Aspect | First-Visit | Every-Visit |
|--------|-------------|-------------|
| **Samples per episode** | At most 1 per state | Can be multiple |
| **Bias** | Unbiased | Biased (for finite samples) |
| **Variance** | Higher | Lower |
| **Convergence** | To $V^\pi$ | To $V^\pi$ (asymptotically) |

Both converge to $V^\pi$ as the number of episodes approaches infinity.

### 2.4 Incremental Mean Update

Instead of storing all returns, use incremental update:

$$V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t - V(S_t) \right]$$

where $\alpha = \frac{1}{N(S_t)}$ for exact averaging, or a constant $\alpha$ for exponential recency-weighted average.

---

## 3. MC Control: Finding Optimal Policies

### 3.1 The Exploration Problem

**Issue**: To evaluate a policy, we need to visit all states. But a deterministic policy might never visit some states.

**Solution**: Exploring starts or ε-soft policies.

### 3.2 Monte Carlo with Exploring Starts (MC-ES)

**Assumption**: Episodes start with every state-action pair having nonzero probability.

**Algorithm**:
```
Initialize:
    Q(s, a) ← arbitrary, for all s, a
    π(s) ← arbitrary, for all s
    Returns(s, a) ← empty list, for all s, a

For each episode:
    Choose S₀, A₀ randomly (exploring starts)
    Generate episode following π
    G ← 0
    For t = T-1, T-2, ..., 0:
        G ← γG + R_{t+1}
        If (S_t, A_t) not in earlier steps:
            Append G to Returns(S_t, A_t)
            Q(S_t, A_t) ← average(Returns(S_t, A_t))
            π(S_t) ← argmax_a Q(S_t, a)
```

### 3.3 On-Policy MC Control (ε-Greedy)

**Idea**: Use ε-soft policy that explores with probability ε.

$$\pi(a|s) = \begin{cases}
1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
\frac{\varepsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}$$

**Algorithm**:
```
Initialize:
    Q(s, a) ← arbitrary, for all s, a
    Returns(s, a) ← empty list, for all s, a
    π ← ε-greedy policy with respect to Q

For each episode:
    Generate episode following π
    G ← 0
    For t = T-1, T-2, ..., 0:
        G ← γG + R_{t+1}
        If (S_t, A_t) not in earlier steps:
            Append G to Returns(S_t, A_t)
            Q(S_t, A_t) ← average(Returns(S_t, A_t))
            π(S_t) ← ε-greedy with respect to Q(S_t, ·)
```

### 3.4 Off-Policy MC Control (Importance Sampling)

**Idea**: Learn about target policy $\pi$ while following behavior policy $b$.

**Importance sampling ratio**:

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

**Ordinary importance sampling**:

$$V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{|\mathcal{T}(s)|}$$

**Weighted importance sampling** (lower variance):

$$V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}}$$

---

## 4. Mathematical Analysis

### 4.1 Convergence of First-Visit MC

**Theorem**: First-visit MC converges to $V^\pi(s)$ for all $s$ as the number of visits to $s$ goes to infinity.

**Proof sketch**:
1. Each return $G_t$ is an unbiased sample: $\mathbb{E}[G_t | S_t = s] = V^\pi(s)$
2. Returns from different episodes are i.i.d.
3. By the Law of Large Numbers, the sample mean converges to the true mean

### 4.2 Bias-Variance Tradeoff

| Method | Bias | Variance |
|--------|------|----------|
| First-Visit MC | Unbiased | Higher |
| Every-Visit MC | Biased (finite samples) | Lower |
| TD(0) | Biased | Lower than MC |

### 4.3 MC vs TD

**Monte Carlo**:
- Uses complete returns (no bootstrapping)
- Unbiased estimates
- Higher variance
- Only works for episodic tasks
- Not affected by Markov property violations

**Temporal Difference**:
- Bootstraps from estimates
- Biased estimates (initially)
- Lower variance
- Works for continuing tasks
- Exploits Markov property

---

## 5. Practical Considerations

### 5.1 When to Use Monte Carlo

✅ Good for:
- Episodic tasks with clear termination
- When you want unbiased estimates
- Non-Markov environments
- Simulated environments where episodes are cheap

❌ Not ideal for:
- Continuing (non-episodic) tasks
- Very long episodes (high variance)
- When you need online learning

### 5.2 Variance Reduction Techniques

1. **Baseline subtraction**: $G_t - b(S_t)$ where $b$ doesn't depend on actions
2. **Return normalization**: $(G_t - \mu) / \sigma$
3. **More episodes**: Variance decreases as $1/n$

### 5.3 Hyperparameters

| Parameter | Typical Values | Notes |
|-----------|---------------|-------|
| γ (discount) | 0.9 - 1.0 | 1.0 for undiscounted episodic |
| ε (exploration) | 0.1 - 0.3 | Or decay schedule |
| Episodes | 10,000+ | MC needs many episodes |

---

## 6. Summary

### First-Visit MC Prediction
```
For each state s visited in episode:
    G ← return from first visit to s
    V(s) ← V(s) + α[G - V(s)]
```

### Every-Visit MC Prediction
```
For each state s visited in episode:
    G ← return from this visit to s
    V(s) ← V(s) + α[G - V(s)]
```

### MC Control (ε-Greedy)
```
For each (s, a) visited in episode:
    G ← return from this visit
    Q(s, a) ← Q(s, a) + α[G - Q(s, a)]
    π(s) ← ε-greedy w.r.t. Q(s, ·)
```

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 5.
2. Singh, S. P., & Sutton, R. S. (1996). Reinforcement learning with replacing eligibility traces. *Machine Learning*, 22(1-3), 123-158.
