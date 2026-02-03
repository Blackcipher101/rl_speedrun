# Temporal Difference Learning

This document covers Temporal Difference (TD) learning methods, which combine ideas from Dynamic Programming and Monte Carlo methods to learn directly from experience without a model.

---

## Prerequisites

- MDP definition and Bellman equations
- Value functions V(s) and Q(s,a)
- Dynamic Programming (for comparison)

---

## 1. The Key Insight: Bootstrapping

### 1.1 Three Approaches to Value Estimation

| Method | Requires Model? | Updates When? | Uses Estimates? |
|--------|----------------|---------------|-----------------|
| **Dynamic Programming** | Yes | After full sweep | Yes (bootstraps) |
| **Monte Carlo** | No | After episode ends | No (waits for G_t) |
| **Temporal Difference** | No | After each step | Yes (bootstraps) |

**TD combines the best of both worlds:**
- Like MC: Learns from experience (no model needed)
- Like DP: Updates estimates based on other estimates (bootstrapping)

### 1.2 The Bootstrapping Idea

In Monte Carlo, we wait for the actual return:

$$V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t - V(S_t) \right]$$

where $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$ requires waiting until episode end.

In TD, we **bootstrap** - use our current estimate:

$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

We replace the unknown $G_t$ with the estimate $R_{t+1} + \gamma V(S_{t+1})$.

---

## 2. TD Prediction: TD(0)

### 2.1 The TD Error

**Definition**: The TD error $\delta_t$ is:

$$\delta_t \triangleq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

**Interpretation:**
- $V(S_t)$: Our current estimate of value at $S_t$
- $R_{t+1} + \gamma V(S_{t+1})$: A better estimate after seeing one transition
- $\delta_t$: The "surprise" - how much we need to update

### 2.2 TD(0) Update Rule

$$V(S_t) \leftarrow V(S_t) + \alpha \cdot \delta_t = V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

**Parameters:**
- $\alpha \in (0, 1]$: Learning rate (step size)
- $\gamma \in [0, 1]$: Discount factor

### 2.3 TD(0) Algorithm for Policy Evaluation

```
Input: policy π to evaluate
Initialize V(s) arbitrarily for all s ∈ S, V(terminal) = 0

For each episode:
    Initialize S
    While S is not terminal:
        A ← action given by π for S
        Take action A, observe R, S'
        V(S) ← V(S) + α[R + γV(S') - V(S)]
        S ← S'
```

### 2.4 Why TD(0) Works

**Intuition**: The TD target $R_{t+1} + \gamma V(S_{t+1})$ is an unbiased estimate of $V^\pi(S_t)$ when $V = V^\pi$.

**Convergence**: TD(0) converges to $V^\pi$ under standard stochastic approximation conditions:
1. Learning rates satisfy: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$
2. All states visited infinitely often

In practice, a fixed small $\alpha$ (e.g., 0.1) often works well.

---

## 3. TD Control: SARSA (On-Policy)

### 3.1 From Prediction to Control

For control, we need to learn $Q(s, a)$ instead of $V(s)$, then derive a policy from Q.

**SARSA** = **S**tate-**A**ction-**R**eward-**S**tate-**A**ction

The name comes from the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ used in each update.

### 3.2 SARSA Update Rule

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

**Key property**: Uses the actual next action $A_{t+1}$ chosen by the policy.

### 3.3 SARSA Algorithm

```
Initialize Q(s, a) arbitrarily for all s, a
Initialize Q(terminal, ·) = 0

For each episode:
    Initialize S
    Choose A from S using policy derived from Q (e.g., ε-greedy)

    While S is not terminal:
        Take action A, observe R, S'
        Choose A' from S' using policy derived from Q (e.g., ε-greedy)
        Q(S, A) ← Q(S, A) + α[R + γQ(S', A') - Q(S, A)]
        S ← S'
        A ← A'
```

### 3.4 On-Policy Nature

SARSA is **on-policy**: it evaluates and improves the policy being used to make decisions.

- The behavior policy (choosing actions) = the target policy (being evaluated)
- If using ε-greedy, SARSA converges to $Q^{\pi_\varepsilon}$, not $Q^*$
- With decaying ε → 0, converges to $Q^*$

---

## 4. TD Control: Q-Learning (Off-Policy)

### 4.1 The Key Difference

Q-Learning uses a **different** update target:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

Instead of the actual next action $A_{t+1}$, it uses the **greedy** action $\arg\max_a Q(S_{t+1}, a)$.

### 4.2 Q-Learning Algorithm

```
Initialize Q(s, a) arbitrarily for all s, a
Initialize Q(terminal, ·) = 0

For each episode:
    Initialize S

    While S is not terminal:
        Choose A from S using policy derived from Q (e.g., ε-greedy)
        Take action A, observe R, S'
        Q(S, A) ← Q(S, A) + α[R + γ·max_a Q(S', a) - Q(S, A)]
        S ← S'
```

### 4.3 Off-Policy Nature

Q-Learning is **off-policy**: the behavior policy differs from the target policy.

- **Behavior policy**: ε-greedy (or any exploratory policy)
- **Target policy**: Greedy with respect to Q (the $\max$)

**Key theorem**: Q-Learning converges to $Q^*$ regardless of the behavior policy, as long as all state-action pairs are visited infinitely often.

### 4.4 Q-Learning vs SARSA Comparison

| Aspect | Q-Learning | SARSA |
|--------|------------|-------|
| **Update target** | $R + \gamma \max_a Q(S', a)$ | $R + \gamma Q(S', A')$ |
| **Policy type** | Off-policy | On-policy |
| **Converges to** | $Q^*$ | $Q^\pi$ (policy being followed) |
| **Exploration** | Learns optimal Q despite exploration | Learns about policy with exploration |
| **Cliff example** | Takes risky optimal path | Takes safer path |

---

## 5. Exploration Strategies

### 5.1 The Exploration-Exploitation Dilemma

- **Exploitation**: Choose the best known action (greedy)
- **Exploration**: Try other actions to discover better options

Without exploration, we might miss the optimal policy.

### 5.2 ε-Greedy Policy

$$\pi(a|s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\varepsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$

- With probability $1-\varepsilon$: choose greedy action
- With probability $\varepsilon$: choose uniformly random

### 5.3 ε-Decay Schedules

Start with high exploration, gradually shift to exploitation:

**Linear decay:**
$$\varepsilon_t = \max(\varepsilon_{\min}, \varepsilon_0 - \text{decay\_rate} \cdot t)$$

**Exponential decay:**
$$\varepsilon_t = \max(\varepsilon_{\min}, \varepsilon_0 \cdot \text{decay}^t)$$

**Typical values:**
- $\varepsilon_0 = 1.0$ (start fully random)
- $\varepsilon_{\min} = 0.01$ (always some exploration)
- decay = 0.995 (per episode)

---

## 6. Mathematical Analysis

### 6.1 TD Update as Gradient Descent

The TD(0) update can be viewed as stochastic gradient descent on the mean squared error:

$$\text{MSE} = \mathbb{E}\left[ (V(S) - V^\pi(S))^2 \right]$$

The TD error $\delta_t$ estimates the gradient direction.

### 6.2 Convergence Guarantees

**Theorem (TD(0) Convergence)**: Under the following conditions:
1. The MDP is finite
2. The learning rate schedule satisfies $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$
3. Every state is visited infinitely often

TD(0) converges to $V^\pi$ with probability 1.

**Theorem (Q-Learning Convergence)**: Under similar conditions, with all state-action pairs visited infinitely often, Q-Learning converges to $Q^*$.

### 6.3 Bias-Variance Tradeoff

| Method | Bias | Variance |
|--------|------|----------|
| Monte Carlo | Unbiased | High (uses full return) |
| TD(0) | Biased (bootstraps) | Lower (uses one-step) |

TD's bias decreases as V approaches $V^\pi$.

---

## 7. Practical Considerations

### 7.1 Hyperparameter Selection

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| $\alpha$ (learning rate) | 0.01 - 0.5 | Too high → unstable, too low → slow |
| $\gamma$ (discount) | 0.9 - 0.999 | Lower → myopic, higher → farsighted |
| $\varepsilon$ (exploration) | 1.0 → 0.01 | Decay from high to low |
| Episodes | 500 - 10000 | Until convergence |

### 7.2 Q-Table Initialization

- **Zeros**: Neutral start, common choice
- **Optimistic**: High initial values encourage exploration (optimistic initialization)
- **Random**: Small random values

### 7.3 Terminal States

Always set $Q(\text{terminal}, a) = 0$ for all actions. This is because:
- No future rewards from terminal states
- Ensures correct bootstrapping

---

## 8. Summary

### TD Prediction (Policy Evaluation)
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

### SARSA (On-Policy Control)
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

### Q-Learning (Off-Policy Control)
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

### Key Differences
- **TD vs MC**: TD bootstraps (faster updates), MC waits for episode end (unbiased)
- **SARSA vs Q-Learning**: SARSA learns about current (exploratory) policy, Q-Learning learns optimal policy

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapters 6-7.
2. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis, Cambridge University.
3. Rummery, G. A., & Niranjan, M. (1994). *On-line Q-learning using connectionist systems*. Technical Report CUED/F-INFENG/TR 166.
