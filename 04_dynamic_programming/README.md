# Dynamic Programming for MDPs

This document covers dynamic programming (DP) algorithms for solving Markov Decision Processes: policy evaluation, policy iteration, and value iteration.

---

## Prerequisites

- MDP definition: $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$
- Bellman equations (expectation and optimality)
- Contraction mapping property

---

## 1. Assumptions for Dynamic Programming

Dynamic programming methods require:

1. **Known model**: The transition probabilities $P(s' \mid s, a)$ and rewards $R(s, a, s')$ are known exactly.

2. **Finite MDP**: The state space $\mathcal{S}$ and action space $\mathcal{A}$ are finite.

3. **Full state enumeration**: We can iterate over all states and actions.

**When DP is applicable**: Planning problems where the environment dynamics are known (e.g., games with known rules, simulated environments, inventory management with known demand distributions).

**When DP is not applicable**: Model-free settings where we only have access to samples from the environment (addressed by Monte Carlo methods and temporal-difference learning).

---

## 2. Policy Evaluation (Prediction)

**Goal**: Given a policy $\pi$, compute its value function $V^\pi$.

### 2.1 The Problem

We want to solve the Bellman expectation equation:

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

This is a system of $|\mathcal{S}|$ linear equations in $|\mathcal{S}|$ unknowns.

### 2.2 Direct Solution (Matrix Inversion)

From the matrix form: $\mathbf{V}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi$

**Complexity**: $O(|\mathcal{S}|^3)$ for matrix inversion.

**Drawback**: Cubic complexity is prohibitive for large state spaces.

### 2.3 Iterative Policy Evaluation

**Idea**: Use the Bellman operator $T^\pi$ as an iterative update rule.

**Algorithm**:

```
Initialize V(s) arbitrarily for all s ∈ S
Repeat until convergence:
    For each s ∈ S:
        V_new(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a) [R(s,a,s') + γV(s')]
    V ← V_new
```

**Update equation**:

$$V_{k+1}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V_k(s')\right]$$

Or in operator notation: $V_{k+1} = T^\pi V_k$

### 2.4 Convergence of Policy Evaluation

**Theorem**: For any initial $V_0$, the sequence $\{V_k\}$ defined by $V_{k+1} = T^\pi V_k$ converges to $V^\pi$.

**Proof**:

1. $T^\pi$ is a $\gamma$-contraction (proven in Bellman equations module)
2. By Banach fixed-point theorem, $V_k \to V^\pi$ as $k \to \infty$
3. Rate: $\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty$

**Convergence criterion**: Stop when $\|V_{k+1} - V_k\|_\infty < \theta$ for some threshold $\theta > 0$.

### 2.5 Synchronous vs. Asynchronous Updates

**Synchronous (full backup)**: Update all states before using any new values. Requires storing $V_k$ and $V_{k+1}$ separately.

**Asynchronous (in-place)**: Update states one at a time, immediately using new values. Can converge faster in practice.

```python
# Synchronous
V_new = np.zeros_like(V)
for s in states:
    V_new[s] = backup(V, s)
V = V_new

# Asynchronous (in-place)
for s in states:
    V[s] = backup(V, s)
```

Both converge to $V^\pi$ under appropriate conditions.

---

## 3. Policy Improvement

**Goal**: Given a value function $V^\pi$, find a better (or equally good) policy.

### 3.1 Policy Improvement Theorem

**Theorem**: Let $\pi$ and $\pi'$ be two policies. If for all $s \in \mathcal{S}$:

$$Q^\pi(s, \pi'(s)) \geq V^\pi(s)$$

then $\pi'$ is at least as good as $\pi$: $V^{\pi'}(s) \geq V^\pi(s)$ for all $s$.

**Proof**:

Starting from any state $s$:

$$V^\pi(s) \leq Q^\pi(s, \pi'(s))$$

Expanding $Q^\pi$:

$$= \sum_{s'} P(s' \mid s, \pi'(s)) \left[R(s, \pi'(s), s') + \gamma V^\pi(s')\right]$$

Applying the same inequality to $V^\pi(s')$:

$$\leq \sum_{s'} P(s' \mid s, \pi'(s)) \left[R(s, \pi'(s), s') + \gamma Q^\pi(s', \pi'(s'))\right]$$

Continuing this expansion:

$$\leq \sum_{s'} P(s' \mid s, \pi'(s)) \left[R(s, \pi'(s), s') + \gamma \sum_{s''} P(s'' \mid s', \pi'(s')) [R + \gamma V^\pi(s'')]\right]$$

Taking this to infinity:

$$\leq \mathbb{E}_{\pi'}\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \mid S_t = s\right] = V^{\pi'}(s)$$

Therefore: $V^{\pi'}(s) \geq V^\pi(s)$ for all $s$. $\square$

### 3.2 Greedy Policy Improvement

**Definition**: The greedy policy with respect to $V$ is:

$$\pi'(s) = \arg\max_{a \in \mathcal{A}} \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V(s')\right]$$

Equivalently:

$$\pi'(s) = \arg\max_{a \in \mathcal{A}} Q^\pi(s, a)$$

**Claim**: The greedy policy $\pi'$ satisfies the condition of the policy improvement theorem.

**Proof**: By definition of $\pi'$:

$$Q^\pi(s, \pi'(s)) = \max_a Q^\pi(s, a) \geq Q^\pi(s, \pi(s)) = V^\pi(s)$$

### 3.3 When Does Improvement Stop?

**Lemma**: If $\pi' = \pi$ (the greedy policy equals the original policy), then $\pi$ is optimal.

**Proof**: If $\pi' = \pi$, then for all $s$:

$$V^\pi(s) = Q^\pi(s, \pi(s)) = \max_a Q^\pi(s, a)$$

This is exactly the Bellman optimality equation. Therefore $V^\pi = V^*$, and $\pi$ is optimal.

---

## 4. Policy Iteration

**Idea**: Alternate between policy evaluation and policy improvement until convergence.

### 4.1 Algorithm

```
Initialize π arbitrarily

Repeat:
    // Policy Evaluation
    Compute V^π (using iterative policy evaluation or matrix inversion)

    // Policy Improvement
    For each s ∈ S:
        π'(s) ← argmax_a Σ_s' P(s'|s,a) [R(s,a,s') + γV^π(s')]

    // Check for convergence
    If π' = π:
        Return π (optimal policy)
    Else:
        π ← π'
```

### 4.2 Convergence

**Theorem**: Policy iteration converges to an optimal policy in a finite number of iterations.

**Proof**:

1. Each policy improvement step produces a strictly better policy (unless already optimal)
2. The number of deterministic policies is finite: at most $|\mathcal{A}|^{|\mathcal{S}|}$
3. We never revisit a previous policy (since values strictly increase)
4. Therefore, the algorithm must terminate at an optimal policy

### 4.3 Complexity Analysis

**Per iteration**:
- Policy evaluation: $O(|\mathcal{S}|^2 \cdot |\mathcal{A}|)$ per sweep, multiple sweeps until convergence (or $O(|\mathcal{S}|^3)$ for direct solution)
- Policy improvement: $O(|\mathcal{S}| \cdot |\mathcal{A}| \cdot |\mathcal{S}|) = O(|\mathcal{S}|^2 \cdot |\mathcal{A}|)$

**Number of iterations**: At most $|\mathcal{A}|^{|\mathcal{S}|}$ (exponential worst case), but typically very few in practice (often $< 10$).

### 4.4 Pseudocode with Details

```python
def policy_iteration(S, A, P, R, gamma, theta=1e-6):
    """
    S: list of states
    A: list of actions
    P: P[s][a] = list of (prob, next_state, reward, done) tuples
    gamma: discount factor
    theta: convergence threshold
    """
    # Initialize arbitrary deterministic policy
    pi = {s: A[0] for s in S}

    while True:
        # Policy Evaluation
        V = policy_evaluation(pi, S, A, P, R, gamma, theta)

        # Policy Improvement
        policy_stable = True
        for s in S:
            old_action = pi[s]
            # Compute Q(s, a) for all actions
            q_values = []
            for a in A:
                q = sum(prob * (r + gamma * V[s_next])
                        for prob, s_next, r, _ in P[s][a])
                q_values.append(q)
            # Greedy action selection
            pi[s] = A[np.argmax(q_values)]
            if pi[s] != old_action:
                policy_stable = False

        if policy_stable:
            return V, pi
```

---

## 5. Value Iteration

**Idea**: Combine policy evaluation and improvement into a single update by using the Bellman optimality operator.

### 5.1 Derivation

Recall that policy iteration does:
1. $V = V^\pi$ (evaluate current policy)
2. $\pi'(s) = \arg\max_a Q^\pi(s, a)$ (improve)

**Key observation**: We don't need to fully evaluate $V^\pi$. We can truncate policy evaluation to a single sweep and still converge.

### 5.2 Algorithm

```
Initialize V(s) arbitrarily for all s ∈ S

Repeat until convergence:
    For each s ∈ S:
        V(s) ← max_a Σ_s' P(s'|s,a) [R(s,a,s') + γV(s')]

Return V (which equals V*)
```

**Update equation**:

$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V_k(s')\right]$$

In operator notation: $V_{k+1} = T^* V_k$

### 5.3 Convergence

**Theorem**: Value iteration converges to $V^*$ for any initialization.

**Proof**:

1. The Bellman optimality operator $T^*$ is a $\gamma$-contraction
2. $V^*$ is the unique fixed point of $T^*$
3. By Banach fixed-point theorem: $V_k \to V^*$
4. Rate: $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$

### 5.4 Extracting the Optimal Policy

After value iteration converges to $V^*$, extract the optimal policy:

$$\pi^*(s) = \arg\max_{a \in \mathcal{A}} \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^*(s')\right]$$

### 5.5 Convergence Criterion

**Standard criterion**: Stop when $\|V_{k+1} - V_k\|_\infty < \theta$

**Relationship to optimality**: When $\|V_{k+1} - V_k\|_\infty < \theta$:

$$\|V_k - V^*\|_\infty < \frac{\theta \gamma}{1 - \gamma}$$

**Derivation**:

From contraction: $\|V_{k+1} - V^*\|_\infty \leq \gamma \|V_k - V^*\|_\infty$

Triangle inequality: $\|V_k - V^*\|_\infty \leq \|V_k - V_{k+1}\|_\infty + \|V_{k+1} - V^*\|_\infty$

Therefore: $\|V_k - V^*\|_\infty \leq \theta + \gamma \|V_k - V^*\|_\infty$

Solving: $\|V_k - V^*\|_\infty \leq \frac{\theta}{1 - \gamma}$

### 5.6 Pseudocode

```python
def value_iteration(S, A, P, R, gamma, theta=1e-6):
    """
    Returns optimal value function and policy.
    """
    V = np.zeros(len(S))

    while True:
        delta = 0
        for s in S:
            v = V[s]
            # Bellman optimality update
            q_values = []
            for a in A:
                q = sum(prob * (r + gamma * V[s_next])
                        for prob, s_next, r, _ in P[s][a])
                q_values.append(q)
            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    # Extract policy
    pi = {}
    for s in S:
        q_values = []
        for a in A:
            q = sum(prob * (r + gamma * V[s_next])
                    for prob, s_next, r, _ in P[s][a])
            q_values.append(q)
        pi[s] = A[np.argmax(q_values)]

    return V, pi
```

---

## 6. Comparison: Policy Iteration vs. Value Iteration

| Aspect | Policy Iteration | Value Iteration |
|--------|-----------------|-----------------|
| **Per iteration** | Full policy evaluation + improvement | Single Bellman optimality backup |
| **Iteration cost** | Higher (multiple evaluation sweeps) | Lower (one sweep) |
| **Number of iterations** | Fewer | More |
| **Total complexity** | Often better for small $\gamma$ | Often better for large $\gamma$ |
| **Memory** | Stores $V$ and $\pi$ | Stores $V$ only |

### 6.1 Generalized Policy Iteration

Policy iteration and value iteration are extremes of a spectrum:
- **Policy iteration**: Evaluate to convergence, then improve
- **Value iteration**: Single evaluation step, then improve

**Generalized policy iteration (GPI)**: Any interleaving of evaluation and improvement steps converges to optimal policy/value.

---

## 7. Complexity Summary

| Algorithm | Time per Iteration | Space |
|-----------|-------------------|-------|
| Policy Evaluation (iterative) | $O(|\mathcal{S}|^2 |\mathcal{A}|)$ per sweep | $O(|\mathcal{S}|)$ |
| Policy Evaluation (direct) | $O(|\mathcal{S}|^3)$ | $O(|\mathcal{S}|^2)$ |
| Policy Improvement | $O(|\mathcal{S}|^2 |\mathcal{A}|)$ | $O(|\mathcal{S}|)$ |
| Value Iteration | $O(|\mathcal{S}|^2 |\mathcal{A}|)$ per sweep | $O(|\mathcal{S}|)$ |

### 7.1 Number of Iterations

For value iteration, to achieve $\|V_k - V^*\|_\infty \leq \epsilon$:

$$k \geq \frac{\log(\|V_0 - V^*\|_\infty / \epsilon)}{\log(1/\gamma)} = O\left(\frac{1}{1-\gamma} \log\frac{1}{\epsilon}\right)$$

---

## 8. Limitations of Dynamic Programming

1. **Curse of dimensionality**: Complexity is polynomial in $|\mathcal{S}|$ and $|\mathcal{A}|$, but these grow exponentially with the number of state/action variables.
   - Example: A state with 10 binary features has $2^{10} = 1024$ states
   - 20 features: over 1 million states

2. **Requires known model**: Must know $P(s' \mid s, a)$ and $R(s, a, s')$ exactly.

3. **Finite state/action spaces**: Standard DP assumes discrete, finite spaces.

4. **Full state enumeration**: Each iteration must loop over all states.

**Solutions in RL**:
- **Model-free methods**: Learn from experience without a model (TD, Q-learning)
- **Function approximation**: Generalize across states (neural networks)
- **Sampling**: Monte Carlo methods avoid full enumeration

---

## 9. Algorithm Summary

### Policy Evaluation

$$V_{k+1}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V_k(s')\right]$$

### Policy Improvement

$$\pi'(s) = \arg\max_{a} \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

### Policy Iteration

Repeat: Evaluate → Improve → Check stability

### Value Iteration

$$V_{k+1}(s) = \max_{a} \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V_k(s')\right]$$

---

## References

1. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
2. Howard, R. A. (1960). *Dynamic Programming and Markov Processes*. MIT Press.
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 4.
4. Bertsekas, D. P. (2012). *Dynamic Programming and Optimal Control* (4th ed.). Athena Scientific.
