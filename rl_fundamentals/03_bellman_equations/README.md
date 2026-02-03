# Bellman Equations

This document provides complete derivations of the Bellman equations, the recursive relationships that form the foundation of dynamic programming in reinforcement learning.

---

## Prerequisites

- MDP definition: $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$
- Return: $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma G_{t+1}$
- Value functions: $V^\pi(s)$, $Q^\pi(s,a)$, $V^*(s)$, $Q^*(s,a)$

---

## 1. Bellman Expectation Equation for $V^\pi$

### 1.1 Statement

For any policy $\pi$ and state $s \in \mathcal{S}$:

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

### 1.2 Full Derivation

**Starting point**: Definition of state-value function.

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$$

**Step 1**: Expand using the recursive definition of return.

$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s]$$

**Step 2**: Use linearity of expectation.

$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} \mid S_t = s] + \gamma \cdot \mathbb{E}_\pi[G_{t+1} \mid S_t = s]$$

**Step 3**: Expand the first term by conditioning on action $A_t$ and next state $S_{t+1}$.

The reward $R_{t+1} = R(S_t, A_t, S_{t+1})$ depends on $(S_t, A_t, S_{t+1})$. Using the law of total expectation:

$$\mathbb{E}_\pi[R_{t+1} \mid S_t = s] = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \cdot R(s, a, s')$$

**Justification**:
- Given $S_t = s$, the action $A_t$ is drawn from $\pi(\cdot \mid s)$
- Given $S_t = s$ and $A_t = a$, the next state $S_{t+1}$ is drawn from $P(\cdot \mid s, a)$
- Given $(S_t, A_t, S_{t+1}) = (s, a, s')$, the reward is $R(s, a, s')$

**Step 4**: Expand the second term similarly.

$$\mathbb{E}_\pi[G_{t+1} \mid S_t = s] = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \cdot \mathbb{E}_\pi[G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']$$

**Step 5**: Apply the Markov property.

By the Markov property, $G_{t+1}$ depends only on $S_{t+1}$, not on $(S_t, A_t)$ given $S_{t+1}$:

$$\mathbb{E}_\pi[G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s'] = \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']$$

**Step 6**: Recognize the recursive structure.

$$\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] = V^\pi(s')$$

**Step 7**: Combine all terms.

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \cdot R(s, a, s') + \gamma \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \cdot V^\pi(s')$$

**Step 8**: Factor out the common summation.

$$\boxed{V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]}$$

### 1.3 Alternative Forms

**Using expected reward** $r(s, a) = \sum_{s'} P(s' \mid s, a) R(s, a, s')$:

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \left[r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')\right]$$

**For deterministic policy** $\pi(s) = a$:

$$V^\pi(s) = \sum_{s'} P(s' \mid s, \pi(s)) \left[R(s, \pi(s), s') + \gamma V^\pi(s')\right]$$

---

## 2. Bellman Expectation Equation for $Q^\pi$

### 2.1 Statement

For any policy $\pi$, state $s \in \mathcal{S}$, and action $a \in \mathcal{A}$:

$$Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a')\right]$$

### 2.2 Full Derivation

**Starting point**: Definition of action-value function.

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$$

**Step 1**: Expand using the recursive definition of return.

$$Q^\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a]$$

**Step 2**: Use linearity of expectation.

$$Q^\pi(s, a) = \mathbb{E}_\pi[R_{t+1} \mid S_t = s, A_t = a] + \gamma \cdot \mathbb{E}_\pi[G_{t+1} \mid S_t = s, A_t = a]$$

**Step 3**: Expand the first term by conditioning on $S_{t+1}$.

$$\mathbb{E}_\pi[R_{t+1} \mid S_t = s, A_t = a] = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \cdot R(s, a, s')$$

**Step 4**: Expand the second term.

$$\mathbb{E}_\pi[G_{t+1} \mid S_t = s, A_t = a] = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \cdot \mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']$$

**Step 5**: Express $\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']$ in terms of $Q^\pi$.

$$\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] = V^\pi(s') = \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a')$$

**Step 6**: Combine all terms.

$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \cdot R(s, a, s') + \gamma \sum_{s'} P(s' \mid s, a) \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')$$

**Step 7**: Factor.

$$\boxed{Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a')\right]}$$

### 2.3 Alternative Form (Using $V^\pi$)

Using the relationship $V^\pi(s') = \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')$:

$$Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

This form is often more practical since it relates $Q^\pi$ directly to $V^\pi$.

---

## 3. Bellman Optimality Equation for $V^*$

### 3.1 Statement

For all states $s \in \mathcal{S}$:

$$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^*(s')\right]$$

### 3.2 Full Derivation

**Starting point**: Definition of optimal value function.

$$V^*(s) = \max_\pi V^\pi(s)$$

**Key insight**: For any state $s$, there exists an optimal action $a^*$ such that taking $a^*$ and then acting optimally achieves $V^*(s)$.

**Step 1**: Express $V^*(s)$ as maximum over first action.

$$V^*(s) = \max_{a \in \mathcal{A}} Q^*(s, a)$$

This follows because the optimal policy is greedy with respect to $Q^*$:
- If we take any action $a$ and then act optimally, we get $Q^*(s, a)$
- The optimal value is achieved by the best first action

**Step 2**: Express $Q^*(s, a)$ in terms of $V^*$.

$$Q^*(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^*(s')\right]$$

**Justification**: Under the optimal policy, after taking action $a$ in state $s$:
- We receive immediate reward $R(s, a, s')$
- We transition to state $s'$ with probability $P(s' \mid s, a)$
- From $s'$, the maximum achievable return is $V^*(s')$

**Step 3**: Substitute.

$$\boxed{V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^*(s')\right]}$$

### 3.3 Why $\max$ Instead of $\sum_a \pi(a \mid s)$?

In the Bellman expectation equation, we average over actions weighted by $\pi(a \mid s)$.

In the Bellman optimality equation, we take the maximum because:
1. The optimal policy is deterministic (for finite MDPs)
2. The optimal policy chooses the best action with probability 1
3. Therefore, the expectation over actions becomes a maximum

---

## 4. Bellman Optimality Equation for $Q^*$

### 4.1 Statement

For all states $s \in \mathcal{S}$ and actions $a \in \mathcal{A}$:

$$Q^*(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a')\right]$$

### 4.2 Full Derivation

**Starting point**: Definition and relationship.

$$Q^*(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^*(s')\right]$$

**Step 1**: Use $V^*(s') = \max_{a'} Q^*(s', a')$.

$$\boxed{Q^*(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a')\right]}$$

### 4.3 Interpretation

The optimal action-value at $(s, a)$ equals:
- The expected immediate reward $\sum_{s'} P(s' \mid s, a) R(s, a, s')$
- Plus the discounted expected optimal future value $\gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')$

---

## 5. Fixed-Point Interpretation

### 5.1 Bellman Operators

Define the **Bellman expectation operator** $T^\pi$ for a given policy $\pi$:

$$(T^\pi V)(s) \triangleq \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V(s')\right]$$

The Bellman expectation equation states that $V^\pi$ is a **fixed point** of $T^\pi$:

$$T^\pi V^\pi = V^\pi$$

Similarly, define the **Bellman optimality operator** $T^*$:

$$(T^* V)(s) \triangleq \max_{a} \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V(s')\right]$$

The Bellman optimality equation states that $V^*$ is a **fixed point** of $T^*$:

$$T^* V^* = V^*$$

### 5.2 Uniqueness of Fixed Points

**Theorem**: For $\gamma < 1$:
1. $T^\pi$ has a unique fixed point, which is $V^\pi$
2. $T^*$ has a unique fixed point, which is $V^*$

**Proof**: Both operators are **contraction mappings** under the supremum norm.

### 5.3 Contraction Mapping Property

**Definition**: An operator $T$ is a $\gamma$-contraction under norm $\|\cdot\|$ if:

$$\|T V_1 - T V_2\| \leq \gamma \|V_1 - V_2\| \quad \forall V_1, V_2$$

**Theorem**: $T^\pi$ and $T^*$ are $\gamma$-contractions under the supremum norm $\|V\|_\infty = \max_s |V(s)|$.

**Proof for $T^\pi$**:

For any $V_1, V_2$:

$$|(T^\pi V_1)(s) - (T^\pi V_2)(s)| = \gamma \left|\sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) [V_1(s') - V_2(s')]\right|$$

$$\leq \gamma \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) |V_1(s') - V_2(s')|$$

$$\leq \gamma \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \|V_1 - V_2\|_\infty$$

$$= \gamma \|V_1 - V_2\|_\infty$$

Therefore: $\|T^\pi V_1 - T^\pi V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$

**Proof for $T^*$**: Similar, using the fact that $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$.

### 5.4 Banach Fixed-Point Theorem

**Theorem (Banach)**: If $T$ is a contraction mapping on a complete metric space, then:
1. $T$ has a unique fixed point $V^*$
2. For any initial $V_0$, the sequence $V_{k+1} = T V_k$ converges to $V^*$
3. The rate of convergence is geometric: $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$

This theorem guarantees that iteratively applying the Bellman operators converges to the unique value function.

---

## 6. Matrix Form of Bellman Equations

### 6.1 Notation

Let $n = |\mathcal{S}|$ and define:
- $\mathbf{V}^\pi \in \mathbb{R}^n$: Value function vector, $[\mathbf{V}^\pi]_i = V^\pi(s_i)$
- $\mathbf{P}^\pi \in \mathbb{R}^{n \times n}$: Transition matrix under $\pi$, $[\mathbf{P}^\pi]_{ij} = \sum_a \pi(a \mid s_i) P(s_j \mid s_i, a)$
- $\mathbf{r}^\pi \in \mathbb{R}^n$: Expected reward vector, $[\mathbf{r}^\pi]_i = \sum_a \pi(a \mid s_i) \sum_{s'} P(s' \mid s_i, a) R(s_i, a, s')$

### 6.2 Bellman Expectation Equation (Matrix Form)

$$\mathbf{V}^\pi = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi$$

**Derivation**: The $i$-th component is:

$$[\mathbf{V}^\pi]_i = [\mathbf{r}^\pi]_i + \gamma \sum_j [\mathbf{P}^\pi]_{ij} [\mathbf{V}^\pi]_j$$

$$V^\pi(s_i) = \sum_a \pi(a \mid s_i) \sum_{s'} P(s' \mid s_i, a) R(s_i, a, s') + \gamma \sum_a \pi(a \mid s_i) \sum_{s_j} P(s_j \mid s_i, a) V^\pi(s_j)$$

This matches the Bellman expectation equation.

### 6.3 Closed-Form Solution

Rearranging the matrix equation:

$$\mathbf{V}^\pi - \gamma \mathbf{P}^\pi \mathbf{V}^\pi = \mathbf{r}^\pi$$

$$(\mathbf{I} - \gamma \mathbf{P}^\pi) \mathbf{V}^\pi = \mathbf{r}^\pi$$

$$\boxed{\mathbf{V}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi}$$

**Existence of inverse**: The matrix $(\mathbf{I} - \gamma \mathbf{P}^\pi)$ is invertible when $\gamma < 1$.

**Proof**: $\mathbf{P}^\pi$ is a stochastic matrix, so all eigenvalues $\lambda$ satisfy $|\lambda| \leq 1$. The eigenvalues of $(\mathbf{I} - \gamma \mathbf{P}^\pi)$ are $(1 - \gamma \lambda)$. Since $|\lambda| \leq 1$ and $\gamma < 1$, we have $|1 - \gamma \lambda| \geq 1 - \gamma > 0$. Thus no eigenvalue is zero, and the matrix is invertible.

### 6.4 Neumann Series Expansion

The inverse can be expressed as:

$$(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} = \sum_{k=0}^{\infty} (\gamma \mathbf{P}^\pi)^k = \mathbf{I} + \gamma \mathbf{P}^\pi + \gamma^2 (\mathbf{P}^\pi)^2 + \cdots$$

This converges because $\|\gamma \mathbf{P}^\pi\| \leq \gamma < 1$.

**Interpretation**: The value function is the sum of expected discounted rewards over all future time steps:
- $\mathbf{r}^\pi$: Immediate expected reward
- $\gamma \mathbf{P}^\pi \mathbf{r}^\pi$: Expected reward at $t+1$, discounted
- $\gamma^2 (\mathbf{P}^\pi)^2 \mathbf{r}^\pi$: Expected reward at $t+2$, discounted
- And so on...

---

## 7. Summary of Bellman Equations

| Equation | Formula |
|----------|---------|
| **Expectation for $V^\pi$** | $V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) [R(s,a,s') + \gamma V^\pi(s')]$ |
| **Expectation for $Q^\pi$** | $Q^\pi(s,a) = \sum_{s'} P(s' \mid s, a) [R(s,a,s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s',a')]$ |
| **Optimality for $V^*$** | $V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) [R(s,a,s') + \gamma V^*(s')]$ |
| **Optimality for $Q^*$** | $Q^*(s,a) = \sum_{s'} P(s' \mid s, a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$ |

### Key Relationships

```
V^π(s) ←--[π(a|s)]-- Q^π(s,a) ←--[P,R,γ]-- V^π(s')
                                              |
                                         [π(a'|s')]
                                              ↓
                                          Q^π(s',a')
```

```
V*(s) ←--[max_a]-- Q*(s,a) ←--[P,R,γ]-- V*(s')
                                           |
                                      [max_a']
                                           ↓
                                       Q*(s',a')
```

### Visual Intuition (ASCII)

```
Bellman Expectation (fixed policy π)

   State s
     |
     | choose action a ~ π(a|s)
     v
   Action a
     |\
     | \ immediate reward R(s,a,s')
     |  \
     |   v
     |  [reward]
     |
     | P(s'|s,a)
     v
   Next state s'
     |
     | value V^π(s')
     v
   [future value]
        \
         \  (reward + γ * future value)
          \
           v
          V^π(s)

Bellman Optimality (best action)

   State s
     |
     | choose best action a*
     v
   Action a*
     |\
     | \ immediate reward R(s,a,s')
     |  \
     |   v
     |  [reward]
     |
     | P(s'|s,a)
     v
   Next state s'
     |
     | best future value max_a' Q*(s',a')
     v
   [best future value]
        \
         \  (reward + γ * best future value)
          \
           v
          V*(s)
```

---

## 8. Computational Implications

| Property | Bellman Expectation | Bellman Optimality |
|----------|--------------------|--------------------|
| Linearity | Linear in $V$ | **Nonlinear** (due to $\max$) |
| Closed-form solution | Yes: $(\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi$ | No |
| Iterative solution | Yes (converges) | Yes (converges) |
| Unique fixed point | Yes | Yes |

The nonlinearity of the Bellman optimality equation means we cannot solve it directly via matrix inversion. Instead, we use iterative methods (value iteration, policy iteration) covered in the next module.

---

## References

1. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 3.
3. Bertsekas, D. P., & Tsitsiklis, J. N. (1996). *Neuro-Dynamic Programming*. Athena Scientific. Chapter 2.
