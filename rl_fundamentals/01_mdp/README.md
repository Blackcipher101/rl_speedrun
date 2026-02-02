# Markov Decision Processes (MDPs)

This document provides a rigorous, first-principles treatment of Markov Decision Processes, the mathematical framework underlying reinforcement learning.

---

## 1. Formal Definition

A **Markov Decision Process** is a 5-tuple:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$

where:

| Symbol | Name | Description |
|--------|------|-------------|
| $\mathcal{S}$ | State space | Finite set of states |
| $\mathcal{A}$ | Action space | Finite set of actions |
| $P$ | Transition probability function | $P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ |
| $R$ | Reward function | $R: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$ |
| $\gamma$ | Discount factor | $\gamma \in [0, 1]$ |

**Assumption**: Throughout this document, we assume $\mathcal{S}$ and $\mathcal{A}$ are finite sets with $|\mathcal{S}| = n$ and $|\mathcal{A}| = m$.

---

## 2. The Markov Property

The defining characteristic of an MDP is the **Markov property**: the future is conditionally independent of the past given the present.

**Definition (Markov Property)**: A stochastic process $\{S_t\}_{t \geq 0}$ satisfies the Markov property if for all $t \geq 0$ and all states $s_0, s_1, \ldots, s_t, s_{t+1} \in \mathcal{S}$:

$$P(S_{t+1} = s_{t+1} \mid S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, \ldots, S_0 = s_0) = P(S_{t+1} = s_{t+1} \mid S_t = s_t, A_t = a_t)$$

**Interpretation**: The state $S_t$ contains all information necessary to predict the distribution of $S_{t+1}$. The history $(S_0, A_0, S_1, A_1, \ldots, S_{t-1}, A_{t-1})$ provides no additional predictive power beyond $S_t$.

**Why this matters**: The Markov property enables recursive decomposition of the decision problem. Without it, we would need to condition on the entire history, making computation intractable.

---

## 3. Transition Probability Function

The transition probability function $P$ specifies the dynamics of the environment.

**Definition**: For states $s, s' \in \mathcal{S}$ and action $a \in \mathcal{A}$:

$$P(s' \mid s, a) \triangleq \Pr\{S_{t+1} = s' \mid S_t = s, A_t = a\}$$

**Properties** (these follow from probability axioms):

1. **Non-negativity**: $P(s' \mid s, a) \geq 0$ for all $s, s' \in \mathcal{S}$, $a \in \mathcal{A}$

2. **Normalization**: For all $s \in \mathcal{S}$, $a \in \mathcal{A}$:
   $$\sum_{s' \in \mathcal{S}} P(s' \mid s, a) = 1$$

**Alternative notation**: We often write $P_{ss'}^a \triangleq P(s' \mid s, a)$ when working with matrices.

**Matrix representation**: For each action $a$, we can define a transition matrix $\mathbf{P}^a \in \mathbb{R}^{n \times n}$ where:

$$[\mathbf{P}^a]_{ij} = P(s_j \mid s_i, a)$$

Each row of $\mathbf{P}^a$ sums to 1 (i.e., $\mathbf{P}^a$ is a stochastic matrix).

---

## 4. Reward Function

The reward function quantifies the immediate desirability of state transitions.

### 4.1 Full Reward Function

**Definition**: The most general form is:

$$R: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$$

where $R(s, a, s')$ is the reward received when transitioning from state $s$ to state $s'$ under action $a$.

### 4.2 Expected Reward Function

In many contexts, we work with the **expected reward** $r(s, a)$:

$$r(s, a) \triangleq \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a] = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \cdot R(s, a, s')$$

### 4.3 Simplified Reward Functions

Depending on the problem structure, rewards may depend on fewer arguments:

| Form | Notation | When Used |
|------|----------|-----------|
| $R(s, a, s')$ | Full | Reward depends on transition |
| $R(s, a)$ | State-action | Reward depends on state and action only |
| $R(s)$ | State-only | Reward depends on state reached |

**Convention in this repository**: Unless otherwise stated, we use $R(s, a, s')$ for generality. The expected reward $r(s, a)$ is computed from this when needed.

### 4.4 Stochastic vs. Deterministic Rewards

**Deterministic rewards**: $R(s, a, s')$ is a fixed value for each $(s, a, s')$ triple.

**Stochastic rewards**: $R_{t+1}$ is a random variable with distribution depending on $(S_t, A_t, S_{t+1})$. In this case, $R(s, a, s')$ represents the expected value:

$$R(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']$$

---

## 5. Discount Factor

The discount factor $\gamma \in [0, 1]$ determines the present value of future rewards.

**Purpose**: Without discounting (or a finite horizon), the sum of rewards could be infinite, making comparisons between policies undefined.

**Interpretation**:

1. **Economic**: $\gamma$ represents the time value of reward. A reward of 1 received $k$ steps in the future is worth $\gamma^k$ today.

2. **Probabilistic**: $\gamma$ can be interpreted as the probability of the process continuing at each step. With probability $(1 - \gamma)$, the process terminates.

3. **Mathematical**: $\gamma < 1$ ensures the infinite sum of discounted rewards converges (for bounded rewards).

**Extreme cases**:
- $\gamma = 0$: Agent is "myopic" — only immediate rewards matter
- $\gamma = 1$: All future rewards count equally (only valid for episodic tasks)

---

## 6. Return (Cumulative Discounted Reward)

The **return** $G_t$ is the total discounted reward from time $t$ onward.

**Definition**:

$$G_t \triangleq \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

**Recursive formulation**: The return satisfies a recursive relationship:

$$G_t = R_{t+1} + \gamma G_{t+1}$$

**Proof**:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + \cdots) = R_{t+1} + \gamma G_{t+1}$$

**Boundedness**: If rewards are bounded, i.e., $|R(s, a, s')| \leq R_{\max}$ for all $(s, a, s')$, then:

$$|G_t| \leq \sum_{k=0}^{\infty} \gamma^k R_{\max} = \frac{R_{\max}}{1 - \gamma}$$

This bound is crucial for proving convergence of dynamic programming algorithms.

---

## 7. Episodic vs. Continuing Tasks

### 7.1 Episodic Tasks

**Definition**: An episodic task has a natural termination point. The agent-environment interaction breaks into **episodes**, each ending in a **terminal state**.

**Notation**: Let $\mathcal{S}^+$ denote the set of all states including terminal states, and $\mathcal{S}$ denote non-terminal states only.

**Properties of terminal states**:
- Once reached, no further transitions or rewards occur
- Formally: if $s_{\text{term}}$ is terminal, then $P(s_{\text{term}} \mid s_{\text{term}}, a) = 1$ and $R(s_{\text{term}}, a, s_{\text{term}}) = 0$ for all $a$

**Return for episodic tasks**: If the episode terminates at time $T$:

$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

When $\gamma = 1$, this simplifies to the undiscounted sum of rewards.

### 7.2 Continuing Tasks

**Definition**: A continuing task has no natural termination. The agent-environment interaction continues indefinitely.

**Requirement**: For continuing tasks, we must have $\gamma < 1$ to ensure $G_t$ is well-defined (finite).

### 7.3 Unified Notation

We can unify episodic and continuing tasks by treating terminal states as **absorbing states** with zero reward:

- Terminal state $s_{\text{term}}$ transitions only to itself: $P(s_{\text{term}} \mid s_{\text{term}}, a) = 1$
- All rewards from terminal states are zero: $R(s_{\text{term}}, a, s') = 0$

With this convention, the infinite sum definition of $G_t$ applies to both cases.

---

## 8. The Agent-Environment Interface

At each discrete time step $t = 0, 1, 2, \ldots$:

1. Agent observes state $S_t \in \mathcal{S}$
2. Agent selects action $A_t \in \mathcal{A}$ according to its policy
3. Environment transitions to $S_{t+1} \sim P(\cdot \mid S_t, A_t)$
4. Agent receives reward $R_{t+1} = R(S_t, A_t, S_{t+1})$

**Trajectory**: A sequence of states, actions, and rewards:

$$\tau = (S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \ldots)$$

**Note on indexing**: $R_{t+1}$ is the reward received after taking action $A_t$ in state $S_t$. This indexing reflects that the reward is a consequence of the state-action pair $(S_t, A_t)$.

---

## 9. Assumptions and Limitations

**Assumptions made in the MDP framework**:

1. **Full observability**: The agent knows the current state $S_t$ exactly. (Relaxed in POMDPs)

2. **Discrete time**: Time proceeds in discrete steps. (Continuous-time MDPs exist but are not covered here)

3. **Finite state and action spaces**: $|\mathcal{S}| < \infty$ and $|\mathcal{A}| < \infty$. (Can be relaxed with appropriate measure theory)

4. **Stationary dynamics**: $P(s' \mid s, a)$ and $R(s, a, s')$ do not change over time.

5. **Known model** (for planning): Dynamic programming assumes $P$ and $R$ are known. (Relaxed in model-free RL)

**When MDPs are insufficient**:

- **Partial observability**: Agent cannot observe the full state → POMDP
- **Non-stationary environments**: Dynamics change over time → Non-stationary MDP
- **Multiple agents**: Agents interact → Multi-agent RL, Game Theory
- **Continuous state/action**: Infinite spaces → Function approximation required

---

## 10. Summary

| Concept | Mathematical Definition |
|---------|------------------------|
| MDP | $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$ |
| Markov Property | $P(S_{t+1} \mid S_t, A_t, \text{history}) = P(S_{t+1} \mid S_t, A_t)$ |
| Transition Probability | $P(s' \mid s, a) = \Pr\{S_{t+1} = s' \mid S_t = s, A_t = a\}$ |
| Expected Reward | $r(s, a) = \sum_{s'} P(s' \mid s, a) R(s, a, s')$ |
| Return | $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ |
| Return (recursive) | $G_t = R_{t+1} + \gamma G_{t+1}$ |

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Puterman, M. L. (2014). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
3. Bertsekas, D. P. (2012). *Dynamic Programming and Optimal Control* (4th ed.). Athena Scientific.
