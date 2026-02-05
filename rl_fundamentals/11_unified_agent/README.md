# Unified Agent: Exploration Strategies & Reward Shaping

This module consolidates core tabular RL concepts into a flexible, configurable agent framework. It demonstrates how to build modular RL systems with pluggable components.

---

## Key Concepts

### 1. The Exploration-Exploitation Tradeoff

The fundamental dilemma in RL:
- **Explore**: Try new actions to discover potentially better rewards
- **Exploit**: Use current knowledge to maximize immediate reward

```
                    Exploration-Exploitation Spectrum

    Pure Exploration                              Pure Exploitation
         ↓                                              ↓
    Random policy                                 Greedy policy
    (learns nothing useful)                      (may miss optimal)
         │                                              │
         └──────────────── Balance ────────────────────┘
                              ↓
                     Effective learning
```

---

## 2. Exploration Strategies

### 2.1 Epsilon-Greedy

**The simplest approach:**

$$a = \begin{cases}
\text{random action} & \text{with probability } \varepsilon \\
\arg\max_a Q(s,a) & \text{with probability } 1-\varepsilon
\end{cases}$$

**Decay schedules:**
- **Exponential**: $\varepsilon_t = \max(\varepsilon_{min}, \varepsilon_0 \cdot \rho^t)$
- **Linear**: $\varepsilon_t = \max(\varepsilon_{min}, \varepsilon_0 - \delta \cdot t)$

| Pros | Cons |
|------|------|
| Simple to implement | Explores uniformly (ignores Q-values) |
| Easy to tune | Exploration independent of uncertainty |
| Well-understood | May over-explore bad actions |

### 2.2 Boltzmann (Softmax) Exploration

**Action selection probability:**

$$\pi(a|s) = \frac{\exp(Q(s,a) / \tau)}{\sum_{a'} \exp(Q(s,a') / \tau)}$$

**Temperature τ controls exploration:**
- $\tau \to \infty$: Uniform random (max exploration)
- $\tau \to 0$: Greedy (max exploitation)

```
Temperature Effect on Action Probabilities

Q-values: [1.0, 2.0, 1.5, 0.5]

τ = 5.0 (high):    [0.21, 0.28, 0.25, 0.17]  (nearly uniform)
τ = 1.0 (medium):  [0.11, 0.43, 0.26, 0.09]  (preference visible)
τ = 0.1 (low):     [0.00, 0.99, 0.01, 0.00]  (nearly greedy)
```

| Pros | Cons |
|------|------|
| Uses Q-value information | Temperature tuning required |
| Prefers better actions | Numerically unstable without care |
| Smooth probability distribution | Doesn't track uncertainty |

### 2.3 Upper Confidence Bound (UCB)

**Action selection:**

$$a = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}} \right]$$

Where:
- $N(s)$: Total visits to state $s$
- $N(s,a)$: Times action $a$ taken in state $s$
- $c$: Exploration constant

**Key insight**: The bonus term $c\sqrt{\ln N(s) / N(s,a)}$ shrinks as an action is taken more often, naturally balancing exploration.

```
UCB Bonus Visualization

Visits:     N(s,a) = 1    10    100    1000
Bonus:              1.66  0.52  0.17   0.05
                    ↓     ↓     ↓      ↓
               High exp  →→→→→→→→  Low exp
```

| Pros | Cons |
|------|------|
| Theoretically optimal regret | Requires visit counting |
| Naturally balances exploration | Memory for counts |
| No hyperparameter decay needed | Complex for large state spaces |

---

## 3. Strategy Comparison

| Strategy | Uses Q-info | Requires Counts | Optimal Regret | Complexity |
|----------|-------------|-----------------|----------------|------------|
| ε-greedy | No | No | O(√T) | Simple |
| Boltzmann | Yes | No | O(√T) | Medium |
| UCB | Yes | Yes | O(log T) | Complex |

**Regret** = Expected cumulative reward loss compared to optimal policy

---

## 4. Reward Shaping

### The Problem

Sparse rewards slow learning:
- Agent must stumble upon reward by random exploration
- Credit assignment over long horizons is difficult

### The Solution: Shaped Rewards

Augment reward with shaping function:
$$R'(s, a, s') = R(s, a, s') + F(s, s')$$

### Potential-Based Shaping

**Theorem (Ng et al., 1999)**: If the shaping function has the form:

$$F(s, s') = \gamma \cdot \phi(s') - \phi(s)$$

where $\phi: S \to \mathbb{R}$ is a potential function, then the **optimal policy is preserved**.

**Proof sketch**: The shaped return equals the original return plus a telescoping sum that depends only on the start and end states, not the policy.

### Example: Distance-Based Potential

For a gridworld with goal at state $g$:

$$\phi(s) = -\text{distance}(s, g)$$

This gives bonus for moving toward the goal:
- Moving closer: $F > 0$ (bonus)
- Moving away: $F < 0$ (penalty)
- Staying same: $F = 0$ (no effect)

```
Potential Function (4x4 GridWorld, Goal at bottom-right)

φ(s) values:
  -6  -5  -4  -3
  -5  -4  -3  -2
  -4  -3  -2  -1
  -3  -2  -1   0  ← Goal

Shaping reward for moving right from (0,0) to (0,1):
F = 0.99 × (-5) - (-6) = -4.95 + 6 = +1.05
```

---

## 5. Module Structure

```
11_unified_agent/
├── README.md                    # This file
├── exploration_strategies.py    # Modular exploration strategies
└── unified_agent.py             # Configurable agent framework
```

### Key Classes

```python
# Exploration strategies
class ExplorationStrategy(ABC):
    def select_action(q_values, state, step) -> int
    def decay()

class EpsilonGreedy(ExplorationStrategy)
class BoltzmannExploration(ExplorationStrategy)
class UCBExploration(ExplorationStrategy)

# Unified agent
class UnifiedAgent:
    def __init__(n_states, n_actions, algorithm, exploration, alpha, gamma, reward_shaping)
    def train(env, n_episodes) -> TrainingMetrics
    def evaluate(env, n_episodes) -> (mean, std)
```

---

## 6. Usage Examples

### Basic Q-Learning

```python
from unified_agent import UnifiedAgent
from exploration_strategies import EpsilonGreedy

agent = UnifiedAgent(
    n_states=16,
    n_actions=4,
    algorithm="q_learning",
    exploration=EpsilonGreedy(epsilon=1.0, decay_rate=0.995),
    alpha=0.1,
    gamma=0.99
)

metrics = agent.train(env, n_episodes=10000)
mean_return, std_return = agent.evaluate(env, n_episodes=100)
```

### SARSA with Boltzmann

```python
agent = UnifiedAgent(
    n_states=16,
    n_actions=4,
    algorithm="sarsa",
    exploration=BoltzmannExploration(temperature=2.0, decay_rate=0.995),
    alpha=0.1,
    gamma=0.99
)
```

### With Reward Shaping

```python
from unified_agent import (
    UnifiedAgent,
    distance_based_potential,
    potential_based_shaping
)

potential_fn = distance_based_potential(goal_state=15, n_states=16, grid_size=4)
shaping_fn = potential_based_shaping(potential_fn, gamma=0.99)

agent = UnifiedAgent(
    n_states=16,
    n_actions=4,
    algorithm="q_learning",
    reward_shaping=shaping_fn
)
```

---

## 7. Running the Demo

```bash
cd rl_fundamentals/11_unified_agent
python unified_agent.py
```

### Expected Output

```
--- Q-Learning + Epsilon-Greedy ---
Episode   100: Avg return (100 ep) = 0.01
...
Final evaluation: 0.73 ± 0.44

--- SARSA + Epsilon-Greedy ---
...
Final evaluation: 0.71 ± 0.45

--- Q-Learning + Boltzmann ---
...
Final evaluation: 0.69 ± 0.46

--- Q-Learning + Reward Shaping ---
...
Final evaluation: 0.74 ± 0.43
```

---

## 8. Key Takeaways

1. **Exploration matters**: Different strategies work better in different environments
2. **ε-greedy is a good default**: Simple, effective, well-understood
3. **Boltzmann uses more information**: Can be better when Q-values are meaningful
4. **UCB is theoretically optimal**: But complex for large state spaces
5. **Reward shaping accelerates learning**: But must be potential-based to preserve optimality

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. Chapter 2 (Multi-armed Bandits).
2. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.
3. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2-3), 235-256.
