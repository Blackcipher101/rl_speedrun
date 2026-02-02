# FrozenLake Environment

Solving the FrozenLake-v1 environment from Gymnasium using dynamic programming.

---

## Environment Description

FrozenLake is a classic gridworld environment where an agent navigates a frozen lake to reach a goal while avoiding holes. The key challenge is that the ice is **slippery**: actions are stochastic.

### Grid Layout (4x4)

```
SFFF    S = Start (state 0)
FHFH    F = Frozen (safe)
FFFH    H = Hole (terminal, reward 0)
HFFG    G = Goal (terminal, reward 1)
```

Map encoding:
- **S**: Starting position (state 0)
- **F**: Frozen surface (safe to walk)
- **H**: Hole (fall in, episode ends with reward 0)
- **G**: Goal (reach it, episode ends with reward 1)

---

## State Space

**States**: $\mathcal{S} = \{0, 1, 2, \ldots, 15\}$

State encoding (row-major order):

| State | Position | Cell Type |
|-------|----------|-----------|
| 0 | (0, 0) | Start (S) |
| 1 | (0, 1) | Frozen (F) |
| 2 | (0, 2) | Frozen (F) |
| 3 | (0, 3) | Frozen (F) |
| 4 | (1, 0) | Frozen (F) |
| 5 | (1, 1) | **Hole (H)** |
| 6 | (1, 2) | Frozen (F) |
| 7 | (1, 3) | **Hole (H)** |
| 8 | (2, 0) | Frozen (F) |
| 9 | (2, 1) | Frozen (F) |
| 10 | (2, 2) | Frozen (F) |
| 11 | (2, 3) | **Hole (H)** |
| 12 | (3, 0) | **Hole (H)** |
| 13 | (3, 1) | Frozen (F) |
| 14 | (3, 2) | Frozen (F) |
| 15 | (3, 3) | **Goal (G)** |

---

## Action Space

**Actions**: $\mathcal{A} = \{0, 1, 2, 3\}$

| Action | Direction |
|--------|-----------|
| 0 | LEFT |
| 1 | DOWN |
| 2 | RIGHT |
| 3 | UP |

---

## Transition Dynamics (Slippery Ice)

This is the key feature of FrozenLake: **stochastic transitions**.

When the agent takes an action, the ice is slippery, so:
- With probability $\frac{1}{3}$: move in the **intended direction**
- With probability $\frac{1}{3}$: move **perpendicular (clockwise)**
- With probability $\frac{1}{3}$: move **perpendicular (counter-clockwise)**

**Example**: If action is RIGHT:
- $P(\text{move right}) = \frac{1}{3}$
- $P(\text{move down}) = \frac{1}{3}$
- $P(\text{move up}) = \frac{1}{3}$

**Boundary handling**: If a movement would go off the grid, the agent stays in place.

**Terminal states**: Holes (5, 7, 11, 12) and Goal (15) are absorbing states.

---

## Reward Structure

$$R(s, a, s') = \begin{cases} 1 & \text{if } s' = 15 \text{ (goal)} \\ 0 & \text{otherwise} \end{cases}$$

The agent only receives reward upon reaching the goal. Falling into a hole gives no reward.

---

## MDP Formulation

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$

| Component | Value |
|-----------|-------|
| $\mathcal{S}$ | $\{0, 1, \ldots, 15\}$ |
| $\mathcal{A}$ | $\{0, 1, 2, 3\}$ (LEFT, DOWN, RIGHT, UP) |
| $P(s' \mid s, a)$ | Stochastic (see above) |
| $R(s, a, s')$ | 1 if $s' = 15$, else 0 |
| $\gamma$ | Typically 0.99 or 1.0 |

---

## Extracting Dynamics from Gymnasium

The transition dynamics are available via `env.P`:

```python
import gymnasium as gym

env = gym.make('FrozenLake-v1', is_slippery=True)

# env.P[s][a] is a list of (probability, next_state, reward, done) tuples
# Example: env.P[0][2] gives transitions for state 0, action RIGHT
```

This allows us to construct the full MDP matrices for dynamic programming.

---

## Expected Results

Due to the stochastic nature, the optimal policy is not intuitive. Even with the optimal policy, success rate is limited by the random transitions.

**Typical optimal value at start (V*(0))**: Approximately 0.74 with $\gamma = 0.99$

The optimal policy generally:
- Avoids actions that lead toward holes
- Accounts for the slip probability

---

## Running the Solver

```bash
python solve_frozenlake.py
```

This will:
1. Load the FrozenLake environment from Gymnasium
2. Extract transition dynamics
3. Solve using value iteration
4. Display optimal value function and policy
5. Run evaluation episodes to verify performance

---

## Key Insights

1. **Stochastic transitions make planning harder**: Even with full knowledge of dynamics, the agent cannot guarantee reaching the goal.

2. **Value function interpretation**: $V^*(s)$ represents the probability of reaching the goal from state $s$ under the optimal policy (when $\gamma = 1$ and reward is 1 for goal only).

3. **Policy may seem counterintuitive**: Sometimes the optimal action avoids the direct path to account for slip risk near holes.

---

## References

1. Gymnasium FrozenLake documentation: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.
