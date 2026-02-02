# GridWorld Environment

A custom 4x4 gridworld environment for demonstrating dynamic programming methods.

---

## Environment Description

GridWorld is a classic reinforcement learning environment where an agent navigates a grid to reach goal states.

### Grid Layout (4x4)

```
+----+----+----+----+
| G  |    |    |    |   Row 0
+----+----+----+----+
|    |    |    |    |   Row 1
+----+----+----+----+
|    |    |    |    |   Row 2
+----+----+----+----+
|    |    |    | G  |   Row 3
+----+----+----+----+
  0    1    2    3
       Columns
```

- **G**: Goal states (terminal) at positions (0,0) and (3,3)
- All other cells are non-terminal states

---

## State Space

**States**: $\mathcal{S} = \{0, 1, 2, \ldots, 15\}$

State encoding (row-major order):

$$s = \text{row} \times 4 + \text{col}$$

| State | Position (row, col) |
|-------|---------------------|
| 0 | (0, 0) - **Terminal** |
| 1 | (0, 1) |
| 2 | (0, 2) |
| 3 | (0, 3) |
| 4 | (1, 0) |
| 5 | (1, 1) |
| 6 | (1, 2) |
| 7 | (1, 3) |
| 8 | (2, 0) |
| 9 | (2, 1) |
| 10 | (2, 2) |
| 11 | (2, 3) |
| 12 | (3, 0) |
| 13 | (3, 1) |
| 14 | (3, 2) |
| 15 | (3, 3) - **Terminal** |

---

## Action Space

**Actions**: $\mathcal{A} = \{0, 1, 2, 3\}$

| Action | Direction | Effect on (row, col) |
|--------|-----------|---------------------|
| 0 | UP | row - 1 |
| 1 | DOWN | row + 1 |
| 2 | LEFT | col - 1 |
| 3 | RIGHT | col + 1 |

---

## Transition Dynamics

**Deterministic transitions**: Actions always succeed (no stochasticity).

$$P(s' \mid s, a) = \begin{cases} 1 & \text{if } s' = \text{result of action } a \text{ from } s \\ 0 & \text{otherwise} \end{cases}$$

**Boundary handling**: If an action would move the agent off the grid, the agent stays in place.

**Terminal states**: States 0 and 15 are absorbing. Once reached:
- All actions lead back to the same state: $P(s_{\text{term}} \mid s_{\text{term}}, a) = 1$
- No further rewards are received

---

## Reward Structure

**Reward function**: $R(s, a, s') = -1$ for all transitions from non-terminal states.

$$R(s, a, s') = \begin{cases} 0 & \text{if } s \in \{0, 15\} \text{ (terminal)} \\ -1 & \text{otherwise} \end{cases}$$

**Interpretation**: The agent incurs a cost of -1 for each step. The optimal policy minimizes the number of steps to reach a terminal state.

---

## MDP Formulation

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$

| Component | Value |
|-----------|-------|
| $\mathcal{S}$ | $\{0, 1, \ldots, 15\}$ |
| $\mathcal{A}$ | $\{0, 1, 2, 3\}$ (UP, DOWN, LEFT, RIGHT) |
| $P(s' \mid s, a)$ | Deterministic (see above) |
| $R(s, a, s')$ | $-1$ for non-terminal, $0$ for terminal |
| $\gamma$ | Typically 1.0 (undiscounted episodic) |

---

## Expected Optimal Policy

Due to symmetry, the optimal policy directs the agent toward the nearest terminal state:

```
Optimal Policy (arrows point to action):
+----+----+----+----+
| *  | ←  | ←  | ↓  |
+----+----+----+----+
| ↑  | ↑  | ↓  | ↓  |
+----+----+----+----+
| ↑  | ↑  | ↓  | ↓  |
+----+----+----+----+
| ↑  | →  | →  | *  |
+----+----+----+----+
```

- States in the upper-left region go toward (0,0)
- States in the lower-right region go toward (3,3)
- The diagonal is a "decision boundary" where either terminal is equally close

---

## Expected Optimal Value Function

With $\gamma = 1$ (undiscounted) and reward of $-1$ per step:

$$V^*(s) = -(\text{Manhattan distance to nearest terminal})$$

```
Optimal Value Function V*(s):
+------+------+------+------+
|  0   | -1   | -2   | -3   |
+------+------+------+------+
| -1   | -2   | -3   | -2   |
+------+------+------+------+
| -2   | -3   | -2   | -1   |
+------+------+------+------+
| -3   | -2   | -1   |  0   |
+------+------+------+------+
```

---

## Running the Solver

```bash
python gridworld_dp.py
```

This will:
1. Construct the GridWorld MDP
2. Solve using both value iteration and policy iteration
3. Display the optimal value function and policy

---

## Implementation Notes

1. **No external dependencies**: The MDP is constructed manually (no gymnasium)
2. **Transition matrix**: $P$ is a $(16 \times 4 \times 16)$ tensor
3. **Reward matrix**: $R$ is a $(16 \times 4 \times 16)$ tensor
4. **Discount factor**: $\gamma = 1.0$ (can be modified to see effect)

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Example 4.1.
