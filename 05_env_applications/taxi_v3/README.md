# Taxi-v3 Environment

Solving the Taxi-v3 environment from Gymnasium using dynamic programming.

---

## Environment Description

Taxi-v3 is a navigation and pick-up/drop-off task where a taxi must:
1. Navigate to a passenger location
2. Pick up the passenger
3. Navigate to the destination
4. Drop off the passenger

### Grid Layout (5x5)

```
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```

- **R, G, Y, B**: Four designated locations (Red, Green, Yellow, Blue)
- **|**: Walls (cannot pass through)
- **:** : Passable areas

---

## State Space

**States**: $\mathcal{S} = \{0, 1, 2, \ldots, 499\}$ (500 discrete states)

State encoding combines four components:

$$s = (\text{taxi\_row} \times 5 + \text{taxi\_col}) \times 5 \times 4 + \text{passenger\_loc} \times 4 + \text{destination}$$

| Component | Range | Description |
|-----------|-------|-------------|
| taxi_row | 0-4 | Taxi's row position |
| taxi_col | 0-4 | Taxi's column position |
| passenger_loc | 0-4 | Passenger location (0=R, 1=G, 2=Y, 3=B, 4=in taxi) |
| destination | 0-3 | Destination (0=R, 1=G, 2=Y, 3=B) |

**Location indices**:
| Index | Location | Position (row, col) |
|-------|----------|---------------------|
| 0 | R (Red) | (0, 0) |
| 1 | G (Green) | (0, 4) |
| 2 | Y (Yellow) | (4, 0) |
| 3 | B (Blue) | (4, 3) |

### State Decoding

```python
def decode_state(s):
    destination = s % 4
    s //= 4
    passenger_loc = s % 5
    s //= 5
    taxi_col = s % 5
    taxi_row = s // 5
    return taxi_row, taxi_col, passenger_loc, destination
```

---

## Action Space

**Actions**: $\mathcal{A} = \{0, 1, 2, 3, 4, 5\}$ (6 discrete actions)

| Action | Name | Effect |
|--------|------|--------|
| 0 | South | Move down |
| 1 | North | Move up |
| 2 | East | Move right |
| 3 | West | Move left |
| 4 | Pickup | Pick up passenger |
| 5 | Dropoff | Drop off passenger |

---

## Transition Dynamics

**Movement (Actions 0-3)**: Deterministic. The taxi moves in the specified direction unless:
- It would hit a wall (stays in place)
- It would go off the grid (stays in place)

**Pickup (Action 4)**:
- If taxi is at passenger location and passenger not in taxi: passenger enters taxi
- Otherwise: illegal action (no state change, penalty)

**Dropoff (Action 5)**:
- If passenger is in taxi and taxi is at destination: successful dropoff (terminal)
- Otherwise: illegal action (penalty)

---

## Reward Structure

| Condition | Reward |
|-----------|--------|
| Successful dropoff at destination | +20 |
| Each time step | -1 |
| Illegal pickup or dropoff | -10 |

**Interpretation**: The agent is penalized for time (-1 per step) and severely penalized for illegal actions (-10). The only positive reward is successful delivery (+20).

---

## MDP Formulation

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$

| Component | Value |
|-----------|-------|
| $\mathcal{S}$ | $\{0, 1, \ldots, 499\}$ |
| $\mathcal{A}$ | $\{0, 1, 2, 3, 4, 5\}$ |
| $P(s' \mid s, a)$ | Deterministic |
| $R(s, a, s')$ | +20 (deliver), -1 (step), -10 (illegal) |
| $\gamma$ | Typically 0.99 or 1.0 |

---

## Expected Optimal Performance

The optimal policy achieves:
- **Average return**: ~8-9 (depends on initial state distribution)
- **Average steps**: ~13 steps per episode

The theoretical minimum number of steps depends on the initial configuration:
- Navigate to passenger: varies
- Pick up: 1 step
- Navigate to destination: varies
- Drop off: 1 step

With the -1 per step penalty and +20 for delivery, optimal returns are typically in the range [7, 12].

---

## Running the Solver

```bash
python solve_taxi.py
```

This will:
1. Load the Taxi-v3 environment from Gymnasium
2. Extract transition dynamics from `env.P`
3. Solve using value iteration
4. Report optimal value statistics
5. Run evaluation episodes

---

## Key Insights

1. **Large state space**: 500 states is larger than FrozenLake (16) or GridWorld (16), demonstrating DP scalability.

2. **Factored state**: The state combines multiple variables (taxi position, passenger location, destination), a common pattern in real-world problems.

3. **Hierarchical structure**: The task has implicit subtasks (navigate → pickup → navigate → dropoff).

4. **Deterministic dynamics**: Unlike FrozenLake, Taxi-v3 has deterministic transitions, making planning more straightforward.

5. **Sparse rewards**: Only terminal actions give positive reward, requiring credit assignment over long sequences.

---

## References

1. Gymnasium Taxi documentation: https://gymnasium.farama.org/environments/toy_text/taxi/
2. Dietterich, T. G. (2000). Hierarchical reinforcement learning with the MAXQ value function decomposition. *JAIR*.
