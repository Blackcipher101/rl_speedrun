# CliffWalking: Q-Learning vs SARSA Showdown

The CliffWalking environment is the classic demonstration of the difference between off-policy (Q-Learning) and on-policy (SARSA) temporal difference learning.

---

## Environment Description

```
. . . . . . . . . . . .
. . . . . . . . . . . .
. . . . . . . . . . . .
S C C C C C C C C C C G
```

A 4×12 grid world where:
- **S**: Start position (bottom-left, state 36)
- **G**: Goal position (bottom-right, state 47)
- **C**: Cliff cells (bottom row, states 37-46)
- **.**: Safe cells

### State Space

- **Total states**: 48 (4 rows × 12 columns)
- **State encoding**: `state = row * 12 + col`
- **Terminal states**: Goal (47), Cliff (37-46)

### Action Space

| Action | Direction |
|--------|-----------|
| 0 | Up |
| 1 | Right |
| 2 | Down |
| 3 | Left |

### Transitions

- **Deterministic**: Actions always move in intended direction
- **Boundaries**: Attempting to move off-grid stays in place
- **Cliff**: Falling off cliff returns to start (S)

### Reward Structure

| Event | Reward |
|-------|--------|
| Each step | -1 |
| Reaching goal | -1 (episode ends) |
| Falling off cliff | -100 (return to start) |

---

## The Q-Learning vs SARSA Difference

This environment perfectly illustrates why on-policy and off-policy methods learn different behaviors:

### Q-Learning (Off-Policy)

$$Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \cdot \max_a Q(S',a) - Q(S,A)]$$

- Uses `max` over next actions → learns **optimal** Q-values
- Ignores exploration behavior during updates
- **Result**: Learns the risky optimal path along the cliff edge
- *"I know the best path theoretically, even if I sometimes fall"*

### SARSA (On-Policy)

$$Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \cdot Q(S',A') - Q(S,A)]$$

- Uses actual next action A' → learns Q-values for **policy being followed**
- Accounts for exploration mistakes in updates
- **Result**: Learns a safer path away from the cliff
- *"I'll stay away from the cliff because I know I sometimes explore randomly"*

---

## Expected Policies

### Q-Learning: The Optimal Path

```
→ → → → → → → → → → → ↓
. . . . . . . . . . . ↓
. . . . . . . . . . . ↓
S C C C C C C C C C C G
```

- Takes the shortest path (12 steps)
- Walks right along the cliff edge
- Optimal if executed perfectly (return = -13)
- **During training**: Often falls due to ε-greedy exploration

### SARSA: The Safe Path

```
→ → → → → → → → → → → ↓
↑ . . . . . . . . . . ↓
↑ . . . . . . . . . . ↓
S C C C C C C C C C C G
```

- Takes a detour away from the cliff
- Longer path but avoids cliff risk
- Suboptimal but safer (return ≈ -17)
- **During training**: Rarely falls, more stable returns

---

## Training Dynamics

### Learning Curves

| Metric | Q-Learning | SARSA |
|--------|------------|-------|
| **Early training** | Highly variable (cliff falls) | More stable |
| **Average return** | Lower during training | Higher during training |
| **Final policy** | Optimal (greedy) | Safe (accounts for ε) |
| **Variance** | High | Low |

### Why the Difference?

**Q-Learning** during ε-greedy exploration:
- Learns Q-values assuming greedy execution
- But actually explores ε-randomly
- Near the cliff, random action → fall → -100 reward
- The learned Q-values don't reflect this risk

**SARSA** during ε-greedy exploration:
- Learns Q-values for the actual ε-greedy policy
- Updates incorporate the risk of random actions
- Near the cliff, Q-values are lowered by exploration failures
- Naturally learns to avoid the risky region

---

## Mathematical Insight

Let's consider state near the cliff with action "move right along edge":

**Q-Learning thinks:**
> "If I move right optimally from here, my value is high"

**SARSA thinks:**
> "If I move right with ε probability of random action, I might fall. My value is lower."

The TD target reveals the difference:

| Method | TD Target |
|--------|-----------|
| Q-Learning | $R + \gamma \cdot \max_a Q(S',a)$ |
| SARSA | $R + \gamma \cdot Q(S', A')$ where A' ~ ε-greedy |

SARSA's target averages over what actually happens, including failures.

---

## Running the Experiment

```bash
cd rl_fundamentals/07_td_applications/cliffwalking
python solve_cliffwalking.py
```

### Expected Output

```
Training Q-Learning... 500 episodes
Training SARSA... 500 episodes

Results:
  Q-Learning - Mean return: -45.2 ± 62.1 (high variance!)
  SARSA      - Mean return: -17.3 ± 3.2  (stable)

Policies saved to: cliffwalking_comparison.png
```

---

## Key Takeaways

1. **Off-policy ≠ always better**: Q-Learning finds optimal but risky policies
2. **On-policy is exploration-aware**: SARSA accounts for actual behavior
3. **Context matters**: In high-risk environments, SARSA might be preferred
4. **During evaluation**: With ε=0, both policies can perform optimally

---

## References

- Sutton & Barto, Chapter 6.5: "Example: Cliff Walking"
- The CliffWalking environment in Gymnasium: `gym.make("CliffWalking-v0")`
