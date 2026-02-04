# Blackjack: Monte Carlo Methods

Blackjack is the classic environment for demonstrating Monte Carlo methods. It's an episodic task where the agent must make decisions with incomplete information.

---

## Environment Description

### The Game

Blackjack (also known as 21) is a card game where:
- Goal: Get cards summing as close to 21 as possible without going over
- Face cards (J, Q, K) = 10 points
- Ace = 1 or 11 points (player's choice)
- Number cards = face value

### Gymnasium Environment

```python
import gymnasium as gym
env = gym.make("Blackjack-v1", natural=False, sab=False)
```

### State Space

The state is a tuple of 3 values:

| Component | Range | Description |
|-----------|-------|-------------|
| Player sum | 4-21 | Current sum of player's cards |
| Dealer showing | 1-10 | Dealer's face-up card (Ace=1) |
| Usable ace | 0/1 | Does player have usable ace? |

**Total states**: ~200 (18 × 10 × 2)

### Action Space

| Action | Meaning |
|--------|---------|
| 0 | **Stick** - Stop taking cards |
| 1 | **Hit** - Take another card |

### Rewards

| Outcome | Reward |
|---------|--------|
| Win | +1 |
| Lose | -1 |
| Draw | 0 |

### Episode Dynamics

1. Player and dealer each get 2 cards (one dealer card face-down)
2. Player chooses to hit or stick repeatedly
3. If player busts (>21): immediate loss
4. After player sticks: dealer reveals and plays (hits until ≥17)
5. Compare final sums to determine winner

---

## Why Blackjack for Monte Carlo?

### Perfect Fit for MC

1. **Episodic**: Clear episode termination (win/lose/draw)
2. **No model needed**: Don't know the deck composition
3. **Stochastic**: Random cards make exact computation impossible
4. **Simple enough**: Small state space for visualization

### Why Not TD?

- MC is natural here because:
  - Returns are only known at episode end
  - Each episode is independent
  - No bootstrapping makes sense (no intermediate rewards)

---

## Optimal Strategy (Simplified)

Basic strategy depends on:
1. Your hand total
2. Dealer's showing card
3. Whether you have a usable ace

### Without Usable Ace

```
Player Sum | Dealer Showing
           | 2  3  4  5  6  7  8  9  10  A
-----------+-------------------------------
    21     | S  S  S  S  S  S  S  S  S   S
    20     | S  S  S  S  S  S  S  S  S   S
    19     | S  S  S  S  S  S  S  S  S   S
    18     | S  S  S  S  S  S  S  S  S   S
    17     | S  S  S  S  S  S  S  S  S   S
    16     | S  S  S  S  S  H  H  H  H   H
    15     | S  S  S  S  S  H  H  H  H   H
    14     | S  S  S  S  S  H  H  H  H   H
    13     | S  S  S  S  S  H  H  H  H   H
    12     | H  H  S  S  S  H  H  H  H   H
   ≤11     | H  H  H  H  H  H  H  H  H   H

S = Stick, H = Hit
```

### With Usable Ace

```
Player Sum | Dealer Showing
           | 2  3  4  5  6  7  8  9  10  A
-----------+-------------------------------
    21     | S  S  S  S  S  S  S  S  S   S
    20     | S  S  S  S  S  S  S  S  S   S
    19     | S  S  S  S  S  S  S  S  S   S
    18     | S  S  S  S  S  S  S  H  H   H
   ≤17     | H  H  H  H  H  H  H  H  H   H
```

---

## MC Learning in Blackjack

### State Representation

Convert tuple to index for Q-table:
```python
def state_to_index(player_sum, dealer_card, usable_ace):
    # player_sum: 4-21 (18 values, index 0-17)
    # dealer_card: 1-10 (10 values, index 0-9)
    # usable_ace: 0 or 1
    return (player_sum - 4) * 20 + (dealer_card - 1) * 2 + usable_ace
```

### Learning Dynamics

1. **Early episodes**: Random policy, learns that busting is bad
2. **Mid training**: Starts to learn when to stick vs hit
3. **Convergence**: Approximates optimal basic strategy

### Expected Performance

- Random policy: ~-30% (lose more than win)
- Optimal policy: ~-5% (house still has edge)

The house edge exists because:
- Dealer plays last (player can bust first)
- Dealer must hit on 16, stick on 17

---

## Visualizations

### Value Function Heatmaps

Two heatmaps showing V(s) for:
1. States without usable ace
2. States with usable ace

X-axis: Dealer showing card (1-10)
Y-axis: Player sum (12-21)

### Policy Visualization

Two grids showing optimal action:
- Green: Stick
- Red: Hit

---

## Running the Solver

```bash
cd rl_fundamentals/10_mc_pg_applications/blackjack
python solve_blackjack.py
```

### Expected Output

```
Training MC Control on Blackjack...
Episodes: 500,000

Evaluation (10,000 episodes):
  Win rate: 42.3%
  Lose rate: 48.1%
  Draw rate: 9.6%
  Expected return: -0.058

Visualizations saved to: blackjack_solution.png
```

---

## Key Insights

1. **MC is well-suited** for games with delayed rewards
2. **Large sample sizes** needed due to high variance
3. **Visualization helps** understand learned policies
4. **House always wins** (slightly) - can't beat the math!

---

## References

- Sutton & Barto, Example 5.1: Blackjack
- Gymnasium Blackjack-v1 documentation
