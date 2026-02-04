# CartPole: REINFORCE Policy Gradient

This demonstrates the REINFORCE algorithm on CartPole - a classic benchmark for policy gradient methods. Unlike our discretized Q-Learning approach, REINFORCE works directly with continuous states using a neural network policy.

---

## Environment Recap

```
        ┌───┐
        │   │  ← Pole (keep upright!)
        │   │
        └───┘
    ════╪════════
      ───┴───     ← Cart (push left/right)
═══════════════════════════════════════
```

### State Space (Continuous)

| Index | Variable | Description |
|-------|----------|-------------|
| 0 | x | Cart position |
| 1 | ẋ | Cart velocity |
| 2 | θ | Pole angle |
| 3 | θ̇ | Pole angular velocity |

### Action Space

| Action | Effect |
|--------|--------|
| 0 | Push left |
| 1 | Push right |

### Reward

+1 for every timestep the pole remains upright (max 500 steps).

**Solved**: Average reward ≥ 195 over 100 consecutive episodes.

---

## Why REINFORCE for CartPole?

### Advantages over Q-Learning

| Aspect | Q-Learning (Discretized) | REINFORCE |
|--------|--------------------------|-----------|
| State handling | Discretize → lose info | Direct continuous input |
| Policy | Deterministic (ε-greedy) | Naturally stochastic |
| Function approx | Q-table | Neural network |
| Scalability | Poor (curse of dimensionality) | Good |

### The Trade-off

- REINFORCE has **higher variance** (Monte Carlo returns)
- But handles **continuous states naturally**
- And learns **stochastic policies** (better exploration)

---

## Algorithm: REINFORCE with Baseline

### Policy Network

```
Input: [x, ẋ, θ, θ̇] (4 dims)
    ↓
Hidden Layer (128 units, ReLU)
    ↓
Hidden Layer (128 units, ReLU)
    ↓
Output Layer (2 units)
    ↓
Softmax → π(left|s), π(right|s)
```

### Value Network (Baseline)

```
Input: [x, ẋ, θ, θ̇] (4 dims)
    ↓
Hidden Layer (128 units, ReLU)
    ↓
Output Layer (1 unit) → V(s)
```

### Update Rules

**Policy update** (gradient ascent on expected return):
$$\theta \leftarrow \theta + \alpha \sum_t (G_t - V(s_t)) \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**Value update** (minimize MSE):
$$w \leftarrow w - \alpha_v \sum_t (G_t - V_w(s_t)) \nabla_w V_w(s_t)$$

---

## Implementation Details

### Return Computation

```python
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns
```

### Return Normalization

Critical for stable training:

```python
returns = torch.tensor(returns)
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

### Policy Loss

```python
policy_loss = []
for log_prob, G in zip(saved_log_probs, returns):
    policy_loss.append(-log_prob * G)  # Negative for gradient ascent
loss = torch.stack(policy_loss).sum()
```

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate (policy) | 1e-3 | Adam optimizer |
| Learning rate (value) | 1e-3 | Adam optimizer |
| γ (discount) | 0.99 | Standard |
| Hidden units | 128 | 2 hidden layers |
| Episodes | 1000-2000 | Until solved |

---

## Expected Learning Curve

```
Episode      Average Length
---------    --------------
0-100        20-40 (random wobbling)
100-300      50-100 (learning balance)
300-500      100-200 (getting good)
500-800      200-400 (nearly solved)
800+         400-500 (solved!)
```

---

## Key Insights

### Why Baseline Helps

Without baseline:
- All positive rewards → all actions reinforced
- High variance in updates
- Slow, unstable learning

With baseline V(s):
- Only actions **better than average** reinforced
- Lower variance
- Faster, more stable convergence

### Entropy Regularization (Optional)

Add entropy bonus to prevent premature convergence:

$$J(\theta) = \mathbb{E}[R] + \beta H(\pi)$$

This encourages exploration by penalizing deterministic policies.

---

## Running the Solver

```bash
cd rl_fundamentals/10_mc_pg_applications/cartpole_reinforce
python solve_cartpole_reinforce.py
```

### Expected Output

```
Training REINFORCE on CartPole...

Episode 100:  Avg length = 45.2
Episode 200:  Avg length = 89.7
Episode 300:  Avg length = 156.3
Episode 400:  Avg length = 234.8
Episode 500:  Avg length = 312.4
...
Episode 800:  Avg length = 487.2

SOLVED at episode 823!
Average length over last 100 episodes: 198.5

Visualization saved to: cartpole_reinforce.png
```

---

## Comparison: Discretized Q-Learning vs REINFORCE

| Metric | Q-Learning | REINFORCE |
|--------|------------|-----------|
| Episodes to solve | ~2000+ (if at all) | ~500-1000 |
| Final performance | Inconsistent | Reliable |
| Memory | O(bins^4) | O(network params) |
| Generalization | Poor | Good |

REINFORCE wins on CartPole because:
1. No information loss from discretization
2. Neural network generalizes across similar states
3. Stochastic policy explores naturally

---

## References

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
- Sutton & Barto, Chapter 13: Policy Gradient Methods
