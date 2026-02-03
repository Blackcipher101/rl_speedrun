# CartPole: Discretized Q-Learning

CartPole demonstrates how to apply tabular Q-Learning to continuous state spaces through **discretization** - converting infinite state spaces into finite bins.

---

## Environment Description

```
        ┌───┐
        │   │  ← Pole
        │   │
        └───┘
    ════╪════════
      ───┴───     ← Cart
═══════════════════════════
```

A pole is attached to a cart moving on a frictionless track. The goal is to prevent the pole from falling over by moving the cart left or right.

### State Space (Continuous)

| Index | Variable | Description | Range |
|-------|----------|-------------|-------|
| 0 | `x` | Cart position | [-4.8, 4.8] |
| 1 | `ẋ` | Cart velocity | [-∞, ∞] |
| 2 | `θ` | Pole angle (radians) | [-0.418, 0.418] |
| 3 | `θ̇` | Pole angular velocity | [-∞, ∞] |

**Problem**: Infinite states → can't use Q-table directly!

### Action Space (Discrete)

| Action | Effect |
|--------|--------|
| 0 | Push cart left |
| 1 | Push cart right |

### Reward

- **+1** for every timestep the pole remains upright
- Episode ends when pole angle > ±12° or cart position > ±2.4
- **Solved**: Average reward ≥ 195 over 100 consecutive episodes

---

## Discretization Strategy

### The Core Idea

Convert continuous states to discrete bins:

```
Continuous: θ = 0.05 radians
                 ↓
Bins:       [-0.2, -0.1, 0, 0.1, 0.2]
                          ↑
Discrete:   bin_index = 2
```

### Bin Design

We create bins for each state variable, focusing on the **relevant ranges**:

| Variable | Bins | Typical Ranges |
|----------|------|----------------|
| Cart position | 6 bins | [-2.4, 2.4] |
| Cart velocity | 6 bins | [-3.0, 3.0] |
| Pole angle | 12 bins | [-0.25, 0.25] rad |
| Angular velocity | 12 bins | [-3.0, 3.0] |

### Total Discrete States

$$|S_{discrete}| = 6 \times 6 \times 12 \times 12 = 5,184 \text{ states}$$

### Discretization Function

```python
def discretize(observation, bins):
    """
    Convert continuous observation to discrete state index.

    Args:
        observation: [x, x_dot, theta, theta_dot]
        bins: List of bin edges for each dimension

    Returns:
        state_index: Integer index into Q-table
    """
    indices = []
    for i, (obs, bin_edges) in enumerate(zip(observation, bins)):
        # np.digitize returns bin index
        idx = np.digitize(obs, bin_edges) - 1
        idx = np.clip(idx, 0, len(bin_edges) - 2)
        indices.append(idx)

    # Convert multi-dimensional index to flat index
    state = 0
    multiplier = 1
    for idx, bin_edges in zip(reversed(indices), reversed(bins)):
        state += idx * multiplier
        multiplier *= len(bin_edges) - 1

    return state
```

---

## Why Discretization Works for CartPole

### Key Insight

CartPole has **smooth dynamics** - nearby states have similar optimal actions:

- If pole is tilting right → push right
- If pole is tilting left → push left

Discretization preserves this structure while enabling tabular methods.

### Trade-offs

| Finer Bins | Coarser Bins |
|------------|--------------|
| More precise | Faster learning |
| Larger Q-table | Smaller Q-table |
| Needs more samples | Generalizes more |
| Slow convergence | May miss details |

---

## Q-Learning with Discretization

### Algorithm

```
Initialize:
    - Create bin edges for each state dimension
    - Initialize Q[discrete_state, action] = 0

For each episode:
    obs = env.reset()
    state = discretize(obs)

    While not done:
        action = ε-greedy(Q[state])
        next_obs, reward, done = env.step(action)
        next_state = discretize(next_obs)

        # Standard Q-Learning update
        Q[state, action] += α * (reward + γ * max(Q[next_state]) - Q[state, action])

        state = next_state
```

### Hyperparameters for CartPole

| Parameter | Value | Notes |
|-----------|-------|-------|
| α (learning rate) | 0.1 | Standard choice |
| γ (discount) | 0.99 | Value future rewards |
| ε (exploration) | 1.0 → 0.01 | Decay over training |
| ε_decay | 0.995 | Per episode |
| n_episodes | 2000 | Until convergence |

---

## Expected Results

### Learning Curve

```
Episode      Average Length
---------    --------------
0-100        20-30 steps (random)
100-500      50-100 steps (learning)
500-1000     150-200 steps (improving)
1000-2000    200+ steps (solved!)
```

### Success Criterion

The environment is "solved" when:
- Average episode length ≥ 195 over 100 consecutive episodes
- Maximum episode length is 500 (timeout)

---

## Limitations of Discretization

### When It Works

✅ Low-dimensional state spaces (< 10 dimensions)
✅ Smooth dynamics
✅ Clear relevant ranges

### When It Fails

❌ High-dimensional spaces (curse of dimensionality)
❌ States with vastly different scales
❌ Fine-grained control needed

### Bin Count Explosion

For a d-dimensional space with b bins per dimension:

$$|S| = b^d$$

| Dimensions | Bins/Dim | Total States |
|------------|----------|--------------|
| 4 | 10 | 10,000 |
| 4 | 20 | 160,000 |
| 10 | 10 | 10 billion! |

**This is why we need function approximation for complex problems.**

---

## Running the Experiment

```bash
cd rl_fundamentals/07_td_applications/cartpole
python solve_cartpole.py
```

### Expected Output

```
Training Q-Learning on CartPole with discretized states...
  5,184 discrete states (6 × 6 × 12 × 12)

Episode 500:  Avg length = 98.4
Episode 1000: Avg length = 187.2
Episode 1500: Avg length = 201.8
Episode 2000: Avg length = 234.5

Solved! Average length >= 195 achieved.
Visualization saved to: cartpole_learning.png
```

---

## Key Takeaways

1. **Discretization bridges continuous and tabular**: Enables Q-tables on continuous spaces
2. **Bin design matters**: Focus bins on the relevant ranges
3. **Trade-off exists**: More bins = more precision but slower learning
4. **Scalability limit**: Only works for low-dimensional problems
5. **Foundation for function approximation**: Understanding discretization helps understand why we need neural networks for complex problems

---

## References

- Gymnasium CartPole-v1: https://gymnasium.farama.org/environments/classic_control/cart_pole/
- Sutton & Barto, Chapter 9: Function Approximation
