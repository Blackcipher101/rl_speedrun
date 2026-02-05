# CartPole: DQN Solution

Solving CartPole-v1 using Deep Q-Networks, demonstrating DQN on a classic continuous-state control problem.

---

## Environment

```
        ┌───┐
        │   │  ← Pole (keep balanced!)
        │   │
        └───┘
    ════╪════════
      ───┴───     ← Cart (push left/right)
═══════════════════════════════════════
```

### State Space (Continuous)

| Index | Variable | Range | Description |
|-------|----------|-------|-------------|
| 0 | x | [-4.8, 4.8] | Cart position |
| 1 | ẋ | (-∞, ∞) | Cart velocity |
| 2 | θ | [-0.42, 0.42] rad | Pole angle |
| 3 | θ̇ | (-∞, ∞) | Pole angular velocity |

### Action Space

| Action | Effect |
|--------|--------|
| 0 | Push cart left |
| 1 | Push cart right |

### Reward & Termination

- **Reward**: +1 for every timestep the pole remains upright
- **Termination**:
  - Pole angle > ±12° (0.21 rad)
  - Cart position > ±2.4
  - Episode length reaches 500 (truncated)
- **Solved**: Average return ≥ 195 over 100 consecutive episodes

---

## Why DQN for CartPole?

### Continuous State Space

CartPole has 4 continuous state variables. Options:

| Approach | Pros | Cons |
|----------|------|------|
| Discretization | Simple, proven | Curse of dimensionality, info loss |
| Linear FA | Simple, stable | Limited expressiveness |
| **Neural Network (DQN)** | Handles continuous states naturally | Requires careful training |

### DQN Advantages

1. **No discretization needed** - Direct continuous input
2. **Generalization** - Similar states share learned features
3. **Scalability** - Same approach works for larger problems
4. **End-to-end learning** - Features learned automatically

---

## Hyperparameters

Optimized settings for CartPole:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden layers | [128, 128] | Enough capacity for CartPole |
| Learning rate | 0.001 | Standard for Adam |
| Batch size | 64 | Balance of stability/speed |
| Buffer size | 50,000 | ~100 episodes of data |
| Target update | 100 steps | Frequent enough for fast learning |
| ε start | 1.0 | Full exploration initially |
| ε end | 0.01 | Keep some exploration |
| ε decay | 0.995 | ~450 episodes to reach ε_min |
| γ | 0.99 | Standard discount |

---

## Expected Performance

### Learning Curve

```
Episode    Avg Length    Epsilon
────────   ──────────    ───────
   50         25          0.78
  100         55          0.61
  150        110          0.47
  200        220          0.37
  250        380          0.29
  300        485          0.22

Evaluation (100 episodes): 492 ± 18
Status: SOLVED!
```

### Typical Results

| Metric | DQN | Double DQN |
|--------|-----|------------|
| Episodes to solve | ~250 | ~200 |
| Final performance | 480 ± 30 | 490 ± 20 |
| Training stability | Good | Better |

---

## Algorithm Details

### Network Architecture

```
State: [x, ẋ, θ, θ̇]
         │
         ▼
    ┌─────────┐
    │ Dense   │ 128 units, ReLU
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Dense   │ 128 units, ReLU
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Dense   │ 2 units, Linear
    └────┬────┘
         │
         ▼
Q-values: [Q(s, left), Q(s, right)]
```

### Training Loop

```python
for episode in range(n_episodes):
    state = env.reset()

    while not done:
        # ε-greedy action
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax(Q(state))

        # Environment step
        next_state, reward, done = env.step(action)

        # Store and train
        buffer.push(state, action, reward, next_state, done)
        if buffer.ready():
            batch = buffer.sample(64)

            # Double DQN target
            best_actions = argmax(Q_online(next_states))
            targets = rewards + γ * Q_target(next_states, best_actions)

            loss = MSE(Q(states, actions), targets)
            update_weights(loss)

            # Periodic target update
            if step % 100 == 0:
                Q_target = Q_online.copy()

    epsilon *= 0.995
```

---

## Running the Code

```bash
cd rl_fundamentals/15_dqn_applications/cartpole_dqn
python solve_cartpole_dqn.py
```

### Output

```
======================================================================
CartPole: DQN and Double DQN
======================================================================

----------------------------------------
Environment Info
----------------------------------------
  State: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
  Actions: 0=push left, 1=push right
  Reward: +1 per timestep
  Solved: avg length >= 195 over 100 episodes
  Max episode length: 500

======================================================================
Training Double DQN (Single Run)
======================================================================
Episode   50 | Avg Return:   24.30 | Epsilon: 0.778
Episode  100 | Avg Return:   58.20 | Epsilon: 0.605
...
Episode  350 | Avg Return:  487.40 | Epsilon: 0.175

Final evaluation: 493.2 +/- 15.8
SOLVED!
```

---

## Comparison: DQN vs Previous Approaches

| Approach | Episodes to Solve | Final Performance |
|----------|-------------------|-------------------|
| Discretized Q-Learning | ~2000+ (if at all) | Inconsistent |
| REINFORCE | ~800-1500 | ~480 |
| **DQN** | ~300 | ~490 |
| **Double DQN** | ~250 | ~495 |

DQN advantages:
- Faster learning (data efficiency via replay)
- More stable (target networks)
- Better final performance (continuous state handling)

---

## Key Insights

1. **Experience replay is crucial** - Breaking correlation enables stable training
2. **Target network prevents oscillation** - Fixed targets = stable optimization
3. **Double DQN helps** - Reduces overestimation for faster, more stable learning
4. **ε-decay matters** - Too fast = poor exploration, too slow = slow convergence

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No improvement | ε decays too fast | Increase decay period |
| Unstable | Learning rate too high | Reduce to 0.0001 |
| Oscillating | Target update too frequent | Increase to 200+ steps |
| Slow learning | Buffer too small | Increase buffer size |
