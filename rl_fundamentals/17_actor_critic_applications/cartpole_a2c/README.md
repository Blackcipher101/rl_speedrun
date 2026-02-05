# CartPole: A2C Solution

Solving CartPole-v1 using Advantage Actor-Critic, demonstrating how combining policy gradients with value function learning improves over pure REINFORCE.

---

## A2C vs REINFORCE on CartPole

### Key Differences

| Aspect | REINFORCE | A2C |
|--------|-----------|-----|
| Baseline | None (raw returns) | Learned V(s) |
| Updates | End of episode | Every n steps |
| Variance | High | Lower (via GAE) |
| Typical convergence | 800-1500 episodes | 200-400 episodes |

### Why A2C is Faster

1. **Learned baseline**: V(s) reduces variance more than constant baseline
2. **Frequent updates**: Don't wait for episode end
3. **GAE**: Tunable bias-variance via λ

---

## Configuration

### Network Architecture

```
State [4] ─────┐
               │
         ┌─────▼─────┐
         │  Shared   │
         │  64 → 64  │
         │  (ReLU)   │
         └─────┬─────┘
               │
       ┌───────┴───────┐
       ▼               ▼
   ┌───────┐       ┌───────┐
   │ Actor │       │Critic │
   │  → 2  │       │  → 1  │
   └───┬───┘       └───┬───┘
       │               │
       ▼               ▼
    π(a|s)          V(s)
```

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Hidden dims | [64, 64] | Shared feature layers |
| Learning rate | 0.001 | Adam optimizer |
| γ | 0.99 | Discount factor |
| λ (GAE) | 0.95 | Advantage estimation |
| Value coef | 0.5 | Critic loss weight |
| Entropy coef | 0.01 | Exploration bonus |
| Update freq | 5 steps | N-step returns |
| Max grad norm | 0.5 | Gradient clipping |

---

## Expected Performance

### Learning Curve

```
Episode    Avg Length    Entropy    V_loss
────────   ──────────    ───────    ──────
   50         28          0.68       2.34
  100         65          0.61       1.86
  150        125          0.54       1.24
  200        210          0.46       0.89
  250        365          0.39       0.53
  300        480          0.31       0.24

Evaluation (100 episodes): 487 ± 21
Status: SOLVED!
```

### Comparison with Other Methods

| Method | Episodes to Solve | Final Performance |
|--------|-------------------|-------------------|
| Q-Learning (discretized) | ~2000+ | Inconsistent |
| REINFORCE | ~1000 | ~480 |
| DQN | ~300 | ~490 |
| **A2C** | ~250 | ~485 |

---

## Training Dynamics

### Entropy Should Decrease

```
Entropy
  ^
  │  ╭────────────
  │ ╱              ╲
  │╱                 ╲──────────────
  └──────────────────────────────────→
                  Episodes

Start: ~0.69 (uniform over 2 actions = log(2))
End: ~0.2-0.4 (confident but not deterministic)
```

### Value Loss Should Decrease

```
Value Loss
  ^
  │╲
  │ ╲
  │  ╲
  │   ╲────────
  │          ──────────────────
  └──────────────────────────────────→
                  Episodes

Indicates critic is learning accurate value estimates
```

---

## Key Components

### 1. Advantage Estimation (GAE)

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    gae = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1-dones[t]) - values[t]
        gae = delta + gamma * lam * gae * (1-dones[t])
        advantages[t] = gae

    return advantages
```

### 2. Combined Loss

```python
# Policy gradient loss (maximize advantage-weighted log prob)
policy_loss = -mean(log_prob * advantage)

# Value function loss (minimize MSE)
value_loss = mean((V(s) - returns)²)

# Entropy bonus (encourage exploration)
entropy_bonus = mean(entropy(π(·|s)))

# Total loss
total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
```

### 3. N-Step Updates

```python
for step in range(n_steps):
    action = sample_action(state)
    next_state, reward, done = env.step(action)
    buffer.store(state, action, reward, value, done)
    state = next_state

# Compute advantages and update
bootstrap = V(state) if not done else 0
advantages = GAE(buffer, bootstrap)
update_networks(buffer, advantages)
```

---

## Running the Code

```bash
cd rl_fundamentals/17_actor_critic_applications/cartpole_a2c
python solve_cartpole_a2c.py
```

### Expected Output

```
======================================================================
CartPole: Advantage Actor-Critic (A2C)
======================================================================

----------------------------------------
Environment Info
----------------------------------------
  State: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
  Actions: 0=push left, 1=push right
  Reward: +1 per timestep
  Solved: avg length >= 195 over 100 episodes

======================================================================
Training A2C
======================================================================
--------------------------------------------------
Episode   50 | Avg Return:   28.40 | Entropy: 0.682 | V_loss: 2.341
Episode  100 | Avg Return:   68.20 | Entropy: 0.608 | V_loss: 1.856
...
Episode  350 | Avg Return:  485.40 | Entropy: 0.312 | V_loss: 0.198
--------------------------------------------------

Final evaluation (100 episodes): 488.5 +/- 19.8
SOLVED!
First solved at episode 285
```

---

## Hyperparameter Effects

### GAE Lambda

| λ | Bias | Variance | Effect |
|---|------|----------|--------|
| 0.0 | High | Low | Fast but inaccurate |
| 0.5 | Medium | Medium | Balanced |
| 0.95 | Low | Medium | Best for most problems |
| 1.0 | None | High | Pure Monte Carlo |

### Entropy Coefficient

| β | Exploration | Convergence |
|---|-------------|-------------|
| 0.001 | Too little | Fast but local optima |
| 0.01 | Balanced | Good default |
| 0.1 | Too much | Slow, may not converge |

### Update Frequency

| n_steps | Bias | Variance | Speed |
|---------|------|----------|-------|
| 1 | High | Low | Fast updates |
| 5 | Medium | Medium | Good balance |
| 20 | Low | High | Slower, more stable |
| Episode | None | Highest | Like REINFORCE |

---

## Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| No learning | Returns flat | Check learning rate |
| Oscillating | Returns unstable | Reduce LR, increase λ |
| Too slow | Takes many episodes | Increase update freq |
| Premature convergence | Gets stuck | Increase entropy coef |

---

## Key Insights

1. **Actor-Critic beats REINFORCE**: Learned baseline is better than none
2. **GAE is powerful**: λ=0.95 works well across many problems
3. **Entropy prevents collapse**: Keep some exploration even late in training
4. **Shared features help**: Same backbone for actor and critic
5. **Frequent updates speed learning**: Don't wait for episode end
