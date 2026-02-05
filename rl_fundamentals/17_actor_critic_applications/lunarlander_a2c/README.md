# LunarLander: A2C Solution

Solving LunarLander-v3 using Advantage Actor-Critic - demonstrating A2C on a more challenging benchmark.

---

## Environment Summary

| Aspect | Specification |
|--------|---------------|
| State dim | 8 (position, velocity, angle, leg contacts) |
| Action dim | 4 (nothing, left, main, right thrusters) |
| Reward range | -100 (crash) to +140 (perfect landing) |
| Solved threshold | Average return ≥ 200 |

---

## A2C Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden layers | [128, 128] | Larger than CartPole |
| Learning rate | 0.0007 | Lower for stability |
| γ | 0.99 | Standard |
| λ (GAE) | 0.95 | Standard |
| Update freq | 8 steps | Longer rollouts |
| Entropy coef | 0.01 | Standard |
| Value coef | 0.5 | Standard |

---

## Expected Performance

```
Episode    Avg Return    Entropy    V_loss
────────   ──────────    ───────    ──────
   50        -165         1.35       45.2
  100        -95          1.28       32.1
  200        -10          1.12       18.5
  300         85          0.95       12.3
  400        145          0.82        8.7
  500        195          0.71        5.4
  600        225          0.62        3.8

First solved at ~550 episodes
Final evaluation: 235 ± 45
```

---

## Comparison with DQN

| Metric | Double DQN | A2C |
|--------|------------|-----|
| Episodes to solve | ~700 | ~550 |
| Final return | ~220 | ~235 |
| Training stability | Good | Good |
| Sample efficiency | Better | Good |

A2C tends to be slightly faster due to:
- On-policy learning is naturally suited to this task
- GAE provides good advantage estimates
- Entropy regularization maintains exploration

---

## Running

```bash
# Prerequisites
pip install gymnasium[box2d]

# Train
cd rl_fundamentals/17_actor_critic_applications/lunarlander_a2c
python solve_lunarlander_a2c.py
```

---

## Tips

1. **Patience**: LunarLander needs ~500-800 episodes
2. **Entropy**: Watch that it doesn't drop too fast
3. **Value loss**: Should decrease steadily
4. **Learning rate**: Lower is more stable for this problem
