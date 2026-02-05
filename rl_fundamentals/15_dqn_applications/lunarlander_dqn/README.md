# LunarLander: DQN Solution

Solving LunarLander-v3 using Double DQN - a more challenging benchmark than CartPole with larger state/action spaces and sparser rewards.

---

## Environment

```
                 *  *  *  *  *
              *              *
           *     LANDER        *
              ┌───────┐
              │   ^   │
              │  / \  │
              └───┬───┘
                 / \
              🔥     🔥   ← Thrusters

    ════════════════╪═══════════════════
                 LANDING PAD
```

### State Space (8 dimensions)

| Index | Variable | Description |
|-------|----------|-------------|
| 0 | x | Horizontal position |
| 1 | y | Vertical position |
| 2 | vx | Horizontal velocity |
| 3 | vy | Vertical velocity |
| 4 | θ | Angle |
| 5 | ω | Angular velocity |
| 6 | leg1 | Left leg contact (0 or 1) |
| 7 | leg2 | Right leg contact (0 or 1) |

### Action Space (4 actions)

| Action | Effect |
|--------|--------|
| 0 | Do nothing |
| 1 | Fire left engine (rotate right) |
| 2 | Fire main engine (thrust up) |
| 3 | Fire right engine (rotate left) |

### Reward Structure

| Event | Reward |
|-------|--------|
| Move toward pad | Positive |
| Move away from pad | Negative |
| Crash | -100 |
| Land safely | +100 to +140 |
| Leg touches ground | +10 each |
| Fire main engine | -0.3 per frame |
| Fire side engine | -0.03 per frame |

**Solved**: Average return ≥ 200 over 100 consecutive episodes.

---

## Why LunarLander is Harder

| Aspect | CartPole | LunarLander |
|--------|----------|-------------|
| State dimension | 4 | 8 |
| Action space | 2 | 4 |
| Episode length | 200-500 | 200-1000+ |
| Reward | Dense (+1/step) | Sparse (landing) |
| Optimal behavior | Single skill | Multiple skills |

### Required Skills

The agent must learn:
1. **Hover** - Control altitude
2. **Stabilize** - Keep upright
3. **Navigate** - Move toward pad
4. **Land** - Soft touchdown
5. **Fuel efficiency** - Minimize thrust

---

## Hyperparameters

Tuned for LunarLander's complexity:

| Parameter | Value | Comparison to CartPole |
|-----------|-------|------------------------|
| Hidden layers | [256, 256] | Larger (vs [128, 128]) |
| Learning rate | 0.0005 | Lower (vs 0.001) |
| Buffer size | 100,000 | Larger (vs 50,000) |
| Min buffer | 5,000 | Larger (vs 1,000) |
| Target update | 200 steps | Less frequent (vs 100) |
| ε decay | 0.9995 | Slower (vs 0.995) |

### Why These Differences?

- **Larger network**: More complex state-action mapping
- **Lower LR**: More stable with larger network
- **Larger buffer**: Longer episodes need more history
- **More warmup**: Better initial data diversity
- **Slower ε decay**: Harder exploration problem

---

## Expected Performance

### Learning Curve

```
Episode    Avg Return    Best    Epsilon
────────   ──────────    ────    ───────
   50        -180        -120     0.975
  100        -110         -50     0.951
  200         -30          80     0.905
  300          45         140     0.861
  400          95         180     0.819
  500         145         210     0.779
  600         185         245     0.741
  700         215         270     0.704

SOLVED at episode ~700!
Average return over last 100: 208.5
```

### Training Time

| Hardware | Approximate Time |
|----------|------------------|
| CPU (M1) | 15-25 minutes |
| CPU (x86) | 20-40 minutes |

---

## Algorithm Notes

### Double DQN for Stability

LunarLander benefits significantly from Double DQN:
- Longer horizons → more bootstrapping → more overestimation
- Sparse rewards → noisy value estimates
- Double DQN reduces both issues

### Importance of Exploration

The environment requires careful exploration:
- Random actions often crash immediately
- Need to discover landing is rewarded
- Slow ε decay allows thorough exploration

---

## Running the Code

### Prerequisites

```bash
pip install gymnasium[box2d]
```

Note: `box2d` extra required for LunarLander.

### Training

```bash
cd rl_fundamentals/15_dqn_applications/lunarlander_dqn
python solve_lunarlander_dqn.py
```

### Expected Output

```
======================================================================
LunarLander: Double DQN
======================================================================

----------------------------------------
Environment Info
----------------------------------------
  State: [x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact]
  Actions: 0=nothing, 1=left engine, 2=main engine, 3=right engine
  Reward: -100 to +100 (crash to perfect landing)
  Solved: avg return >= 200 over 100 episodes

======================================================================
Training Double DQN
======================================================================
--------------------------------------------------
Episode   50 | Avg Return:  -178.35 | Best:  -95.42 | Epsilon: 0.975
Episode  100 | Avg Return:  -112.28 | Best:  -45.12 | Epsilon: 0.951
...
Episode  700 | Avg Return:   215.42 | Best:  268.35 | Epsilon: 0.704

SOLVED at episode 692!
Average return over last 100 episodes: 208.73
--------------------------------------------------

Evaluating trained agent...

Final evaluation (100 episodes): 224.58 +/- 45.32
SOLVED!
```

---

## Visualization

The training produces a plot showing:
1. **Learning curve**: Return vs episode (smoothed)
2. **Return distribution**: Histogram of recent returns

A successful agent should show:
- Returns trending from ~-150 to ~200+
- Distribution centered above 200 with low variance

---

## Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| No progress | Returns stuck at -100 to -200 | Increase buffer warmup |
| Slow learning | Takes >1000 episodes | Check learning rate |
| Unstable | Returns oscillate wildly | Reduce LR, increase target update freq |
| Crashes persist | Never learns to land | Slower ε decay |

### Common Fixes

1. **More exploration**: ε_decay = 0.9998 (even slower)
2. **Larger buffer**: buffer_size = 200000
3. **More warmup**: min_buffer = 10000
4. **Stabler targets**: target_update = 500

---

## Comparison with Other Approaches

| Algorithm | Episodes to Solve | Final Return |
|-----------|-------------------|--------------|
| Random policy | Never | -150 ± 80 |
| REINFORCE | ~1500-2000 | 180 ± 60 |
| **Double DQN** | ~600-800 | 220 ± 45 |
| A2C (Week 6) | ~400-600 | 240 ± 35 |

---

## Key Insights

1. **Patience required**: LunarLander needs longer training than CartPole
2. **Hyperparameters matter**: Default DQN settings won't work well
3. **Double DQN helps significantly**: Overestimation is a real problem here
4. **Exploration is critical**: Random crashing provides poor learning signal
