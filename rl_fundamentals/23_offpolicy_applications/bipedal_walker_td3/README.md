# BipedalWalker with TD3

Teaching a robot to walk using Twin Delayed DDPG. This is one of the harder standard benchmarks.

---

## Environment

| Property | Value |
|----------|-------|
| State dim | 24 (hull angle/velocity, joint angles/speeds, lidar readings) |
| Action space | Box(4): continuous torques in [-1, 1] for hip/knee joints |
| Reward | +300 total for walking to end, -100 for falling |
| Solved threshold | Average return >= 300 over 100 episodes |

---

## TD3 Configuration

| Parameter | Value |
|-----------|-------|
| Hidden dim | 256 |
| Actor lr | 3e-4 |
| Critic lr | 3e-4 |
| Tau | 0.005 |
| Policy delay | 2 |
| Target noise | 0.2 |
| Noise clip | 0.5 |
| Exploration noise | 0.1 |
| Batch size | 256 |
| Buffer size | 1,000,000 |
| Warmup | 10,000 steps |
| Max episodes | 1000 |

---

## Running

```bash
python rl_fundamentals/23_offpolicy_applications/bipedal_walker_td3/solve_bipedal_walker_td3.py
```

This takes significantly longer than Pendulum (~30-60 minutes).

---

## Results

| Metric | Value |
|--------|-------|
| Final eval (100 eps) | **291.4 +/- 0.8** |
| Best training avg | 288.9 (ep 1000) |
| Breakthrough episode | ~500 (returns go positive) |
| Peak training avg | 275.5 (ep 850) |

The agent learns a very consistent walking gait (std 0.8) but falls just short of the strict 300 threshold. BipedalWalker-v3 is notoriously hard — the random terrain variants cause occasional stumbles that lower the average. Longer training or hyperparameter tuning could push past 300.
