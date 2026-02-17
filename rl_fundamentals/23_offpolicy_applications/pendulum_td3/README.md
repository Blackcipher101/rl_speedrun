# Pendulum with TD3

Solving Pendulum-v1 using Twin Delayed DDPG. Comparing TD3's stability improvements over DDPG.

---

## Environment

| Property | Value |
|----------|-------|
| State dim | 3 (cos θ, sin θ, angular velocity) |
| Action space | Box(1): torque in [-2, 2] |
| Reward | -(θ² + 0.1·θ̇² + 0.001·u²) |
| Solved threshold | Average return >= -200 over 100 episodes |

---

## TD3 Configuration

| Parameter | Value |
|-----------|-------|
| Hidden dim | 256 |
| Actor lr | 1e-4 |
| Critic lr | 1e-3 |
| Tau | 0.005 |
| Policy delay | 2 |
| Target noise | 0.2 |
| Noise clip | 0.5 |
| Exploration noise | 0.1 (Gaussian) |
| Batch size | 64 |

---

## Running

```bash
python rl_fundamentals/23_offpolicy_applications/pendulum_td3/solve_pendulum_td3.py
```
