# Pendulum with DDPG

Solving Pendulum-v1 using Deep Deterministic Policy Gradient. The simplest continuous control benchmark.

---

## Environment

| Property | Value |
|----------|-------|
| State dim | 3 (cos θ, sin θ, angular velocity) |
| Action space | Box(1): torque in [-2, 2] |
| Reward | -(θ² + 0.1·θ̇² + 0.001·u²) |
| Solved threshold | Average return >= -200 over 100 episodes |

---

## DDPG Configuration

| Parameter | Value |
|-----------|-------|
| Hidden dim | 256 |
| Actor lr | 1e-4 |
| Critic lr | 1e-3 |
| Tau | 0.005 |
| Batch size | 64 |
| Buffer size | 100,000 |
| Noise | OU (sigma=0.2) |
| Warmup | 1,000 steps |

---

## Running

```bash
python rl_fundamentals/23_offpolicy_applications/pendulum_ddpg/solve_pendulum_ddpg.py
```
