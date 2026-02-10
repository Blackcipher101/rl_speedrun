# LunarLander with PPO

Solving LunarLander-v3 using Proximal Policy Optimization with discrete actions.

---

## Environment

| Property | Value |
|----------|-------|
| State dim | 8 (x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact) |
| Action space | Discrete(4): nothing, left engine, main engine, right engine |
| Reward range | -100 (crash) to +100 (perfect landing) |
| Solved threshold | Average return >= 200 over 100 episodes |

---

## PPO Configuration

| Parameter | Value |
|-----------|-------|
| Hidden dim | 128 |
| Learning rate | 5e-4 |
| Clip range | 0.2 |
| n_steps | 512 |
| n_epochs | 10 |
| Batch size | 64 |
| GAE lambda | 0.95 |
| Entropy coef | 0.01 |
| Max episodes | 3000 |

Solved at episode ~2331 with evaluation score 201.8 +/- 98.9.

---

## Running

```bash
python rl_fundamentals/19_ppo_applications/lunarlander_ppo/solve_lunarlander_ppo.py
```

---

## PPO vs A2C on LunarLander

PPO's clipped objective provides more stable learning than A2C:
- A2C can suffer from large policy updates that crash performance
- PPO's clipping keeps updates conservative and steady
- Multi-epoch training makes PPO more sample-efficient
