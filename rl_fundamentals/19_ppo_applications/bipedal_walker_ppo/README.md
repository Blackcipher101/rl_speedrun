# BipedalWalker with PPO

Solving BipedalWalker-v3 using PPO with continuous (Gaussian) actions. This is significantly harder than the discrete environments.

---

## Environment

| Property | Value |
|----------|-------|
| State dim | 24 (hull angle/velocity, joint angles/speeds, lidar readings) |
| Action space | Box(4): continuous torques in [-1, 1] for hip/knee joints |
| Reward | +300 total for walking to end, -100 for falling |
| Solved threshold | Average return >= 300 over 100 episodes |

---

## Gaussian Policy

Unlike discrete environments that use a Categorical distribution, BipedalWalker requires a continuous policy:

$$\pi(a|s) = \mathcal{N}(\mu_\theta(s), \sigma^2)$$

- **Mean** ($\mu$): State-dependent, output of the actor network
- **Std** ($\sigma$): Learnable parameter (state-independent), initialized to 1.0
- **Log-prob**: Summed across the 4 independent action dimensions
- **Actions**: Clamped to [-1, 1] after sampling

---

## PPO Configuration

| Parameter | Value |
|-----------|-------|
| Hidden dim | 64 |
| Architecture | Separate actor/critic networks |
| Learning rate | 3e-4 |
| Clip range | 0.2 |
| n_steps | 2048 |
| n_epochs | 10 |
| Batch size | 64 |
| GAE lambda | 0.95 |
| Entropy coef | 0.0 |
| Max episodes | 3000 |

Note: Entropy coefficient is 0.0 because the Gaussian policy already provides natural exploration through its learned standard deviation.

---

## Running

```bash
python rl_fundamentals/19_ppo_applications/bipedal_walker_ppo/solve_bipedal_walker_ppo.py
```

This takes significantly longer than discrete environments (~10-30 minutes).

---

## Results & Challenges

BipedalWalker is one of the harder standard benchmarks. Our basic PPO implementation shows learning (from -106 to +29 peak avg return) but does not fully solve the environment at 300. SB3 typically needs 1-5M timesteps with additional features.

Key challenges encountered:
- **Shared vs separate networks**: Shared backbones cause value loss to corrupt policy features. Separate actor/critic networks are essential for continuous control.
- **Log-prob consistency**: Must compute log_prob on the clamped action in both `get_action()` and `evaluate()` to keep the PPO ratio consistent.
- **High clip fractions**: With 4D continuous actions, the probability ratio diverges quickly over multiple epochs.
- **Entropy collapse**: log_std needs clamping to prevent the policy from becoming too deterministic too fast.

### What would help solve it
- Observation normalization (running mean/std)
- Learning rate annealing
- Reward scaling
- More training timesteps (1-5M)
- Larger networks (256 hidden)
