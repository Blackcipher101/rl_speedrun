# MuZero

Learning to plan without knowing the rules. Extends AlphaZero by learning the environment model, eliminating the need for a game simulator during planning.

---

## From AlphaZero to MuZero

| Aspect | AlphaZero | MuZero |
|--------|-----------|--------|
| Game rules needed? | Yes (for MCTS simulation) | No |
| State representation | Raw board | Learned latent state |
| MCTS transitions | Use game simulator | Use dynamics network |
| Value estimation | Network on real state | Network on latent state |
| Reward prediction | Known from rules | Learned |

---

## Three Learned Functions

```
Observation ──→ h(obs) ──→ hidden_state ──→ f(h) ──→ (policy, value)
                              │
                    action ──→│
                              ▼
                          g(h, a) ──→ (next_h, reward) ──→ f(next_h) ──→ (policy, value)
                              │
                    action ──→│
                              ▼
                          g(h, a) ──→ (next_h, reward) ──→ f(next_h) ──→ ...

h = Representation:  observation → hidden_state
g = Dynamics:        (hidden_state, action) → (next_hidden, reward)
f = Prediction:      hidden_state → (policy, value)
```

---

## Training: Unrolled K-Step Imagination

```
Real game:     s₀ ──a₀──→ s₁ ──a₁──→ s₂ ──a₂──→ s₃

Imagination:   h(s₀) ──a₀──→ g(h,a₀) ──a₁──→ g(h,a₁) ──a₂──→ g(h,a₂)
                 │              │                │                │
                f(h)           f(h)             f(h)            f(h)
                 │              │                │                │
              (π₀,v₀)       (π₁,v₁)          (π₂,v₂)        (π₃,v₃)
                 │              │                │                │
Loss:         CE(π₀,π̂₀)    CE(π₁,π̂₁)       CE(π₂,π̂₂)     CE(π₃,π̂₃)
              MSE(v₀,v̂₀)   MSE(v₁,v̂₁)      MSE(v₂,v̂₂)    MSE(v₃,v̂₃)
                            MSE(r₁,r̂₁)      MSE(r₂,r̂₂)    MSE(r₃,r̂₃)
```

---

## Key Classes

| Class | Description |
|-------|-------------|
| `RepresentationNet` | h(obs) → hidden_state |
| `DynamicsNet` | g(h, a) → (next_h, reward) |
| `PredictionNet` | f(h) → (policy, value) |
| `MuZeroNetwork` | Wraps all three networks |
| `MuZeroMCTS` | MCTS operating in latent space |
| `MuZeroReplayBuffer` | Stores full game trajectories |
| `MuZeroTrainer` | Self-play + unrolled training loop |

---

## Hyperparameters (TicTacToe)

| Param | Value |
|-------|-------|
| hidden_dim | 128 |
| n_iterations | 80 |
| n_simulations | 80 |
| unroll_steps (K) | 5 |
| td_steps | 10 |
| lr | 5e-4 |
| c_puct | 1.25 |

---

## Running

```bash
python rl_fundamentals/26_muzero/muzero.py
```

Expected: ~70% win rate vs random on TicTacToe. Lower than AlphaZero (~94%) because MuZero must also learn the game dynamics from scratch.

---

## References

- Schrittwieser et al., 2020 — "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
