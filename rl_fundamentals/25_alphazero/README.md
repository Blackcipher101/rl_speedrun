# AlphaZero

Self-play reinforcement learning combining MCTS with a policy-value neural network. Replaces random rollouts with learned evaluation.

---

## From MCTS to AlphaZero

```
Pure MCTS                          AlphaZero MCTS
├── Selection: UCB1                ├── Selection: PUCT (uses network prior)
├── Expansion: add children        ├── Expansion: add children with P(a|s)
├── Simulation: random rollout ──→ ├── Evaluation: network predicts v(s)
└── Backpropagation               └── Backpropagation
```

### PUCT vs UCB1

```
UCB1:  Q(s,a) + c · √(ln N(s) / N(s,a))
PUCT:  Q(s,a) + c · P(a|s) · √N(s) / (1 + N(s,a))
                     └─────┘
                   network prior
```

The network prior P(a|s) guides search toward promising moves.

---

## Self-Play Training Loop

```
┌─────────────────────────────────────────────┐
│                                             │
│  ┌──────────┐    ┌───────────┐    ┌──────┐ │
│  │ Self-Play │──→│   Train   │──→│ Eval  │─┘
│  │ (MCTS +  │    │  Network  │    │ New   │
│  │ Network) │    │  on Data  │    │vs Old │
│  └──────────┘    └───────────┘    └──────┘
│       │
│  Collect (state, MCTS_policy, outcome)
│       │
│  Loss = MSE(v, outcome) + CE(π, MCTS_policy)
└─────────────────────────────────────────────┘
```

---

## Network Architecture

```
Input: (3, H, W)      ← current player / opponent / color channels
    │
    ▼
Conv 3×3 + BN + ReLU  ← input projection
    │
    ▼
Residual Blocks × N   ← shared trunk
    │
    ├──→ Policy Head: Conv 1×1 → FC → softmax → P(a|s)
    │
    └──→ Value Head:  Conv 1×1 → FC → tanh → v(s) ∈ [-1, 1]
```

---

## Key Classes

| Class | Description |
|-------|-------------|
| `PolicyValueNet` | Shared trunk → policy head + value head |
| `ResidualBlock` | Conv → BN → ReLU → Conv → BN + skip |
| `AlphaZeroMCTS` | MCTS with PUCT selection and network evaluation |
| `SelfPlayBuffer` | Stores (state, policy, outcome) training data |
| `AlphaZeroTrainer` | Full self-play → train → evaluate loop |

---

## Hyperparameters

| Param | TicTacToe | ConnectFour |
|-------|-----------|-------------|
| n_iterations | 50-100 | 200 |
| n_self_play_games | 20-25 | 50 |
| n_simulations | 50-100 | 200 |
| n_channels | 64 | 128 |
| n_res_blocks | 2 | 4 |
| lr | 1e-3 | 1e-3 |
| c_puct | 1.0 | 1.0 |

---

## Running

```bash
python rl_fundamentals/25_alphazero/alphazero.py
```

Expected: ~94% win rate vs random on TicTacToe after 50 iterations (~2 min).

---

## References

- Silver et al., 2017 — "Mastering the game of Go without human knowledge"
- Silver et al., 2018 — "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"
