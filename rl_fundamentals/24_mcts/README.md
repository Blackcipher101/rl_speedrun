# Monte Carlo Tree Search (MCTS)

Pure MCTS with UCB1 selection — no neural networks. Uses random rollouts for value estimation.

---

## Algorithm

MCTS builds a search tree incrementally through four repeated phases:

```
         Selection          Expansion         Simulation       Backpropagation
            │                  │                  │                  │
    ┌───────▼───────┐  ┌──────▼──────┐  ┌───────▼───────┐  ┌──────▼──────┐
    │  Walk tree    │  │  Add child  │  │ Random rollout│  │ Update N, Q │
    │  using UCB1   │→ │  nodes for  │→ │ to terminal   │→ │ back to     │
    │  to leaf      │  │  legal moves│  │ state         │  │ root        │
    └───────────────┘  └─────────────┘  └───────────────┘  └─────────────┘
```

### UCB1 Formula

```
UCB1(s, a) = Q(s,a) + c · √(ln N(s) / N(s,a))
             ├─────┘   └──────────────────────┘
           exploit            explore
```

- `Q(s,a)` = average value (win rate) of action a from state s
- `N(s)` = visit count of parent state
- `N(s,a)` = visit count of child after action a
- `c` = exploration constant (√2 ≈ 1.41)

---

## Key Classes

| Class | Description |
|-------|-------------|
| `Game` | Abstract base class for two-player games |
| `TicTacToe` | 3×3 board, actions 0-8, players +1/-1 |
| `MCTSNode` | Tree node with visit count N and value sum W |
| `MCTS` | Search engine: select → expand → simulate → backprop |

---

## Running

```bash
python rl_fundamentals/24_mcts/mcts.py
```

Expected: MCTS achieves ~97% win rate vs random player on TicTacToe.

---

## References

- Kocsis & Szepesvári, 2006 — "Bandit based Monte-Carlo Planning"
- Coulom, 2006 — "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search"
