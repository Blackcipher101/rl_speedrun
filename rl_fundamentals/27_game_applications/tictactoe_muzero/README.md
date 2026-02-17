# TicTacToe with MuZero

MuZero learns to play TicTacToe without knowing the game rules. It learns its own dynamics model and plans in a learned latent space.

## Running

```bash
python rl_fundamentals/27_game_applications/tictactoe_muzero/solve_tictactoe_muzero.py
```

Expected: ~70% win rate vs random. Lower than AlphaZero (~94%) because MuZero must also learn game dynamics. Runtime: ~5-10 min.
