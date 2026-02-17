"""
Solve TicTacToe with pure MCTS.

Demonstrates that MCTS achieves near-perfect play on TicTacToe
with sufficient simulations, using only random rollouts.

Author: RL Speedrun
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '24_mcts'))
from mcts import TicTacToe, MCTS, mcts_player, random_player, play_tournament

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("TicTacToe: Pure MCTS Solver")
    print("=" * 60)

    game = TicTacToe()
    np.random.seed(42)

    # Sweep simulation counts
    print("\n" + "-" * 40)
    print("Win Rate vs Simulation Count")
    print("-" * 40)
    sim_counts = [10, 25, 50, 100, 200, 500, 1000]
    win_rates_p1 = []
    win_rates_p2 = []
    rand_fn = random_player(game)

    for n_sims in sim_counts:
        fn = mcts_player(game, n_simulations=n_sims)
        # MCTS as player 1
        w1, l1, d1 = play_tournament(game, fn, rand_fn, n_games=50)
        wr1 = (w1 + 0.5 * d1) / 50
        # MCTS as player 2
        w2, l2, d2 = play_tournament(game, rand_fn, fn, n_games=50)
        wr2 = (l2 + 0.5 * d2) / 50
        win_rates_p1.append(wr1)
        win_rates_p2.append(wr2)
        print(f"  {n_sims:5d} sims | P1: {wr1*100:5.1f}% ({w1}W/{l1}L/{d1}D) | "
              f"P2: {wr2*100:5.1f}% ({l2}W/{w2}L/{d2}D)")

    # Self-play
    print("\n" + "-" * 40)
    print("MCTS (1000 sims) vs MCTS (1000 sims)")
    print("-" * 40)
    fn1 = mcts_player(game, n_simulations=1000)
    fn2 = mcts_player(game, n_simulations=1000)
    w, l, d = play_tournament(game, fn1, fn2, n_games=50)
    print(f"  P1 wins: {w}, P2 wins: {l}, Draws: {d}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(sim_counts, [wr * 100 for wr in win_rates_p1], 'bo-',
                 linewidth=2, markersize=8, label='MCTS as P1')
    axes[0].plot(sim_counts, [wr * 100 for wr in win_rates_p2], 'rs-',
                 linewidth=2, markersize=8, label='MCTS as P2')
    axes[0].axhline(y=100, color='green', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('MCTS Simulations')
    axes[0].set_ylabel('Win Rate vs Random (%)')
    axes[0].set_title('MCTS Win Rate vs Simulation Budget')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bar chart: best results
    best_fn = mcts_player(game, n_simulations=1000)
    w_p1, l_p1, d_p1 = play_tournament(game, best_fn, rand_fn, n_games=100)
    w_p2, l_p2, d_p2 = play_tournament(game, rand_fn, best_fn, n_games=100)

    labels = ['MCTS (P1)\nvs Random', 'Random vs\nMCTS (P2)']
    wins = [w_p1, l_p2]
    losses = [l_p1, w_p2]
    draws = [d_p1, d_p2]
    x = np.arange(len(labels))
    width = 0.25
    axes[1].bar(x - width, wins, width, color='green', alpha=0.7, label='MCTS Wins')
    axes[1].bar(x, losses, width, color='red', alpha=0.7, label='Random Wins')
    axes[1].bar(x + width, draws, width, color='gray', alpha=0.7, label='Draws')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Games (out of 100)')
    axes[1].set_title('MCTS (1000 sims) vs Random')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'tictactoe_mcts.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    main()
