"""
Solve Connect Four with AlphaZero self-play.

Connect Four is a much harder benchmark than TicTacToe (6x7 board, 7 actions).
AlphaZero learns through self-play with a deeper network.

Author: RL Speedrun
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '25_alphazero'))
from alphazero import AlphaZeroTrainer, AlphaZeroMCTS
from games import ConnectFour

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    print("=" * 70)
    print("Connect Four: AlphaZero Self-Play")
    print("=" * 70)

    print("\n" + "-" * 40)
    print("Environment Info")
    print("-" * 40)
    print("  Board: 6x7, gravity-based column placement")
    print("  Actions: 7 (one per column)")
    print("  Win condition: 4 in a row (any direction)")

    np.random.seed(42)
    torch.manual_seed(42)

    game = ConnectFour()
    trainer = AlphaZeroTrainer(
        game,
        n_iterations=100,
        n_self_play_games=30,
        n_training_steps=200,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        n_simulations=100,
        c_puct=1.0,
        n_channels=128,
        n_blocks=4,
        temperature_threshold=15,
        n_eval_games=20,
        buffer_capacity=100000,
    )

    print("\n" + "-" * 40)
    print("AlphaZero Configuration")
    print("-" * 40)
    print("  Network: 128 channels, 4 residual blocks")
    print("  Simulations: 100 per move")
    print("  Self-play: 30 games per iteration, 100 iterations")

    print("\n" + "=" * 70)
    print("Training (this takes ~15-30 minutes)...")
    print("=" * 70)
    print("-" * 60)
    model, history = trainer.train()
    print("-" * 60)

    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Evaluation (50 games vs random)")
    print("-" * 60)
    final_wr = trainer._evaluate_vs_random(n_games=50)
    print(f"  Win rate vs random: {final_wr*100:.0f}%")

    if final_wr >= 0.80:
        print("  STRONG PLAY achieved!")
    else:
        print(f"  Still improving (got {final_wr*100:.0f}%)")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['iteration'], history['policy_loss'], 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Policy Loss')
    axes[0, 0].set_title('Policy Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['iteration'], history['value_loss'], 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Value Loss')
    axes[0, 1].set_title('Value Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['iteration'],
                     [wr * 100 for wr in history['win_rate_vs_random']],
                     'g-', linewidth=1.5)
    axes[1, 0].axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80%')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Win Rate (%)')
    axes[1, 0].set_title('Win Rate vs Random')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['iteration'], history['total_loss'], 'k-', linewidth=1.5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Total Loss')
    axes[1, 1].set_title('Total Loss')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'connect4_alphazero.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    main()
