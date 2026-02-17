"""
Solve TicTacToe with AlphaZero self-play.

Trains a policy-value network through self-play + MCTS,
then evaluates against random and pure MCTS opponents.

Author: RL Speedrun
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '25_alphazero'))
from alphazero import AlphaZeroTrainer, AlphaZeroMCTS
from games import TicTacToe

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("TicTacToe: AlphaZero Self-Play")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    game = TicTacToe()
    trainer = AlphaZeroTrainer(
        game,
        n_iterations=80,
        n_self_play_games=25,
        n_training_steps=150,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        n_simulations=80,
        c_puct=1.0,
        n_channels=64,
        n_blocks=2,
        temperature_threshold=8,
        n_eval_games=20,
        buffer_capacity=50000,
    )

    print("\nTraining AlphaZero...")
    print("-" * 60)
    model, history = trainer.train()

    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Evaluation (100 games vs random)")
    print("-" * 60)
    final_wr = trainer._evaluate_vs_random(n_games=100)
    print(f"  Win rate vs random: {final_wr*100:.0f}%")

    if final_wr >= 0.90:
        print("  STRONG PLAY achieved!")
    else:
        print(f"  Still learning (need >= 90%, got {final_wr*100:.0f}%)")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history['iteration'], history['policy_loss'], 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Policy Loss')
    axes[0, 0].set_title('Policy Loss (Cross-Entropy)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['iteration'], history['value_loss'], 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Value Loss')
    axes[0, 1].set_title('Value Loss (MSE)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['iteration'],
                     [wr * 100 for wr in history['win_rate_vs_random']],
                     'g-', linewidth=1.5)
    axes[1, 0].axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90%')
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
    save_path = os.path.join(os.path.dirname(__file__), 'tictactoe_alphazero.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    main()
