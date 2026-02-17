"""
Solve TicTacToe with MuZero — learning without knowing the rules.

MuZero never receives the game rules. It learns its own dynamics model
and plans in a learned latent space.

Author: RL Speedrun
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '26_muzero'))
from muzero import MuZeroTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '25_alphazero'))
from games import TicTacToe

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("TicTacToe: MuZero (Learning Without Rules)")
    print("=" * 60)

    print("\n  Key insight: MuZero never gets the game rules!")
    print("  It learns state transitions and rewards from experience.")

    np.random.seed(42)
    torch.manual_seed(42)

    game = TicTacToe()
    trainer = MuZeroTrainer(
        game,
        n_iterations=80,
        n_self_play_games=25,
        n_training_steps=200,
        batch_size=64,
        lr=5e-4,
        weight_decay=1e-4,
        n_simulations=80,
        c_puct=1.25,
        hidden_dim=128,
        unroll_steps=5,
        td_steps=10,
        temperature_threshold=8,
        n_eval_games=20,
        buffer_capacity=2000,
    )

    print("\n" + "-" * 40)
    print("MuZero Configuration")
    print("-" * 40)
    print("  Hidden dim: 128")
    print("  Simulations: 80 per move")
    print("  Unroll steps (K): 5")
    print("  Networks: Representation + Dynamics + Prediction")

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    print("-" * 60)
    network, history = trainer.train()
    print("-" * 60)

    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Evaluation (40 games vs random)")
    print("-" * 60)
    trainer.n_eval_games = 40
    final_wr = trainer._evaluate_vs_random()
    print(f"  Win rate vs random: {final_wr*100:.0f}%")
    print(f"\n  (Compare: AlphaZero achieves ~94% — MuZero is harder")
    print(f"   because it must also learn the game dynamics)")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history['iteration'], history['policy_loss'], 'b-',
                     label='Policy', linewidth=1.5)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Policy Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['iteration'], history['value_loss'], 'r-',
                     label='Value', linewidth=1.5)
    axes[0, 1].plot(history['iteration'], history['reward_loss'], 'g-',
                     label='Reward', linewidth=1.5)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Value & Reward Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['iteration'],
                     [wr * 100 for wr in history['win_rate_vs_random']],
                     'g-', linewidth=1.5)
    axes[1, 0].axhline(y=70, color='green', linestyle='--', alpha=0.5, label='70%')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Win Rate (%)')
    axes[1, 0].set_title('MuZero vs Random')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['iteration'], history['total_loss'], 'k-', linewidth=1.5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Total Loss')
    axes[1, 1].set_title('Total Training Loss')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'tictactoe_muzero.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")


if __name__ == "__main__":
    main()
