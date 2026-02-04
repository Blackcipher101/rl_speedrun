"""
Blackjack Solver using Monte Carlo Methods

This script trains an agent to play Blackjack using First-Visit Monte Carlo
control with ε-greedy exploration.

State: (player_sum, dealer_showing, usable_ace)
Actions: 0=stick, 1=hit

Author: RL Speedrun
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Tuple, Dict, List
import os


def state_to_tuple(obs) -> Tuple[int, int, int]:
    """Convert observation to hashable tuple."""
    return (obs[0], obs[1], int(obs[2]))


def mc_control_blackjack(
    env,
    n_episodes: int = 500000,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    epsilon_decay: float = 1.0,
    epsilon_min: float = 0.01,
    seed: int = 42
) -> Tuple[Dict, Dict, List[float]]:
    """
    First-Visit MC Control for Blackjack.

    Args:
        env: Blackjack environment
        n_episodes: Number of training episodes
        gamma: Discount factor (1.0 for undiscounted)
        epsilon: Exploration rate
        epsilon_decay: Epsilon decay per episode
        epsilon_min: Minimum epsilon
        seed: Random seed

    Returns:
        Q: Action-value function {state: [Q(s,stick), Q(s,hit)]}
        policy: Learned policy {state: action}
        episode_returns: Return of each episode
    """
    rng = np.random.default_rng(seed)

    # Initialize Q and counts
    Q = defaultdict(lambda: np.zeros(2))
    N = defaultdict(lambda: np.zeros(2))
    policy = {}
    episode_returns = []

    current_epsilon = epsilon

    for ep in range(n_episodes):
        # Generate episode
        episode = []
        obs, _ = env.reset(seed=seed + ep)
        state = state_to_tuple(obs)
        done = False

        while not done:
            # ε-greedy action selection
            if state not in Q or rng.random() < current_epsilon:
                action = rng.integers(2)
            else:
                action = np.argmax(Q[state])

            next_obs, reward, done, truncated, _ = env.step(action)
            episode.append((state, action, reward))

            if not done:
                state = state_to_tuple(next_obs)

        # Calculate returns and update Q (first-visit)
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                N[state][action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state][action]
                policy[state] = np.argmax(Q[state])

        episode_returns.append(G)

        # Decay epsilon
        current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)

        # Progress
        if (ep + 1) % 100000 == 0:
            recent_return = np.mean(episode_returns[-10000:])
            print(f"Episode {ep + 1:,}: Avg return (last 10k) = {recent_return:.3f}")

    return dict(Q), policy, episode_returns


def evaluate_policy(
    env,
    policy: Dict,
    n_episodes: int = 10000,
    seed: int = 0
) -> Dict:
    """
    Evaluate a Blackjack policy.

    Returns:
        stats: Dictionary with win/lose/draw rates and expected return
    """
    wins = 0
    losses = 0
    draws = 0
    total_return = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = state_to_tuple(obs)
        done = False

        while not done:
            if state in policy:
                action = policy[state]
            else:
                # Default: stick if sum >= 17
                action = 0 if state[0] >= 17 else 1

            obs, reward, done, _, _ = env.step(action)
            if not done:
                state = state_to_tuple(obs)

        total_return += reward
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    return {
        "win_rate": wins / n_episodes,
        "lose_rate": losses / n_episodes,
        "draw_rate": draws / n_episodes,
        "expected_return": total_return / n_episodes
    }


def visualize_policy_and_values(Q: Dict, policy: Dict, save_path: str = None):
    """
    Create visualization of learned policy and value function.

    Creates 2x2 grid:
    - Top: Value functions (no ace, with ace)
    - Bottom: Policies (no ace, with ace)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Player sum range: 12-21 (below 12, always hit)
    player_sums = range(12, 22)
    dealer_cards = range(1, 11)

    for usable_ace in [0, 1]:
        # Extract V and policy for this ace status
        V = np.zeros((10, 10))  # 10 player sums x 10 dealer cards
        P = np.zeros((10, 10))

        for i, player_sum in enumerate(player_sums):
            for j, dealer_card in enumerate(dealer_cards):
                state = (player_sum, dealer_card, usable_ace)
                if state in Q:
                    V[i, j] = np.max(Q[state])
                    P[i, j] = policy.get(state, 1)  # Default hit
                else:
                    V[i, j] = 0
                    P[i, j] = 1  # Default hit

        # Plot value function
        ax_v = axes[0, usable_ace]
        im = ax_v.imshow(V, cmap='RdYlGn', aspect='auto',
                         extent=[0.5, 10.5, 21.5, 11.5])
        ax_v.set_xlabel('Dealer Showing')
        ax_v.set_ylabel('Player Sum')
        ace_str = "With Usable Ace" if usable_ace else "Without Usable Ace"
        ax_v.set_title(f'Value Function V(s) - {ace_str}')
        ax_v.set_xticks(range(1, 11))
        ax_v.set_xticklabels(['A'] + list(range(2, 11)))
        ax_v.set_yticks(range(12, 22))
        plt.colorbar(im, ax=ax_v, label='V(s)')

        # Plot policy
        ax_p = axes[1, usable_ace]
        # 0=stick (green), 1=hit (red)
        policy_colors = np.where(P == 0, 0, 1)
        im2 = ax_p.imshow(policy_colors, cmap='RdYlGn_r', aspect='auto',
                          extent=[0.5, 10.5, 21.5, 11.5], vmin=0, vmax=1)
        ax_p.set_xlabel('Dealer Showing')
        ax_p.set_ylabel('Player Sum')
        ax_p.set_title(f'Policy π(s) - {ace_str}')
        ax_p.set_xticks(range(1, 11))
        ax_p.set_xticklabels(['A'] + list(range(2, 11)))
        ax_p.set_yticks(range(12, 22))

        # Add text annotations for policy
        for i, player_sum in enumerate(player_sums):
            for j, dealer_card in enumerate(dealer_cards):
                action = 'S' if P[i, j] == 0 else 'H'
                color = 'white' if P[i, j] == 1 else 'black'
                ax_p.text(dealer_card, player_sum, action,
                         ha='center', va='center', fontsize=8,
                         fontweight='bold', color=color)

    # Add legend for policy
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Stick'),
        Patch(facecolor='red', label='Hit')
    ]
    axes[1, 1].legend(handles=legend_elements, loc='lower right')

    plt.suptitle('Blackjack: Monte Carlo Learning Results', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")

    plt.show()


def main():
    print("=" * 70)
    print("Blackjack Solver: Monte Carlo Control")
    print("=" * 70)

    # Create environment
    env = gym.make("Blackjack-v1", natural=False, sab=False)

    print("\nEnvironment:")
    print("  State: (player_sum, dealer_showing, usable_ace)")
    print("  Actions: 0=stick, 1=hit")
    print("  Reward: +1 win, -1 lose, 0 draw")

    # Hyperparameters
    config = {
        "n_episodes": 500000,
        "gamma": 1.0,
        "epsilon": 1.0,
        "epsilon_decay": 0.99999,
        "epsilon_min": 0.01,
        "seed": 42
    }

    print("\nHyperparameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Train
    print("\nTraining...")
    print("-" * 50)
    Q, policy, returns = mc_control_blackjack(env, **config)

    # Evaluate
    print("-" * 50)
    print("\nEvaluating learned policy (10,000 episodes)...")
    stats = evaluate_policy(env, policy, n_episodes=10000, seed=0)

    print(f"\nResults:")
    print(f"  Win rate:  {stats['win_rate']*100:.1f}%")
    print(f"  Lose rate: {stats['lose_rate']*100:.1f}%")
    print(f"  Draw rate: {stats['draw_rate']*100:.1f}%")
    print(f"  Expected return: {stats['expected_return']:.3f}")

    # Compare with random policy
    print("\nComparison with random policy...")
    random_policy = {}  # Empty = will use default
    random_stats = evaluate_policy(env, random_policy, n_episodes=10000, seed=0)
    print(f"  Random policy expected return: {random_stats['expected_return']:.3f}")
    print(f"  Improvement: {stats['expected_return'] - random_stats['expected_return']:.3f}")

    # Print some learned policy decisions
    print("\nSample learned decisions:")
    sample_states = [
        (20, 10, 0), (16, 10, 0), (12, 6, 0),
        (20, 1, 0), (18, 10, 1), (13, 2, 0)
    ]
    for state in sample_states:
        if state in policy:
            action = "STICK" if policy[state] == 0 else "HIT"
            ace_str = "with ace" if state[2] else "no ace"
            print(f"  Sum={state[0]}, Dealer={state[1]}, {ace_str}: {action}")

    # Visualize
    save_path = os.path.join(os.path.dirname(__file__), 'blackjack_solution.png')
    visualize_policy_and_values(Q, policy, save_path)

    # Plot learning curve
    fig, ax = plt.subplots(figsize=(10, 4))
    window = 10000
    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='darkgreen', linewidth=1)
    ax.axhline(y=stats['expected_return'], color='red', linestyle='--',
               label=f'Final: {stats["expected_return"]:.3f}')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return (smoothed)')
    ax.set_title('Blackjack MC Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    curve_path = os.path.join(os.path.dirname(__file__), 'blackjack_learning_curve.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    print(f"Learning curve saved to: {curve_path}")
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
