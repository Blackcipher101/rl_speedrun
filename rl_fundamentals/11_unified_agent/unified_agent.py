"""
Unified Tabular RL Agent

A configurable agent that consolidates Q-Learning and SARSA with:
- Pluggable exploration strategies (ε-greedy, Boltzmann, UCB)
- Optional reward shaping
- Comprehensive logging and visualization

This module demonstrates how to build a flexible agent framework
that can switch between different algorithms and exploration methods.

Author: RL Speedrun
"""

import numpy as np
from typing import Tuple, List, Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Import exploration strategies (relative import when used as module)
try:
    from .exploration_strategies import (
        ExplorationStrategy, EpsilonGreedy, BoltzmannExploration, UCBExploration
    )
except ImportError:
    from exploration_strategies import (
        ExplorationStrategy, EpsilonGreedy, BoltzmannExploration, UCBExploration
    )


class Algorithm(Enum):
    """Supported TD algorithms."""
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    episode_returns: List[float]
    episode_lengths: List[int]
    epsilon_history: List[float]
    q_value_history: List[float]  # Mean Q-value per episode


class UnifiedAgent:
    """
    Unified Tabular RL Agent supporting Q-Learning and SARSA.

    The agent is configurable with different:
    - Learning algorithms (Q-Learning vs SARSA)
    - Exploration strategies (ε-greedy, Boltzmann, UCB)
    - Reward shaping functions

    Key equations:
        Q-Learning (off-policy):
            Q(S,A) ← Q(S,A) + α[R + γ·max_a Q(S',a) - Q(S,A)]

        SARSA (on-policy):
            Q(S,A) ← Q(S,A) + α[R + γ·Q(S',A') - Q(S,A)]
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        algorithm: str = "q_learning",
        exploration: Optional[ExplorationStrategy] = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        reward_shaping: Optional[Callable[[int, int, bool], float]] = None,
        seed: int = 42
    ):
        """
        Initialize the unified agent.

        Args:
            n_states: Number of states in the environment
            n_actions: Number of actions
            algorithm: 'q_learning' or 'sarsa'
            exploration: Exploration strategy (default: EpsilonGreedy)
            alpha: Learning rate
            gamma: Discount factor
            reward_shaping: Optional function F(s, s', done) for shaping reward
            seed: Random seed
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Set algorithm
        if algorithm.lower() == "q_learning":
            self.algorithm = Algorithm.Q_LEARNING
        elif algorithm.lower() == "sarsa":
            self.algorithm = Algorithm.SARSA
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Set exploration strategy
        if exploration is None:
            self.exploration = EpsilonGreedy(seed=seed)
        else:
            self.exploration = exploration

        # Reward shaping function
        self.reward_shaping = reward_shaping

        # Initialize Q-table
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        """Select action using the exploration strategy."""
        return self.exploration.select_action(
            self.Q[state],
            state=state
        )

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: Optional[int] = None,
        done: bool = False
    ) -> float:
        """
        Update Q-values based on the selected algorithm.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (required for SARSA)
            done: Whether episode terminated

        Returns:
            TD error (for monitoring)
        """
        # Apply reward shaping if provided
        shaped_reward = reward
        if self.reward_shaping is not None:
            shaped_reward += self.reward_shaping(state, next_state, done)

        # Compute TD target based on algorithm
        if self.algorithm == Algorithm.Q_LEARNING:
            # Off-policy: use max Q
            if done:
                td_target = shaped_reward
            else:
                td_target = shaped_reward + self.gamma * np.max(self.Q[next_state])
        else:  # SARSA
            # On-policy: use actual next action
            if next_action is None:
                raise ValueError("SARSA requires next_action")
            if done:
                td_target = shaped_reward
            else:
                td_target = shaped_reward + self.gamma * self.Q[next_state, next_action]

        # TD error
        td_error = td_target - self.Q[state, action]

        # Update Q-value
        self.Q[state, action] += self.alpha * td_error

        return td_error

    def train(
        self,
        env,
        n_episodes: int = 1000,
        max_steps: int = 1000,
        verbose: bool = True
    ) -> TrainingMetrics:
        """
        Train the agent on an environment.

        Args:
            env: Gymnasium environment
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Print progress

        Returns:
            TrainingMetrics with training history
        """
        metrics = TrainingMetrics(
            episode_returns=[],
            episode_lengths=[],
            epsilon_history=[],
            q_value_history=[]
        )

        for episode in range(n_episodes):
            state, _ = env.reset(seed=self.seed + episode)
            episode_return = 0.0
            episode_length = 0

            # For SARSA: select initial action
            if self.algorithm == Algorithm.SARSA:
                action = self.select_action(state)

            for step in range(max_steps):
                # Select action (Q-Learning selects each step, SARSA uses pre-selected)
                if self.algorithm == Algorithm.Q_LEARNING:
                    action = self.select_action(state)

                # Take action
                next_state, reward, done, truncated, _ = env.step(action)
                episode_return += reward
                episode_length += 1

                # For SARSA: select next action before update
                next_action = None
                if self.algorithm == Algorithm.SARSA and not (done or truncated):
                    next_action = self.select_action(next_state)

                # Update Q-values
                self.update(
                    state, action, reward, next_state,
                    next_action=next_action,
                    done=done or truncated
                )

                if done or truncated:
                    break

                # Transition
                state = next_state
                if self.algorithm == Algorithm.SARSA:
                    action = next_action

            # Track metrics
            metrics.episode_returns.append(episode_return)
            metrics.episode_lengths.append(episode_length)
            metrics.q_value_history.append(np.mean(self.Q))

            # Track exploration parameter
            if hasattr(self.exploration, 'epsilon'):
                metrics.epsilon_history.append(self.exploration.epsilon)
            elif hasattr(self.exploration, 'temperature'):
                metrics.epsilon_history.append(self.exploration.temperature)
            else:
                metrics.epsilon_history.append(0)

            # Decay exploration
            self.exploration.decay()

            # Progress logging
            if verbose and (episode + 1) % 100 == 0:
                avg_return = np.mean(metrics.episode_returns[-100:])
                print(f"Episode {episode + 1:5d}: Avg return (100 ep) = {avg_return:.2f}")

        return metrics

    def get_policy(self) -> np.ndarray:
        """Extract greedy policy from Q-table."""
        return np.argmax(self.Q, axis=1)

    def get_value_function(self) -> np.ndarray:
        """Get state-value function V(s) = max_a Q(s,a)."""
        return np.max(self.Q, axis=1)

    def evaluate(
        self,
        env,
        n_episodes: int = 100,
        seed: int = 0
    ) -> Tuple[float, float]:
        """
        Evaluate the learned policy (greedy, no exploration).

        Returns:
            mean_return: Average return
            std_return: Standard deviation
        """
        returns = []
        policy = self.get_policy()

        for ep in range(n_episodes):
            state, _ = env.reset(seed=seed + ep)
            episode_return = 0.0
            done = False
            truncated = False

            while not done and not truncated:
                action = policy[state]
                state, reward, done, truncated, _ = env.step(action)
                episode_return += reward

            returns.append(episode_return)

        return np.mean(returns), np.std(returns)


# =============================================================================
# Reward Shaping Utilities
# =============================================================================

def potential_based_shaping(
    potential_fn: Callable[[int], float],
    gamma: float
) -> Callable[[int, int, bool], float]:
    """
    Create a potential-based reward shaping function.

    F(s, s') = γ * φ(s') - φ(s)

    Theorem (Ng et al., 1999): Potential-based shaping preserves
    the optimal policy. The shaped reward R' = R + F has the same
    optimal policy as the original reward R.

    Args:
        potential_fn: φ(s) - potential function for each state
        gamma: Discount factor

    Returns:
        Shaping function F(s, s', done) -> float
    """
    def shaping_reward(state: int, next_state: int, done: bool) -> float:
        if done:
            # Terminal state has zero potential
            return -potential_fn(state)
        return gamma * potential_fn(next_state) - potential_fn(state)

    return shaping_reward


def distance_based_potential(
    goal_state: int,
    n_states: int,
    grid_size: int = 4
) -> Callable[[int], float]:
    """
    Create potential function based on distance to goal.

    φ(s) = -distance(s, goal)

    Closer to goal = higher potential = larger shaping bonus.

    Args:
        goal_state: Goal state index
        n_states: Total number of states
        grid_size: Grid dimension (assumes square grid)

    Returns:
        Potential function φ(s)
    """
    goal_row, goal_col = goal_state // grid_size, goal_state % grid_size

    def potential(state: int) -> float:
        row, col = state // grid_size, state % grid_size
        # Manhattan distance, negated so closer = higher potential
        distance = abs(row - goal_row) + abs(col - goal_col)
        return -distance

    return potential


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    import gymnasium as gym
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Unified Agent Demo: Comparing Algorithms and Exploration Strategies")
    print("=" * 70)

    # Create environment
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    # Configuration for experiments
    n_episodes = 10000
    n_eval = 100

    results = {}

    # 1. Q-Learning with Epsilon-Greedy
    print("\n--- Q-Learning + Epsilon-Greedy ---")
    agent_q_eps = UnifiedAgent(
        n_states=16, n_actions=4,
        algorithm="q_learning",
        exploration=EpsilonGreedy(
            epsilon=1.0, epsilon_min=0.01, decay_rate=0.9995
        ),
        alpha=0.1, gamma=0.99, seed=42
    )
    metrics_q_eps = agent_q_eps.train(env, n_episodes=n_episodes)
    mean_ret, std_ret = agent_q_eps.evaluate(env, n_episodes=n_eval)
    results['Q-Learning + ε-greedy'] = (metrics_q_eps, mean_ret, std_ret)
    print(f"Final evaluation: {mean_ret:.3f} ± {std_ret:.3f}")

    # 2. SARSA with Epsilon-Greedy
    print("\n--- SARSA + Epsilon-Greedy ---")
    agent_s_eps = UnifiedAgent(
        n_states=16, n_actions=4,
        algorithm="sarsa",
        exploration=EpsilonGreedy(
            epsilon=1.0, epsilon_min=0.01, decay_rate=0.9995
        ),
        alpha=0.1, gamma=0.99, seed=42
    )
    metrics_s_eps = agent_s_eps.train(env, n_episodes=n_episodes)
    mean_ret, std_ret = agent_s_eps.evaluate(env, n_episodes=n_eval)
    results['SARSA + ε-greedy'] = (metrics_s_eps, mean_ret, std_ret)
    print(f"Final evaluation: {mean_ret:.3f} ± {std_ret:.3f}")

    # 3. Q-Learning with Boltzmann
    print("\n--- Q-Learning + Boltzmann ---")
    agent_q_boltz = UnifiedAgent(
        n_states=16, n_actions=4,
        algorithm="q_learning",
        exploration=BoltzmannExploration(
            temperature=2.0, temperature_min=0.1, decay_rate=0.9995
        ),
        alpha=0.1, gamma=0.99, seed=42
    )
    metrics_q_boltz = agent_q_boltz.train(env, n_episodes=n_episodes)
    mean_ret, std_ret = agent_q_boltz.evaluate(env, n_episodes=n_eval)
    results['Q-Learning + Boltzmann'] = (metrics_q_boltz, mean_ret, std_ret)
    print(f"Final evaluation: {mean_ret:.3f} ± {std_ret:.3f}")

    # 4. Q-Learning with reward shaping
    print("\n--- Q-Learning + Reward Shaping ---")
    potential_fn = distance_based_potential(goal_state=15, n_states=16, grid_size=4)
    shaping_fn = potential_based_shaping(potential_fn, gamma=0.99)

    agent_q_shaped = UnifiedAgent(
        n_states=16, n_actions=4,
        algorithm="q_learning",
        exploration=EpsilonGreedy(
            epsilon=1.0, epsilon_min=0.01, decay_rate=0.9995
        ),
        alpha=0.1, gamma=0.99,
        reward_shaping=shaping_fn,
        seed=42
    )
    metrics_q_shaped = agent_q_shaped.train(env, n_episodes=n_episodes)
    mean_ret, std_ret = agent_q_shaped.evaluate(env, n_episodes=n_eval)
    results['Q-Learning + Shaping'] = (metrics_q_shaped, mean_ret, std_ret)
    print(f"Final evaluation: {mean_ret:.3f} ± {std_ret:.3f}")

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    window = 500
    for name, (metrics, mean_ret, std_ret) in results.items():
        smoothed = np.convolve(
            metrics.episode_returns,
            np.ones(window) / window,
            mode='valid'
        )
        axes[0].plot(smoothed, label=f"{name} ({mean_ret:.2f})", alpha=0.8)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return (smoothed)')
    axes[0].set_title('Learning Curves Comparison')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Final performance bar chart
    names = list(results.keys())
    means = [results[n][1] for n in names]
    stds = [results[n][2] for n in names]

    colors = ['steelblue', 'green', 'orange', 'purple']
    bars = axes[1].bar(range(len(names)), means, yerr=stds, capsize=5, color=colors)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([n.replace(' + ', '\n') for n in names], fontsize=8)
    axes[1].set_ylabel('Mean Return')
    axes[1].set_title('Final Performance Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('unified_agent_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison to: unified_agent_comparison.png")
    plt.show()

    # Print best policy
    print("\nBest Policy (Q-Learning + ε-greedy):")
    action_symbols = ['←', '↓', '→', '↑']
    policy = agent_q_eps.get_policy()
    for row in range(4):
        row_str = ""
        for col in range(4):
            state = row * 4 + col
            if state in [5, 7, 11, 12]:
                row_str += " H "
            elif state == 15:
                row_str += " G "
            else:
                row_str += f" {action_symbols[policy[state]]} "
        print(row_str)

    env.close()
