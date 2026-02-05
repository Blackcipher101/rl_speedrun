"""
Exploration Strategies for Reinforcement Learning

This module provides modular exploration strategies that can be plugged into
any tabular RL agent. Includes:
- Epsilon-Greedy (with various decay schedules)
- Boltzmann (Softmax) Exploration
- Upper Confidence Bound (UCB)

The exploration-exploitation tradeoff is fundamental to RL:
- Too much exploration: slow learning, suboptimal performance
- Too little exploration: may miss optimal policy, get stuck

Author: RL Speedrun
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class ExplorationStrategy(ABC):
    """
    Abstract base class for exploration strategies.

    All strategies must implement:
    - select_action(): Choose action given Q-values
    - decay(): Update exploration parameters (if applicable)
    """

    @abstractmethod
    def select_action(
        self,
        q_values: np.ndarray,
        state: Optional[int] = None,
        step: int = 0
    ) -> int:
        """
        Select an action based on Q-values and exploration strategy.

        Args:
            q_values: Q(s, a) values for current state, shape (n_actions,)
            state: Current state index (needed for some strategies like UCB)
            step: Current timestep (for time-dependent exploration)

        Returns:
            Selected action index
        """
        pass

    def decay(self):
        """Update exploration parameters after each episode."""
        pass

    def reset(self):
        """Reset strategy to initial state."""
        pass


class EpsilonGreedy(ExplorationStrategy):
    """
    Epsilon-Greedy Exploration Strategy.

    With probability ε: select random action (exploration)
    With probability 1-ε: select greedy action (exploitation)

    Supports multiple decay schedules:
    - None: constant epsilon
    - 'linear': ε_t = max(ε_min, ε_0 - decay_rate * t)
    - 'exponential': ε_t = max(ε_min, ε_0 * decay_rate^t)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        decay_rate: float = 0.995,
        decay_type: str = 'exponential',
        seed: int = 42
    ):
        """
        Initialize epsilon-greedy strategy.

        Args:
            epsilon: Initial exploration probability
            epsilon_min: Minimum exploration probability
            decay_rate: Decay factor per episode
            decay_type: 'exponential', 'linear', or None
            seed: Random seed for reproducibility
        """
        self.epsilon_init = epsilon
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.decay_type = decay_type
        self.rng = np.random.default_rng(seed)
        self.episode = 0

    def select_action(
        self,
        q_values: np.ndarray,
        state: Optional[int] = None,
        step: int = 0
    ) -> int:
        """Select action using ε-greedy policy."""
        if self.rng.random() < self.epsilon:
            # Random action (exploration)
            return self.rng.integers(len(q_values))
        else:
            # Greedy action (exploitation)
            # Break ties randomly
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return self.rng.choice(best_actions)

    def decay(self):
        """Apply epsilon decay after episode."""
        self.episode += 1

        if self.decay_type == 'exponential':
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.decay_rate
            )
        elif self.decay_type == 'linear':
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_init - self.decay_rate * self.episode
            )
        # If decay_type is None, epsilon stays constant

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.epsilon_init
        self.episode = 0


class BoltzmannExploration(ExplorationStrategy):
    """
    Boltzmann (Softmax) Exploration Strategy.

    Action selection probability:
        P(a|s) = exp(Q(s,a) / τ) / Σ_a' exp(Q(s,a') / τ)

    Temperature τ controls exploration:
    - High τ → uniform distribution (exploration)
    - Low τ → greedy selection (exploitation)
    - τ → 0 → deterministic greedy
    - τ → ∞ → uniform random

    Advantage over ε-greedy: Uses Q-value information
    (prefers actions with higher Q even when exploring)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        temperature_min: float = 0.1,
        decay_rate: float = 0.995,
        seed: int = 42
    ):
        """
        Initialize Boltzmann exploration.

        Args:
            temperature: Initial temperature τ
            temperature_min: Minimum temperature
            decay_rate: Temperature decay per episode
            seed: Random seed
        """
        self.temperature_init = temperature
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.decay_rate = decay_rate
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        q_values: np.ndarray,
        state: Optional[int] = None,
        step: int = 0
    ) -> int:
        """
        Select action using Boltzmann distribution.

        Uses numerical stability trick: subtract max before exp.
        """
        # Numerical stability
        scaled_q = q_values / self.temperature
        scaled_q = scaled_q - np.max(scaled_q)

        # Compute softmax probabilities
        exp_q = np.exp(np.clip(scaled_q, -20, 20))
        probs = exp_q / (np.sum(exp_q) + 1e-10)

        # Sample action
        return self.rng.choice(len(q_values), p=probs)

    def get_action_probabilities(self, q_values: np.ndarray) -> np.ndarray:
        """Return action probabilities for visualization."""
        scaled_q = q_values / self.temperature
        scaled_q = scaled_q - np.max(scaled_q)
        exp_q = np.exp(np.clip(scaled_q, -20, 20))
        return exp_q / (np.sum(exp_q) + 1e-10)

    def decay(self):
        """Decay temperature after episode."""
        self.temperature = max(
            self.temperature_min,
            self.temperature * self.decay_rate
        )

    def reset(self):
        """Reset temperature to initial value."""
        self.temperature = self.temperature_init


class UCBExploration(ExplorationStrategy):
    """
    Upper Confidence Bound (UCB) Exploration Strategy.

    Action selection:
        a = argmax_a [Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))]

    Where:
    - N(s): number of times state s has been visited
    - N(s,a): number of times action a was taken in state s
    - c: exploration constant (controls exploration bonus)

    The bonus term decreases as actions are taken more often,
    naturally balancing exploration and exploitation.

    Theoretical property: Achieves logarithmic regret (optimal).
    """

    def __init__(
        self,
        c: float = 2.0,
        n_states: int = 16,
        n_actions: int = 4,
        seed: int = 42
    ):
        """
        Initialize UCB exploration.

        Args:
            c: Exploration constant (higher = more exploration)
            n_states: Number of states (for tracking visit counts)
            n_actions: Number of actions
            seed: Random seed
        """
        self.c = c
        self.n_states = n_states
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

        # Visit counts
        self.state_visits = np.zeros(n_states)
        self.state_action_visits = np.zeros((n_states, n_actions))

    def select_action(
        self,
        q_values: np.ndarray,
        state: Optional[int] = None,
        step: int = 0
    ) -> int:
        """
        Select action using UCB criterion.

        If any action hasn't been tried, select it first.
        Otherwise, use UCB formula.
        """
        if state is None:
            raise ValueError("UCB requires state for visit counting")

        n_actions = len(q_values)

        # Update state visit count
        self.state_visits[state] += 1

        # Check for unexplored actions
        action_visits = self.state_action_visits[state]
        if np.any(action_visits == 0):
            # Select random unvisited action
            unvisited = np.where(action_visits == 0)[0]
            action = self.rng.choice(unvisited)
        else:
            # UCB selection
            N_s = self.state_visits[state]
            N_sa = self.state_action_visits[state]

            # UCB bonus: c * sqrt(ln(N(s)) / N(s,a))
            bonus = self.c * np.sqrt(np.log(N_s) / N_sa)

            # Select action with highest UCB value
            ucb_values = q_values + bonus
            action = np.argmax(ucb_values)

        # Update action visit count
        self.state_action_visits[state, action] += 1

        return action

    def get_ucb_values(self, q_values: np.ndarray, state: int) -> np.ndarray:
        """Return UCB values for visualization."""
        N_s = self.state_visits[state]
        N_sa = self.state_action_visits[state]

        if N_s == 0:
            return q_values

        # Avoid division by zero
        N_sa = np.maximum(N_sa, 1e-10)
        bonus = self.c * np.sqrt(np.log(max(N_s, 1)) / N_sa)

        return q_values + bonus

    def decay(self):
        """UCB doesn't need decay (exploration naturally decreases)."""
        pass

    def reset(self):
        """Reset visit counts."""
        self.state_visits = np.zeros(self.n_states)
        self.state_action_visits = np.zeros((self.n_states, self.n_actions))


# =============================================================================
# Utility Functions
# =============================================================================

def create_exploration_strategy(
    strategy_type: str,
    **kwargs
) -> ExplorationStrategy:
    """
    Factory function to create exploration strategies.

    Args:
        strategy_type: 'epsilon_greedy', 'boltzmann', or 'ucb'
        **kwargs: Strategy-specific parameters

    Returns:
        ExplorationStrategy instance
    """
    strategies = {
        'epsilon_greedy': EpsilonGreedy,
        'boltzmann': BoltzmannExploration,
        'ucb': UCBExploration
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_type}. "
                        f"Available: {list(strategies.keys())}")

    return strategies[strategy_type](**kwargs)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Exploration Strategies Demo")
    print("=" * 60)

    # Test Q-values (4 actions)
    q_values = np.array([1.0, 2.0, 1.5, 0.5])
    print(f"\nQ-values: {q_values}")
    print(f"Optimal action: {np.argmax(q_values)}")

    # Test each strategy
    n_samples = 10000

    # 1. Epsilon-Greedy
    print("\n--- Epsilon-Greedy (ε=0.1) ---")
    eps_greedy = EpsilonGreedy(epsilon=0.1, decay_type=None, seed=42)
    actions = [eps_greedy.select_action(q_values) for _ in range(n_samples)]
    action_counts = np.bincount(actions, minlength=4)
    print(f"Action distribution: {action_counts / n_samples}")

    # 2. Boltzmann
    print("\n--- Boltzmann (τ=0.5) ---")
    boltzmann = BoltzmannExploration(temperature=0.5, seed=42)
    actions = [boltzmann.select_action(q_values) for _ in range(n_samples)]
    action_counts = np.bincount(actions, minlength=4)
    print(f"Action distribution: {action_counts / n_samples}")
    print(f"Theoretical probs: {boltzmann.get_action_probabilities(q_values)}")

    # 3. UCB (needs state)
    print("\n--- UCB (c=2.0) ---")
    ucb = UCBExploration(c=2.0, n_states=1, n_actions=4, seed=42)
    actions = [ucb.select_action(q_values, state=0) for _ in range(n_samples)]
    action_counts = np.bincount(actions, minlength=4)
    print(f"Action distribution: {action_counts / n_samples}")

    # Visualize decay schedules
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Epsilon decay
    eps_exp = EpsilonGreedy(epsilon=1.0, decay_rate=0.99, decay_type='exponential')
    eps_lin = EpsilonGreedy(epsilon=1.0, decay_rate=0.01, decay_type='linear')

    eps_exp_values = []
    eps_lin_values = []
    for _ in range(200):
        eps_exp_values.append(eps_exp.epsilon)
        eps_lin_values.append(eps_lin.epsilon)
        eps_exp.decay()
        eps_lin.decay()

    axes[0].plot(eps_exp_values, label='Exponential (0.99)')
    axes[0].plot(eps_lin_values, label='Linear (0.01/ep)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Epsilon')
    axes[0].set_title('Epsilon Decay Schedules')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Temperature decay
    boltz = BoltzmannExploration(temperature=2.0, decay_rate=0.99)
    temp_values = []
    for _ in range(200):
        temp_values.append(boltz.temperature)
        boltz.decay()

    axes[1].plot(temp_values, color='orange')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Temperature')
    axes[1].set_title('Boltzmann Temperature Decay')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exploration_decay.png', dpi=150, bbox_inches='tight')
    print("\nSaved decay visualization to: exploration_decay.png")
    plt.show()
