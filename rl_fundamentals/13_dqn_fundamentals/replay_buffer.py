"""
Experience Replay Buffer for DQN

Experience replay breaks correlation between consecutive samples and enables
data-efficient learning by reusing past experiences multiple times.

Key insight: RL data is not i.i.d. - consecutive samples are highly correlated.
Replay buffer stores transitions and samples random mini-batches for training.

Author: RL Speedrun
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Transition:
    """Single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-size circular buffer for storing experience tuples.

    Memory complexity: O(capacity * state_dim)
    Sample complexity: O(batch_size)

    The buffer overwrites oldest experiences when full (FIFO).
    """

    def __init__(self, capacity: int, state_dim: int, seed: int = 42):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state vectors
            seed: Random seed for reproducible sampling
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.position = 0  # Current write position
        self.size = 0      # Current number of stored transitions

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.

        Overwrites oldest transition if buffer is full.

        Args:
            state: Current state s_t
            action: Action taken a_t
            reward: Reward received r_{t+1}
            next_state: Next state s_{t+1}
            done: Whether episode terminated
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each is a numpy array with batch_size rows
        """
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} from buffer with {self.size} transitions")

        # Sample random indices without replacement
        indices = self.rng.choice(self.size, size=batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return self.size >= batch_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) - samples important transitions more often.

    Priority is based on TD error: transitions with high error are sampled more
    frequently because they have more to teach the agent.

    p_i = |TD_error_i| + epsilon  (proportional prioritization)
    P(i) = p_i^alpha / sum(p_j^alpha)

    Importance sampling weights correct for the non-uniform sampling:
    w_i = (N * P(i))^(-beta)

    Reference: Schaul et al., 2015 - "Prioritized Experience Replay"
    """

    def __init__(self, capacity: int, state_dim: int,
                 alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, epsilon: float = 1e-6,
                 seed: int = 42):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            state_dim: State dimension
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full)
            beta_increment: How much to increase beta per sample call
            epsilon: Small constant to ensure non-zero priorities
            seed: Random seed
        """
        super().__init__(capacity, state_dim, seed)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Priority storage
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add transition with maximum priority (to ensure it's sampled at least once)."""
        # New transitions get max priority
        self.priorities[self.position] = self.max_priority
        super().push(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch weighted by priorities.

        Returns:
            states, actions, rewards, next_states, dones, weights, indices
            - weights: Importance sampling weights for loss scaling
            - indices: Buffer indices (needed for priority updates)
        """
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} from buffer with {self.size} transitions")

        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs = probs / (probs.sum() + 1e-10)

        # Sample indices based on priorities
        indices = self.rng.choice(self.size, size=batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        # w_i = (N * P(i))^(-beta), normalized by max weight
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Anneal beta toward 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights.astype(np.float32),
            indices
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on new TD errors.

        Called after computing TD errors for a sampled batch.

        Args:
            indices: Buffer indices of sampled transitions
            td_errors: Corresponding TD errors |r + gamma*max Q' - Q|
        """
        priorities = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Experience Replay Buffer Demo")
    print("=" * 60)

    # Create buffer
    state_dim = 4
    buffer = ReplayBuffer(capacity=1000, state_dim=state_dim, seed=42)

    print(f"\nBuffer created: capacity={buffer.capacity}, state_dim={state_dim}")
    print(f"Initial size: {len(buffer)}")

    # Add some transitions
    print("\nAdding 100 random transitions...")
    rng = np.random.default_rng(42)
    for i in range(100):
        state = rng.random(state_dim).astype(np.float32)
        action = rng.integers(2)
        reward = rng.random()
        next_state = rng.random(state_dim).astype(np.float32)
        done = rng.random() < 0.1
        buffer.push(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)}")
    print(f"Ready for batch of 32? {buffer.is_ready(32)}")

    # Sample a batch
    print("\nSampling batch of 32...")
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Next states shape: {next_states.shape}")
    print(f"  Dones shape: {dones.shape}")

    # Test circular buffer
    print("\nTesting circular buffer (adding 1500 transitions to capacity-1000 buffer)...")
    for i in range(1500):
        buffer.push(
            rng.random(state_dim).astype(np.float32),
            rng.integers(2),
            float(i),  # Use index as reward to verify overwriting
            rng.random(state_dim).astype(np.float32),
            False
        )

    print(f"Buffer size after 1600 pushes: {len(buffer)} (capped at {buffer.capacity})")

    # Check that old values were overwritten
    _, _, rewards, _, _ = buffer.sample(100)
    print(f"Reward range in sample: [{rewards.min():.0f}, {rewards.max():.0f}]")
    print("(Should be >= 600 since oldest transitions were overwritten)")

    # Prioritized replay demo
    print("\n" + "=" * 60)
    print("Prioritized Experience Replay Demo")
    print("=" * 60)

    per_buffer = PrioritizedReplayBuffer(
        capacity=1000, state_dim=state_dim,
        alpha=0.6, beta=0.4, seed=42
    )

    # Add transitions
    for i in range(200):
        per_buffer.push(
            rng.random(state_dim).astype(np.float32),
            rng.integers(2),
            rng.random(),
            rng.random(state_dim).astype(np.float32),
            False
        )

    print(f"\nPER buffer size: {len(per_buffer)}")

    # Sample with priorities
    states, actions, rewards, next_states, dones, weights, indices = per_buffer.sample(32)
    print(f"Sampled batch with importance weights")
    print(f"  Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"  Beta (IS exponent): {per_buffer.beta:.3f}")

    # Update priorities with fake TD errors
    td_errors = rng.random(32) * 10  # Random TD errors
    per_buffer.update_priorities(indices, td_errors)
    print(f"  Updated priorities for sampled indices")
    print(f"  Max priority: {per_buffer.max_priority:.3f}")
