"""
Entropy Regularization for Policy Gradient Methods

Entropy measures the "randomness" of a probability distribution.
Adding entropy bonus to the objective encourages exploration by
penalizing overly deterministic policies.

H(π) = -sum_a π(a|s) * log π(a|s)

Benefits:
1. Prevents premature convergence to suboptimal deterministic policies
2. Maintains exploration throughout training
3. Leads to more robust policies

Author: RL Speedrun
"""

import numpy as np
from typing import Tuple


def compute_entropy(probs: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute entropy of a probability distribution.

    H(π) = -sum_a π(a) * log π(a)

    Args:
        probs: Action probabilities [π(a_0), π(a_1), ...]
        epsilon: Small constant for numerical stability

    Returns:
        Entropy value (higher = more random)
    """
    # Clip to avoid log(0)
    probs = np.clip(probs, epsilon, 1.0)
    return -np.sum(probs * np.log(probs))


def compute_batch_entropy(probs_batch: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute entropy for a batch of probability distributions.

    Args:
        probs_batch: Shape (batch_size, n_actions)
        epsilon: Numerical stability constant

    Returns:
        Entropies for each distribution in batch
    """
    probs_batch = np.clip(probs_batch, epsilon, 1.0)
    return -np.sum(probs_batch * np.log(probs_batch), axis=1)


def entropy_regularized_loss(
    log_probs: np.ndarray,
    advantages: np.ndarray,
    probs: np.ndarray,
    entropy_coef: float = 0.01
) -> Tuple[float, float, float]:
    """
    Compute policy gradient loss with entropy regularization.

    L = -E[log π(a|s) * A(s,a)] - β * H(π)

    where:
    - First term: Standard policy gradient (maximize)
    - Second term: Entropy bonus (maximize → explore)

    Args:
        log_probs: Log probabilities of taken actions
        advantages: Advantage estimates
        probs: Full action probability distributions
        entropy_coef: Entropy bonus coefficient (β)

    Returns:
        total_loss: Combined loss
        policy_loss: Policy gradient component
        entropy: Mean entropy (for logging)
    """
    # Policy gradient loss: -E[log π(a|s) * A(s,a)]
    policy_loss = -np.mean(log_probs * advantages)

    # Entropy bonus
    entropies = compute_batch_entropy(probs)
    mean_entropy = np.mean(entropies)

    # Total loss (subtract entropy because we want to maximize it)
    total_loss = policy_loss - entropy_coef * mean_entropy

    return total_loss, policy_loss, mean_entropy


def entropy_schedule(
    step: int,
    initial_coef: float = 0.01,
    final_coef: float = 0.001,
    decay_steps: int = 100000
) -> float:
    """
    Linearly decay entropy coefficient during training.

    Early training: High entropy → lots of exploration
    Late training: Low entropy → exploit learned policy

    Args:
        step: Current training step
        initial_coef: Starting entropy coefficient
        final_coef: Final entropy coefficient
        decay_steps: Steps over which to decay

    Returns:
        Current entropy coefficient
    """
    if step >= decay_steps:
        return final_coef

    decay_rate = (initial_coef - final_coef) / decay_steps
    return initial_coef - decay_rate * step


class EntropyRegularizer:
    """
    Manages entropy regularization for policy gradient training.

    Supports:
    - Fixed entropy coefficient
    - Linear decay schedule
    - Adaptive based on policy entropy
    """

    def __init__(
        self,
        initial_coef: float = 0.01,
        final_coef: float = 0.001,
        decay_steps: int = 100000,
        min_entropy: float = 0.1,
        adaptive: bool = False
    ):
        """
        Initialize entropy regularizer.

        Args:
            initial_coef: Starting coefficient
            final_coef: Final coefficient (for decay)
            decay_steps: Steps for linear decay
            min_entropy: Target minimum entropy (for adaptive)
            adaptive: Whether to adaptively adjust coefficient
        """
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.decay_steps = decay_steps
        self.min_entropy = min_entropy
        self.adaptive = adaptive

        self.current_coef = initial_coef
        self.step = 0

    def get_coefficient(self) -> float:
        """Get current entropy coefficient."""
        return self.current_coef

    def update(self, current_entropy: float = None) -> float:
        """
        Update entropy coefficient.

        Args:
            current_entropy: Current policy entropy (for adaptive mode)

        Returns:
            Updated coefficient
        """
        self.step += 1

        if self.adaptive and current_entropy is not None:
            # Adaptive: increase coef if entropy too low
            if current_entropy < self.min_entropy:
                self.current_coef = min(self.initial_coef, self.current_coef * 1.01)
            else:
                self.current_coef = max(self.final_coef, self.current_coef * 0.999)
        else:
            # Linear decay
            self.current_coef = entropy_schedule(
                self.step, self.initial_coef, self.final_coef, self.decay_steps
            )

        return self.current_coef


def max_entropy(n_actions: int) -> float:
    """
    Compute maximum possible entropy for given action space.

    Max entropy achieved by uniform distribution: H = log(n_actions)

    Args:
        n_actions: Number of possible actions

    Returns:
        Maximum entropy value
    """
    return np.log(n_actions)


def normalized_entropy(probs: np.ndarray) -> float:
    """
    Compute entropy normalized by maximum possible entropy.

    Returns value in [0, 1]:
    - 0: Deterministic policy
    - 1: Uniform random policy

    Args:
        probs: Action probabilities

    Returns:
        Normalized entropy [0, 1]
    """
    n_actions = len(probs)
    max_ent = max_entropy(n_actions)
    current_ent = compute_entropy(probs)
    return current_ent / max_ent if max_ent > 0 else 0


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Entropy Regularization Demo")
    print("=" * 60)

    # Example distributions
    n_actions = 4
    max_ent = max_entropy(n_actions)

    print(f"\nAction space: {n_actions} actions")
    print(f"Maximum entropy: {max_ent:.4f} (= log({n_actions}))")

    print("\n" + "-" * 40)
    print("Entropy for Different Distributions")
    print("-" * 40)

    distributions = {
        "Uniform [0.25, 0.25, 0.25, 0.25]": np.array([0.25, 0.25, 0.25, 0.25]),
        "Slightly biased [0.4, 0.3, 0.2, 0.1]": np.array([0.4, 0.3, 0.2, 0.1]),
        "Very biased [0.7, 0.1, 0.1, 0.1]": np.array([0.7, 0.1, 0.1, 0.1]),
        "Near deterministic [0.97, 0.01, 0.01, 0.01]": np.array([0.97, 0.01, 0.01, 0.01]),
    }

    for name, probs in distributions.items():
        ent = compute_entropy(probs)
        norm_ent = normalized_entropy(probs)
        print(f"{name}")
        print(f"  Entropy: {ent:.4f} ({norm_ent*100:.1f}% of max)")

    # Batch entropy
    print("\n" + "-" * 40)
    print("Batch Entropy Computation")
    print("-" * 40)

    batch_probs = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.1, 0.1, 0.1],
        [0.97, 0.01, 0.01, 0.01],
    ])

    batch_ent = compute_batch_entropy(batch_probs)
    print(f"Batch shape: {batch_probs.shape}")
    print(f"Entropies: {np.round(batch_ent, 4)}")
    print(f"Mean entropy: {np.mean(batch_ent):.4f}")

    # Entropy-regularized loss
    print("\n" + "-" * 40)
    print("Entropy-Regularized Policy Gradient Loss")
    print("-" * 40)

    log_probs = np.array([-1.0, -0.5, -2.0, -0.3, -1.5])  # Log probs of taken actions
    advantages = np.array([1.0, -0.5, 2.0, 0.5, -1.0])     # Advantages
    probs_batch = np.array([
        [0.4, 0.3, 0.2, 0.1],
        [0.6, 0.2, 0.1, 0.1],
        [0.3, 0.3, 0.2, 0.2],
        [0.7, 0.1, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25],
    ])

    for ent_coef in [0.0, 0.01, 0.1]:
        total, policy, entropy = entropy_regularized_loss(
            log_probs, advantages, probs_batch, ent_coef
        )
        print(f"entropy_coef={ent_coef:.2f}:")
        print(f"  Policy loss: {policy:.4f}")
        print(f"  Mean entropy: {entropy:.4f}")
        print(f"  Total loss: {total:.4f}")

    # Entropy decay schedule
    print("\n" + "-" * 40)
    print("Entropy Coefficient Schedule")
    print("-" * 40)

    steps = [0, 25000, 50000, 75000, 100000, 150000]
    for step in steps:
        coef = entropy_schedule(step, initial_coef=0.01, final_coef=0.001, decay_steps=100000)
        print(f"Step {step:6d}: coefficient = {coef:.4f}")

    # EntropyRegularizer class
    print("\n" + "-" * 40)
    print("EntropyRegularizer Class Demo")
    print("-" * 40)

    regularizer = EntropyRegularizer(
        initial_coef=0.01, final_coef=0.001, decay_steps=1000
    )

    print("Linear decay schedule:")
    for _ in range(5):
        for _ in range(200):
            regularizer.update()
        print(f"  Step {regularizer.step:4d}: coef = {regularizer.get_coefficient():.5f}")

    # Adaptive mode
    print("\nAdaptive mode:")
    adaptive_reg = EntropyRegularizer(
        initial_coef=0.01, final_coef=0.001, min_entropy=0.5, adaptive=True
    )

    entropies = [1.2, 0.8, 0.4, 0.3, 0.2, 0.6, 0.8, 1.0]  # Simulated policy entropies
    for ent in entropies:
        coef = adaptive_reg.update(current_entropy=ent)
        print(f"  Entropy={ent:.1f} → coef={coef:.5f}")
