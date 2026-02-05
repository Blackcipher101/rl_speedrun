"""
Advantage Estimation for Actor-Critic Methods

The advantage function A(s,a) = Q(s,a) - V(s) measures how much better
an action is compared to the average action in that state.

Using advantages instead of raw returns:
1. Reduces variance (baselines subtract average value)
2. Centers gradients (positive for good, negative for bad)
3. Enables credit assignment (identifies which actions helped)

Author: RL Speedrun
"""

import numpy as np
from typing import List, Tuple


def compute_returns(rewards: List[float], gamma: float = 0.99) -> np.ndarray:
    """
    Compute discounted returns G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}.

    Args:
        rewards: List of rewards [r_0, r_1, ..., r_T]
        gamma: Discount factor

    Returns:
        Array of returns [G_0, G_1, ..., G_T]
    """
    T = len(rewards)
    returns = np.zeros(T)

    # Compute backwards for efficiency
    G = 0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns


def compute_td_errors(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute TD errors: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t).

    TD error is a one-step estimate of the advantage.

    Args:
        rewards: Rewards received
        values: V(s_t) estimates
        next_values: V(s_{t+1}) estimates
        dones: Episode termination flags
        gamma: Discount factor

    Returns:
        TD errors (one-step advantages)
    """
    # Bootstrap from next state unless episode terminated
    bootstrap = gamma * next_values * (1 - dones.astype(np.float32))
    td_errors = rewards + bootstrap - values
    return td_errors


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE interpolates between:
    - TD (lambda=0): Low variance, high bias
    - MC (lambda=1): High variance, low bias

    A^GAE_t = sum_{l=0}^{infinity} (gamma*lambda)^l * delta_{t+l}

    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    Reference: Schulman et al., 2015 - "High-Dimensional Continuous Control
    Using Generalized Advantage Estimation"

    Args:
        rewards: Rewards [r_0, ..., r_T]
        values: Value estimates [V(s_0), ..., V(s_T)]
        next_values: Next state values [V(s_1), ..., V(s_{T+1})]
        dones: Done flags
        gamma: Discount factor
        gae_lambda: GAE parameter (0=TD, 1=MC)

    Returns:
        advantages: GAE advantages
        returns: Advantages + values (used as value targets)
    """
    T = len(rewards)
    advantages = np.zeros(T)

    # Compute TD errors
    td_errors = compute_td_errors(rewards, values, next_values, dones, gamma)

    # GAE: exponentially-weighted sum of TD errors
    gae = 0
    for t in reversed(range(T)):
        # Reset GAE at episode boundaries
        if dones[t]:
            gae = 0

        gae = td_errors[t] + gamma * gae_lambda * gae * (1 - dones[t])
        advantages[t] = gae

    # Returns = advantages + values (used as targets for value function)
    returns = advantages + values

    return advantages, returns


def compute_n_step_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    n_steps: int = 5
) -> np.ndarray:
    """
    Compute n-step returns.

    G_t^{(n)} = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1} + gamma^n*V(s_{t+n})

    Interpolates between TD(0) (n=1) and MC (n=infinity).

    Args:
        rewards: Rewards
        values: Value estimates (including V(s_T) for bootstrap)
        dones: Done flags
        gamma: Discount factor
        n_steps: Number of steps

    Returns:
        n-step returns
    """
    T = len(rewards)
    returns = np.zeros(T)

    for t in range(T):
        G = 0
        gamma_power = 1.0

        for k in range(n_steps):
            if t + k >= T:
                break

            G += gamma_power * rewards[t + k]
            gamma_power *= gamma

            # If episode ends, don't bootstrap
            if dones[t + k]:
                break
        else:
            # Bootstrap from value at t+n if we didn't hit episode end
            if t + n_steps < len(values):
                G += gamma_power * values[t + n_steps]

        returns[t] = G

    return returns


def normalize_advantages(advantages: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize advantages to have zero mean and unit variance.

    This helps with:
    - Consistent gradient magnitudes across episodes
    - Better credit assignment (clearly positive/negative)
    - Improved training stability

    Args:
        advantages: Raw advantages
        epsilon: Small constant for numerical stability

    Returns:
        Normalized advantages
    """
    mean = np.mean(advantages)
    std = np.std(advantages)
    return (advantages - mean) / (std + epsilon)


class AdvantageEstimator:
    """
    Unified interface for different advantage estimation methods.

    Supports:
    - 'mc': Monte Carlo returns (no bootstrapping)
    - 'td': TD error (one-step advantage)
    - 'nstep': N-step returns
    - 'gae': Generalized Advantage Estimation
    """

    def __init__(
        self,
        method: str = 'gae',
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_steps: int = 5,
        normalize: bool = True
    ):
        """
        Initialize advantage estimator.

        Args:
            method: 'mc', 'td', 'nstep', or 'gae'
            gamma: Discount factor
            gae_lambda: Lambda for GAE (ignored for other methods)
            n_steps: Steps for n-step returns (ignored for mc/td/gae)
            normalize: Whether to normalize advantages
        """
        self.method = method
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.normalize = normalize

    def compute(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages and returns using the configured method.

        Args:
            rewards: Rewards
            values: V(s_t) estimates
            next_values: V(s_{t+1}) estimates
            dones: Episode termination flags

        Returns:
            advantages: Estimated advantages
            returns: Value function targets
        """
        if self.method == 'mc':
            # Monte Carlo: full returns, no bootstrapping
            returns = compute_returns(rewards, self.gamma)
            advantages = returns - values

        elif self.method == 'td':
            # TD(0): one-step advantage
            advantages = compute_td_errors(rewards, values, next_values, dones, self.gamma)
            returns = advantages + values

        elif self.method == 'nstep':
            # N-step returns
            returns = compute_n_step_returns(rewards, values, dones, self.gamma, self.n_steps)
            advantages = returns - values

        elif self.method == 'gae':
            # Generalized Advantage Estimation
            advantages, returns = compute_gae(
                rewards, values, next_values, dones,
                self.gamma, self.gae_lambda
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

        if self.normalize:
            advantages = normalize_advantages(advantages)

        return advantages, returns


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Advantage Estimation Demo")
    print("=" * 60)

    # Simulate an episode
    np.random.seed(42)
    T = 10
    rewards = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 10])  # Big reward at end
    values = np.array([5, 5, 5, 5, 5, 3, 2, 3, 5, 8])     # Value estimates
    next_values = np.array([5, 5, 5, 5, 3, 2, 3, 5, 8, 0])  # V(s')
    dones = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])       # Episode ends at t=9

    gamma = 0.99

    print("\nRewards:", rewards)
    print("Values:", values)
    print("Dones:", dones)

    # Monte Carlo returns
    print("\n" + "-" * 40)
    print("Monte Carlo Returns")
    print("-" * 40)
    mc_returns = compute_returns(rewards, gamma)
    mc_advantages = mc_returns - values
    print(f"Returns:    {np.round(mc_returns, 2)}")
    print(f"Advantages: {np.round(mc_advantages, 2)}")

    # TD errors
    print("\n" + "-" * 40)
    print("TD Errors (One-Step Advantages)")
    print("-" * 40)
    td_errors = compute_td_errors(rewards, values, next_values, dones, gamma)
    print(f"TD errors: {np.round(td_errors, 2)}")

    # GAE
    print("\n" + "-" * 40)
    print("Generalized Advantage Estimation (lambda=0.95)")
    print("-" * 40)
    gae_advantages, gae_returns = compute_gae(
        rewards, values, next_values, dones, gamma, gae_lambda=0.95
    )
    print(f"GAE advantages: {np.round(gae_advantages, 2)}")
    print(f"GAE returns:    {np.round(gae_returns, 2)}")

    # Compare different lambda values
    print("\n" + "-" * 40)
    print("Effect of Lambda on GAE")
    print("-" * 40)
    for lam in [0.0, 0.5, 0.95, 1.0]:
        adv, _ = compute_gae(rewards, values, next_values, dones, gamma, gae_lambda=lam)
        print(f"lambda={lam:.2f}: {np.round(adv, 2)}")

    # Normalization
    print("\n" + "-" * 40)
    print("Advantage Normalization")
    print("-" * 40)
    raw_adv = gae_advantages
    norm_adv = normalize_advantages(raw_adv)
    print(f"Raw advantages:        {np.round(raw_adv, 2)}")
    print(f"Normalized advantages: {np.round(norm_adv, 2)}")
    print(f"  Mean: {np.mean(norm_adv):.6f}, Std: {np.std(norm_adv):.6f}")

    # AdvantageEstimator class demo
    print("\n" + "-" * 40)
    print("AdvantageEstimator Class")
    print("-" * 40)

    for method in ['mc', 'td', 'nstep', 'gae']:
        estimator = AdvantageEstimator(method=method, gamma=gamma, normalize=False)
        adv, ret = estimator.compute(rewards, values, next_values, dones)
        print(f"{method:5s}: adv_mean={np.mean(adv):6.2f}, adv_std={np.std(adv):6.2f}")
