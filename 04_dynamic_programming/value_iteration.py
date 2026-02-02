"""
Value Iteration Algorithm for Finite MDPs

This module implements value iteration using the Bellman optimality operator.
The implementation uses explicit full backups with no sampling.

Mathematical Foundation:
    Value iteration applies the Bellman optimality operator T* iteratively:

        V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V_k(s')]

    The operator T* is a gamma-contraction, guaranteeing convergence to V*.
"""

import numpy as np
from typing import Tuple, Optional


def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    theta: float = 1e-8,
    max_iterations: int = 10000,
    V_init: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute optimal value function and policy using value iteration.

    This implementation performs synchronous (full) backups using the
    Bellman optimality equation:

        V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V_k(s')]

    Parameters
    ----------
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probability matrix.
        P[s, a, s'] = P(s' | s, a) = probability of transitioning to s'
        given state s and action a.
        Must satisfy: P[s, a, :].sum() == 1 for all s, a.

    R : np.ndarray, shape (n_states, n_actions, n_states)
        Reward matrix.
        R[s, a, s'] = immediate reward for transition (s, a) -> s'.

    gamma : float
        Discount factor in [0, 1).
        For continuing tasks, must be strictly less than 1.

    theta : float, default=1e-8
        Convergence threshold.
        Algorithm stops when max|V_{k+1}(s) - V_k(s)| < theta.

    max_iterations : int, default=10000
        Maximum number of iterations.

    V_init : np.ndarray, optional, shape (n_states,)
        Initial value function. If None, initialized to zeros.

    Returns
    -------
    V : np.ndarray, shape (n_states,)
        Optimal value function V*(s) for each state.

    policy : np.ndarray, shape (n_states,), dtype=int
        Optimal deterministic policy.
        policy[s] = optimal action in state s.

    iterations : int
        Number of iterations until convergence.

    Raises
    ------
    ValueError
        If inputs have invalid shapes or values.

    Examples
    --------
    >>> # Simple 2-state, 2-action MDP
    >>> P = np.array([
    ...     [[0.9, 0.1], [0.1, 0.9]],  # State 0
    ...     [[0.5, 0.5], [0.5, 0.5]]   # State 1
    ... ])
    >>> R = np.array([
    ...     [[0, 1], [0, 0]],  # State 0
    ...     [[1, 0], [0, 1]]   # State 1
    ... ])
    >>> V, policy, iters = value_iteration(P, R, gamma=0.9)
    """
    # Validate inputs
    _validate_inputs(P, R, gamma, theta)

    n_states, n_actions, _ = P.shape

    # Initialize value function
    if V_init is None:
        V = np.zeros(n_states)
    else:
        V = V_init.copy()

    # Main iteration loop
    for iteration in range(max_iterations):
        V_new = _bellman_optimality_update(V, P, R, gamma)

        # Check convergence: ||V_{k+1} - V_k||_infinity < theta
        delta = np.max(np.abs(V_new - V))
        V = V_new

        if delta < theta:
            break

    # Extract greedy policy from V*
    policy = _extract_greedy_policy(V, P, R, gamma)

    return V, policy, iteration + 1


def _bellman_optimality_update(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Apply one iteration of the Bellman optimality operator.

    Computes: V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V_k(s')]

    Parameters
    ----------
    V : np.ndarray, shape (n_states,)
        Current value function estimate.
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R : np.ndarray, shape (n_states, n_actions, n_states)
        Rewards.
    gamma : float
        Discount factor.

    Returns
    -------
    V_new : np.ndarray, shape (n_states,)
        Updated value function after one Bellman backup.
    """
    n_states, n_actions, _ = P.shape

    # Compute Q(s, a) for all state-action pairs
    # Q[s, a] = sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
    #
    # Using einsum for clarity:
    # Q[s,a] = sum_s' P[s,a,s'] * (R[s,a,s'] + gamma * V[s'])

    # Expected immediate reward: E[R | s, a]
    expected_reward = np.einsum('sas,sas->sa', P, R)

    # Expected future value: gamma * E[V(s') | s, a]
    expected_future_value = gamma * np.einsum('sas,s->sa', P, V)

    # Q-values
    Q = expected_reward + expected_future_value

    # V(s) = max_a Q(s, a)
    V_new = np.max(Q, axis=1)

    return V_new


def _extract_greedy_policy(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Extract the greedy policy with respect to a value function.

    Computes: pi(s) = argmax_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V(s')]

    Parameters
    ----------
    V : np.ndarray, shape (n_states,)
        Value function (typically V*).
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R : np.ndarray, shape (n_states, n_actions, n_states)
        Rewards.
    gamma : float
        Discount factor.

    Returns
    -------
    policy : np.ndarray, shape (n_states,), dtype=int
        Greedy policy where policy[s] is the action maximizing Q(s, a).
    """
    # Compute Q-values
    expected_reward = np.einsum('sas,sas->sa', P, R)
    expected_future_value = gamma * np.einsum('sas,s->sa', P, V)
    Q = expected_reward + expected_future_value

    # Greedy action selection
    policy = np.argmax(Q, axis=1)

    return policy


def compute_q_from_v(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Compute action-value function Q from state-value function V.

    Q(s, a) = sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V(s')]

    Parameters
    ----------
    V : np.ndarray, shape (n_states,)
        State-value function.
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R : np.ndarray, shape (n_states, n_actions, n_states)
        Rewards.
    gamma : float
        Discount factor.

    Returns
    -------
    Q : np.ndarray, shape (n_states, n_actions)
        Action-value function.
    """
    expected_reward = np.einsum('sas,sas->sa', P, R)
    expected_future_value = gamma * np.einsum('sas,s->sa', P, V)
    return expected_reward + expected_future_value


def _validate_inputs(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    theta: float
) -> None:
    """Validate input parameters for value iteration."""
    # Check types
    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a numpy array")
    if not isinstance(R, np.ndarray):
        raise TypeError("R must be a numpy array")

    # Check shapes
    if P.ndim != 3:
        raise ValueError(f"P must be 3-dimensional, got {P.ndim}")
    if R.ndim != 3:
        raise ValueError(f"R must be 3-dimensional, got {R.ndim}")
    if P.shape != R.shape:
        raise ValueError(f"P and R must have same shape: {P.shape} vs {R.shape}")

    n_states, n_actions, n_states2 = P.shape
    if n_states != n_states2:
        raise ValueError("P must have shape (n_states, n_actions, n_states)")

    # Check transition probabilities
    if np.any(P < 0):
        raise ValueError("Transition probabilities must be non-negative")

    row_sums = P.sum(axis=2)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("Transition probabilities must sum to 1 for each (s, a)")

    # Check gamma
    if not 0 <= gamma < 1:
        raise ValueError(f"gamma must be in [0, 1), got {gamma}")

    # Check theta
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")


# ============================================================================
# Demonstration with a simple MDP
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Value Iteration Demo: Simple 3-State MDP")
    print("=" * 60)

    # Define a simple MDP
    # States: 0, 1, 2 (state 2 is terminal)
    # Actions: 0 (left), 1 (right)

    n_states = 3
    n_actions = 2

    # Transition probabilities P[s, a, s']
    P = np.zeros((n_states, n_actions, n_states))

    # From state 0:
    # - Action 0 (left): stay in 0
    # - Action 1 (right): go to 1
    P[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0

    # From state 1:
    # - Action 0 (left): go to 0
    # - Action 1 (right): go to 2 (terminal)
    P[1, 0, 0] = 1.0
    P[1, 1, 2] = 1.0

    # State 2 is terminal (absorbing)
    P[2, 0, 2] = 1.0
    P[2, 1, 2] = 1.0

    # Rewards R[s, a, s']
    R = np.zeros((n_states, n_actions, n_states))

    # Reward of +1 for reaching terminal state 2
    R[1, 1, 2] = 1.0

    gamma = 0.9

    print(f"\nMDP Configuration:")
    print(f"  States: {n_states}")
    print(f"  Actions: {n_actions} (0=left, 1=right)")
    print(f"  Discount factor: {gamma}")
    print(f"  Terminal state: 2 (reward +1)")

    # Run value iteration
    V, policy, iterations = value_iteration(P, R, gamma, theta=1e-8)

    print(f"\nResults after {iterations} iterations:")
    print(f"\nOptimal Value Function V*:")
    for s in range(n_states):
        print(f"  V*({s}) = {V[s]:.6f}")

    print(f"\nOptimal Policy pi*:")
    action_names = ["left", "right"]
    for s in range(n_states):
        print(f"  pi*({s}) = {policy[s]} ({action_names[policy[s]]})")

    # Verify: compute Q-values
    Q = compute_q_from_v(V, P, R, gamma)
    print(f"\nQ-values (for verification):")
    for s in range(n_states):
        for a in range(n_actions):
            print(f"  Q*({s}, {a}) = {Q[s, a]:.6f}")

    # Expected values:
    # V*(2) = 0 (terminal)
    # V*(1) = max(0 + 0.9*V*(0), 1 + 0.9*V*(2)) = max(0.9*V*(0), 1)
    # V*(0) = max(0 + 0.9*V*(0), 0 + 0.9*V*(1))
    #
    # If optimal is to go right from 0 and 1:
    # V*(1) = 1 (go right to terminal, get +1)
    # V*(0) = 0.9 * V*(1) = 0.9 * 1 = 0.9 (go right to 1)

    print("\n" + "=" * 60)
    print("Expected values (analytical):")
    print("  V*(0) = 0.9 (go right to state 1)")
    print("  V*(1) = 1.0 (go right to terminal, get +1)")
    print("  V*(2) = 0.0 (terminal state)")
    print("=" * 60)
