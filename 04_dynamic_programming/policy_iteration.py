"""
Policy Iteration Algorithm for Finite MDPs

This module implements policy iteration with separate policy evaluation
and policy improvement steps.

Mathematical Foundation:
    Policy iteration alternates between:

    1. Policy Evaluation: Compute V^π by solving the Bellman expectation equation
       V^π(s) = sum_a π(a|s) sum_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]

    2. Policy Improvement: Update policy to be greedy w.r.t. V^π
       π'(s) = argmax_a sum_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]

    Convergence is guaranteed in finite iterations due to the policy
    improvement theorem and finite policy space.
"""

import numpy as np
from typing import Tuple, Optional


def policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    theta: float = 1e-8,
    max_iterations: int = 1000,
    policy_init: Optional[np.ndarray] = None,
    eval_method: str = "iterative"
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute optimal value function and policy using policy iteration.

    Policy iteration alternates between:
    1. Policy Evaluation: Compute V^π for current policy π
    2. Policy Improvement: Update π to be greedy w.r.t. V^π

    Parameters
    ----------
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probability matrix.
        P[s, a, s'] = P(s' | s, a).

    R : np.ndarray, shape (n_states, n_actions, n_states)
        Reward matrix.
        R[s, a, s'] = immediate reward for transition (s, a) -> s'.

    gamma : float
        Discount factor in [0, 1).

    theta : float, default=1e-8
        Convergence threshold for policy evaluation.

    max_iterations : int, default=1000
        Maximum number of policy iteration cycles.

    policy_init : np.ndarray, optional, shape (n_states,), dtype=int
        Initial policy. If None, initialized to action 0 for all states.

    eval_method : str, default="iterative"
        Method for policy evaluation:
        - "iterative": Iterative policy evaluation (Bellman backups)
        - "direct": Direct solution via matrix inversion (O(n^3))

    Returns
    -------
    V : np.ndarray, shape (n_states,)
        Optimal value function V*(s).

    policy : np.ndarray, shape (n_states,), dtype=int
        Optimal deterministic policy.

    iterations : int
        Number of policy iteration cycles.

    Examples
    --------
    >>> P = np.array([
    ...     [[0.9, 0.1], [0.1, 0.9]],
    ...     [[0.5, 0.5], [0.5, 0.5]]
    ... ])
    >>> R = np.array([
    ...     [[0, 1], [0, 0]],
    ...     [[1, 0], [0, 1]]
    ... ])
    >>> V, policy, iters = policy_iteration(P, R, gamma=0.9)
    """
    _validate_inputs(P, R, gamma, theta)

    n_states, n_actions, _ = P.shape

    # Initialize policy
    if policy_init is None:
        policy = np.zeros(n_states, dtype=int)
    else:
        policy = policy_init.copy()

    for iteration in range(max_iterations):
        # Policy Evaluation
        if eval_method == "direct":
            V = policy_evaluation_direct(P, R, gamma, policy)
        else:
            V = policy_evaluation_iterative(P, R, gamma, policy, theta)

        # Policy Improvement
        new_policy, policy_stable = policy_improvement(V, P, R, gamma)

        if policy_stable:
            # Policy has converged
            return V, policy, iteration + 1

        policy = new_policy

    # Return best found if max iterations reached
    return V, policy, max_iterations


def policy_evaluation_iterative(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    policy: np.ndarray,
    theta: float = 1e-8,
    max_iterations: int = 10000
) -> np.ndarray:
    """
    Compute V^π using iterative policy evaluation.

    Applies the Bellman expectation operator T^π iteratively:
        V_{k+1}(s) = sum_{s'} P(s'|s,π(s)) [R(s,π(s),s') + γV_k(s')]

    Parameters
    ----------
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R : np.ndarray, shape (n_states, n_actions, n_states)
        Rewards.
    gamma : float
        Discount factor.
    policy : np.ndarray, shape (n_states,), dtype=int
        Deterministic policy to evaluate.
    theta : float
        Convergence threshold.
    max_iterations : int
        Maximum iterations.

    Returns
    -------
    V : np.ndarray, shape (n_states,)
        Value function V^π.
    """
    n_states = P.shape[0]
    V = np.zeros(n_states)

    for _ in range(max_iterations):
        V_new = _bellman_expectation_update(V, P, R, gamma, policy)

        delta = np.max(np.abs(V_new - V))
        V = V_new

        if delta < theta:
            break

    return V


def policy_evaluation_direct(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    policy: np.ndarray
) -> np.ndarray:
    """
    Compute V^π by direct matrix inversion.

    Solves: V^π = (I - γP^π)^{-1} r^π

    where P^π[s,s'] = P(s'|s,π(s)) and r^π[s] = sum_{s'} P(s'|s,π(s)) R(s,π(s),s')

    Parameters
    ----------
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R : np.ndarray, shape (n_states, n_actions, n_states)
        Rewards.
    gamma : float
        Discount factor.
    policy : np.ndarray, shape (n_states,), dtype=int
        Deterministic policy to evaluate.

    Returns
    -------
    V : np.ndarray, shape (n_states,)
        Value function V^π.
    """
    n_states = P.shape[0]

    # Build transition matrix P^π
    # P_pi[s, s'] = P(s' | s, π(s))
    P_pi = np.zeros((n_states, n_states))
    for s in range(n_states):
        P_pi[s, :] = P[s, policy[s], :]

    # Build reward vector r^π
    # r_pi[s] = sum_{s'} P(s'|s,π(s)) * R(s,π(s),s')
    r_pi = np.zeros(n_states)
    for s in range(n_states):
        r_pi[s] = np.sum(P[s, policy[s], :] * R[s, policy[s], :])

    # Solve (I - γP^π) V = r^π
    A = np.eye(n_states) - gamma * P_pi
    V = np.linalg.solve(A, r_pi)

    return V


def _bellman_expectation_update(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    policy: np.ndarray
) -> np.ndarray:
    """
    Apply one iteration of the Bellman expectation operator for policy π.

    V_{k+1}(s) = sum_{s'} P(s'|s,π(s)) [R(s,π(s),s') + γV_k(s')]

    Parameters
    ----------
    V : np.ndarray, shape (n_states,)
        Current value estimate.
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R : np.ndarray, shape (n_states, n_actions, n_states)
        Rewards.
    gamma : float
        Discount factor.
    policy : np.ndarray, shape (n_states,), dtype=int
        Current policy.

    Returns
    -------
    V_new : np.ndarray, shape (n_states,)
        Updated value function.
    """
    n_states = P.shape[0]
    V_new = np.zeros(n_states)

    for s in range(n_states):
        a = policy[s]
        # V(s) = sum_{s'} P(s'|s,a) * [R(s,a,s') + γV(s')]
        V_new[s] = np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))

    return V_new


def policy_improvement(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float
) -> Tuple[np.ndarray, bool]:
    """
    Perform policy improvement: make policy greedy w.r.t. V.

    π'(s) = argmax_a sum_{s'} P(s'|s,a) [R(s,a,s') + γV(s')]

    Parameters
    ----------
    V : np.ndarray, shape (n_states,)
        Current value function.
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R : np.ndarray, shape (n_states, n_actions, n_states)
        Rewards.
    gamma : float
        Discount factor.

    Returns
    -------
    new_policy : np.ndarray, shape (n_states,), dtype=int
        Improved policy (greedy w.r.t. V).
    policy_stable : bool
        True if V is already optimal (greedy policy unchanged).
    """
    n_states, n_actions, _ = P.shape

    # Compute Q-values for all state-action pairs
    # Q[s, a] = sum_{s'} P(s'|s,a) * [R(s,a,s') + γV(s')]
    expected_reward = np.einsum('sas,sas->sa', P, R)
    expected_future_value = gamma * np.einsum('sas,s->sa', P, V)
    Q = expected_reward + expected_future_value

    # Extract greedy policy
    new_policy = np.argmax(Q, axis=1)

    # Check if policy is already greedy (i.e., V = V*)
    # Policy is stable if max_a Q(s,a) == V(s) for all s
    # Equivalently: V(s) == Q(s, π(s)) and π'(s) achieves the max
    max_Q = np.max(Q, axis=1)
    policy_stable = np.allclose(V, max_Q)

    return new_policy, policy_stable


def compute_q_from_v(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Compute Q-values from a value function V.

    Q(s, a) = sum_{s'} P(s'|s,a) [R(s,a,s') + γV(s')]

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
    """Validate input parameters."""
    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a numpy array")
    if not isinstance(R, np.ndarray):
        raise TypeError("R must be a numpy array")

    if P.ndim != 3:
        raise ValueError(f"P must be 3-dimensional, got {P.ndim}")
    if R.ndim != 3:
        raise ValueError(f"R must be 3-dimensional, got {R.ndim}")
    if P.shape != R.shape:
        raise ValueError(f"P and R must have same shape: {P.shape} vs {R.shape}")

    n_states, n_actions, n_states2 = P.shape
    if n_states != n_states2:
        raise ValueError("P must have shape (n_states, n_actions, n_states)")

    if np.any(P < 0):
        raise ValueError("Transition probabilities must be non-negative")

    row_sums = P.sum(axis=2)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("Transition probabilities must sum to 1")

    if not 0 <= gamma < 1:
        raise ValueError(f"gamma must be in [0, 1), got {gamma}")

    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Policy Iteration Demo: Simple 3-State MDP")
    print("=" * 60)

    # Same MDP as value_iteration.py demo
    n_states = 3
    n_actions = 2

    P = np.zeros((n_states, n_actions, n_states))
    P[0, 0, 0] = 1.0  # State 0, action left: stay
    P[0, 1, 1] = 1.0  # State 0, action right: go to 1
    P[1, 0, 0] = 1.0  # State 1, action left: go to 0
    P[1, 1, 2] = 1.0  # State 1, action right: go to terminal
    P[2, 0, 2] = 1.0  # Terminal: absorbing
    P[2, 1, 2] = 1.0

    R = np.zeros((n_states, n_actions, n_states))
    R[1, 1, 2] = 1.0  # Reward +1 for reaching terminal

    gamma = 0.9

    print(f"\nMDP Configuration:")
    print(f"  States: {n_states}")
    print(f"  Actions: {n_actions} (0=left, 1=right)")
    print(f"  Discount factor: {gamma}")
    print(f"  Terminal state: 2 (reward +1)")

    # Run policy iteration with iterative evaluation
    print("\n--- Iterative Policy Evaluation ---")
    V_iter, policy_iter, iters_iter = policy_iteration(
        P, R, gamma, theta=1e-8, eval_method="iterative"
    )
    print(f"Converged in {iters_iter} policy iteration cycles")

    # Run policy iteration with direct (matrix) evaluation
    print("\n--- Direct (Matrix) Policy Evaluation ---")
    V_direct, policy_direct, iters_direct = policy_iteration(
        P, R, gamma, theta=1e-8, eval_method="direct"
    )
    print(f"Converged in {iters_direct} policy iteration cycles")

    # Results
    print(f"\nOptimal Value Function V*:")
    for s in range(n_states):
        print(f"  V*({s}) = {V_iter[s]:.6f}")

    print(f"\nOptimal Policy pi*:")
    action_names = ["left", "right"]
    for s in range(n_states):
        print(f"  pi*({s}) = {policy_iter[s]} ({action_names[policy_iter[s]]})")

    # Verify both methods agree
    print(f"\nVerification:")
    print(f"  V (iterative) == V (direct): {np.allclose(V_iter, V_direct)}")
    print(f"  Policy agreement: {np.array_equal(policy_iter, policy_direct)}")

    # Compare with analytical solution
    print("\n" + "=" * 60)
    print("Expected values (analytical):")
    print("  V*(0) = 0.9 (go right to state 1)")
    print("  V*(1) = 1.0 (go right to terminal, get +1)")
    print("  V*(2) = 0.0 (terminal state)")
    print("=" * 60)
