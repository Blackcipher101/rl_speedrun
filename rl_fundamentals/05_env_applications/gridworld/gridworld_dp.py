"""
GridWorld Dynamic Programming Solver

A 4x4 gridworld with deterministic transitions, solved using value iteration
and policy iteration.

Grid Layout:
    +----+----+----+----+
    | G  |    |    |    |   Row 0
    +----+----+----+----+
    |    |    |    |    |   Row 1
    +----+----+----+----+
    |    |    |    |    |   Row 2
    +----+----+----+----+
    |    |    |    | G  |   Row 3
    +----+----+----+----+
      0    1    2    3

Terminal states: (0,0) and (3,3)
Reward: -1 for each step (except from terminal states)
Transitions: Deterministic
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ============================================================================
# GridWorld MDP Construction
# ============================================================================

class GridWorld:
    """
    4x4 GridWorld environment with deterministic transitions.

    Attributes
    ----------
    grid_size : int
        Size of the grid (4 for 4x4).
    n_states : int
        Total number of states (16).
    n_actions : int
        Number of actions (4: UP, DOWN, LEFT, RIGHT).
    terminal_states : set
        Set of terminal state indices.
    P : np.ndarray
        Transition probability matrix P[s, a, s'].
    R : np.ndarray
        Reward matrix R[s, a, s'].
    """

    # Action definitions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    ACTION_ARROWS = ['↑', '↓', '←', '→']

    def __init__(self, grid_size: int = 4):
        """
        Initialize GridWorld.

        Parameters
        ----------
        grid_size : int
            Size of the square grid.
        """
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4

        # Terminal states: top-left (0,0) and bottom-right (n-1, n-1)
        self.terminal_states = {0, self.n_states - 1}

        # Build transition and reward matrices
        self.P, self.R = self._build_mdp()

    def _state_to_pos(self, state: int) -> tuple:
        """Convert state index to (row, col) position."""
        return divmod(state, self.grid_size)

    def _pos_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) position to state index."""
        return row * self.grid_size + col

    def _build_mdp(self) -> tuple:
        """
        Construct the MDP transition and reward matrices.

        Returns
        -------
        P : np.ndarray, shape (n_states, n_actions, n_states)
            Transition probabilities.
        R : np.ndarray, shape (n_states, n_actions, n_states)
            Rewards.
        """
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions, self.n_states))

        for s in range(self.n_states):
            row, col = self._state_to_pos(s)

            for a in range(self.n_actions):
                if s in self.terminal_states:
                    # Terminal states are absorbing with zero reward
                    P[s, a, s] = 1.0
                    R[s, a, s] = 0.0
                else:
                    # Compute next state based on action
                    next_row, next_col = row, col

                    if a == self.UP:
                        next_row = max(0, row - 1)
                    elif a == self.DOWN:
                        next_row = min(self.grid_size - 1, row + 1)
                    elif a == self.LEFT:
                        next_col = max(0, col - 1)
                    elif a == self.RIGHT:
                        next_col = min(self.grid_size - 1, col + 1)

                    next_state = self._pos_to_state(next_row, next_col)

                    # Deterministic transition
                    P[s, a, next_state] = 1.0

                    # Reward of -1 for each step
                    R[s, a, next_state] = -1.0

        return P, R


# ============================================================================
# Dynamic Programming Algorithms
# ============================================================================

def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float,
                    theta: float = 1e-8, max_iter: int = 10000) -> tuple:
    """
    Value iteration algorithm.

    V_{k+1}(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V_k(s')]

    Parameters
    ----------
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R : np.ndarray, shape (n_states, n_actions, n_states)
        Rewards.
    gamma : float
        Discount factor.
    theta : float
        Convergence threshold.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    V : np.ndarray, shape (n_states,)
        Optimal value function.
    policy : np.ndarray, shape (n_states,), dtype=int
        Optimal policy.
    iterations : int
        Number of iterations.
    """
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)

    for iteration in range(max_iter):
        V_new = np.zeros(n_states)

        for s in range(n_states):
            q_values = []
            for a in range(n_actions):
                q = np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))
                q_values.append(q)
            V_new[s] = max(q_values)

        delta = np.max(np.abs(V_new - V))
        V = V_new

        if delta < theta:
            break

    # Extract greedy policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = [np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))
                    for a in range(n_actions)]
        policy[s] = np.argmax(q_values)

    return V, policy, iteration + 1


def policy_evaluation(P: np.ndarray, R: np.ndarray, gamma: float,
                      policy: np.ndarray, theta: float = 1e-8,
                      max_iter: int = 10000) -> np.ndarray:
    """
    Iterative policy evaluation.

    V_{k+1}(s) = sum_{s'} P(s'|s,π(s)) [R(s,π(s),s') + gamma * V_k(s')]

    Parameters
    ----------
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R : np.ndarray, shape (n_states, n_actions, n_states)
        Rewards.
    gamma : float
        Discount factor.
    policy : np.ndarray, shape (n_states,), dtype=int
        Policy to evaluate.
    theta : float
        Convergence threshold.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    V : np.ndarray, shape (n_states,)
        Value function V^π.
    """
    n_states = P.shape[0]
    V = np.zeros(n_states)

    for _ in range(max_iter):
        V_new = np.zeros(n_states)

        for s in range(n_states):
            a = policy[s]
            V_new[s] = np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))

        delta = np.max(np.abs(V_new - V))
        V = V_new

        if delta < theta:
            break

    return V


def policy_iteration(P: np.ndarray, R: np.ndarray, gamma: float,
                     theta: float = 1e-8, max_iter: int = 1000) -> tuple:
    """
    Policy iteration algorithm.

    Alternates between policy evaluation and policy improvement.

    Parameters
    ----------
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probabilities.
    R : np.ndarray, shape (n_states, n_actions, n_states)
        Rewards.
    gamma : float
        Discount factor.
    theta : float
        Convergence threshold for policy evaluation.
    max_iter : int
        Maximum policy iteration cycles.

    Returns
    -------
    V : np.ndarray, shape (n_states,)
        Optimal value function.
    policy : np.ndarray, shape (n_states,), dtype=int
        Optimal policy.
    iterations : int
        Number of policy iteration cycles.
    """
    n_states, n_actions, _ = P.shape

    # Initialize with arbitrary policy
    policy = np.zeros(n_states, dtype=int)

    for iteration in range(max_iter):
        # Policy Evaluation
        V = policy_evaluation(P, R, gamma, policy, theta)

        # Policy Improvement
        policy_stable = True
        new_policy = np.zeros(n_states, dtype=int)

        for s in range(n_states):
            q_values = [np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))
                        for a in range(n_actions)]
            new_policy[s] = np.argmax(q_values)

            if new_policy[s] != policy[s]:
                policy_stable = False

        policy = new_policy

        if policy_stable:
            break

    return V, policy, iteration + 1


# ============================================================================
# Visualization Functions
# ============================================================================

def print_value_function(V: np.ndarray, grid_size: int = 4) -> None:
    """Print value function as a grid."""
    print("\nOptimal Value Function V*(s):")
    print("+" + "-------+" * grid_size)
    for row in range(grid_size):
        row_str = "|"
        for col in range(grid_size):
            s = row * grid_size + col
            row_str += f" {V[s]:5.2f} |"
        print(row_str)
        print("+" + "-------+" * grid_size)


def print_policy(policy: np.ndarray, grid_size: int = 4,
                 terminal_states: set = None) -> None:
    """Print policy as a grid with arrows."""
    if terminal_states is None:
        terminal_states = {0, 15}

    arrows = ['↑', '↓', '←', '→']

    print("\nOptimal Policy π*(s):")
    print("+" + "---+" * grid_size)
    for row in range(grid_size):
        row_str = "|"
        for col in range(grid_size):
            s = row * grid_size + col
            if s in terminal_states:
                row_str += " * |"  # Terminal state
            else:
                row_str += f" {arrows[policy[s]]} |"
        print(row_str)
        print("+" + "---+" * grid_size)


def print_policy_detailed(policy: np.ndarray, grid_size: int = 4,
                          terminal_states: set = None) -> None:
    """Print policy with action names."""
    if terminal_states is None:
        terminal_states = {0, 15}

    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    print("\nPolicy Details:")
    for row in range(grid_size):
        for col in range(grid_size):
            s = row * grid_size + col
            if s in terminal_states:
                print(f"  State {s:2d} ({row},{col}): TERMINAL")
            else:
                print(f"  State {s:2d} ({row},{col}): {action_names[policy[s]]}")


def visualize_solution(V: np.ndarray, policy: np.ndarray, grid_size: int = 4,
                       terminal_states: set = None, save_path: str = None) -> None:
    """
    Create a visualization of the value function and policy.

    Parameters
    ----------
    V : np.ndarray
        Value function.
    policy : np.ndarray
        Policy (action indices).
    grid_size : int
        Size of the grid.
    terminal_states : set
        Set of terminal state indices.
    save_path : str, optional
        Path to save the figure.
    """
    if terminal_states is None:
        terminal_states = {0, grid_size * grid_size - 1}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Reshape value function to grid
    V_grid = V.reshape(grid_size, grid_size)

    # ===== Left plot: Value Function Heatmap =====
    ax1 = axes[0]
    im = ax1.imshow(V_grid, cmap='RdYlGn', aspect='equal')

    # Add value annotations
    for i in range(grid_size):
        for j in range(grid_size):
            s = i * grid_size + j
            color = 'white' if abs(V_grid[i, j]) > 1.5 else 'black'
            if s in terminal_states:
                ax1.text(j, i, f'G\n{V_grid[i, j]:.1f}', ha='center', va='center',
                        fontsize=12, fontweight='bold', color=color)
            else:
                ax1.text(j, i, f'{V_grid[i, j]:.1f}', ha='center', va='center',
                        fontsize=12, color=color)

    ax1.set_xticks(range(grid_size))
    ax1.set_yticks(range(grid_size))
    ax1.set_xticklabels(range(grid_size))
    ax1.set_yticklabels(range(grid_size))
    ax1.set_xlabel('Column', fontsize=12)
    ax1.set_ylabel('Row', fontsize=12)
    ax1.set_title('Value Function V*(s)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='V*(s)')

    # Add grid lines
    for i in range(grid_size + 1):
        ax1.axhline(i - 0.5, color='black', linewidth=1)
        ax1.axvline(i - 0.5, color='black', linewidth=1)

    # ===== Right plot: Policy with arrows =====
    ax2 = axes[1]

    # Create a grid showing terminal states
    policy_grid = np.zeros((grid_size, grid_size))
    for s in terminal_states:
        i, j = divmod(s, grid_size)
        policy_grid[i, j] = 1

    ax2.imshow(policy_grid, cmap='Greens', alpha=0.3, aspect='equal')

    # Arrow directions: UP, DOWN, LEFT, RIGHT
    arrow_dx = [0, 0, -0.3, 0.3]
    arrow_dy = [-0.3, 0.3, 0, 0]
    arrow_symbols = ['↑', '↓', '←', '→']

    for i in range(grid_size):
        for j in range(grid_size):
            s = i * grid_size + j
            if s in terminal_states:
                ax2.text(j, i, '★', ha='center', va='center', fontsize=24,
                        color='green', fontweight='bold')
            else:
                action = policy[s]
                ax2.annotate('', xy=(j + arrow_dx[action], i + arrow_dy[action]),
                            xytext=(j, i),
                            arrowprops=dict(arrowstyle='->', color='darkblue',
                                          lw=2, mutation_scale=20))
                ax2.text(j, i + 0.35, arrow_symbols[action], ha='center', va='center',
                        fontsize=10, color='darkblue')

    ax2.set_xticks(range(grid_size))
    ax2.set_yticks(range(grid_size))
    ax2.set_xticklabels(range(grid_size))
    ax2.set_yticklabels(range(grid_size))
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)
    ax2.set_title('Optimal Policy π*(s)', fontsize=14, fontweight='bold')
    ax2.set_xlim(-0.5, grid_size - 0.5)
    ax2.set_ylim(grid_size - 0.5, -0.5)

    # Add grid lines
    for i in range(grid_size + 1):
        ax2.axhline(i - 0.5, color='black', linewidth=1)
        ax2.axvline(i - 0.5, color='black', linewidth=1)

    # Legend
    star_patch = mpatches.Patch(color='green', alpha=0.5, label='Terminal (Goal)')
    ax2.legend(handles=[star_patch], loc='upper right')

    plt.suptitle('GridWorld Solution (4x4, Deterministic)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nVisualization saved to: {save_path}")

    plt.close()


# ============================================================================
# Main Solver
# ============================================================================

def solve_gridworld(gamma: float = 1.0, theta: float = 1e-8) -> None:
    """
    Solve GridWorld using dynamic programming methods.

    Parameters
    ----------
    gamma : float
        Discount factor. Default is 1.0 (undiscounted episodic).
    theta : float
        Convergence threshold.
    """
    print("=" * 60)
    print("GridWorld Dynamic Programming Solver")
    print("=" * 60)

    # Create environment
    env = GridWorld(grid_size=4)

    print(f"\nEnvironment Configuration:")
    print(f"  Grid size: {env.grid_size}x{env.grid_size}")
    print(f"  States: {env.n_states}")
    print(f"  Actions: {env.n_actions} (UP, DOWN, LEFT, RIGHT)")
    print(f"  Terminal states: {env.terminal_states}")
    print(f"  Discount factor (γ): {gamma}")

    # Note: For undiscounted (gamma=1), we need gamma < 1 for convergence
    # guarantees in infinite horizon. However, GridWorld is episodic
    # (terminates), so gamma=1 works. For the algorithm, we use gamma=0.9999
    # when gamma=1 to ensure numerical stability.
    effective_gamma = 0.9999 if gamma == 1.0 else gamma

    print("\n" + "-" * 60)
    print("Solving with Value Iteration...")
    print("-" * 60)

    V_vi, policy_vi, iters_vi = value_iteration(
        env.P, env.R, effective_gamma, theta=theta
    )

    print(f"Converged in {iters_vi} iterations")
    print_value_function(V_vi, env.grid_size)
    print_policy(policy_vi, env.grid_size, env.terminal_states)

    print("\n" + "-" * 60)
    print("Solving with Policy Iteration...")
    print("-" * 60)

    V_pi, policy_pi, iters_pi = policy_iteration(
        env.P, env.R, effective_gamma, theta=theta
    )

    print(f"Converged in {iters_pi} policy iteration cycles")
    print_value_function(V_pi, env.grid_size)
    print_policy(policy_pi, env.grid_size, env.terminal_states)

    # Verify both methods agree
    print("\n" + "-" * 60)
    print("Verification")
    print("-" * 60)
    print(f"Value functions match: {np.allclose(V_vi, V_pi)}")
    print(f"Policies match: {np.array_equal(policy_vi, policy_pi)}")

    # Print detailed policy
    print_policy_detailed(policy_vi, env.grid_size, env.terminal_states)

    # Expected values analysis
    print("\n" + "-" * 60)
    print("Expected vs Computed Values")
    print("-" * 60)
    print("For undiscounted rewards, V*(s) = -(steps to nearest terminal)")

    expected_V = np.array([
        0, -1, -2, -3,
        -1, -2, -3, -2,
        -2, -3, -2, -1,
        -3, -2, -1, 0
    ], dtype=float)

    print("\nExpected V*(s):")
    print_value_function(expected_V, env.grid_size)

    # Check approximate match (gamma=0.9999 gives slightly different values)
    max_diff = np.max(np.abs(V_vi - expected_V))
    print(f"\nMax difference from expected: {max_diff:.6f}")
    print(f"(Small difference due to γ={effective_gamma} vs γ=1.0)")

    # Generate and save visualization
    save_path = Path(__file__).parent / "gridworld_solution.png"
    visualize_solution(V_vi, policy_vi, env.grid_size, env.terminal_states, str(save_path))


if __name__ == "__main__":
    solve_gridworld(gamma=1.0)
