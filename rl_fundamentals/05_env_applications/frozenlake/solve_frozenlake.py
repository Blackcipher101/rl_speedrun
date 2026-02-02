"""
FrozenLake Dynamic Programming Solver

Solves the FrozenLake-v1 environment from Gymnasium using value iteration.
Extracts full transition dynamics from env.P and applies exact DP methods.

Environment: 4x4 grid with slippery ice
    SFFF    S = Start, F = Frozen, H = Hole, G = Goal
    FHFH
    FFFH
    HFFG

Key features:
    - Stochastic transitions (1/3 probability each for intended + perpendicular)
    - Reward: +1 for reaching goal, 0 otherwise
    - Terminal states: Holes (reward 0) and Goal (reward 1)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Warning: gymnasium not installed. Using hardcoded dynamics.")


# ============================================================================
# Extract MDP from Gymnasium Environment
# ============================================================================

def extract_mdp_from_gym(env_name: str = 'FrozenLake-v1',
                          is_slippery: bool = True) -> tuple:
    """
    Extract transition and reward matrices from a Gymnasium environment.

    Parameters
    ----------
    env_name : str
        Name of the Gymnasium environment.
    is_slippery : bool
        Whether to use slippery ice dynamics.

    Returns
    -------
    P : np.ndarray, shape (n_states, n_actions, n_states)
        Transition probability matrix.
    R : np.ndarray, shape (n_states, n_actions, n_states)
        Reward matrix.
    n_states : int
        Number of states.
    n_actions : int
        Number of actions.
    env : gymnasium.Env
        The environment object.
    """
    env = gym.make(env_name, is_slippery=is_slippery)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize matrices
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions, n_states))

    # Get the underlying environment (unwrap from TimeLimit wrapper)
    unwrapped_env = env.unwrapped

    # Extract dynamics from env.P
    # env.P[s][a] is a list of (probability, next_state, reward, done) tuples
    for s in range(n_states):
        for a in range(n_actions):
            for prob, next_state, reward, done in unwrapped_env.P[s][a]:
                P[s, a, next_state] += prob
                # Reward is associated with the transition
                R[s, a, next_state] = reward

    return P, R, n_states, n_actions, env


def get_hardcoded_frozenlake_mdp() -> tuple:
    """
    Return hardcoded FrozenLake 4x4 MDP for use without gymnasium.

    This is the standard FrozenLake-v1 with is_slippery=True.
    """
    n_states = 16
    n_actions = 4

    # Map layout
    # SFFF  0  1  2  3
    # FHFH  4  5  6  7
    # FFFH  8  9  10 11
    # HFFG  12 13 14 15

    # Terminal states: Holes at 5, 7, 11, 12; Goal at 15
    holes = {5, 7, 11, 12}
    goal = 15
    terminal_states = holes | {goal}

    # Actions: LEFT=0, DOWN=1, RIGHT=2, UP=3
    # Movement deltas: (row_delta, col_delta)
    action_deltas = {
        0: (0, -1),   # LEFT
        1: (1, 0),    # DOWN
        2: (0, 1),    # RIGHT
        3: (-1, 0),   # UP
    }

    # For slippery ice, actual movement is:
    # - 1/3 intended direction
    # - 1/3 perpendicular clockwise
    # - 1/3 perpendicular counter-clockwise
    # Perpendicular mappings:
    perpendicular = {
        0: [3, 1],  # LEFT -> UP, DOWN
        1: [0, 2],  # DOWN -> LEFT, RIGHT
        2: [1, 3],  # RIGHT -> DOWN, UP
        3: [2, 0],  # UP -> RIGHT, LEFT
    }

    def state_to_pos(s):
        return divmod(s, 4)

    def pos_to_state(row, col):
        return row * 4 + col

    def move(row, col, action):
        dr, dc = action_deltas[action]
        new_row = max(0, min(3, row + dr))
        new_col = max(0, min(3, col + dc))
        return pos_to_state(new_row, new_col)

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions, n_states))

    for s in range(n_states):
        if s in terminal_states:
            # Terminal states are absorbing
            for a in range(n_actions):
                P[s, a, s] = 1.0
                R[s, a, s] = 0.0
        else:
            row, col = state_to_pos(s)
            for a in range(n_actions):
                # Intended direction
                next_s = move(row, col, a)
                P[s, a, next_s] += 1/3

                # Perpendicular directions
                for perp_a in perpendicular[a]:
                    next_s = move(row, col, perp_a)
                    P[s, a, next_s] += 1/3

                # Reward for reaching goal
                if P[s, a, goal] > 0:
                    R[s, a, goal] = 1.0

    return P, R, n_states, n_actions


# ============================================================================
# Value Iteration (Standalone Implementation)
# ============================================================================

def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float,
                    theta: float = 1e-8, max_iter: int = 10000) -> tuple:
    """
    Value iteration algorithm.

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
                # Q(s, a) = sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
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
            row_str += f" {V[s]:5.3f} |"
        print(row_str)
        print("+" + "-------+" * grid_size)


def print_policy(policy: np.ndarray, grid_size: int = 4,
                 terminal_states: set = None) -> None:
    """Print policy as a grid with arrows."""
    if terminal_states is None:
        terminal_states = {5, 7, 11, 12, 15}  # FrozenLake holes + goal

    # Actions: LEFT=0, DOWN=1, RIGHT=2, UP=3
    arrows = ['←', '↓', '→', '↑']

    print("\nOptimal Policy π*(s):")
    print("+" + "---+" * grid_size)
    for row in range(grid_size):
        row_str = "|"
        for col in range(grid_size):
            s = row * grid_size + col
            if s in terminal_states:
                if s == 15:
                    row_str += " G |"  # Goal
                else:
                    row_str += " H |"  # Hole
            else:
                row_str += f" {arrows[policy[s]]} |"
        print(row_str)
        print("+" + "---+" * grid_size)


def print_environment_map(grid_size: int = 4) -> None:
    """Print the FrozenLake map."""
    map_chars = "SFFFFFFHFHFFFHFHFFGHFFG"[:grid_size*grid_size]
    default_map = ['S', 'F', 'F', 'F',
                   'F', 'H', 'F', 'H',
                   'F', 'F', 'F', 'H',
                   'H', 'F', 'F', 'G']

    print("\nFrozenLake Map:")
    print("+" + "---+" * grid_size)
    for row in range(grid_size):
        row_str = "|"
        for col in range(grid_size):
            s = row * grid_size + col
            row_str += f" {default_map[s]} |"
        print(row_str)
        print("+" + "---+" * grid_size)
    print("S=Start, F=Frozen, H=Hole, G=Goal")


def visualize_frozenlake_solution(V: np.ndarray, policy: np.ndarray,
                                   grid_size: int = 4, save_path: str = None) -> None:
    """
    Create a visualization of FrozenLake solution.

    Parameters
    ----------
    V : np.ndarray
        Value function.
    policy : np.ndarray
        Policy (action indices).
    grid_size : int
        Size of the grid.
    save_path : str, optional
        Path to save the figure.
    """
    # FrozenLake map
    cell_types = ['S', 'F', 'F', 'F',
                  'F', 'H', 'F', 'H',
                  'F', 'F', 'F', 'H',
                  'H', 'F', 'F', 'G']

    holes = {5, 7, 11, 12}
    goal = {15}
    start = {0}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    V_grid = V.reshape(grid_size, grid_size)

    # ===== Left plot: Value Function =====
    ax1 = axes[0]

    # Create custom colormap data
    cmap_data = np.zeros((grid_size, grid_size, 4))
    for i in range(grid_size):
        for j in range(grid_size):
            s = i * grid_size + j
            if s in holes:
                cmap_data[i, j] = [0.2, 0.2, 0.2, 1.0]  # Dark gray for holes
            elif s in goal:
                cmap_data[i, j] = [0.2, 0.8, 0.2, 1.0]  # Green for goal
            else:
                # Blue gradient based on value
                intensity = V[s] / max(V.max(), 0.01)
                cmap_data[i, j] = [0.1, 0.3 + 0.5 * intensity, 0.9, 1.0]

    ax1.imshow(cmap_data, aspect='equal')

    # Add value and cell type annotations
    for i in range(grid_size):
        for j in range(grid_size):
            s = i * grid_size + j
            if s in holes:
                ax1.text(j, i, f'H\n0.00', ha='center', va='center',
                        fontsize=11, color='white', fontweight='bold')
            elif s in goal:
                ax1.text(j, i, f'G\n{V[s]:.2f}', ha='center', va='center',
                        fontsize=11, color='white', fontweight='bold')
            elif s in start:
                ax1.text(j, i, f'S\n{V[s]:.2f}', ha='center', va='center',
                        fontsize=11, color='white', fontweight='bold')
            else:
                ax1.text(j, i, f'{V[s]:.2f}', ha='center', va='center',
                        fontsize=11, color='white')

    ax1.set_xticks(range(grid_size))
    ax1.set_yticks(range(grid_size))
    ax1.set_title('Value Function V*(s)\n(≈ Prob. of reaching goal)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')

    for i in range(grid_size + 1):
        ax1.axhline(i - 0.5, color='white', linewidth=2)
        ax1.axvline(i - 0.5, color='white', linewidth=2)

    # ===== Right plot: Policy =====
    ax2 = axes[1]

    # Background colors
    bg_colors = np.zeros((grid_size, grid_size, 4))
    for i in range(grid_size):
        for j in range(grid_size):
            s = i * grid_size + j
            if s in holes:
                bg_colors[i, j] = [0.2, 0.2, 0.2, 1.0]
            elif s in goal:
                bg_colors[i, j] = [0.2, 0.8, 0.2, 1.0]
            elif s in start:
                bg_colors[i, j] = [0.9, 0.9, 0.2, 1.0]
            else:
                bg_colors[i, j] = [0.7, 0.85, 1.0, 1.0]

    ax2.imshow(bg_colors, aspect='equal')

    # Actions: LEFT=0, DOWN=1, RIGHT=2, UP=3
    arrow_dx = [-0.3, 0, 0.3, 0]
    arrow_dy = [0, 0.3, 0, -0.3]
    arrow_symbols = ['←', '↓', '→', '↑']

    for i in range(grid_size):
        for j in range(grid_size):
            s = i * grid_size + j
            if s in holes:
                ax2.text(j, i, '☠', ha='center', va='center', fontsize=20)
            elif s in goal:
                ax2.text(j, i, '🎯', ha='center', va='center', fontsize=20)
            else:
                action = policy[s]
                ax2.annotate('', xy=(j + arrow_dx[action], i + arrow_dy[action]),
                            xytext=(j, i),
                            arrowprops=dict(arrowstyle='->', color='darkblue',
                                          lw=2.5, mutation_scale=20))

    ax2.set_xticks(range(grid_size))
    ax2.set_yticks(range(grid_size))
    ax2.set_title('Optimal Policy π*(s)\n(Slippery: 1/3 chance each direction)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.set_xlim(-0.5, grid_size - 0.5)
    ax2.set_ylim(grid_size - 0.5, -0.5)

    for i in range(grid_size + 1):
        ax2.axhline(i - 0.5, color='black', linewidth=2)
        ax2.axvline(i - 0.5, color='black', linewidth=2)

    # Legend
    patches = [
        mpatches.Patch(color=[0.9, 0.9, 0.2], label='Start'),
        mpatches.Patch(color=[0.7, 0.85, 1.0], label='Frozen'),
        mpatches.Patch(color=[0.2, 0.2, 0.2], label='Hole'),
        mpatches.Patch(color=[0.2, 0.8, 0.2], label='Goal'),
    ]
    ax2.legend(handles=patches, loc='upper left', fontsize=9)

    plt.suptitle('FrozenLake-v1 Solution (4x4, Slippery)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nVisualization saved to: {save_path}")

    plt.close()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(env, policy: np.ndarray, n_episodes: int = 1000,
                    max_steps: int = 100) -> tuple:
    """
    Evaluate a policy by running episodes.

    Parameters
    ----------
    env : gymnasium.Env
        The environment.
    policy : np.ndarray
        Policy to evaluate.
    n_episodes : int
        Number of episodes.
    max_steps : int
        Maximum steps per episode.

    Returns
    -------
    success_rate : float
        Fraction of episodes reaching the goal.
    avg_return : float
        Average return per episode.
    """
    successes = 0
    total_return = 0.0

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0.0

        for _ in range(max_steps):
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            state = next_state

            if terminated or truncated:
                break

        total_return += episode_return
        if episode_return > 0:  # Reached goal
            successes += 1

    return successes / n_episodes, total_return / n_episodes


# ============================================================================
# Main Solver
# ============================================================================

def solve_frozenlake(gamma: float = 0.99, theta: float = 1e-8) -> None:
    """
    Solve FrozenLake using value iteration.

    Parameters
    ----------
    gamma : float
        Discount factor.
    theta : float
        Convergence threshold.
    """
    print("=" * 60)
    print("FrozenLake Dynamic Programming Solver")
    print("=" * 60)

    # Load environment and extract MDP
    if GYM_AVAILABLE:
        print("\nLoading FrozenLake-v1 from Gymnasium...")
        P, R, n_states, n_actions, env = extract_mdp_from_gym(
            'FrozenLake-v1', is_slippery=True
        )
    else:
        print("\nUsing hardcoded FrozenLake dynamics...")
        P, R, n_states, n_actions = get_hardcoded_frozenlake_mdp()
        env = None

    print_environment_map()

    print(f"\nEnvironment Configuration:")
    print(f"  States: {n_states}")
    print(f"  Actions: {n_actions} (LEFT, DOWN, RIGHT, UP)")
    print(f"  Slippery: True (1/3 probability each direction)")
    print(f"  Discount factor (γ): {gamma}")

    # Terminal states
    holes = {5, 7, 11, 12}
    goal = {15}
    terminal_states = holes | goal

    # Solve with value iteration
    print("\n" + "-" * 60)
    print("Solving with Value Iteration...")
    print("-" * 60)

    V, policy, iterations = value_iteration(P, R, gamma, theta)

    print(f"Converged in {iterations} iterations")

    print_value_function(V)
    print_policy(policy, terminal_states=terminal_states)

    # Analysis
    print("\n" + "-" * 60)
    print("Analysis")
    print("-" * 60)

    print(f"\nValue at start state V*(0): {V[0]:.4f}")
    print("(This approximates the probability of reaching the goal")
    print(" from the start under the optimal policy when γ ≈ 1)")

    # Show Q-values for start state
    print("\nQ-values at start state (state 0):")
    action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    for a in range(n_actions):
        q = np.sum(P[0, a, :] * (R[0, a, :] + gamma * V))
        marker = " <-- optimal" if a == policy[0] else ""
        print(f"  Q(0, {action_names[a]:5s}) = {q:.4f}{marker}")

    # Evaluate policy if environment is available
    if env is not None:
        print("\n" + "-" * 60)
        print("Policy Evaluation (Monte Carlo)")
        print("-" * 60)

        n_eval_episodes = 10000
        print(f"Running {n_eval_episodes} evaluation episodes...")

        success_rate, avg_return = evaluate_policy(
            env, policy, n_episodes=n_eval_episodes
        )

        print(f"\nResults:")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average return: {avg_return:.4f}")
        print(f"  Expected from V*(0): {V[0]:.4f}")

        # Compare random policy
        print("\nComparing with random policy...")
        random_policy = np.random.randint(0, n_actions, size=n_states)
        random_success, random_return = evaluate_policy(
            env, random_policy, n_episodes=n_eval_episodes
        )
        print(f"  Random success rate: {random_success:.2%}")
        print(f"  Optimal success rate: {success_rate:.2%}")
        print(f"  Improvement: {(success_rate - random_success)/random_success*100:.1f}%")

        env.close()

    # Generate and save visualization
    save_path = Path(__file__).parent / "frozenlake_solution.png"
    visualize_frozenlake_solution(V, policy, save_path=str(save_path))

    print("\n" + "=" * 60)
    print("Solution Complete")
    print("=" * 60)


if __name__ == "__main__":
    solve_frozenlake(gamma=0.99)
