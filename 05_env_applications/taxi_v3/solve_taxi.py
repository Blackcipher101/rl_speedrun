"""
Taxi-v3 Dynamic Programming Solver

Solves the Taxi-v3 environment from Gymnasium using value iteration.
Extracts full transition dynamics from env.P and applies exact DP methods.

Environment: 5x5 grid with 4 designated locations
    - Taxi must pick up passenger and deliver to destination
    - 500 discrete states (5x5 grid × 5 passenger locations × 4 destinations)
    - 6 actions (South, North, East, West, Pickup, Dropoff)
    - Deterministic transitions

Rewards:
    - +20 for successful delivery
    - -1 per time step
    - -10 for illegal pickup/dropoff
"""

import numpy as np

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Warning: gymnasium not installed. Cannot run Taxi-v3 solver.")


# ============================================================================
# Extract MDP from Gymnasium Environment
# ============================================================================

def extract_mdp_from_gym(env_name: str = 'Taxi-v3') -> tuple:
    """
    Extract transition and reward matrices from a Gymnasium environment.

    Parameters
    ----------
    env_name : str
        Name of the Gymnasium environment.

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
    env = gym.make(env_name)

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
                R[s, a, next_state] = reward

    return P, R, n_states, n_actions, env


# ============================================================================
# State Encoding/Decoding
# ============================================================================

def decode_state(s: int) -> tuple:
    """
    Decode a Taxi-v3 state into its components.

    Parameters
    ----------
    s : int
        State index (0-499).

    Returns
    -------
    taxi_row : int
        Taxi's row position (0-4).
    taxi_col : int
        Taxi's column position (0-4).
    passenger_loc : int
        Passenger location (0=R, 1=G, 2=Y, 3=B, 4=in taxi).
    destination : int
        Destination (0=R, 1=G, 2=Y, 3=B).
    """
    destination = s % 4
    s //= 4
    passenger_loc = s % 5
    s //= 5
    taxi_col = s % 5
    taxi_row = s // 5
    return taxi_row, taxi_col, passenger_loc, destination


def encode_state(taxi_row: int, taxi_col: int,
                 passenger_loc: int, destination: int) -> int:
    """Encode state components into a state index."""
    return ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination


# Location positions (row, col)
LOCATIONS = {
    0: (0, 0),  # R (Red)
    1: (0, 4),  # G (Green)
    2: (4, 0),  # Y (Yellow)
    3: (4, 3),  # B (Blue)
}
LOCATION_NAMES = ['R', 'G', 'Y', 'B', 'in_taxi']


# ============================================================================
# Value Iteration
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

def print_state_info(s: int, V: np.ndarray = None, policy: np.ndarray = None) -> None:
    """Print information about a specific state."""
    taxi_row, taxi_col, pass_loc, dest = decode_state(s)
    action_names = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']

    print(f"State {s}:")
    print(f"  Taxi position: ({taxi_row}, {taxi_col})")
    print(f"  Passenger: {LOCATION_NAMES[pass_loc]}")
    print(f"  Destination: {LOCATION_NAMES[dest]}")

    if V is not None:
        print(f"  V*(s) = {V[s]:.2f}")
    if policy is not None:
        print(f"  π*(s) = {action_names[policy[s]]}")


def print_value_statistics(V: np.ndarray) -> None:
    """Print statistics about the value function."""
    print("\nValue Function Statistics:")
    print(f"  Min V*(s):  {V.min():.2f}")
    print(f"  Max V*(s):  {V.max():.2f}")
    print(f"  Mean V*(s): {V.mean():.2f}")
    print(f"  Std V*(s):  {V.std():.2f}")

    # Find best and worst states
    best_state = np.argmax(V)
    worst_state = np.argmin(V)

    print(f"\nBest state (highest V*):")
    print_state_info(best_state, V)

    print(f"\nWorst state (lowest V*):")
    print_state_info(worst_state, V)


def print_grid_for_state(env, s: int) -> None:
    """Print the rendered grid for a specific state."""
    if env is None:
        return

    # Temporarily render the environment at this state
    env_render = gym.make('Taxi-v3', render_mode='ansi')
    env_render.reset()
    env_render.unwrapped.s = s
    print(env_render.render())
    env_render.close()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(env, policy: np.ndarray, n_episodes: int = 1000,
                    max_steps: int = 200) -> tuple:
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
        Fraction of episodes completing successfully.
    avg_return : float
        Average return per episode.
    avg_steps : float
        Average steps per episode.
    """
    successes = 0
    total_return = 0.0
    total_steps = 0

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_return = 0.0
        episode_steps = 0

        for step in range(max_steps):
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            episode_steps += 1
            state = next_state

            if terminated or truncated:
                break

        total_return += episode_return
        total_steps += episode_steps

        # Success if positive return (reached goal)
        if episode_return > 0:
            successes += 1

    avg_steps = total_steps / n_episodes

    return successes / n_episodes, total_return / n_episodes, avg_steps


# ============================================================================
# Main Solver
# ============================================================================

def solve_taxi(gamma: float = 0.99, theta: float = 1e-8) -> None:
    """
    Solve Taxi-v3 using value iteration.

    Parameters
    ----------
    gamma : float
        Discount factor.
    theta : float
        Convergence threshold.
    """
    print("=" * 60)
    print("Taxi-v3 Dynamic Programming Solver")
    print("=" * 60)

    if not GYM_AVAILABLE:
        print("\nError: gymnasium is required to run this solver.")
        print("Install with: pip install gymnasium")
        return

    # Load environment and extract MDP
    print("\nLoading Taxi-v3 from Gymnasium...")
    P, R, n_states, n_actions, env = extract_mdp_from_gym('Taxi-v3')

    print(f"\nEnvironment Configuration:")
    print(f"  States: {n_states}")
    print(f"  Actions: {n_actions} (South, North, East, West, Pickup, Dropoff)")
    print(f"  Transitions: Deterministic")
    print(f"  Discount factor (γ): {gamma}")

    print("\nReward Structure:")
    print("  +20 for successful delivery")
    print("  -1 per time step")
    print("  -10 for illegal pickup/dropoff")

    # Solve with value iteration
    print("\n" + "-" * 60)
    print("Solving with Value Iteration...")
    print("-" * 60)

    V, policy, iterations = value_iteration(P, R, gamma, theta)

    print(f"Converged in {iterations} iterations")

    print_value_statistics(V)

    # Show example states
    print("\n" + "-" * 60)
    print("Example States and Optimal Actions")
    print("-" * 60)

    action_names = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']

    # Initial state example: taxi at (0,0), passenger at R, dest G
    example_state = encode_state(0, 0, 0, 1)  # Taxi at R, pass at R, dest G
    print(f"\nExample 1: Taxi at R(0,0), Passenger at R, Destination G")
    print_state_info(example_state, V, policy)

    # Mid-episode state: passenger in taxi
    example_state2 = encode_state(2, 2, 4, 1)  # Taxi mid-grid, pass in taxi, dest G
    print(f"\nExample 2: Taxi at (2,2), Passenger in taxi, Destination G")
    print_state_info(example_state2, V, policy)

    # Near delivery
    example_state3 = encode_state(0, 4, 4, 1)  # Taxi at G, pass in taxi, dest G
    print(f"\nExample 3: Taxi at G(0,4), Passenger in taxi, Destination G")
    print_state_info(example_state3, V, policy)

    # Evaluate policy
    print("\n" + "-" * 60)
    print("Policy Evaluation (Monte Carlo)")
    print("-" * 60)

    n_eval_episodes = 10000
    print(f"Running {n_eval_episodes} evaluation episodes...")

    success_rate, avg_return, avg_steps = evaluate_policy(
        env, policy, n_episodes=n_eval_episodes
    )

    print(f"\nResults with Optimal Policy:")
    print(f"  Success rate: {success_rate:.2%}")
    print(f"  Average return: {avg_return:.2f}")
    print(f"  Average steps: {avg_steps:.1f}")

    # Compare with random policy
    print("\nComparing with random policy...")
    random_policy = np.random.randint(0, n_actions, size=n_states)
    random_success, random_return, random_steps = evaluate_policy(
        env, random_policy, n_episodes=n_eval_episodes, max_steps=500
    )
    print(f"  Random success rate: {random_success:.2%}")
    print(f"  Random average return: {random_return:.2f}")
    print(f"  Random average steps: {random_steps:.1f}")

    print(f"\nImprovement:")
    print(f"  Return improvement: {avg_return - random_return:.2f}")

    # Value function by passenger location
    print("\n" + "-" * 60)
    print("Value Analysis by Passenger Location")
    print("-" * 60)

    for pass_loc in range(5):
        # Get all states with this passenger location
        states = [s for s in range(n_states)
                  if decode_state(s)[2] == pass_loc]
        values = V[states]
        print(f"  Passenger at {LOCATION_NAMES[pass_loc]:7s}: "
              f"mean V* = {values.mean():6.2f}, "
              f"min = {values.min():6.2f}, max = {values.max():6.2f}")

    env.close()

    print("\n" + "=" * 60)
    print("Solution Complete")
    print("=" * 60)


if __name__ == "__main__":
    solve_taxi(gamma=0.99)
