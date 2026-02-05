"""
Benchmarking Framework for RL Algorithms

Systematic comparison of implemented RL algorithms across multiple
environments with statistical analysis and visualization.

Algorithms benchmarked:
- Q-Learning (TD)
- SARSA (TD)
- Monte Carlo Control
- REINFORCE (Policy Gradient)

Environments:
- FrozenLake-v1 (discrete, stochastic)
- CliffWalking-v0 (discrete, deterministic)
- Taxi-v3 (discrete, larger state space)

Author: RL Speedrun
"""

import numpy as np
import gymnasium as gym
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class BenchmarkResult:
    """Container for a single benchmark run result."""
    algorithm: str
    environment: str
    run_id: int
    episode_returns: List[float]
    episode_lengths: List[int]
    training_time: float
    final_mean_return: float
    final_std_return: float
    episodes_to_solve: Optional[int] = None


@dataclass
class AggregatedResult:
    """Aggregated results across multiple runs."""
    algorithm: str
    environment: str
    n_runs: int
    mean_returns: np.ndarray  # Mean across runs at each episode
    std_returns: np.ndarray   # Std across runs at each episode
    mean_training_time: float
    std_training_time: float
    mean_final_return: float
    std_final_return: float
    solve_rate: float  # Fraction of runs that solved
    mean_episodes_to_solve: Optional[float] = None


# =============================================================================
# Algorithm Implementations (Wrappers)
# =============================================================================

def run_q_learning(
    env,
    n_episodes: int,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    seed: int = 42
) -> Tuple[List[float], List[int]]:
    """Run Q-Learning and return episode returns and lengths."""
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    episode_returns = []
    episode_lengths = []
    epsilon = epsilon_start

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_return = 0.0
        ep_length = 0

        while not done and not truncated:
            # Epsilon-greedy
            if rng.random() < epsilon:
                action = rng.integers(n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, truncated, _ = env.step(action)

            # Q-Learning update
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            ep_return += reward
            ep_length += 1

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return episode_returns, episode_lengths


def run_sarsa(
    env,
    n_episodes: int,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    seed: int = 42
) -> Tuple[List[float], List[int]]:
    """Run SARSA and return episode returns and lengths."""
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    episode_returns = []
    episode_lengths = []
    epsilon = epsilon_start

    def eps_greedy(state):
        if rng.random() < epsilon:
            return rng.integers(n_actions)
        return np.argmax(Q[state])

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        action = eps_greedy(state)
        done = False
        truncated = False
        ep_return = 0.0
        ep_length = 0

        while not done and not truncated:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = eps_greedy(next_state)

            # SARSA update
            td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            action = next_action
            ep_return += reward
            ep_length += 1

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return episode_returns, episode_lengths


def run_monte_carlo(
    env,
    n_episodes: int,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.9995,
    seed: int = 42
) -> Tuple[List[float], List[int]]:
    """Run First-Visit MC Control and return episode returns and lengths."""
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions))
    episode_returns = []
    episode_lengths = []
    epsilon = epsilon_start

    for ep in range(n_episodes):
        # Generate episode
        trajectory = []
        state, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False

        while not done and not truncated:
            if rng.random() < epsilon:
                action = rng.integers(n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, truncated, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state

        # Compute returns and update
        G = 0
        visited = set()
        for t in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[t]
            G = gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                N[state, action] += 1
                Q[state, action] += (G - Q[state, action]) / N[state, action]

        episode_returns.append(G if trajectory else 0)
        episode_lengths.append(len(trajectory))
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return episode_returns, episode_lengths


# =============================================================================
# Benchmark Runner
# =============================================================================

ALGORITHMS = {
    'Q-Learning': run_q_learning,
    'SARSA': run_sarsa,
    'Monte Carlo': run_monte_carlo,
}

ENVIRONMENTS = {
    'FrozenLake-v1': {'make': lambda: gym.make("FrozenLake-v1", is_slippery=True), 'solve_threshold': 0.7},
    'CliffWalking-v0': {'make': lambda: gym.make("CliffWalking-v0"), 'solve_threshold': -20},
    'Taxi-v3': {'make': lambda: gym.make("Taxi-v3"), 'solve_threshold': 8},
}


def run_single_benchmark(
    algorithm_name: str,
    env_name: str,
    n_episodes: int,
    run_id: int,
    seed: int
) -> BenchmarkResult:
    """Run a single benchmark experiment."""
    env_config = ENVIRONMENTS[env_name]
    env = env_config['make']()
    algorithm_fn = ALGORITHMS[algorithm_name]

    # Time the training
    start_time = time.time()
    returns, lengths = algorithm_fn(env, n_episodes, seed=seed)
    training_time = time.time() - start_time

    # Evaluate final performance
    final_returns = returns[-100:] if len(returns) >= 100 else returns
    final_mean = np.mean(final_returns)
    final_std = np.std(final_returns)

    # Check for solve
    solve_threshold = env_config['solve_threshold']
    episodes_to_solve = None
    window = 100
    for i in range(window, len(returns)):
        if np.mean(returns[i-window:i]) >= solve_threshold:
            episodes_to_solve = i
            break

    env.close()

    return BenchmarkResult(
        algorithm=algorithm_name,
        environment=env_name,
        run_id=run_id,
        episode_returns=returns,
        episode_lengths=lengths,
        training_time=training_time,
        final_mean_return=final_mean,
        final_std_return=final_std,
        episodes_to_solve=episodes_to_solve
    )


def run_benchmark(
    algorithms: Optional[List[str]] = None,
    environments: Optional[List[str]] = None,
    n_episodes: int = 5000,
    n_runs: int = 5,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[str, AggregatedResult]]:
    """
    Run benchmark across multiple algorithms and environments.

    Args:
        algorithms: List of algorithm names (default: all)
        environments: List of environment names (default: all)
        n_episodes: Episodes per run
        n_runs: Number of runs per configuration
        seed: Base random seed
        verbose: Print progress

    Returns:
        Nested dict: results[algorithm][environment] = AggregatedResult
    """
    if algorithms is None:
        algorithms = list(ALGORITHMS.keys())
    if environments is None:
        environments = list(ENVIRONMENTS.keys())

    results = {}

    for alg_name in algorithms:
        results[alg_name] = {}

        for env_name in environments:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Benchmarking: {alg_name} on {env_name}")
                print(f"{'='*60}")

            run_results = []
            for run in range(n_runs):
                if verbose:
                    print(f"  Run {run + 1}/{n_runs}...", end=" ")

                result = run_single_benchmark(
                    alg_name, env_name, n_episodes,
                    run_id=run, seed=seed + run * 1000
                )
                run_results.append(result)

                if verbose:
                    print(f"Final return: {result.final_mean_return:.2f}")

            # Aggregate results
            all_returns = np.array([r.episode_returns for r in run_results])
            training_times = [r.training_time for r in run_results]
            final_returns = [r.final_mean_return for r in run_results]
            solve_episodes = [r.episodes_to_solve for r in run_results if r.episodes_to_solve]

            aggregated = AggregatedResult(
                algorithm=alg_name,
                environment=env_name,
                n_runs=n_runs,
                mean_returns=np.mean(all_returns, axis=0),
                std_returns=np.std(all_returns, axis=0),
                mean_training_time=np.mean(training_times),
                std_training_time=np.std(training_times),
                mean_final_return=np.mean(final_returns),
                std_final_return=np.std(final_returns),
                solve_rate=len(solve_episodes) / n_runs,
                mean_episodes_to_solve=np.mean(solve_episodes) if solve_episodes else None
            )

            results[alg_name][env_name] = aggregated

            if verbose:
                print(f"\n  Aggregated results ({n_runs} runs):")
                print(f"    Final return: {aggregated.mean_final_return:.2f} ± {aggregated.std_final_return:.2f}")
                print(f"    Training time: {aggregated.mean_training_time:.2f}s")
                print(f"    Solve rate: {aggregated.solve_rate * 100:.0f}%")

    return results


def print_summary_table(results: Dict[str, Dict[str, AggregatedResult]]):
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    algorithms = list(results.keys())
    environments = list(results[algorithms[0]].keys())

    # Header
    header = f"{'Environment':<20}"
    for alg in algorithms:
        header += f"{alg:>15}"
    print(header)
    print("-" * 80)

    # Results
    for env in environments:
        row = f"{env:<20}"
        for alg in algorithms:
            result = results[alg][env]
            row += f"{result.mean_final_return:>12.2f} ± {result.std_final_return:.1f}"
        print(row)

    print("=" * 80)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RL Algorithm Benchmark")
    print("=" * 70)

    # Run benchmark (reduced episodes for demo)
    results = run_benchmark(
        algorithms=['Q-Learning', 'SARSA', 'Monte Carlo'],
        environments=['FrozenLake-v1', 'CliffWalking-v0'],
        n_episodes=5000,
        n_runs=3,
        seed=42,
        verbose=True
    )

    # Print summary
    print_summary_table(results)

    # Save results for visualization
    import pickle
    with open('benchmark_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nResults saved to: benchmark_results.pkl")
