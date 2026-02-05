"""
Visualization Utilities for RL Benchmark Results

Generates publication-quality comparison plots:
- Learning curves with confidence intervals
- Final performance bar charts
- Sample efficiency comparisons
- Algorithm ranking heatmaps

Author: RL Speedrun
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import pickle
import os

# Import benchmark types
try:
    from benchmark import AggregatedResult, run_benchmark
except ImportError:
    from .benchmark import AggregatedResult, run_benchmark


def smooth_curve(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_learning_curves(
    results: Dict[str, Dict[str, AggregatedResult]],
    environment: str,
    window: int = 100,
    figsize: Tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot learning curves for all algorithms on a single environment.

    Shows mean ± std across runs with smoothing.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (alg_name, env_results) in enumerate(results.items()):
        if environment not in env_results:
            continue

        result = env_results[environment]
        mean = smooth_curve(result.mean_returns, window)
        std = smooth_curve(result.std_returns, window)

        episodes = np.arange(len(mean))
        color = colors[i % len(colors)]

        ax.plot(episodes, mean, label=alg_name, color=color, linewidth=2)
        ax.fill_between(
            episodes,
            mean - std,
            mean + std,
            alpha=0.2,
            color=color
        )

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Return (smoothed)', fontsize=12)
    ax.set_title(f'Learning Curves: {environment}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_final_performance(
    results: Dict[str, Dict[str, AggregatedResult]],
    environments: Optional[List[str]] = None,
    figsize: Tuple = (12, 5),
    save_path: Optional[str] = None
):
    """
    Plot bar chart of final performance across algorithms and environments.
    """
    algorithms = list(results.keys())

    if environments is None:
        environments = list(results[algorithms[0]].keys())

    n_algs = len(algorithms)
    n_envs = len(environments)
    x = np.arange(n_envs)
    width = 0.8 / n_algs

    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, alg in enumerate(algorithms):
        means = []
        stds = []
        for env in environments:
            result = results[alg][env]
            means.append(result.mean_final_return)
            stds.append(result.std_final_return)

        offset = (i - n_algs / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, means, width,
            label=alg,
            color=colors[i % len(colors)],
            yerr=stds,
            capsize=3
        )

    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Final Return (mean ± std)', fontsize=12)
    ax.set_title('Final Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(environments)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_sample_efficiency(
    results: Dict[str, Dict[str, AggregatedResult]],
    environment: str,
    threshold: float,
    figsize: Tuple = (8, 5),
    save_path: Optional[str] = None
):
    """
    Plot episodes needed to reach a performance threshold.
    """
    algorithms = []
    episodes_to_solve = []

    for alg_name, env_results in results.items():
        if environment not in env_results:
            continue

        result = env_results[environment]

        # Find when mean crosses threshold
        mean = result.mean_returns
        window = 100
        for i in range(window, len(mean)):
            if np.mean(mean[i-window:i]) >= threshold:
                algorithms.append(alg_name)
                episodes_to_solve.append(i)
                break
        else:
            # Didn't solve
            algorithms.append(alg_name)
            episodes_to_solve.append(len(mean))

    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    bars = ax.bar(
        algorithms, episodes_to_solve,
        color=[colors[i % len(colors)] for i in range(len(algorithms))]
    )

    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Episodes to Solve', fontsize=12)
    ax.set_title(f'Sample Efficiency: {environment} (threshold={threshold})', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, episodes_to_solve):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            str(val),
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_algorithm_ranking(
    results: Dict[str, Dict[str, AggregatedResult]],
    figsize: Tuple = (8, 6),
    save_path: Optional[str] = None
):
    """
    Plot heatmap of algorithm rankings across environments.
    """
    algorithms = list(results.keys())
    environments = list(results[algorithms[0]].keys())

    # Compute rankings (1 = best)
    rankings = np.zeros((len(algorithms), len(environments)))

    for j, env in enumerate(environments):
        returns = [(i, results[alg][env].mean_final_return) for i, alg in enumerate(algorithms)]
        sorted_returns = sorted(returns, key=lambda x: x[1], reverse=True)

        for rank, (alg_idx, _) in enumerate(sorted_returns):
            rankings[alg_idx, j] = rank + 1

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(rankings, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(np.arange(len(environments)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(environments, fontsize=10)
    ax.set_yticklabels(algorithms, fontsize=10)

    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(environments)):
            text = ax.text(
                j, i, f'{int(rankings[i, j])}',
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='white' if rankings[i, j] > 1.5 else 'black'
            )

    ax.set_title('Algorithm Rankings (1 = Best)', fontsize=14)
    plt.colorbar(im, ax=ax, label='Rank')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def generate_all_plots(
    results: Dict[str, Dict[str, AggregatedResult]],
    output_dir: str = '.'
):
    """Generate all comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    environments = list(results[list(results.keys())[0]].keys())

    # Learning curves for each environment
    for env in environments:
        env_clean = env.replace('-', '_').replace('.', '_')
        plot_learning_curves(
            results, env,
            save_path=os.path.join(output_dir, f'learning_curves_{env_clean}.png')
        )

    # Final performance comparison
    plot_final_performance(
        results,
        save_path=os.path.join(output_dir, 'final_performance.png')
    )

    # Algorithm ranking heatmap
    plot_algorithm_ranking(
        results,
        save_path=os.path.join(output_dir, 'algorithm_ranking.png')
    )

    print(f"\nAll plots saved to: {output_dir}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Check if benchmark results exist
    results_file = 'benchmark_results.pkl'

    if os.path.exists(results_file):
        print("Loading existing benchmark results...")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    else:
        print("Running benchmark (this may take a while)...")
        results = run_benchmark(
            algorithms=['Q-Learning', 'SARSA', 'Monte Carlo'],
            environments=['FrozenLake-v1', 'CliffWalking-v0'],
            n_episodes=3000,
            n_runs=3,
            seed=42,
            verbose=True
        )

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

    # Generate all plots
    generate_all_plots(results, output_dir='benchmark_plots')

    plt.show()
