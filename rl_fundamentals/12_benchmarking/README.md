# Benchmarking Framework

A systematic framework for comparing RL algorithms across multiple environments with statistical rigor.

---

## Overview

This module provides tools for:
- Running controlled experiments across algorithms and environments
- Aggregating results from multiple runs for statistical validity
- Generating publication-quality comparison visualizations
- Analyzing sample efficiency and convergence properties

---

## Benchmarked Algorithms

| Algorithm | Type | Key Property |
|-----------|------|--------------|
| Q-Learning | TD Control | Off-policy, learns optimal Q |
| SARSA | TD Control | On-policy, learns policy Q |
| Monte Carlo | Episode-based | Unbiased, high variance |

---

## Benchmark Environments

| Environment | States | Actions | Challenge |
|-------------|--------|---------|-----------|
| FrozenLake-v1 | 16 | 4 | Stochastic transitions |
| CliffWalking-v0 | 48 | 4 | Risky optimal path |
| Taxi-v3 | 500 | 6 | Large state space |

---

## Usage

### Quick Benchmark

```python
from benchmark import run_benchmark

results = run_benchmark(
    algorithms=['Q-Learning', 'SARSA', 'Monte Carlo'],
    environments=['FrozenLake-v1', 'CliffWalking-v0'],
    n_episodes=5000,
    n_runs=5,
    seed=42
)
```

### Generate Visualizations

```python
from comparison_plots import generate_all_plots

generate_all_plots(results, output_dir='benchmark_plots')
```

---

## Output Metrics

### Per-Run Metrics

- Episode returns (full history)
- Episode lengths
- Training time
- Final mean/std return
- Episodes to solve (if applicable)

### Aggregated Metrics

- Mean/std returns across runs
- Solve rate (% of runs that solved)
- Mean episodes to solve
- Training time statistics

---

## Visualizations

### 1. Learning Curves

Shows mean ± std across runs with smoothing:
- X-axis: Episode number
- Y-axis: Return (smoothed)
- Shaded region: Standard deviation

### 2. Final Performance Bar Chart

Compares algorithms by final return:
- Error bars show std across runs
- Grouped by environment

### 3. Sample Efficiency

Episodes needed to reach threshold:
- Lower is better
- Measures learning speed

### 4. Algorithm Ranking Heatmap

Ranks algorithms per environment:
- 1 = best performer
- Color-coded for quick comparison

---

## Running the Benchmark

```bash
cd rl_fundamentals/12_benchmarking

# Run benchmark
python benchmark.py

# Generate plots (uses saved results)
python comparison_plots.py
```

---

## Extending the Framework

### Adding New Algorithm

```python
# In benchmark.py
def run_my_algorithm(env, n_episodes, **kwargs):
    # Your implementation
    return episode_returns, episode_lengths

ALGORITHMS['My Algorithm'] = run_my_algorithm
```

### Adding New Environment

```python
ENVIRONMENTS['MyEnv-v1'] = {
    'make': lambda: gym.make("MyEnv-v1"),
    'solve_threshold': 100  # What counts as "solved"
}
```

---

## Statistical Considerations

1. **Multiple runs**: Default 5 runs per configuration
2. **Seed control**: Different seeds per run for variance
3. **Smoothing**: Moving average for visual clarity
4. **Confidence intervals**: Std shown, not confidence intervals

For publication, consider:
- More runs (10-30)
- Bootstrap confidence intervals
- Statistical tests (t-test, Mann-Whitney)

---

## Module Structure

```
12_benchmarking/
├── README.md           # This file
├── benchmark.py        # Core benchmark runner
└── comparison_plots.py # Visualization utilities
```
