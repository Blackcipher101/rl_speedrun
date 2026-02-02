# RL Speedrun 🏃‍♂️💨

> *"Why spend months learning RL when you can mass produce it in days?"* - Me, probably sleep deprived

Welcome to my chaotic journey through Reinforcement Learning. This repo is basically me speedrunning RL concepts, writing everything from scratch, and pretending I know what I'm doing.

## What's This?

A personal RL learning repo where I implement algorithms from first principles. No fancy libraries doing the heavy lifting - just raw NumPy energy and questionable life choices.

## Current Speedrun: Dynamic Programming

```
📁 rl_fundamentals/          <- The "I actually understand math" section
├── 01_mdp/                  <- MDPs: fancy way to say "states go brrr"
├── 02_value_functions/      <- V(s) and Q(s,a) - the OG value bros
├── 03_bellman_equations/    <- Bellman said: "it's recursive, deal with it"
├── 04_dynamic_programming/  <- When you know everything about the world
└── 05_env_applications/     <- Actually making things work
    ├── gridworld/           <- Baby's first MDP
    ├── frozenlake/          <- Slippery boi simulator
    └── taxi_v3/             <- Uber but worse
```

## The Visuals

Because staring at numbers is boring:

### GridWorld - The Classic
<p align="center">
  <img src="rl_fundamentals/05_env_applications/gridworld/gridworld_solution.png" width="600"/>
</p>

*Optimal policy for a 4x4 grid. Terminal states at corners. Agent just wants to go home.*

### FrozenLake - Slippery When Wet
<p align="center">
  <img src="rl_fundamentals/05_env_applications/frozenlake/frozenlake_solution.png" width="600"/>
</p>

*That feeling when you try to go right but physics says "nah". 1/3 chance of actually going where you want.*

### Taxi-v3 - 500 States of Pain
<p align="center">
  <img src="rl_fundamentals/05_env_applications/taxi_v3/taxi_value_heatmap.png" width="500"/>
</p>

*Value function heatmap. Higher = closer to dropping off passengers and escaping this nightmare.*

## Quick Start

```bash
# Install the goods
pip install -r rl_fundamentals/requirements.txt

# Watch GridWorld get solved
python rl_fundamentals/05_env_applications/gridworld/gridworld_dp.py

# See a drunk agent navigate ice
python rl_fundamentals/05_env_applications/frozenlake/solve_frozenlake.py

# Experience Taxi dispatcher hell
python rl_fundamentals/05_env_applications/taxi_v3/solve_taxi.py
```

## Speedrun Progress

- [x] **Dynamic Programming** - When you have the cheat codes (full model)
- [ ] **Monte Carlo** - When you just yolo sample everything
- [ ] **Temporal Difference** - The "best of both worlds" arc
- [ ] **Q-Learning** - The famous one
- [ ] **Policy Gradient** - Going continuous
- [ ] **Actor-Critic** - Why choose when you can have both
- [ ] **Deep RL** - Neural networks enter the chat

## Philosophy

This repo follows the ancient wisdom:

1. **Understand the math** - Actually derive things, no hand-waving
2. **Implement from scratch** - Suffering builds character
3. **Visualize everything** - Pretty pictures > walls of numbers
4. **Keep it real** - Comments are for future confused me

## Resources I'm Stealing From

- Sutton & Barto's RL Book (the bible)
- David Silver's lectures (goated)
- OpenAI Spinning Up (documentation supremacy)
- Stack Overflow (no shame)

---

*Currently speedrunning: Dynamic Programming*

*Next up: Monte Carlo Methods (sampling my way to victory)*

**Stars appreciated, issues tolerated, PRs celebrated** ⭐
