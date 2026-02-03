# RL Speedrun 🏃‍♂️💨

> *"Why spend months learning RL when you can mass produce it in days?"* - Me, probably sleep deprived

Welcome to my chaotic journey through Reinforcement Learning. This repo is basically me speedrunning RL concepts, writing everything from scratch, and pretending I know what I'm doing.

## What's This?

A personal RL learning repo where I implement algorithms from first principles. No fancy libraries doing the heavy lifting - just raw NumPy energy and questionable life choices.

## Repository Structure

```
📁 rl_fundamentals/          <- The "I actually understand math" section
├── 01_mdp/                  <- MDPs: fancy way to say "states go brrr"
├── 02_value_functions/      <- V(s) and Q(s,a) - the OG value bros
├── 03_bellman_equations/    <- Bellman said: "it's recursive, deal with it"
├── 04_dynamic_programming/  <- When you know everything about the world
├── 05_env_applications/     <- DP in action
│   ├── gridworld/           <- Baby's first MDP
│   ├── frozenlake/          <- Slippery boi simulator
│   └── taxi_v3/             <- Uber but worse
├── 06_temporal_difference/  <- Learning from experience, one step at a time
│   ├── q_learning.py        <- Off-policy: "I'll learn the optimal path even while exploring"
│   └── sarsa.py             <- On-policy: "I'll learn about what I actually do"
└── 07_td_applications/      <- TD algorithms in the wild
    ├── cliffwalking/        <- Q-Learning vs SARSA showdown
    └── cartpole/            <- Continuous states? Just discretize it bro
```

---

## Week 1: Dynamic Programming

*"When you have God mode enabled (full model knowledge)"*

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

---

## Week 2: Temporal Difference Learning

*"Model-free vibes - learning from experience without knowing the rules"*

### CliffWalking - The Q-Learning vs SARSA Showdown
<p align="center">
  <img src="rl_fundamentals/07_td_applications/cliffwalking/cliffwalking_comparison.png" width="700"/>
</p>

*Q-Learning: "I'll walk the edge, YOLO"*
*SARSA: "I'd rather live, thanks"*

The classic demonstration of off-policy vs on-policy learning:
- **Q-Learning** finds the risky optimal path (right along the cliff edge)
- **SARSA** finds the safer path (stays away from the cliff because it knows it might slip)

### CartPole - Discretization Station
<p align="center">
  <img src="rl_fundamentals/07_td_applications/cartpole/cartpole_learning.png" width="700"/>
</p>

*Continuous state space? Just chop it into bins and pretend it's discrete.*

---

## Quick Start

```bash
# Install the goods
pip install -r rl_fundamentals/requirements.txt

# === Week 1: Dynamic Programming ===
# Watch GridWorld get solved
python rl_fundamentals/05_env_applications/gridworld/gridworld_dp.py

# See a drunk agent navigate ice
python rl_fundamentals/05_env_applications/frozenlake/solve_frozenlake.py

# Experience Taxi dispatcher hell
python rl_fundamentals/05_env_applications/taxi_v3/solve_taxi.py

# === Week 2: Temporal Difference ===
# The classic Q-Learning vs SARSA comparison
python rl_fundamentals/07_td_applications/cliffwalking/solve_cliffwalking.py

# Discretized Q-Learning on CartPole
python rl_fundamentals/07_td_applications/cartpole/solve_cartpole.py
```

---

## Speedrun Progress

- [x] **Dynamic Programming** - When you have the cheat codes (full model)
- [x] **Temporal Difference** - Q-Learning & SARSA (model-free vibes)
- [ ] **Monte Carlo** - When you just yolo sample full episodes
- [ ] **Policy Gradient** - Going continuous
- [ ] **Actor-Critic** - Why choose when you can have both
- [ ] **Deep RL** - Neural networks enter the chat

---

## The Algorithms

### Week 1: Dynamic Programming (Model-Based)

| Algorithm | Update Rule | Requires Model? |
|-----------|-------------|-----------------|
| **Value Iteration** | V(s) ← max_a Σ P(s'|s,a)[R + γV(s')] | Yes |
| **Policy Iteration** | Evaluate → Improve → Repeat | Yes |

### Week 2: Temporal Difference (Model-Free)

| Algorithm | Update Rule | Policy Type |
|-----------|-------------|-------------|
| **Q-Learning** | Q(S,A) ← Q(S,A) + α[R + γ·max_a Q(S',a) - Q(S,A)] | Off-policy |
| **SARSA** | Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)] | On-policy |

The key difference? Q-Learning uses `max` (what's optimal), SARSA uses the actual next action (what we'll really do).

---

## Philosophy

This repo follows the ancient wisdom:

1. **Understand the math** - Actually derive things, no hand-waving
2. **Implement from scratch** - Suffering builds character
3. **Visualize everything** - Pretty pictures > walls of numbers
4. **Keep it real** - Comments are for future confused me

---

## Resources I'm Stealing From

- Sutton & Barto's RL Book (the bible)
- David Silver's lectures (goated)
- OpenAI Spinning Up (documentation supremacy)
- Stack Overflow (no shame)

---

*Currently speedrunning: Temporal Difference Learning* ✓

*Next up: Policy Gradients (going continuous)*

**Stars appreciated, issues tolerated, PRs celebrated** ⭐
