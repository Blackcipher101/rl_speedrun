# Deep Q-Networks (DQN) Fundamentals

This module implements DQN from scratch using pure NumPy, covering the core innovations that enabled deep reinforcement learning to achieve human-level performance on Atari games.

---

## The Challenge: Combining Deep Learning with RL

### The Deadly Triad

Combining three elements leads to instability and divergence:

```
┌─────────────────────────────────────────────────────────────┐
│                     THE DEADLY TRIAD                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Function Approximation                                 │
│      └─ Neural networks generalize but can diverge         │
│                                                             │
│   2. Bootstrapping                                          │
│      └─ TD learning updates from own estimates             │
│                                                             │
│   3. Off-Policy Learning                                    │
│      └─ Learning from different behavior policy            │
│                                                             │
│   All three together → Unstable learning, divergence       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

DQN solves this with two key innovations:
1. **Experience Replay** - Breaks correlation, enables data reuse
2. **Target Networks** - Stabilizes learning targets

---

## 1. Experience Replay

### The Problem: Correlated Samples

In standard online RL:
- Consecutive samples are highly correlated (same trajectory)
- Neural networks expect i.i.d. data
- Single experience used once, then discarded

```
Online Learning (Unstable):
t=0: (s₀, a₀, r₁, s₁) → update → forget
t=1: (s₁, a₁, r₂, s₂) → update → forget  ← Correlated with t=0!
t=2: (s₂, a₂, r₃, s₃) → update → forget  ← Correlated with t=1!
```

### The Solution: Replay Buffer

Store transitions and sample randomly:

```
                    Experience Replay Buffer
                ┌────────────────────────────────┐
                │  (s₀, a₀, r₁, s₁, done)        │
                │  (s₁, a₁, r₂, s₂, done)        │
                │  (s₂, a₂, r₃, s₃, done)        │
      push() →  │  ...                           │  → sample(batch)
                │  (sₙ, aₙ, rₙ₊₁, sₙ₊₁, done)    │
                └────────────────────────────────┘
                        Circular buffer
                     (oldest overwritten)
```

### Implementation

```python
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros(capacity, dtype=int)
        self.rewards = np.zeros(capacity)
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros(capacity, dtype=bool)
        self.position = 0
        self.size = 0

    def push(self, s, a, r, s_next, done):
        self.states[self.position] = s
        # ... store other components
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (self.states[indices], self.actions[indices], ...)
```

### Benefits

| Aspect | Online Learning | Experience Replay |
|--------|-----------------|-------------------|
| Sample correlation | High | Low (random sampling) |
| Data efficiency | 1x (use once) | Nx (reuse many times) |
| Learning stability | Unstable | Stable |
| Memory | O(1) | O(buffer_size) |

---

## 2. Target Networks

### The Problem: Moving Targets

In Q-learning, we minimize:
$$L = (Q(s,a) - y)^2$$

Where the target is:
$$y = r + \gamma \max_{a'} Q(s', a')$$

**Problem**: Both $Q(s,a)$ and $\max_{a'} Q(s', a')$ use the same network!
- Every update changes the target
- "Chasing a moving target"
- Leads to oscillation and divergence

```
Without Target Network:
                    ┌─────────────┐
     Q(s,a) ←───────│  Q-Network  │←─────── Q(s',a')
                    └─────────────┘
                          ↑
                    Both change together!
```

### The Solution: Separate Target Network

Use a separate, slowly-updated network for targets:

```
With Target Network:
                    ┌─────────────┐
     Q(s,a) ←───────│  Q-Network  │  (updated every step)
                    └─────────────┘

                    ┌─────────────┐
    Q_target(s',a')─│Target Network│  (updated every C steps)
                    └─────────────┘
```

### Update Strategies

**Hard Update** (Original DQN):
$$\theta^- \leftarrow \theta \quad \text{every } C \text{ steps}$$

**Soft Update** (DDPG-style):
$$\theta^- \leftarrow \tau \theta + (1-\tau) \theta^- \quad \text{every step}$$

Typical values:
- Hard: $C = 100$ to $10000$ steps
- Soft: $\tau = 0.001$ to $0.01$

---

## 3. The Overestimation Problem

### Why Q-Learning Overestimates

The max operator introduces positive bias:

$$\mathbb{E}[\max_a Q(s,a)] \geq \max_a \mathbb{E}[Q(s,a)]$$

**Intuition**: If Q-values have noise, max picks the highest noise, not the best action.

```
True Q-values:     [1.0, 1.0, 1.0]
Noisy estimates:   [1.2, 0.8, 1.1]  (random noise)
max picks:         1.2 (overestimate!)
True max:          1.0
```

### Accumulation Over Time

The overestimation compounds through bootstrapping:

```
Episode 1: Q(s,a) slightly too high
Episode 2: Target uses max Q(s',a') → even higher
Episode 3: Further overestimation
...
Eventually: Q-values explode or become unstable
```

**Solution**: Double DQN (covered in 14_dqn_improvements/)

---

## 4. DQN Algorithm

### Complete Algorithm

```
Algorithm: Deep Q-Network (DQN)
────────────────────────────────────────────────
Initialize Q-network with random weights θ
Initialize target network with weights θ⁻ = θ
Initialize replay buffer D with capacity N

for episode = 1 to M:
    s ← env.reset()
    for t = 1 to T:
        # ε-greedy action selection
        with probability ε:
            a ← random action
        otherwise:
            a ← argmax_a Q(s, a; θ)

        # Execute action
        s', r, done ← env.step(a)

        # Store transition
        D.push(s, a, r, s', done)

        # Sample and train
        if len(D) > min_buffer_size:
            batch ← D.sample(batch_size)
            states, actions, rewards, next_states, dones ← batch

            # Compute targets using target network
            y = r + γ · max_a' Q(s', a'; θ⁻) · (1 - done)

            # Update Q-network
            L = MSE(Q(s, a; θ), y)
            θ ← θ - α∇L

            # Update target network (every C steps)
            if step % C == 0:
                θ⁻ ← θ

        s ← s'
        if done: break

    # Decay exploration
    ε ← max(ε_min, ε · decay)
```

### Loss Function

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}\left[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2\right]$$

---

## 5. Neural Network Architecture

### Network Structure

```
Input Layer: state_dim (e.g., 4 for CartPole)
    │
    ▼
Hidden Layer 1: 64 units, ReLU
    │
    ▼
Hidden Layer 2: 64 units, ReLU
    │
    ▼
Output Layer: action_dim (e.g., 2 for CartPole)
    │
    ▼
Q-values: Q(s, left), Q(s, right)
```

### Forward Pass

$$h_1 = \text{ReLU}(W_1 \cdot s + b_1)$$
$$h_2 = \text{ReLU}(W_2 \cdot h_1 + b_2)$$
$$Q(s, \cdot) = W_3 \cdot h_2 + b_3$$

### Gradient Computation

Using backpropagation:
$$\frac{\partial L}{\partial \theta} = 2(Q(s,a;\theta) - y) \cdot \frac{\partial Q(s,a;\theta)}{\partial \theta}$$

---

## 6. Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Learning rate | 1e-4 to 1e-3 | Lower for stability |
| Batch size | 32 to 128 | Larger = more stable |
| Buffer size | 10k to 1M | Depends on problem |
| γ (discount) | 0.99 | Standard |
| Target update freq | 100 to 10000 | Higher = more stable |
| ε start | 1.0 | Full exploration |
| ε end | 0.01 to 0.1 | Some exploration remains |
| ε decay | 0.995 to 0.9999 | Per episode |

---

## 7. Module Structure

```
13_dqn_fundamentals/
├── README.md           # This file
├── replay_buffer.py    # Experience replay implementation
├── target_network.py   # Target network management
└── dqn.py              # Complete DQN algorithm
```

---

## 8. Running the Code

```bash
cd rl_fundamentals/13_dqn_fundamentals

# Test replay buffer
python replay_buffer.py

# Test target network
python target_network.py

# Train DQN on CartPole
python dqn.py
```

### Expected Output

```
Episode   50 | Avg Return:   22.50 | Epsilon: 0.778
Episode  100 | Avg Return:   45.30 | Epsilon: 0.605
Episode  150 | Avg Return:   89.70 | Epsilon: 0.471
Episode  200 | Avg Return:  156.20 | Epsilon: 0.366
Episode  250 | Avg Return:  312.80 | Epsilon: 0.285
Episode  300 | Avg Return:  478.50 | Epsilon: 0.222

Evaluation: Mean return = 487.2 +/- 25.3
SOLVED! (Mean return >= 195)
```

---

## 9. Key Insights

### Why DQN Works

1. **Experience Replay**
   - Breaks temporal correlation → stable gradients
   - Reuses data → sample efficient
   - Larger effective dataset → better generalization

2. **Target Networks**
   - Fixed targets → stable optimization
   - Prevents feedback loop → no oscillation
   - Delayed updates → smoothed learning

3. **Together**
   - Both necessary for stable deep RL
   - Neither alone is sufficient

### Limitations

| Issue | Description | Solution |
|-------|-------------|----------|
| Overestimation | max operator biased | Double DQN |
| Sample efficiency | Still needs many samples | Prioritized replay |
| Discrete actions only | Can't handle continuous | DDPG, SAC |
| High variance | Noisy gradients | Dueling DQN |

---

## 10. References

1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

2. Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. *arXiv preprint arXiv:1312.5602*.

3. Lin, L. J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning*, 8(3-4), 293-321.

4. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. Chapter 16.5-16.7.
