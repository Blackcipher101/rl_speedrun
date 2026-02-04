# Policy Gradient Methods

Policy gradient methods directly optimize the policy without learning a value function first. Instead of learning Q(s,a) and deriving a policy, we parameterize the policy and optimize it using gradient ascent.

---

## Prerequisites

- Basic calculus (gradients, chain rule)
- Probability distributions and expectations
- Neural networks (for function approximation)

---

## 1. The Key Insight: Direct Policy Optimization

### 1.1 Value-Based vs Policy-Based

| Aspect | Value-Based (Q-Learning) | Policy-Based (REINFORCE) |
|--------|--------------------------|--------------------------|
| **What we learn** | Q(s, a) or V(s) | π(a\|s) directly |
| **Policy derivation** | Implicit (argmax Q) | Explicit (parameterized) |
| **Action space** | Discrete (finite) | Discrete or continuous |
| **Stochasticity** | Deterministic (ε-greedy hack) | Naturally stochastic |
| **Convergence** | To local optimum of Q | To local optimum of J(θ) |

### 1.2 Why Policy Gradients?

1. **Continuous actions**: Can handle continuous action spaces naturally
2. **Stochastic policies**: Can learn optimal stochastic policies
3. **Convergence**: Guaranteed convergence to local optimum
4. **Prior knowledge**: Can embed domain knowledge in policy structure

---

## 2. Policy Parameterization

### 2.1 Softmax Policy (Discrete Actions)

For discrete actions, use softmax over action preferences:

$$\pi_\theta(a|s) = \frac{e^{h(s,a,\theta)}}{\sum_{a'} e^{h(s,a',\theta)}}$$

where $h(s,a,\theta)$ is a preference function (e.g., linear or neural network).

**Linear preference**:
$$h(s,a,\theta) = \theta^\top \phi(s,a)$$

### 2.2 Gaussian Policy (Continuous Actions)

For continuous actions, use Gaussian:

$$\pi_\theta(a|s) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(a - \mu_\theta(s))^2}{2\sigma^2}\right)$$

where $\mu_\theta(s)$ is the mean (function of state) and $\sigma$ is the standard deviation.

---

## 3. The Policy Gradient Theorem

### 3.1 Objective Function

We want to maximize expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ is a trajectory.

### 3.2 The Gradient

**Policy Gradient Theorem**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the return from time $t$.

### 3.3 Derivation (Simplified)

Starting from:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_\tau P(\tau|\theta) R(\tau)$$

Taking the gradient:
$$\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P(\tau|\theta) R(\tau)$$

Using the log-derivative trick: $\nabla_\theta P(\tau|\theta) = P(\tau|\theta) \nabla_\theta \log P(\tau|\theta)$

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) R(\tau) = \mathbb{E}_{\tau}\left[\nabla_\theta \log P(\tau|\theta) R(\tau)\right]$$

Since $P(\tau|\theta) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)$:

$$\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

(Transition dynamics don't depend on θ!)

---

## 4. REINFORCE Algorithm

### 4.1 Basic REINFORCE

**Algorithm**:
```
Initialize policy parameters θ randomly
For each episode:
    Generate trajectory τ = (s₀, a₀, r₀, ..., sₜ, aₜ, rₜ) following π_θ
    For t = 0, 1, ..., T-1:
        G_t ← Σₖ₌ₜᵀ γᵏ⁻ᵗ rₖ  (return from t)
        θ ← θ + α · γᵗ · G_t · ∇_θ log π_θ(aₜ|sₜ)
```

### 4.2 REINFORCE with Baseline

**Issue**: High variance in gradient estimates.

**Solution**: Subtract a baseline $b(s_t)$ that doesn't depend on actions:

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b(s_t))\right]$$

**Common baseline**: $b(s) = V^\pi(s)$ (learned value function)

**Why baselines work**:
- $\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = 0$ (baseline doesn't change expected gradient)
- But reduces variance: $\text{Var}[G_t - b] < \text{Var}[G_t]$ when $b \approx \mathbb{E}[G_t]$

### 4.3 Algorithm with Baseline

```
Initialize policy parameters θ
Initialize value function parameters w
For each episode:
    Generate trajectory τ following π_θ
    For t = 0, 1, ..., T-1:
        G_t ← Σₖ₌ₜᵀ γᵏ⁻ᵗ rₖ
        δ_t ← G_t - V_w(sₜ)           # Advantage estimate
        w ← w + α_w · δ_t · ∇_w V_w(sₜ)  # Update value function
        θ ← θ + α_θ · γᵗ · δ_t · ∇_θ log π_θ(aₜ|sₜ)  # Update policy
```

---

## 5. Return Normalization

### 5.1 The Problem

Returns can have very different scales across episodes:
- Episode 1: G = 100
- Episode 2: G = 500
- Episode 3: G = -50

This causes unstable gradients.

### 5.2 Normalization Techniques

**Per-episode normalization**:
$$\hat{G}_t = \frac{G_t - \mu_G}{\sigma_G + \epsilon}$$

where $\mu_G$ and $\sigma_G$ are computed over the episode's returns.

**Running normalization**:
$$\hat{G}_t = \frac{G_t - \mu_{running}}{\sigma_{running} + \epsilon}$$

Using exponential moving average for $\mu$ and $\sigma$.

### 5.3 Why It Helps

- Keeps gradients in a reasonable range
- Prevents large updates from outlier episodes
- Improves training stability

---

## 6. Implementation Details

### 6.1 Neural Network Policy

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
```

### 6.2 Computing Log Probabilities

For discrete actions with softmax:
```python
probs = policy(state)
dist = Categorical(probs)
action = dist.sample()
log_prob = dist.log_prob(action)
```

### 6.3 Gradient Update

```python
# Compute policy gradient loss
policy_loss = []
for log_prob, G in zip(saved_log_probs, returns):
    policy_loss.append(-log_prob * G)  # Negative for gradient ascent
loss = torch.stack(policy_loss).sum()

# Backpropagate
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 7. Practical Considerations

### 7.1 Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Learning rate | 1e-4 to 1e-2 | Start small, increase if stable |
| γ (discount) | 0.99 | Standard choice |
| Hidden units | 64-256 | Depends on problem complexity |
| Batch size | 1-32 episodes | More = lower variance |

### 7.2 Common Issues

1. **High variance**: Use baselines, normalize returns
2. **Slow convergence**: Increase batch size, tune learning rate
3. **Reward scaling**: Normalize or clip rewards
4. **Entropy collapse**: Add entropy bonus to encourage exploration

### 7.3 Entropy Regularization

Add entropy term to encourage exploration:

$$J(\theta) = \mathbb{E}[R(\tau)] + \beta H(\pi_\theta)$$

where $H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$

---

## 8. Summary

### REINFORCE Update

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \gamma^t G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

### REINFORCE with Baseline

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \gamma^t (G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t|s_t)$$

### Key Equations

| Concept | Equation |
|---------|----------|
| Objective | $J(\theta) = \mathbb{E}_{\pi_\theta}[R(\tau)]$ |
| Policy Gradient | $\nabla J = \mathbb{E}[\nabla \log \pi_\theta(a\|s) G_t]$ |
| Log-derivative trick | $\nabla P = P \nabla \log P$ |

---

## References

1. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). Chapter 13.
3. Schulman, J., et al. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*.
