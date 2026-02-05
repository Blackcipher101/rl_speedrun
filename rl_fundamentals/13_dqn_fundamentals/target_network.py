"""
Target Network Management for DQN

Target networks provide stable learning targets, preventing the "moving target"
problem that arises when using the same network for both action selection and
Q-value computation.

Key insight: Without target networks, the targets y = r + gamma * max Q(s',a')
shift with every update, causing instability and divergence.

Two update strategies:
1. Hard update: Copy weights every C steps (original DQN)
2. Soft update: Exponential moving average θ' = τθ + (1-τ)θ' (smoother)

Author: RL Speedrun
"""

import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy


class NumpyNetwork:
    """
    Simple feedforward neural network in NumPy.

    Architecture: state_dim -> hidden1 -> hidden2 -> action_dim

    This is designed to be copied for target networks.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 seed: int = 42):
        """
        Initialize network with Xavier initialization.

        Args:
            state_dim: Input dimension (state size)
            action_dim: Output dimension (number of actions)
            hidden_dims: List of hidden layer sizes
            seed: Random seed for weight initialization
        """
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        # Build network parameters
        self.params = {}
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/He initialization."""
        dims = [self.state_dim] + self.hidden_dims + [self.action_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]

            # Xavier initialization for hidden layers
            # He initialization for ReLU layers
            if i < len(dims) - 2:  # Hidden layers with ReLU
                scale = np.sqrt(2.0 / fan_in)  # He init
            else:  # Output layer
                scale = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier init

            self.params[f'W{i}'] = self.rng.standard_normal((fan_in, fan_out)) * scale
            self.params[f'b{i}'] = np.zeros(fan_out)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            state: Input state(s), shape (state_dim,) or (batch, state_dim)

        Returns:
            Q-values for each action, shape (action_dim,) or (batch, action_dim)
        """
        # Ensure 2D input
        single_input = state.ndim == 1
        if single_input:
            state = state.reshape(1, -1)

        x = state
        n_layers = len(self.hidden_dims) + 1

        for i in range(n_layers):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            x = x @ W + b

            # ReLU activation for hidden layers only
            if i < n_layers - 1:
                x = np.maximum(0, x)

        if single_input:
            x = x.squeeze(0)

        return x

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return a deep copy of network parameters."""
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set network parameters from a dictionary."""
        for k, v in params.items():
            self.params[k] = v.copy()

    def copy(self) -> 'NumpyNetwork':
        """Create a deep copy of this network."""
        new_net = NumpyNetwork(
            self.state_dim, self.action_dim,
            self.hidden_dims, seed=0  # Seed doesn't matter for copy
        )
        new_net.set_params(self.get_params())
        return new_net


class TargetNetworkManager:
    """
    Manages target network updates for stable DQN training.

    Supports two update strategies:
    1. Hard update: θ_target = θ_online (every C steps)
    2. Soft update: θ_target = τ*θ_online + (1-τ)*θ_target (every step)
    """

    def __init__(self, online_network: NumpyNetwork,
                 update_type: str = "hard",
                 update_freq: int = 100,
                 tau: float = 0.005):
        """
        Initialize target network manager.

        Args:
            online_network: The network being trained
            update_type: "hard" or "soft"
            update_freq: Steps between hard updates (ignored for soft)
            tau: Soft update interpolation factor (ignored for hard)
        """
        self.online_network = online_network
        self.target_network = online_network.copy()
        self.update_type = update_type
        self.update_freq = update_freq
        self.tau = tau
        self.step_count = 0

    def get_target_network(self) -> NumpyNetwork:
        """Return the target network for computing TD targets."""
        return self.target_network

    def step(self) -> bool:
        """
        Perform one step of target network management.

        Call this after each training step.

        Returns:
            True if target network was updated, False otherwise
        """
        self.step_count += 1
        updated = False

        if self.update_type == "hard":
            # Hard update every C steps
            if self.step_count % self.update_freq == 0:
                self.hard_update()
                updated = True

        elif self.update_type == "soft":
            # Soft update every step
            self.soft_update()
            updated = True

        return updated

    def hard_update(self) -> None:
        """
        Hard update: Copy online network weights to target network.

        θ_target = θ_online
        """
        self.target_network.set_params(self.online_network.get_params())

    def soft_update(self) -> None:
        """
        Soft update: Exponential moving average of weights.

        θ_target = τ * θ_online + (1 - τ) * θ_target

        Typical τ = 0.001 to 0.01 (slow update for stability)
        """
        online_params = self.online_network.get_params()
        target_params = self.target_network.get_params()

        for key in online_params:
            target_params[key] = (
                self.tau * online_params[key] +
                (1 - self.tau) * target_params[key]
            )

        self.target_network.set_params(target_params)

    def force_sync(self) -> None:
        """Force synchronize target to online (used at start of training)."""
        self.hard_update()


def compute_td_targets(
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    target_network: NumpyNetwork,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute TD targets for DQN using target network.

    y_i = r_i + gamma * max_a' Q_target(s'_i, a')  if not done
    y_i = r_i                                       if done

    Args:
        rewards: Batch of rewards (batch_size,)
        next_states: Batch of next states (batch_size, state_dim)
        dones: Batch of done flags (batch_size,)
        target_network: Target Q-network
        gamma: Discount factor

    Returns:
        TD targets (batch_size,)
    """
    # Get Q-values for next states from target network
    next_q_values = target_network.forward(next_states)

    # Max Q-value for each next state
    max_next_q = np.max(next_q_values, axis=1)

    # TD targets: r + gamma * max Q'(s', a') * (1 - done)
    targets = rewards + gamma * max_next_q * (1 - dones.astype(np.float32))

    return targets


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Target Network Management Demo")
    print("=" * 60)

    # Create networks
    state_dim = 4
    action_dim = 2
    hidden_dims = [64, 64]

    online_net = NumpyNetwork(state_dim, action_dim, hidden_dims, seed=42)
    print(f"\nNetwork architecture: {state_dim} -> {hidden_dims} -> {action_dim}")

    # Test forward pass
    state = np.random.randn(state_dim).astype(np.float32)
    q_values = online_net.forward(state)
    print(f"\nSingle state forward pass:")
    print(f"  Input shape: {state.shape}")
    print(f"  Output (Q-values): {q_values}")

    # Batch forward pass
    batch_states = np.random.randn(32, state_dim).astype(np.float32)
    batch_q = online_net.forward(batch_states)
    print(f"\nBatch forward pass:")
    print(f"  Input shape: {batch_states.shape}")
    print(f"  Output shape: {batch_q.shape}")

    # Test hard update manager
    print("\n" + "-" * 40)
    print("Hard Update Strategy")
    print("-" * 40)

    hard_manager = TargetNetworkManager(
        online_net, update_type="hard", update_freq=10
    )

    # Simulate training steps
    print("\nSimulating 25 training steps (update every 10):")
    for step in range(25):
        # Simulate modifying online network
        online_net.params['W0'] += 0.01 * np.random.randn(*online_net.params['W0'].shape)

        updated = hard_manager.step()
        if updated:
            print(f"  Step {step + 1}: Target network UPDATED")

    # Check divergence between online and target
    online_q = online_net.forward(state)
    target_q = hard_manager.get_target_network().forward(state)
    print(f"\nAfter 25 steps:")
    print(f"  Online Q-values:  {online_q}")
    print(f"  Target Q-values:  {target_q}")
    print(f"  Difference: {np.abs(online_q - target_q).mean():.4f}")

    # Test soft update manager
    print("\n" + "-" * 40)
    print("Soft Update Strategy (tau=0.1)")
    print("-" * 40)

    # Reset online network
    online_net = NumpyNetwork(state_dim, action_dim, hidden_dims, seed=42)
    soft_manager = TargetNetworkManager(
        online_net, update_type="soft", tau=0.1
    )

    print("\nSimulating 25 training steps (continuous soft update):")
    divergences = []
    for step in range(25):
        # Simulate modifying online network
        online_net.params['W0'] += 0.01 * np.random.randn(*online_net.params['W0'].shape)

        soft_manager.step()

        # Track divergence
        online_q = online_net.forward(state)
        target_q = soft_manager.get_target_network().forward(state)
        div = np.abs(online_q - target_q).mean()
        divergences.append(div)

        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1}: Divergence = {div:.4f}")

    # Test TD target computation
    print("\n" + "-" * 40)
    print("TD Target Computation")
    print("-" * 40)

    batch_size = 8
    rewards = np.random.randn(batch_size).astype(np.float32)
    next_states = np.random.randn(batch_size, state_dim).astype(np.float32)
    dones = np.array([0, 0, 0, 1, 0, 0, 1, 0], dtype=np.bool_)

    target_net = soft_manager.get_target_network()
    targets = compute_td_targets(rewards, next_states, dones, target_net, gamma=0.99)

    print(f"\nBatch TD targets (gamma=0.99):")
    print(f"  Rewards: {rewards[:4]}...")
    print(f"  Dones:   {dones[:4]}...")
    print(f"  Targets: {targets[:4]}...")
    print(f"\nNote: Targets where done=True equal rewards (no bootstrap)")
    done_mask = dones.astype(bool)
    print(f"  Done indices: {np.where(done_mask)[0]}")
    print(f"  Targets at done: {targets[done_mask]}")
    print(f"  Rewards at done: {rewards[done_mask]}")
