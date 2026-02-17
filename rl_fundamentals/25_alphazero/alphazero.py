"""
AlphaZero — Self-play reinforcement learning with MCTS and neural network guidance.

Replaces MCTS random rollouts with a policy-value network.
The network provides both prior action probabilities P(a|s) and
a value estimate v(s), eliminating the need for simulation to terminal states.

Reference: Silver et al., 2017 — "Mastering the game of Go without human knowledge"

Author: RL Speedrun
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict, List, Tuple, Optional
from collections import deque

from games import TicTacToe, ConnectFour

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# Policy-Value Network
# ==============================================================================

class ResidualBlock(nn.Module):
    """Residual block: Conv → BN → ReLU → Conv → BN + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class PolicyValueNet(nn.Module):
    """Shared trunk → policy head + value head.

    Input: (batch, 3, H, W) — 3 channels (current player, opponent, color)
    Policy head: Conv 1x1 → flatten → Linear → action_size (log softmax)
    Value head: Conv 1x1 → flatten → Linear → Tanh
    """

    def __init__(self, board_size: Tuple[int, int], action_size: int,
                 n_channels: int = 64, n_blocks: int = 2):
        super().__init__()
        h, w = board_size
        self.action_size = action_size

        # Input projection
        self.input_conv = nn.Conv2d(3, n_channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(n_channels)

        # Residual trunk
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(n_channels) for _ in range(n_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(n_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * h * w, action_size)

        # Value head
        self.value_conv = nn.Conv2d(n_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(h * w, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (log_policy, value)."""
        # Trunk
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        log_policy = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return log_policy, v


# ==============================================================================
# AlphaZero MCTS
# ==============================================================================

class AlphaZeroNode:
    """MCTS node with neural network prior P(a|s)."""

    def __init__(self, prior: float):
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, 'AlphaZeroNode'] = {}
        self.state: Optional[np.ndarray] = None

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class AlphaZeroMCTS:
    """MCTS guided by neural network (no random rollouts).

    Uses PUCT instead of UCB1:
        PUCT(s,a) = Q(s,a) + c_puct * P(a|s) * sqrt(N(parent)) / (1 + N(s,a))
    """

    def __init__(self, game, model: PolicyValueNet, n_simulations: int = 100,
                 c_puct: float = 1.0, device: str = 'cpu'):
        self.game = game
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.device = device

    @torch.no_grad()
    def search(self, state: np.ndarray, player: int) -> np.ndarray:
        """Run MCTS from state, return visit-count policy vector."""
        root = AlphaZeroNode(prior=0)
        root.state = state.copy()

        # Expand root with network evaluation
        self._expand(root, state, player)

        # Add Dirichlet noise to root for exploration
        legal = self.game.get_legal_actions(state)
        noise = np.random.dirichlet([0.3] * len(legal))
        for i, action in enumerate(legal):
            if action in root.children:
                root.children[action].prior = (
                    0.75 * root.children[action].prior + 0.25 * noise[i]
                )

        for _ in range(self.n_simulations):
            node = root
            search_path = [node]
            current_state = state.copy()
            current_player = player

            # Selection: walk tree using PUCT
            while node.is_expanded():
                action, node = self._select_child(node)
                current_state, current_player = self.game.step(
                    current_state, action, current_player
                )
                search_path.append(node)

            # Check terminal
            reward = self.game.get_reward(current_state, player)
            if reward is not None:
                # Terminal node — backpropagate actual reward
                self._backpropagate(search_path, reward)
                continue

            # Expansion + evaluation
            value = self._expand(node, current_state, current_player)
            # Value is from current_player's perspective, convert to root player's
            if current_player != player:
                value = -value
            self._backpropagate(search_path, value)

        # Build policy from visit counts
        policy = np.zeros(self.game.action_size, dtype=np.float32)
        for action, child in root.children.items():
            policy[action] = child.visit_count
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        return policy

    def _select_child(self, node: AlphaZeroNode) -> Tuple[int, AlphaZeroNode]:
        """Select child with highest PUCT score."""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        sqrt_parent = math.sqrt(node.visit_count)
        for action, child in node.children.items():
            puct = child.q_value() + self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            if puct > best_score:
                best_score = puct
                best_action = action
                best_child = child
        return best_action, best_child

    def _expand(self, node: AlphaZeroNode, state: np.ndarray, player: int) -> float:
        """Expand node using network. Returns value estimate."""
        encoded = self.game.encode_state(state, player)
        encoded_tensor = torch.FloatTensor(encoded).unsqueeze(0).to(self.device)
        log_policy, value = self.model(encoded_tensor)
        policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
        value = value.item()

        # Mask illegal moves and renormalize
        legal = self.game.get_legal_actions(state)
        legal_mask = np.zeros(self.game.action_size, dtype=np.float32)
        for a in legal:
            legal_mask[a] = 1.0
        policy = policy * legal_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        else:
            # Uniform over legal moves
            policy = legal_mask / legal_mask.sum()

        for action in legal:
            node.children[action] = AlphaZeroNode(prior=policy[action])

        node.state = state.copy()
        return value

    def _backpropagate(self, search_path: List[AlphaZeroNode], value: float):
        """Update visit counts and values along search path."""
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value


# ==============================================================================
# Self-Play Buffer
# ==============================================================================

class SelfPlayBuffer:
    """Stores (encoded_state, mcts_policy, outcome) tuples from self-play."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, encoded_state: np.ndarray, policy: np.ndarray, value: float):
        self.buffer.append((encoded_state, policy, value))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)),
                                   replace=False)
        states, policies, values = [], [], []
        for i in indices:
            s, p, v = self.buffer[i]
            states.append(s)
            policies.append(p)
            values.append(v)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(policies)),
            torch.FloatTensor(np.array(values)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# Self-Play Game
# ==============================================================================

def self_play_game(game, model, n_simulations: int = 100, c_puct: float = 1.0,
                   temperature_threshold: int = 15, device: str = 'cpu'):
    """Play one game of self-play using MCTS.

    Returns list of (encoded_state, mcts_policy, player) tuples.
    After game ends, outcomes are assigned: +1 for winner, -1 for loser, 0 draw.
    """
    mcts = AlphaZeroMCTS(game, model, n_simulations=n_simulations,
                          c_puct=c_puct, device=device)
    state = game.get_initial_state()
    player = 1
    history = []  # (encoded_state, mcts_policy, player)
    move_count = 0

    while not game.is_terminal(state):
        policy = mcts.search(state, player)
        encoded = game.encode_state(state, player)
        history.append((encoded, policy, player))

        # Temperature-based action selection
        if move_count < temperature_threshold:
            # Sample proportional to visit counts
            action = np.random.choice(game.action_size, p=policy)
        else:
            # Greedy
            action = int(np.argmax(policy))

        state, player = game.step(state, action, player)
        move_count += 1

    # Assign outcomes
    training_data = []
    for encoded, policy, p in history:
        reward = game.get_reward(state, p)
        if reward is None:
            reward = 0.0
        training_data.append((encoded, policy, reward))

    return training_data


# ==============================================================================
# Trainer
# ==============================================================================

class AlphaZeroTrainer:
    """Full AlphaZero training loop: self-play → train → evaluate."""

    def __init__(self, game, n_iterations: int = 100, n_self_play_games: int = 25,
                 n_training_steps: int = 200, batch_size: int = 64,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 n_simulations: int = 100, c_puct: float = 1.0,
                 n_channels: int = 64, n_blocks: int = 2,
                 temperature_threshold: int = 15,
                 n_eval_games: int = 20, eval_threshold: float = 0.55,
                 buffer_capacity: int = 50000, device: str = 'cpu'):
        self.game = game
        self.n_iterations = n_iterations
        self.n_self_play_games = n_self_play_games
        self.n_training_steps = n_training_steps
        self.batch_size = batch_size
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.temperature_threshold = temperature_threshold
        self.n_eval_games = n_eval_games
        self.eval_threshold = eval_threshold
        self.device = device

        self.model = PolicyValueNet(
            game.board_size, game.action_size,
            n_channels=n_channels, n_blocks=n_blocks
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.buffer = SelfPlayBuffer(capacity=buffer_capacity)

        self.history = {
            'iteration': [],
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'win_rate_vs_random': [],
        }

    def train(self):
        """Full training loop."""
        for iteration in range(1, self.n_iterations + 1):
            # 1. Self-play
            self.model.eval()
            for _ in range(self.n_self_play_games):
                game_data = self_play_game(
                    self.game, self.model,
                    n_simulations=self.n_simulations,
                    c_puct=self.c_puct,
                    temperature_threshold=self.temperature_threshold,
                    device=self.device,
                )
                for encoded, policy, value in game_data:
                    self.buffer.push(encoded, policy, value)

            # 2. Train
            if len(self.buffer) < self.batch_size:
                continue
            self.model.train()
            total_p_loss = 0
            total_v_loss = 0
            n_steps = min(self.n_training_steps, len(self.buffer) // self.batch_size)
            n_steps = max(n_steps, 1)
            for _ in range(n_steps):
                p_loss, v_loss = self._train_step()
                total_p_loss += p_loss
                total_v_loss += v_loss
            avg_p_loss = total_p_loss / n_steps
            avg_v_loss = total_v_loss / n_steps

            # 3. Evaluate vs random
            win_rate = self._evaluate_vs_random()

            self.history['iteration'].append(iteration)
            self.history['policy_loss'].append(avg_p_loss)
            self.history['value_loss'].append(avg_v_loss)
            self.history['total_loss'].append(avg_p_loss + avg_v_loss)
            self.history['win_rate_vs_random'].append(win_rate)

            if iteration % 10 == 0 or iteration == 1:
                print(f"Iter {iteration:4d} | P_loss: {avg_p_loss:.4f} | "
                      f"V_loss: {avg_v_loss:.4f} | "
                      f"Win vs Random: {win_rate*100:.0f}% | "
                      f"Buffer: {len(self.buffer)}")

        return self.model, self.history

    def _train_step(self) -> Tuple[float, float]:
        """Single gradient step. Returns (policy_loss, value_loss)."""
        states, target_policies, target_values = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)

        log_policy, value = self.model(states)

        # Policy loss: cross-entropy (using target as soft labels)
        policy_loss = -torch.sum(target_policies * log_policy, dim=1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(value, target_values)

        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item()

    @torch.no_grad()
    def _evaluate_vs_random(self, n_games: int = None) -> float:
        """Play against random, return win rate."""
        if n_games is None:
            n_games = self.n_eval_games
        self.model.eval()
        mcts = AlphaZeroMCTS(self.game, self.model, n_simulations=self.n_simulations // 2,
                              c_puct=self.c_puct, device=self.device)
        wins = 0
        for game_i in range(n_games):
            state = self.game.get_initial_state()
            player = 1
            # Alternate who goes first
            az_player = 1 if game_i % 2 == 0 else -1

            while not self.game.is_terminal(state):
                if player == az_player:
                    policy = mcts.search(state, player)
                    action = int(np.argmax(policy))
                else:
                    legal = self.game.get_legal_actions(state)
                    action = np.random.choice(legal)
                state, player = self.game.step(state, action, player)

            reward = self.game.get_reward(state, az_player)
            if reward is not None and reward > 0:
                wins += 1
            elif reward == 0:
                wins += 0.5

        return wins / n_games


# ==============================================================================
# Demo
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AlphaZero — TicTacToe Self-Play Training")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    game = TicTacToe()
    trainer = AlphaZeroTrainer(
        game,
        n_iterations=50,
        n_self_play_games=20,
        n_training_steps=100,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        n_simulations=50,
        c_puct=1.0,
        n_channels=64,
        n_blocks=2,
        temperature_threshold=8,
        n_eval_games=20,
        buffer_capacity=50000,
    )

    print("\nTraining...")
    print("-" * 60)
    model, history = trainer.train()

    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Evaluation (40 games vs random)")
    print("-" * 60)
    final_wr = trainer._evaluate_vs_random(n_games=40)
    print(f"  Win rate vs random: {final_wr*100:.0f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history['iteration'], history['total_loss'], 'b-', linewidth=1.5)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['iteration'],
                 [wr * 100 for wr in history['win_rate_vs_random']],
                 'g-', linewidth=1.5)
    axes[1].axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95%')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Win Rate vs Random (%)')
    axes[1].set_title('AlphaZero vs Random')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'alphazero_demo.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
