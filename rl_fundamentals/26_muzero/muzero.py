"""
MuZero — Learning to plan without knowing the rules.

Unlike AlphaZero which requires a perfect game simulator for MCTS,
MuZero learns its own model of the environment and plans in a learned latent space.

Three learned functions:
- h(observation) -> hidden_state        (representation)
- g(hidden_state, action) -> next_hidden_state, reward  (dynamics)
- f(hidden_state) -> policy, value      (prediction)

Reference: Schrittwieser et al., 2020 — "Mastering Atari, Go, Chess
and Shogi by Planning with a Learned Model"

Author: RL Speedrun
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '25_alphazero'))

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque

from games import TicTacToe

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# Learned Model Components
# ==============================================================================

class RepresentationNet(nn.Module):
    """h(observation) -> hidden_state.
    Encodes raw observation into a latent representation.
    """

    def __init__(self, input_channels: int, board_h: int, board_w: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_h * board_w, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class DynamicsNet(nn.Module):
    """g(hidden_state, action) -> (next_hidden_state, reward).
    Predicts the next latent state and immediate reward.
    """

    def __init__(self, hidden_dim: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + action_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_state: torch.Tensor,
                action_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([hidden_state, action_onehot], dim=1)
        next_hidden = self.net(x)
        reward = torch.tanh(self.reward_head(next_hidden))
        return next_hidden, reward


class PredictionNet(nn.Module):
    """f(hidden_state) -> (policy, value).
    Same role as AlphaZero's policy-value heads but on latent states.
    """

    def __init__(self, hidden_dim: int, action_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(hidden_state)
        log_policy = F.log_softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return log_policy, value


class MuZeroNetwork(nn.Module):
    """Wraps all three networks into one module."""

    def __init__(self, input_channels: int, board_h: int, board_w: int,
                 action_size: int, hidden_dim: int = 64):
        super().__init__()
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.representation = RepresentationNet(input_channels, board_h, board_w, hidden_dim)
        self.dynamics = DynamicsNet(hidden_dim, action_size)
        self.prediction = PredictionNet(hidden_dim, action_size)

    def initial_inference(self, observation: torch.Tensor):
        """h(obs) then f(h) -> (hidden_state, log_policy, value)."""
        hidden = self.representation(observation)
        log_policy, value = self.prediction(hidden)
        return hidden, log_policy, value

    def recurrent_inference(self, hidden_state: torch.Tensor,
                             action: torch.Tensor):
        """g(h, a) then f(g) -> (next_hidden, reward, log_policy, value)."""
        action_onehot = F.one_hot(action.long(), self.action_size).float()
        next_hidden, reward = self.dynamics(hidden_state, action_onehot)
        log_policy, value = self.prediction(next_hidden)
        return next_hidden, reward, log_policy, value


# ==============================================================================
# MuZero MCTS (in latent space)
# ==============================================================================

class MuZeroNode:
    """MCTS node operating in latent space."""

    def __init__(self, prior: float):
        self.hidden_state: Optional[torch.Tensor] = None
        self.reward: float = 0.0
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, 'MuZeroNode'] = {}

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MuZeroMCTS:
    """MCTS in learned latent space.

    Key difference from AlphaZero MCTS:
    - Root: use representation network h(obs) to get hidden state
    - Expansion: use dynamics network g(h, a) to predict next hidden state
    - Evaluation: use prediction network f(h) to get policy and value
    - No game rules needed during search!
    """

    def __init__(self, network: MuZeroNetwork, action_size: int,
                 n_simulations: int = 50, c_puct: float = 1.25,
                 device: str = 'cpu'):
        self.network = network
        self.action_size = action_size
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.device = device

    @torch.no_grad()
    def search(self, observation: np.ndarray,
               legal_actions: Optional[List[int]] = None) -> np.ndarray:
        """Run MCTS from observation, return visit count policy."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        hidden, log_policy, value = self.network.initial_inference(obs_tensor)
        policy = torch.exp(log_policy).squeeze(0).cpu().numpy()

        root = MuZeroNode(prior=0)
        root.hidden_state = hidden

        # Mask illegal moves if provided
        if legal_actions is not None:
            mask = np.zeros(self.action_size, dtype=np.float32)
            for a in legal_actions:
                mask[a] = 1.0
            policy = policy * mask
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy /= policy_sum
            else:
                policy = mask / mask.sum()

        # Expand root
        actions_to_expand = legal_actions if legal_actions is not None else list(range(self.action_size))
        for a in actions_to_expand:
            root.children[a] = MuZeroNode(prior=policy[a])

        # Add Dirichlet noise to root
        noise = np.random.dirichlet([0.3] * len(actions_to_expand))
        for i, a in enumerate(actions_to_expand):
            root.children[a].prior = 0.75 * root.children[a].prior + 0.25 * noise[i]

        for _ in range(self.n_simulations):
            node = root
            search_path = [node]

            # Selection
            while node.is_expanded():
                action, node = self._select_child(node)
                search_path.append(node)

            # Expansion in latent space
            parent = search_path[-2]
            action_tensor = torch.LongTensor([action]).to(self.device)
            next_hidden, reward, log_policy, value = self.network.recurrent_inference(
                parent.hidden_state, action_tensor
            )
            node.hidden_state = next_hidden
            node.reward = reward.item()

            policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
            # In latent space we don't know legal moves — expand all
            for a in range(self.action_size):
                node.children[a] = MuZeroNode(prior=policy[a])

            # Convert value to root player's perspective
            # Depth from root: even depth = root's player, odd = opponent
            depth = len(search_path) - 1
            value_float = value.item()
            if depth % 2 == 1:
                value_float = -value_float
            self._backpropagate(search_path, value_float)

        # Build policy from visit counts
        result_policy = np.zeros(self.action_size, dtype=np.float32)
        for a, child in root.children.items():
            result_policy[a] = child.visit_count
        policy_sum = result_policy.sum()
        if policy_sum > 0:
            result_policy /= policy_sum
        return result_policy

    def _select_child(self, node: MuZeroNode) -> Tuple[int, MuZeroNode]:
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

    def _backpropagate(self, search_path: List[MuZeroNode], value: float):
        """All values stored from root player's perspective (same as AlphaZero)."""
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value


# ==============================================================================
# Replay Buffer with Trajectory Storage
# ==============================================================================

class GameHistory:
    """Stores a complete game trajectory."""

    def __init__(self):
        self.observations = []  # encoded observations
        self.actions = []
        self.rewards = []       # immediate rewards at each step
        self.policies = []      # MCTS policies at each step
        self.root_values = []   # MCTS root values at each step
        self.player_at_step = []

    def store(self, observation, action, reward, policy, root_value, player):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.root_values.append(root_value)
        self.player_at_step.append(player)

    def __len__(self):
        return len(self.observations)


class MuZeroReplayBuffer:
    """Stores full game trajectories for unrolled training."""

    def __init__(self, capacity: int = 1000, unroll_steps: int = 5,
                 td_steps: int = 10, discount: float = 1.0):
        self.buffer = deque(maxlen=capacity)
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.discount = discount

    def store_game(self, game_history: GameHistory):
        self.buffer.append(game_history)

    def sample_batch(self, batch_size: int) -> dict:
        """Sample positions and return K-step unroll targets."""
        games = [self.buffer[i] for i in
                 np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)),
                                  replace=True)]
        observations = []
        actions_list = []      # shape will be (batch, K)
        target_values = []     # shape will be (batch, K+1)
        target_rewards = []    # shape will be (batch, K)
        target_policies = []   # shape will be (batch, K+1, action_size)

        for game in games:
            if len(game) == 0:
                continue
            pos = np.random.randint(len(game))
            observations.append(game.observations[pos])

            actions_k = []
            values_k = []
            rewards_k = []
            policies_k = []

            # Value at current position
            values_k.append(self._compute_target_value(game, pos))
            policies_k.append(game.policies[pos])

            for k in range(1, self.unroll_steps + 1):
                idx = pos + k
                if idx < len(game):
                    actions_k.append(game.actions[idx - 1])
                    rewards_k.append(game.rewards[idx - 1])
                    values_k.append(self._compute_target_value(game, idx))
                    policies_k.append(game.policies[idx] if idx < len(game) else
                                      np.zeros_like(game.policies[0]))
                else:
                    # Past game end — pad with zeros
                    actions_k.append(0)
                    rewards_k.append(0.0)
                    values_k.append(0.0)
                    policies_k.append(np.zeros_like(game.policies[0]))

            actions_list.append(actions_k)
            target_values.append(values_k)
            target_rewards.append(rewards_k)
            target_policies.append(policies_k)

        return {
            'observations': np.array(observations),
            'actions': np.array(actions_list),
            'target_values': np.array(target_values),
            'target_rewards': np.array(target_rewards),
            'target_policies': np.array(target_policies),
        }

    def _compute_target_value(self, game: GameHistory, pos: int) -> float:
        """Value target: actual game outcome from perspective of player at this step.
        For board games, the root_values store the actual outcome per player.
        """
        if pos < len(game.root_values):
            return game.root_values[pos]
        return 0.0

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# MuZero Trainer
# ==============================================================================

class MuZeroTrainer:
    """MuZero training loop: self-play → train on unrolled trajectories."""

    def __init__(self, game, n_iterations: int = 100, n_self_play_games: int = 25,
                 n_training_steps: int = 100, batch_size: int = 64,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 n_simulations: int = 50, c_puct: float = 1.25,
                 hidden_dim: int = 64, unroll_steps: int = 5,
                 td_steps: int = 10, temperature_threshold: int = 15,
                 n_eval_games: int = 20, buffer_capacity: int = 1000,
                 device: str = 'cpu'):
        self.game = game
        self.n_iterations = n_iterations
        self.n_self_play_games = n_self_play_games
        self.n_training_steps = n_training_steps
        self.batch_size = batch_size
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.temperature_threshold = temperature_threshold
        self.n_eval_games = n_eval_games
        self.device = device
        self.unroll_steps = unroll_steps

        self.network = MuZeroNetwork(
            input_channels=3, board_h=game.board_size[0], board_w=game.board_size[1],
            action_size=game.action_size, hidden_dim=hidden_dim
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.buffer = MuZeroReplayBuffer(
            capacity=buffer_capacity, unroll_steps=unroll_steps,
            td_steps=td_steps, discount=1.0
        )

        self.history = {
            'iteration': [],
            'policy_loss': [],
            'value_loss': [],
            'reward_loss': [],
            'total_loss': [],
            'win_rate_vs_random': [],
        }

    def train(self):
        """Full training loop."""
        for iteration in range(1, self.n_iterations + 1):
            # 1. Self-play
            self.network.eval()
            for _ in range(self.n_self_play_games):
                game_history = self._self_play_game()
                self.buffer.store_game(game_history)

            # 2. Train
            if len(self.buffer) < 5:
                continue
            self.network.train()
            total_p, total_v, total_r = 0, 0, 0
            n_steps = min(self.n_training_steps, len(self.buffer) * 5)
            n_steps = max(n_steps, 1)
            for _ in range(n_steps):
                p_loss, v_loss, r_loss = self._train_step()
                total_p += p_loss
                total_v += v_loss
                total_r += r_loss
            avg_p = total_p / n_steps
            avg_v = total_v / n_steps
            avg_r = total_r / n_steps

            # 3. Evaluate
            win_rate = self._evaluate_vs_random()

            self.history['iteration'].append(iteration)
            self.history['policy_loss'].append(avg_p)
            self.history['value_loss'].append(avg_v)
            self.history['reward_loss'].append(avg_r)
            self.history['total_loss'].append(avg_p + avg_v + avg_r)
            self.history['win_rate_vs_random'].append(win_rate)

            if iteration % 10 == 0 or iteration == 1:
                print(f"Iter {iteration:4d} | P: {avg_p:.4f} V: {avg_v:.4f} "
                      f"R: {avg_r:.4f} | Win: {win_rate*100:.0f}% | "
                      f"Games: {len(self.buffer)}")

        return self.network, self.history

    def _self_play_game(self) -> GameHistory:
        """Play one self-play game, collecting trajectory data."""
        mcts = MuZeroMCTS(self.network, self.game.action_size,
                           n_simulations=self.n_simulations,
                           c_puct=self.c_puct, device=self.device)
        state = self.game.get_initial_state()
        player = 1
        history = GameHistory()
        move_count = 0
        game_players = []  # track which player made each move

        while not self.game.is_terminal(state):
            encoded = self.game.encode_state(state, player)
            legal = self.game.get_legal_actions(state)

            with torch.no_grad():
                policy = mcts.search(encoded, legal_actions=legal)
                # Get root value estimate
                obs_tensor = torch.FloatTensor(encoded).unsqueeze(0).to(self.device)
                _, _, root_value = self.network.initial_inference(obs_tensor)
                root_val = root_value.item()

            # Temperature-based selection
            if move_count < self.temperature_threshold:
                legal_probs = np.array([policy[a] for a in legal])
                if legal_probs.sum() > 0:
                    legal_probs /= legal_probs.sum()
                    action = legal[np.random.choice(len(legal), p=legal_probs)]
                else:
                    action = np.random.choice(legal)
            else:
                action = int(np.argmax(policy))

            history.store(encoded, action, 0.0, policy, root_val, player)
            game_players.append(player)
            state, player = self.game.step(state, action, player)
            move_count += 1

        # Assign rewards: only the last step gets the game outcome
        # All intermediate steps have reward 0
        # Each step's reward/value is from the perspective of the player at that step
        if len(history) > 0:
            last_player = game_players[-1]
            last_reward = self.game.get_reward(state, last_player)
            if last_reward is None:
                last_reward = 0.0
            history.rewards[-1] = last_reward
            # Also set root values to actual outcome for better targets
            for i in range(len(history)):
                p = game_players[i]
                outcome = self.game.get_reward(state, p)
                if outcome is None:
                    outcome = 0.0
                history.root_values[i] = outcome

        return history

    def _train_step(self) -> Tuple[float, float, float]:
        """Single training step with K-step unrolling."""
        batch = self.buffer.sample_batch(self.batch_size)

        obs = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        target_values = torch.FloatTensor(batch['target_values']).to(self.device)
        target_rewards = torch.FloatTensor(batch['target_rewards']).to(self.device)
        target_policies = torch.FloatTensor(batch['target_policies']).to(self.device)

        # Initial inference
        hidden, log_policy, value = self.network.initial_inference(obs)

        # Loss at step 0
        policy_loss = -torch.sum(target_policies[:, 0] * log_policy, dim=1).mean()
        value_loss = F.mse_loss(value.squeeze(1), target_values[:, 0])
        reward_loss = torch.tensor(0.0, device=self.device)

        # Unroll K steps
        for k in range(self.unroll_steps):
            action_k = actions[:, k]
            hidden, pred_reward, log_policy, value = self.network.recurrent_inference(
                hidden, action_k
            )

            policy_loss += -torch.sum(target_policies[:, k + 1] * log_policy, dim=1).mean()
            value_loss += F.mse_loss(value.squeeze(1), target_values[:, k + 1])
            reward_loss += F.mse_loss(pred_reward.squeeze(1), target_rewards[:, k])

        # Scale losses
        scale = 1.0 / (self.unroll_steps + 1)
        total_loss = scale * (policy_loss + value_loss + reward_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        return (scale * policy_loss).item(), (scale * value_loss).item(), \
               (scale * reward_loss).item()

    @torch.no_grad()
    def _evaluate_vs_random(self) -> float:
        """Play against random, return win rate."""
        self.network.eval()
        mcts = MuZeroMCTS(self.network, self.game.action_size,
                           n_simulations=self.n_simulations // 2,
                           c_puct=self.c_puct, device=self.device)
        wins = 0
        for game_i in range(self.n_eval_games):
            state = self.game.get_initial_state()
            player = 1
            muzero_player = 1 if game_i % 2 == 0 else -1

            while not self.game.is_terminal(state):
                if player == muzero_player:
                    encoded = self.game.encode_state(state, player)
                    legal = self.game.get_legal_actions(state)
                    policy = mcts.search(encoded, legal_actions=legal)
                    action = int(np.argmax(policy))
                else:
                    legal = self.game.get_legal_actions(state)
                    action = np.random.choice(legal)
                state, player = self.game.step(state, action, player)

            reward = self.game.get_reward(state, muzero_player)
            if reward is not None and reward > 0:
                wins += 1
            elif reward == 0:
                wins += 0.5

        return wins / self.n_eval_games


# ==============================================================================
# Demo
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MuZero — TicTacToe (Learning Without Rules)")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    game = TicTacToe()
    trainer = MuZeroTrainer(
        game,
        n_iterations=80,
        n_self_play_games=25,
        n_training_steps=200,
        batch_size=64,
        lr=5e-4,
        weight_decay=1e-4,
        n_simulations=80,
        c_puct=1.25,
        hidden_dim=128,
        unroll_steps=5,
        td_steps=10,
        temperature_threshold=8,
        n_eval_games=20,
        buffer_capacity=2000,
    )

    print("\nTraining (MuZero learns game dynamics implicitly)...")
    print("-" * 60)
    network, history = trainer.train()

    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Evaluation (40 games vs random)")
    print("-" * 60)
    trainer.n_eval_games = 40
    final_wr = trainer._evaluate_vs_random()
    print(f"  Win rate vs random: {final_wr*100:.0f}%")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(history['iteration'], history['policy_loss'], 'b-', label='Policy', linewidth=1.5)
    axes[0].plot(history['iteration'], history['value_loss'], 'r-', label='Value', linewidth=1.5)
    axes[0].plot(history['iteration'], history['reward_loss'], 'g-', label='Reward', linewidth=1.5)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('MuZero Loss Components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['iteration'], history['total_loss'], 'k-', linewidth=1.5)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Total Loss')
    axes[1].set_title('Total Training Loss')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['iteration'],
                 [wr * 100 for wr in history['win_rate_vs_random']],
                 'g-', linewidth=1.5)
    axes[2].axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90%')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Win Rate vs Random (%)')
    axes[2].set_title('MuZero vs Random')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'muzero_demo.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
