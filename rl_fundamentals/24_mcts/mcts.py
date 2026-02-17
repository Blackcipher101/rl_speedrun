"""
Pure Monte Carlo Tree Search (MCTS) with UCB1 selection.

No neural networks -- uses random rollouts for value estimation.
Demonstrates the four phases: selection, expansion, simulation, backpropagation.

Includes a TicTacToe implementation for demonstration.

Author: RL Speedrun
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ==============================================================================
# Game Interface
# ==============================================================================

class Game(ABC):
    """Abstract base class for two-player zero-sum games."""

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Return the starting game state."""
        pass

    @abstractmethod
    def get_legal_actions(self, state: np.ndarray) -> List[int]:
        """Return list of legal action indices for current state."""
        pass

    @abstractmethod
    def step(self, state: np.ndarray, action: int) -> np.ndarray:
        """Apply action to state, return new state."""
        pass

    @abstractmethod
    def is_terminal(self, state: np.ndarray) -> bool:
        """Return True if game is over."""
        pass

    @abstractmethod
    def get_reward(self, state: np.ndarray, player: int) -> float:
        """Return reward for player in terminal state. +1 win, -1 loss, 0 draw."""
        pass

    @abstractmethod
    def get_current_player(self, state: np.ndarray) -> int:
        """Return current player (1 or -1)."""
        pass

    @abstractmethod
    def clone_state(self, state: np.ndarray) -> np.ndarray:
        """Return a deep copy of state."""
        pass

    @abstractmethod
    def render(self, state: np.ndarray) -> str:
        """Return string representation of state."""
        pass


# ==============================================================================
# TicTacToe
# ==============================================================================

class TicTacToe(Game):
    """3x3 TicTacToe. State: flat array of 9 values {-1, 0, 1}.
    Player 1 = +1, Player 2 = -1. Actions: 0-8 (board positions)."""

    def get_initial_state(self) -> np.ndarray:
        return np.zeros(9, dtype=np.float32)

    def get_legal_actions(self, state: np.ndarray) -> List[int]:
        return [i for i in range(9) if state[i] == 0]

    def step(self, state: np.ndarray, action: int) -> np.ndarray:
        new_state = state.copy()
        player = self.get_current_player(state)
        new_state[action] = player
        return new_state

    def is_terminal(self, state: np.ndarray) -> bool:
        if self._check_winner(state) != 0:
            return True
        return len(self.get_legal_actions(state)) == 0

    def get_reward(self, state: np.ndarray, player: int) -> float:
        winner = self._check_winner(state)
        if winner == player:
            return 1.0
        elif winner == -player:
            return -1.0
        return 0.0

    def get_current_player(self, state: np.ndarray) -> int:
        pieces = np.sum(state != 0)
        return 1 if pieces % 2 == 0 else -1

    def clone_state(self, state: np.ndarray) -> np.ndarray:
        return state.copy()

    def render(self, state: np.ndarray) -> str:
        symbols = {1: 'X', -1: 'O', 0: '.'}
        board = state.reshape(3, 3)
        lines = []
        for row in board:
            lines.append(' '.join(symbols[int(v)] for v in row))
        return '\n'.join(lines)

    def _check_winner(self, state: np.ndarray) -> int:
        """Return +1 if player 1 wins, -1 if player 2 wins, 0 otherwise."""
        board = state.reshape(3, 3)
        # Rows and columns
        for i in range(3):
            if abs(board[i].sum()) == 3:
                return int(np.sign(board[i].sum()))
            if abs(board[:, i].sum()) == 3:
                return int(np.sign(board[:, i].sum()))
        # Diagonals
        diag1 = board[0, 0] + board[1, 1] + board[2, 2]
        diag2 = board[0, 2] + board[1, 1] + board[2, 0]
        if abs(diag1) == 3:
            return int(np.sign(diag1))
        if abs(diag2) == 3:
            return int(np.sign(diag2))
        return 0


# ==============================================================================
# MCTS Node
# ==============================================================================

class MCTSNode:
    """Single node in the search tree."""

    def __init__(self, state: np.ndarray, parent=None, action: Optional[int] = None,
                 player: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action = action              # action that led to this node
        self.player = player              # player who took that action
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count: int = 0         # N(s)
        self.value_sum: float = 0.0       # W(s) - total value
        self.is_expanded: bool = False

    def q_value(self) -> float:
        """Average value Q(s) = W(s) / N(s)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb1(self, c: float = 1.41) -> float:
        """UCB1 = Q(s) + c * sqrt(ln(N_parent) / N(s))."""
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.q_value()
        exploration = c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return exploitation + exploration


# ==============================================================================
# Pure MCTS
# ==============================================================================

class MCTS:
    """Monte Carlo Tree Search with random rollout evaluation.

    The four MCTS phases:
    1. Selection:  Walk tree using UCB1 until reaching unexpanded node
    2. Expansion:  Create child nodes for all legal actions
    3. Simulation: Random rollout from expanded node to terminal state
    4. Backprop:   Update visit counts and values back to root
    """

    def __init__(self, game: Game, n_simulations: int = 1000, c_ucb: float = 1.41):
        self.game = game
        self.n_simulations = n_simulations
        self.c_ucb = c_ucb

    def search(self, state: np.ndarray) -> np.ndarray:
        """Run n_simulations from state, return visit-count policy vector.

        Returns:
            policy: np.array of shape (max_actions,) with visit count proportions
        """
        root_player = self.game.get_current_player(state)
        root = MCTSNode(state=self.game.clone_state(state), player=root_player)

        for _ in range(self.n_simulations):
            node = self._select(root)

            if not self.game.is_terminal(node.state):
                self._expand(node)
                # Pick a random child to simulate from
                if node.children:
                    child_action = np.random.choice(list(node.children.keys()))
                    node = node.children[child_action]

            reward = self._simulate(node.state, root_player)
            self._backpropagate(node, reward)

        # Build policy from root visit counts
        policy = np.zeros(9)  # max actions for TicTacToe
        for action, child in root.children.items():
            policy[action] = child.visit_count
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        return policy

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk tree using UCB1 until reaching unexpanded or terminal node."""
        while node.is_expanded and node.children:
            if self.game.is_terminal(node.state):
                return node
            # Select child with highest UCB1
            best_child = max(node.children.values(), key=lambda c: c.ucb1(self.c_ucb))
            node = best_child
        return node

    def _expand(self, node: MCTSNode):
        """Create child nodes for all legal actions."""
        if node.is_expanded:
            return
        legal_actions = self.game.get_legal_actions(node.state)
        current_player = self.game.get_current_player(node.state)
        for action in legal_actions:
            child_state = self.game.step(node.state, action)
            child = MCTSNode(
                state=child_state,
                parent=node,
                action=action,
                player=current_player,
            )
            node.children[action] = child
        node.is_expanded = True

    def _simulate(self, state: np.ndarray, root_player: int) -> float:
        """Random rollout from state to terminal, return reward for root_player."""
        sim_state = self.game.clone_state(state)
        while not self.game.is_terminal(sim_state):
            legal = self.game.get_legal_actions(sim_state)
            action = np.random.choice(legal)
            sim_state = self.game.step(sim_state, action)
        return self.game.get_reward(sim_state, root_player)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Walk up tree updating visit counts and value sums."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += reward
            node = node.parent

    def get_action(self, state: np.ndarray, temperature: float = 0.0) -> int:
        """Select action from MCTS policy.

        Args:
            temperature: 0 = greedy (argmax), >0 = sample proportional to N^(1/temp)
        """
        policy = self.search(state)
        if temperature == 0:
            return int(np.argmax(policy))
        else:
            # Temperature-based sampling
            legal = self.game.get_legal_actions(state)
            visit_counts = np.array([policy[a] for a in legal])
            if visit_counts.sum() == 0:
                return np.random.choice(legal)
            visit_counts = visit_counts ** (1.0 / temperature)
            probs = visit_counts / visit_counts.sum()
            return legal[np.random.choice(len(legal), p=probs)]


# ==============================================================================
# Game Playing Utilities
# ==============================================================================

def play_game(game: Game, player1_fn, player2_fn) -> int:
    """Play a game between two players. Returns winner (1, -1, or 0 for draw)."""
    state = game.get_initial_state()
    while not game.is_terminal(state):
        player = game.get_current_player(state)
        if player == 1:
            action = player1_fn(state)
        else:
            action = player2_fn(state)
        state = game.step(state, action)
    winner = game._check_winner(state) if hasattr(game, '_check_winner') else 0
    if winner == 0:
        return 0
    return winner


def random_player(game: Game):
    """Return a function that picks random legal moves."""
    def player_fn(state):
        legal = game.get_legal_actions(state)
        return np.random.choice(legal)
    return player_fn


def mcts_player(game: Game, n_simulations: int = 500):
    """Return a function that picks moves using MCTS."""
    mcts = MCTS(game, n_simulations=n_simulations)
    def player_fn(state):
        return mcts.get_action(state, temperature=0)
    return player_fn


# ==============================================================================
# Demo
# ==============================================================================

def play_tournament(game, p1_fn, p2_fn, n_games=100):
    """Play n_games, return (p1_wins, p2_wins, draws)."""
    p1_wins = 0
    p2_wins = 0
    draws = 0
    for _ in range(n_games):
        result = play_game(game, p1_fn, p2_fn)
        if result == 1:
            p1_wins += 1
        elif result == -1:
            p2_wins += 1
        else:
            draws += 1
    return p1_wins, p2_wins, draws


if __name__ == "__main__":
    print("=" * 60)
    print("Monte Carlo Tree Search (MCTS) — TicTacToe Demo")
    print("=" * 60)

    game = TicTacToe()
    np.random.seed(42)

    # --- MCTS vs Random ---
    print("\n" + "-" * 40)
    print("MCTS (500 sims) vs Random — 100 games")
    print("-" * 40)
    mcts_fn = mcts_player(game, n_simulations=500)
    rand_fn = random_player(game)

    wins, losses, draws = play_tournament(game, mcts_fn, rand_fn, n_games=100)
    print(f"  MCTS wins:  {wins}")
    print(f"  Random wins: {losses}")
    print(f"  Draws:       {draws}")
    mcts_vs_random = (wins, losses, draws)

    # --- Random vs MCTS (MCTS plays second) ---
    print("\n" + "-" * 40)
    print("Random vs MCTS (500 sims) — 100 games")
    print("-" * 40)
    wins2, losses2, draws2 = play_tournament(game, rand_fn, mcts_fn, n_games=100)
    print(f"  Random wins: {wins2}")
    print(f"  MCTS wins:   {losses2}")
    print(f"  Draws:        {draws2}")
    random_vs_mcts = (wins2, losses2, draws2)

    # --- MCTS vs MCTS ---
    print("\n" + "-" * 40)
    print("MCTS (500 sims) vs MCTS (500 sims) — 50 games")
    print("-" * 40)
    mcts_fn2 = mcts_player(game, n_simulations=500)
    wins3, losses3, draws3 = play_tournament(game, mcts_fn, mcts_fn2, n_games=50)
    print(f"  Player 1 wins: {wins3}")
    print(f"  Player 2 wins: {losses3}")
    print(f"  Draws:          {draws3}")
    mcts_vs_mcts = (wins3, losses3, draws3)

    # --- Simulation count sweep ---
    print("\n" + "-" * 40)
    print("Win rate vs simulation count")
    print("-" * 40)
    sim_counts = [10, 50, 100, 200, 500]
    win_rates = []
    for n_sims in sim_counts:
        fn = mcts_player(game, n_simulations=n_sims)
        w, l, d = play_tournament(game, fn, rand_fn, n_games=50)
        wr = (w + 0.5 * d) / 50
        win_rates.append(wr)
        print(f"  {n_sims:4d} sims: {wr*100:.0f}% ({w}W/{l}L/{d}D)")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: MCTS vs Random results
    labels = ['MCTS\nvs Random', 'Random\nvs MCTS', 'MCTS\nvs MCTS']
    data = [mcts_vs_random, random_vs_mcts, mcts_vs_mcts]
    x = np.arange(len(labels))
    width = 0.25
    for i, (matchup, label) in enumerate(zip(data, labels)):
        axes[0].bar(x[i] - width, matchup[0], width, color='green', alpha=0.7,
                     label='P1 Wins' if i == 0 else '')
        axes[0].bar(x[i], matchup[1], width, color='red', alpha=0.7,
                     label='P2 Wins' if i == 0 else '')
        axes[0].bar(x[i] + width, matchup[2], width, color='gray', alpha=0.7,
                     label='Draws' if i == 0 else '')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Games')
    axes[0].set_title('MCTS Tournament Results')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Win rate vs simulation count
    axes[1].plot(sim_counts, [wr * 100 for wr in win_rates], 'bo-', linewidth=2, markersize=8)
    axes[1].axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect')
    axes[1].set_xlabel('MCTS Simulations')
    axes[1].set_ylabel('Win Rate vs Random (%)')
    axes[1].set_title('Win Rate vs Simulation Count')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'mcts_demo.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
