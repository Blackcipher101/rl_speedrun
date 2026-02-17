"""
Game implementations for AlphaZero: TicTacToe and Connect Four.

Both games provide tensor-encoded board representations suitable for
convolutional neural network input (3 channels: current player, opponent, color).

Author: RL Speedrun
"""

import numpy as np
from typing import List, Optional, Tuple


# ==============================================================================
# TicTacToe
# ==============================================================================

class TicTacToe:
    """3x3 TicTacToe with tensor encoding for neural networks.

    State: flat np.array(9) with {-1, 0, 1}.
    Player 1 = +1, Player 2 = -1.
    Encoding: 3 channels (3, 3, 3) — current player, opponent, color plane.
    """

    board_size = (3, 3)
    action_size = 9

    def get_initial_state(self) -> np.ndarray:
        return np.zeros(9, dtype=np.float32)

    def get_legal_actions(self, state: np.ndarray) -> List[int]:
        return [i for i in range(9) if state[i] == 0]

    def get_legal_mask(self, state: np.ndarray) -> np.ndarray:
        """Return binary mask of legal actions."""
        return (state == 0).astype(np.float32)

    def step(self, state: np.ndarray, action: int, player: int) -> Tuple[np.ndarray, int]:
        """Apply action, return (new_state, next_player)."""
        new_state = state.copy()
        new_state[action] = player
        return new_state, -player

    def get_current_player(self, state: np.ndarray) -> int:
        pieces = np.sum(state != 0)
        return 1 if pieces % 2 == 0 else -1

    def is_terminal(self, state: np.ndarray) -> bool:
        if self._check_winner(state) != 0:
            return True
        return len(self.get_legal_actions(state)) == 0

    def get_reward(self, state: np.ndarray, player: int) -> Optional[float]:
        """Return reward for player in terminal state. None if not terminal."""
        if not self.is_terminal(state):
            return None
        winner = self._check_winner(state)
        if winner == player:
            return 1.0
        elif winner == -player:
            return -1.0
        return 0.0

    def get_canonical_state(self, state: np.ndarray, player: int) -> np.ndarray:
        """Flip board so current player is always +1."""
        return state * player

    def encode_state(self, state: np.ndarray, player: int) -> np.ndarray:
        """Encode state as 3-channel tensor (3, 3, 3).

        Channel 0: current player's pieces (1 where player has piece)
        Channel 1: opponent's pieces (1 where opponent has piece)
        Channel 2: color plane (all 1s if player==1, all 0s if player==-1)
        """
        canonical = self.get_canonical_state(state, player)
        board = canonical.reshape(3, 3)
        encoded = np.zeros((3, 3, 3), dtype=np.float32)
        encoded[0] = (board == 1).astype(np.float32)   # current player
        encoded[1] = (board == -1).astype(np.float32)   # opponent
        encoded[2] = 1.0 if player == 1 else 0.0        # color plane
        return encoded

    def render(self, state: np.ndarray) -> str:
        symbols = {1: 'X', -1: 'O', 0: '.'}
        board = state.reshape(3, 3)
        lines = []
        for row in board:
            lines.append(' '.join(symbols[int(v)] for v in row))
        return '\n'.join(lines)

    def _check_winner(self, state: np.ndarray) -> int:
        board = state.reshape(3, 3)
        for i in range(3):
            if abs(board[i].sum()) == 3:
                return int(np.sign(board[i].sum()))
            if abs(board[:, i].sum()) == 3:
                return int(np.sign(board[:, i].sum()))
        diag1 = board[0, 0] + board[1, 1] + board[2, 2]
        diag2 = board[0, 2] + board[1, 1] + board[2, 0]
        if abs(diag1) == 3:
            return int(np.sign(diag1))
        if abs(diag2) == 3:
            return int(np.sign(diag2))
        return 0


# ==============================================================================
# Connect Four
# ==============================================================================

class ConnectFour:
    """Connect Four (6x7 board). Gravity-based column placement.

    State: flat np.array(42) with {-1, 0, 1}.
    Player 1 = +1, Player 2 = -1.
    Actions: 0-6 (column to drop piece).
    Encoding: 3 channels (3, 6, 7).
    """

    board_size = (6, 7)
    action_size = 7
    rows = 6
    cols = 7

    def get_initial_state(self) -> np.ndarray:
        return np.zeros(42, dtype=np.float32)

    def get_legal_actions(self, state: np.ndarray) -> List[int]:
        """Legal actions are columns that aren't full (top row is empty)."""
        board = state.reshape(6, 7)
        return [c for c in range(7) if board[0, c] == 0]

    def get_legal_mask(self, state: np.ndarray) -> np.ndarray:
        board = state.reshape(6, 7)
        mask = np.zeros(7, dtype=np.float32)
        for c in range(7):
            if board[0, c] == 0:
                mask[c] = 1.0
        return mask

    def step(self, state: np.ndarray, action: int, player: int) -> Tuple[np.ndarray, int]:
        """Drop piece in column, return (new_state, next_player)."""
        new_state = state.copy()
        board = new_state.reshape(6, 7)
        # Find lowest empty row in column
        for row in range(5, -1, -1):
            if board[row, action] == 0:
                board[row, action] = player
                break
        return new_state, -player

    def get_current_player(self, state: np.ndarray) -> int:
        pieces = np.sum(state != 0)
        return 1 if pieces % 2 == 0 else -1

    def is_terminal(self, state: np.ndarray) -> bool:
        if self._check_winner(state) != 0:
            return True
        return len(self.get_legal_actions(state)) == 0

    def get_reward(self, state: np.ndarray, player: int) -> Optional[float]:
        if not self.is_terminal(state):
            return None
        winner = self._check_winner(state)
        if winner == player:
            return 1.0
        elif winner == -player:
            return -1.0
        return 0.0

    def get_canonical_state(self, state: np.ndarray, player: int) -> np.ndarray:
        return state * player

    def encode_state(self, state: np.ndarray, player: int) -> np.ndarray:
        """Encode state as 3-channel tensor (3, 6, 7)."""
        canonical = self.get_canonical_state(state, player)
        board = canonical.reshape(6, 7)
        encoded = np.zeros((3, 6, 7), dtype=np.float32)
        encoded[0] = (board == 1).astype(np.float32)
        encoded[1] = (board == -1).astype(np.float32)
        encoded[2] = 1.0 if player == 1 else 0.0
        return encoded

    def render(self, state: np.ndarray) -> str:
        symbols = {1: 'X', -1: 'O', 0: '.'}
        board = state.reshape(6, 7)
        lines = ['  '.join(str(c) for c in range(7))]
        lines.append('-' * 20)
        for row in board:
            lines.append('  '.join(symbols[int(v)] for v in row))
        return '\n'.join(lines)

    def _check_winner(self, state: np.ndarray) -> int:
        """Check for 4-in-a-row in all directions."""
        board = state.reshape(6, 7)
        # Direction vectors: horizontal, vertical, diag-down-right, diag-up-right
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(6):
            for c in range(7):
                if board[r, c] == 0:
                    continue
                player = board[r, c]
                for dr, dc in directions:
                    # Check if 4 in a row starting from (r, c)
                    end_r = r + 3 * dr
                    end_c = c + 3 * dc
                    if end_r < 0 or end_r >= 6 or end_c < 0 or end_c >= 7:
                        continue
                    if all(board[r + i * dr, c + i * dc] == player for i in range(4)):
                        return int(player)
        return 0
