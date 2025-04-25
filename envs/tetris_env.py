import numpy as np
import random
from collections import deque

# Super Rotation System (SRS) piece definitions and wall-kicks
# Each rotation is represented as a numpy array (4x4 for I/O, 3x3 for others, padded to 4x4)

# Tetromino rotation matrices (4x4) per SRS
PIECES = {
    'I': [
        np.array([[0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=int),
        np.array([[0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0]], dtype=int),
        np.array([[0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]], dtype=int)
    ],
    'O': [
             np.array([[0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [0, 0, 0, 0]], dtype=int)
         ] * 4,
    'T': [
        np.array([[0, 0, 0, 0],
                  [0, 1, 1, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 1, 1],
                  [0, 0, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 0, 0]], dtype=int)
    ],
    'J': [
        np.array([[0, 0, 0, 0],
                  [1, 1, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [1, 1, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 1, 1, 0],
                  [0, 0, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]], dtype=int)
    ],
    'L': [
        np.array([[0, 0, 0, 0],
                  [1, 1, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [1, 1, 1, 0],
                  [0, 0, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 1, 0]], dtype=int)
    ],
    'S': [
        np.array([[0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [1, 1, 0, 0],
                  [0, 0, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1, 1, 0],
                  [1, 1, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 0, 0]], dtype=int)
    ],
    'Z': [
        np.array([[0, 0, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 1, 0],
                  [0, 1, 0, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 1, 0, 0],
                  [0, 1, 1, 0]], dtype=int),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [1, 1, 0, 0],
                  [1, 0, 0, 0]], dtype=int)
    ]
}

# Map piece letters to unique IDs for rendering/colors
PIECE_IDS = {'I': 1, 'O': 2, 'T': 3, 'J': 4, 'L': 5, 'S': 6, 'Z': 7}

# SRS wall-kick data (unchanged)
JLSTZ_WALL_KICKS = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)]
}
I_WALL_KICKS = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)]
}

class TetrisEnv:
    def __init__(self, width=10, height=20, next_queue=3, hold=False, max_steps=500):
        self.width = width
        self.height = height
        self.next_queue = next_queue
        self.hold_enabled = hold
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=int)
        self._new_bag()
        self.next_pieces = deque([self._draw_piece() for _ in range(self.next_queue)])
        self.current_piece = self._draw_piece()
        self.current_rotation = 0
        self.hold_piece = None
        self.hold_used = False
        self.score = 0
        self.step_count = 0
        return self._get_observation()

    def step(self, action):
        # action: dict with keys 'type', 'x', 'rotation'
        a_type = action['type']
        x = action['x']
        rot = action['rotation']
        old_rot = self.current_rotation
        # choose proper kick set
        kick_set = I_WALL_KICKS if self.current_piece == 'I' else JLSTZ_WALL_KICKS
        placed = False
        for dx, dy in kick_set.get((old_rot, rot), [(0, 0)]):
            if self._attempt_place(x + dx, rot, dy):
                placed = True
                break
        if not placed:
            # invalid move, no placement
            pass
        lines = self._clear_lines()
        self.score += lines
        self.step_count += 1
        self.current_piece = self.next_pieces.popleft()
        self.next_pieces.append(self._draw_piece())
        self.current_rotation = 0
        self.hold_used = False
        done = self.step_count >= self.max_steps or self._game_over()
        info = {'lines_cleared': lines}
        return self._get_observation(), lines, done, info

    def legal_actions(self):
        actions = []
        rotations = len(PIECES[self.current_piece])
        for r in range(rotations):
            shape = PIECES[self.current_piece][r]
            w = shape.shape[1]
            for x in range(self.width - w + 1):
                if self._get_drop_height(shape, x) is not None:
                    actions.append((x, r))
        return actions

    def render(self):
        for row in self.board:
            print(''.join(['X' if cell else '.' for cell in row]))
        print(f"Score: {self.score}, Steps: {self.step_count}\n")

    def _new_bag(self):
        self.bag = list(PIECES.keys())
        random.shuffle(self.bag)

    def _draw_piece(self):
        if not self.bag:
            self._new_bag()
        return self.bag.pop()

    def _attempt_place(self, x, rot, dy=0):
        shape = PIECES[self.current_piece][rot]
        y0 = self._get_drop_height(shape, x)
        if y0 is None:
            return False
        y = y0 + dy
        if y < 0 or y + shape.shape[0] > self.height:
            return False
        piece_id = PIECE_IDS[self.current_piece]
        # place shape
        for i in range(shape.shape[0]):
            for j in range(shape.shape[1]):
                if shape[i, j]:
                    self.board[y + i, x + j] = piece_id
        self.current_rotation = rot
        return True

    def _get_drop_height(self, shape, x):
        for y in range(self.height - shape.shape[0], -1, -1):
            if not self._collides(shape, x, y):
                return y
        return None

    def _collides(self, shape, x, y):
        for i in range(shape.shape[0]):
            for j in range(shape.shape[1]):
                if shape[i, j] and self.board[y + i, x + j]:
                    return True
        return False

    def _clear_lines(self):
        lines = 0
        new_board = self.board.copy()
        for i in range(self.height):
            if all(new_board[i, :]):
                lines += 1
                new_board[1:i + 1] = new_board[0:i]
                new_board[0] = np.zeros(self.width, dtype=int)
        self.board = new_board
        return lines

    def _game_over(self):
        return np.any(self.board[0, :])

    def _get_observation(self):
        return {
            'board': self.board.copy(),
            'current_piece': self.current_piece,
            'current_rotation': self.current_rotation,
            'next_pieces': list(self.next_pieces),
            'hold_piece': self.hold_piece,
            'steps_remaining': self.max_steps - self.step_count,
            'score': self.score
        }
