"""Two-player Snake game window for the mzML Explorer easter egg."""

import random
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QKeyEvent

BLOCK = 8  # pixels per block
COLS = 40  # number of columns
ROWS = 40  # number of rows
WIDTH = BLOCK * COLS  # 320 px
HEIGHT = BLOCK * ROWS  # 320 px

# Colours
COL_BG = QColor("#1a1a2e")
COL_GRID = QColor("#16213e")
# Player 1 (arrow keys) – green
COL_P1_HEAD = QColor("#4ade80")
COL_P1_BODY = QColor("#22c55e")
# Player 2 (WASD) – blue
COL_P2_HEAD = QColor("#60a5fa")
COL_P2_BODY = QColor("#3b82f6")
COL_FOOD = QColor("#f87171")
COL_TEXT = QColor("#e2e8f0")

TICK_MS = 120  # ms per step

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
OPPOSITES = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class SnakeField(QWidget):
    """Two-player snake canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(WIDTH, HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._reset()

    # ------------------------------------------------------------------
    # Game state
    # ------------------------------------------------------------------

    def _reset(self):
        # Player 1 starts on the left half, moving right
        p1x, p1y = COLS // 4, ROWS // 2
        self._s1 = [(p1x, p1y), (p1x - 1, p1y), (p1x - 2, p1y)]
        self._d1 = RIGHT
        self._nd1 = RIGHT
        self._score1 = 0

        # Player 2 starts on the right half, moving left
        p2x, p2y = (COLS * 3) // 4, ROWS // 2
        self._s2 = [(p2x, p2y), (p2x + 1, p2y), (p2x + 2, p2y)]
        self._d2 = LEFT
        self._nd2 = LEFT
        self._score2 = 0

        self._food = self._random_food()
        self._running = False
        self._game_over = False
        self._loser = None  # 1 or 2
        self._update_title()
        self.update()

    def _random_food(self):
        occupied = set(self._s1) | set(self._s2)
        while True:
            pos = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
            if pos not in occupied:
                return pos

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def _step(self):
        self._d1 = self._nd1
        self._d2 = self._nd2

        h1x, h1y = self._s1[0]
        h2x, h2y = self._s2[0]
        new_h1 = ((h1x + self._d1[0]) % COLS, (h1y + self._d1[1]) % ROWS)
        new_h2 = ((h2x + self._d2[0]) % COLS, (h2y + self._d2[1]) % ROWS)

        # Determine which snakes will grow this tick
        eat1 = new_h1 == self._food
        eat2 = new_h2 == self._food

        # Compute bodies after moving (before collision check)
        body1_after = [new_h1] + (self._s1 if eat1 else self._s1[:-1])
        body2_after = [new_h2] + (self._s2 if eat2 else self._s2[:-1])

        dead1 = False
        dead2 = False

        # P1 hits itself
        if new_h1 in body1_after[1:]:
            dead1 = True
        # P2 hits itself
        if new_h2 in body2_after[1:]:
            dead2 = True
        # P1 hits P2's body
        if new_h1 in body2_after:
            dead1 = True
        # P2 hits P1's body
        if new_h2 in body1_after:
            dead2 = True
        # Head-on collision – both die, call it a draw → both lose
        if new_h1 == new_h2:
            dead1 = True
            dead2 = True

        if dead1 or dead2:
            if dead1 and dead2:
                self._loser = 0  # draw
            elif dead1:
                self._loser = 1
            else:
                self._loser = 2
            self._game_over = True
            self._running = False
            self._timer.stop()
            self._update_title()
            self.update()
            return

        # Commit moves
        self._s1 = body1_after
        self._s2 = body2_after

        if eat1:
            self._score1 += 1
            self._food = self._random_food()
        elif eat2:
            self._score2 += 1
            self._food = self._random_food()

        self._update_title()
        self.update()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()

        p1_map = {
            Qt.Key.Key_Up: UP,
            Qt.Key.Key_Down: DOWN,
            Qt.Key.Key_Left: LEFT,
            Qt.Key.Key_Right: RIGHT,
        }
        p2_map = {
            Qt.Key.Key_W: UP,
            Qt.Key.Key_S: DOWN,
            Qt.Key.Key_A: LEFT,
            Qt.Key.Key_D: RIGHT,
        }

        started = False
        if key in p1_map:
            new_dir = p1_map[key]
            if new_dir != OPPOSITES.get(self._d1):
                self._nd1 = new_dir
            started = True
        if key in p2_map:
            new_dir = p2_map[key]
            if new_dir != OPPOSITES.get(self._d2):
                self._nd2 = new_dir
            started = True

        if started:
            if self._game_over:
                self._reset()
            elif not self._running:
                self._running = True
                self._timer.start(TICK_MS)
        elif key == Qt.Key.Key_R:
            self._timer.stop()
            self._reset()

        event.accept()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Background
        painter.fillRect(0, 0, WIDTH, HEIGHT, COL_BG)

        # Subtle grid
        painter.setPen(COL_GRID)
        for col in range(0, WIDTH, BLOCK):
            painter.drawLine(col, 0, col, HEIGHT)
        for row in range(0, HEIGHT, BLOCK):
            painter.drawLine(0, row, WIDTH, row)

        # Food
        fx, fy = self._food
        painter.fillRect(fx * BLOCK, fy * BLOCK, BLOCK, BLOCK, COL_FOOD)

        # Snake 2 (draw first so P1 head is always on top)
        for i, (sx, sy) in enumerate(self._s2):
            colour = COL_P2_HEAD if i == 0 else COL_P2_BODY
            painter.fillRect(sx * BLOCK + 1, sy * BLOCK + 1, BLOCK - 1, BLOCK - 1, colour)

        # Snake 1
        for i, (sx, sy) in enumerate(self._s1):
            colour = COL_P1_HEAD if i == 0 else COL_P1_BODY
            painter.fillRect(sx * BLOCK + 1, sy * BLOCK + 1, BLOCK - 1, BLOCK - 1, colour)

        # Overlay messages
        if not self._running:
            painter.setPen(COL_TEXT)
            f = painter.font()
            f.setPointSize(11)
            f.setBold(True)
            painter.setFont(f)
            if self._game_over:
                if self._loser == 0:
                    result = "DRAW!"
                elif self._loser == 1:
                    result = "Player 2 (WASD) wins!"
                else:
                    result = "Player 1 (\u2190\u2191\u2192\u2193) wins!"
                msg = f"GAME OVER – {result}\nP1: {self._score1}   P2: {self._score2}\n\nPress any arrow / WASD key to restart"
            else:
                msg = "TWO-PLAYER SNAKE\n\nP1: Arrow keys   P2: WASD\nPress any movement key to start\nR to reset"
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, msg)

        painter.end()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_title(self):
        win = self.window()
        if self._game_over:
            if self._loser == 0:
                result = "Draw!"
            elif self._loser == 1:
                result = "P2 wins!"
            else:
                result = "P1 wins!"
            win.setWindowTitle(f"Snake – Game Over – {result} (P1: {self._score1}  P2: {self._score2})")
        else:
            win.setWindowTitle(f"Snake – P1 (\u2190\u2191\u2192\u2193): {self._score1}   P2 (WASD): {self._score2}")


class SnakeWindow(QWidget):
    """Top-level window hosting the two-player snake game."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Snake")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._field = SnakeField(self)
        layout.addWidget(self._field)
        self.setFixedSize(self._field.sizeHint())
        self._field.setFocus()
