"""Multi-player Snake game window for the mzML Explorer easter egg."""

import random
import collections
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPainter, QColor, QKeyEvent, QFont

BLOCK = 8  # pixels per block
COLS = 40  # number of columns
ROWS = 40  # number of rows
WIDTH = BLOCK * COLS  # 320 px
HEIGHT = BLOCK * ROWS  # 320 px

COL_BG = QColor("#1a1a2e")
COL_GRID = QColor("#16213e")
COL_FOOD = QColor("#f87171")
COL_TEXT = QColor("#e2e8f0")

TICK_MS = 120  # ms per step

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
OPPOSITES = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# (head_color, body_color, display_label)
_SNAKE_DEFS = [
    (QColor("#4ade80"), QColor("#22c55e"), "P1 \u2190\u2191\u2192\u2193"),
    (QColor("#60a5fa"), QColor("#3b82f6"), "P2 WASD"),
    (QColor("#fbbf24"), QColor("#f59e0b"), "P3 IJKM"),
    (QColor("#f472b6"), QColor("#ec4899"), "AI"),
    (QColor("#a78bfa"), QColor("#8b5cf6"), "AI"),
    (QColor("#34d399"), QColor("#10b981"), "AI"),
    (QColor("#fb923c"), QColor("#f97316"), "AI"),
]

# Starting (col, row, initial_direction) for up to 7 snakes
_STARTS = [
    (COLS // 4, ROWS // 2, RIGHT),
    ((COLS * 3) // 4, ROWS // 2, LEFT),
    (COLS // 2, ROWS // 4, DOWN),
    (COLS // 2, (ROWS * 3) // 4, UP),
    (COLS // 4, ROWS // 4, RIGHT),
    ((COLS * 3) // 4, ROWS // 4, LEFT),
    (COLS // 4, (ROWS * 3) // 4, RIGHT),
]

# Logo bounding box in grid coordinates (inclusive) – used for visibility check
_LOGO_ROW_MIN = 13
_LOGO_ROW_MAX = 26
_LOGO_COL_MIN = 6
_LOGO_COL_MAX = 33

# Lobby / game states
_ST_LOBBY_PLAYERS = 0
_ST_LOBBY_AI = 1
_ST_LOBBY_READY = 2
_ST_RUNNING = 3
_ST_GAME_OVER = 4


def _bfs_dir(head, food, occupied):
    """BFS from *head* toward *food* avoiding *occupied* cells.
    Returns the first direction to take, or a safe fallback, or None."""
    queue = collections.deque([(head, None)])
    visited = {head}
    while queue:
        pos, first = queue.popleft()
        x, y = pos
        for d in (UP, DOWN, LEFT, RIGHT):
            npos = ((x + d[0]) % COLS, (y + d[1]) % ROWS)
            if npos in visited or npos in occupied:
                continue
            fd = first if first is not None else d
            if npos == food:
                return fd
            visited.add(npos)
            queue.append((npos, fd))
    # No path to food – return any safe direction
    x, y = head
    for d in (UP, DOWN, LEFT, RIGHT):
        if ((x + d[0]) % COLS, (y + d[1]) % ROWS) not in occupied:
            return d
    return None


class _Snake:
    """State for one snake (human or AI)."""

    __slots__ = ("idx", "is_ai", "body", "direction", "next_dir", "score", "alive", "color_head", "color_body", "label")

    def __init__(self, idx: int, is_ai: bool):
        self.idx = idx
        self.is_ai = is_ai
        self.score = 0
        self.alive = True
        sx, sy, d = _STARTS[idx]
        dx, dy = d
        self.body = [((sx - dx * i) % COLS, (sy - dy * i) % ROWS) for i in range(3)]
        self.direction = d
        self.next_dir = d
        ch, cb, lbl = _SNAKE_DEFS[idx]
        self.color_head = ch
        self.color_body = cb
        self.label = lbl


class SnakeField(QWidget):
    """Multi-player snake canvas."""

    # ----- class-level key maps (evaluated once) -----
    _P1_MAP = {
        Qt.Key.Key_Up: UP,
        Qt.Key.Key_Down: DOWN,
        Qt.Key.Key_Left: LEFT,
        Qt.Key.Key_Right: RIGHT,
    }
    _P2_MAP = {
        Qt.Key.Key_W: UP,
        Qt.Key.Key_S: DOWN,
        Qt.Key.Key_A: LEFT,
        Qt.Key.Key_D: RIGHT,
    }
    _P3_MAP = {
        Qt.Key.Key_I: UP,
        Qt.Key.Key_K: DOWN,
        Qt.Key.Key_J: LEFT,
        Qt.Key.Key_M: RIGHT,
    }
    _PLAYER_MAPS = [_P1_MAP, _P2_MAP, _P3_MAP]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(WIDTH, HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._state = _ST_LOBBY_PLAYERS
        self._num_players = 1
        self._num_ai = 0
        self._snakes: list[_Snake] = []
        self._food = (COLS // 2, ROWS // 2)
        self._show_logo = True
        self._update_title()

    # ------------------------------------------------------------------ state

    def _enter_lobby(self):
        self._timer.stop()
        self._state = _ST_LOBBY_PLAYERS
        self._num_players = 1
        self._num_ai = 0
        self._snakes = []
        self._show_logo = True
        self._update_title()
        self.update()

    def _start_game(self):
        self._snakes = [_Snake(i, is_ai=False) for i in range(self._num_players)] + [_Snake(self._num_players + i, is_ai=True) for i in range(self._num_ai)]
        self._food = self._random_food()
        self._show_logo = True
        self._state = _ST_RUNNING
        self._timer.start(TICK_MS)
        self._update_title()
        self.update()

    # ----------------------------------------------------------------- helpers

    def _occupied(self) -> set:
        s: set = set()
        for sn in self._snakes:
            if sn.alive:
                s.update(sn.body)
        return s

    def _random_food(self):
        occ = self._occupied()
        while True:
            pos = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
            if pos not in occ:
                return pos

    # ------------------------------------------------------------------- AI

    def _ai_direction(self, snake: _Snake):
        """BFS toward food; exclude own body (minus head+tail) and all others."""
        occ: set = set()
        for s in self._snakes:
            if not s.alive:
                continue
            if s is snake:
                # exclude own head (start) and tail (will vacate)
                occ.update(s.body[1:-1])
            else:
                occ.update(s.body)
        d = _bfs_dir(snake.body[0], self._food, occ)
        if d is None or d == OPPOSITES.get(snake.direction):
            d = snake.direction
        return d

    # ------------------------------------------------------------------- loop

    def _step(self):
        # AI decisions first
        for s in self._snakes:
            if s.alive and s.is_ai:
                s.next_dir = self._ai_direction(s)

        # Lock directions (prevent reversal)
        for s in self._snakes:
            if s.alive and s.next_dir != OPPOSITES.get(s.direction):
                s.direction = s.next_dir

        # Compute new heads
        new_heads: dict = {}
        for s in self._snakes:
            if s.alive:
                hx, hy = s.body[0]
                dx, dy = s.direction
                new_heads[s.idx] = ((hx + dx) % COLS, (hy + dy) % ROWS)

        eat = {idx: (nh == self._food) for idx, nh in new_heads.items()}

        # Tentative new bodies
        new_bodies: dict = {}
        for s in self._snakes:
            if s.alive:
                nh = new_heads[s.idx]
                tail = list(s.body) if eat[s.idx] else list(s.body[:-1])
                new_bodies[s.idx] = [nh] + tail

        # Collision detection
        dead: set = set()
        for s in self._snakes:
            if not s.alive:
                continue
            nh = new_heads[s.idx]
            if nh in new_bodies[s.idx][1:]:
                dead.add(s.idx)
                continue
            for other in self._snakes:
                if other.idx == s.idx or not other.alive:
                    continue
                if nh in new_bodies[other.idx]:
                    dead.add(s.idx)
                    break

        # Apply moves
        food_eaten = False
        for s in self._snakes:
            if not s.alive:
                continue
            if s.idx in dead:
                s.alive = False
            else:
                s.body = new_bodies[s.idx]
                if eat[s.idx]:
                    s.score += 1
                    food_eaten = True

        if food_eaten:
            self._food = self._random_food()

        # Logo visibility: hide once any snake enters the logo area
        if self._show_logo:
            for s in self._snakes:
                if not s.alive:
                    continue
                for cx, cy in s.body:
                    if _LOGO_ROW_MIN <= cy <= _LOGO_ROW_MAX and _LOGO_COL_MIN <= cx <= _LOGO_COL_MAX:
                        self._show_logo = False
                        break
                if not self._show_logo:
                    break

        # Game-over check
        alive_now = [s for s in self._snakes if s.alive]
        total = len(self._snakes)
        if (total == 1 and not alive_now) or (total > 1 and len(alive_now) <= 1):
            self._state = _ST_GAME_OVER
            self._timer.stop()

        self._update_title()
        self.update()

    # ----------------------------------------------------------------- input

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()

        if self._state == _ST_LOBBY_PLAYERS:
            if key == Qt.Key.Key_1:
                self._num_players = 1
                self._state = _ST_LOBBY_AI
                self.update()
            elif key == Qt.Key.Key_2:
                self._num_players = 2
                self._state = _ST_LOBBY_AI
                self.update()
            elif key == Qt.Key.Key_3:
                self._num_players = 3
                self._state = _ST_LOBBY_AI
                self.update()

        elif self._state == _ST_LOBBY_AI:
            _NUM = {
                Qt.Key.Key_0: 0,
                Qt.Key.Key_1: 1,
                Qt.Key.Key_2: 2,
                Qt.Key.Key_3: 3,
                Qt.Key.Key_4: 4,
            }
            if key in _NUM:
                n = _NUM[key]
                if self._num_players + n <= 7:
                    self._num_ai = n
                    self._state = _ST_LOBBY_READY
                    self.update()

        elif self._state == _ST_LOBBY_READY:
            if key == Qt.Key.Key_Space:
                self._start_game()
            elif key == Qt.Key.Key_R:
                self._enter_lobby()

        elif self._state == _ST_GAME_OVER:
            self._enter_lobby()

        elif self._state == _ST_RUNNING:
            for pidx, ctrl in enumerate(self._PLAYER_MAPS):
                if pidx >= self._num_players:
                    break
                if key in ctrl:
                    s = self._snakes[pidx]
                    if s.alive:
                        nd = ctrl[key]
                        if nd != OPPOSITES.get(s.direction):
                            s.next_dir = nd
            if key == Qt.Key.Key_R:
                self._enter_lobby()

        event.accept()

    # --------------------------------------------------------------- painting

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        painter.fillRect(0, 0, WIDTH, HEIGHT, COL_BG)

        # Subtle grid
        painter.setPen(COL_GRID)
        for c in range(0, WIDTH, BLOCK):
            painter.drawLine(c, 0, c, HEIGHT)
        for r in range(0, HEIGHT, BLOCK):
            painter.drawLine(0, r, WIDTH, r)

        if self._state in (_ST_LOBBY_PLAYERS, _ST_LOBBY_AI, _ST_LOBBY_READY):
            self._paint_lobby(painter)
        else:
            self._paint_game(painter)
            if self._show_logo:
                self._paint_logo(painter)
            if self._state == _ST_GAME_OVER:
                self._paint_game_over(painter)

        painter.end()

    def _paint_lobby(self, painter: QPainter):
        # Title
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        painter.setFont(f)
        painter.setPen(COL_TEXT)
        painter.drawText(QRect(0, 14, WIDTH, 32), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "SNAKE")

        # Color legend for already-chosen snakes
        total = self._num_players + self._num_ai
        f2 = QFont()
        f2.setPointSize(9)
        painter.setFont(f2)
        legend_x = WIDTH // 2 - 68
        for i in range(total):
            ch, cb, lbl = _SNAKE_DEFS[i]
            role = "AI" if i >= self._num_players else lbl
            sy = 54 + i * 19
            painter.fillRect(legend_x, sy, 10, 10, ch)
            painter.setPen(ch)
            painter.drawText(legend_x + 14, sy + 10, role)

        # State-specific prompt
        f3 = QFont()
        f3.setPointSize(10)
        f3.setBold(True)
        painter.setFont(f3)
        painter.setPen(COL_TEXT)

        if self._state == _ST_LOBBY_PLAYERS:
            msg = "How many players?\n\n1  –  Single player  (\u2190\u2191\u2192\u2193)\n2  –  Two players   (+ WASD)\n3  –  Three players (+ IJKM)"
            painter.drawText(QRect(0, HEIGHT // 2 - 55, WIDTH, 150), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, msg)

        elif self._state == _ST_LOBBY_AI:
            max_ai = min(4, 7 - self._num_players)
            msg = f"How many AI snakes?\n\nPress 0 \u2013 {max_ai}"
            painter.drawText(QRect(0, HEIGHT // 2 - 30, WIDTH, 100), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, msg)

        elif self._state == _ST_LOBBY_READY:
            msg = "Press SPACE to start\n\nR \u2013 back to menu"
            painter.drawText(QRect(0, HEIGHT // 2 + 20, WIDTH, 80), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, msg)

    def _paint_game(self, painter: QPainter):
        # Food
        fx, fy = self._food
        painter.fillRect(fx * BLOCK, fy * BLOCK, BLOCK, BLOCK, COL_FOOD)

        # Snakes (reverse order so P1 head draws on top)
        for s in reversed(self._snakes):
            for i, (cx, cy) in enumerate(s.body):
                col = s.color_head if i == 0 else s.color_body
                if not s.alive:
                    col = QColor(col.red() // 3, col.green() // 3, col.blue() // 3)
                painter.fillRect(cx * BLOCK + 1, cy * BLOCK + 1, BLOCK - 1, BLOCK - 1, col)

        # Score strip at top
        f = QFont()
        f.setPointSize(7)
        f.setBold(True)
        painter.setFont(f)
        x = 2
        for s in self._snakes:
            painter.setPen(s.color_head)
            name = "AI" if s.is_ai else s.label.split()[0]
            painter.drawText(x, 10, f"{name}:{s.score}")
            x += 40

    def _paint_logo(self, painter: QPainter):
        logo_rect = QRect(
            _LOGO_COL_MIN * BLOCK,
            _LOGO_ROW_MIN * BLOCK,
            (_LOGO_COL_MAX - _LOGO_COL_MIN + 1) * BLOCK,
            (_LOGO_ROW_MAX - _LOGO_ROW_MIN + 1) * BLOCK,
        )
        f = QFont()
        f.setPointSize(18)
        f.setBold(True)
        painter.setFont(f)
        painter.setPen(QColor(255, 255, 255, 55))
        painter.drawText(logo_rect, Qt.AlignmentFlag.AlignCenter, "mzML\nExplorer")

    def _paint_game_over(self, painter: QPainter):
        painter.fillRect(0, HEIGHT // 2 - 52, WIDTH, 114, QColor(26, 26, 46, 210))
        f = QFont()
        f.setPointSize(11)
        f.setBold(True)
        painter.setFont(f)
        painter.setPen(COL_TEXT)

        alive = [s for s in self._snakes if s.alive]
        if len(alive) == 1:
            w = alive[0]
            result = f"{'AI' if w.is_ai else w.label} wins!"
        else:
            best = max(self._snakes, key=lambda s: s.score)
            ties = sum(1 for s in self._snakes if s.score == best.score)
            result = "Draw!" if ties > 1 else f"{'AI' if best.is_ai else best.label} wins!"

        scores = "  ".join(f"{'AI' if s.is_ai else s.label.split()[0]}:{s.score}" for s in self._snakes)
        msg = f"GAME OVER \u2013 {result}\n{scores}\n\nAny key for menu"
        painter.drawText(QRect(0, HEIGHT // 2 - 52, WIDTH, 114), Qt.AlignmentFlag.AlignCenter, msg)

    # ------------------------------------------------------------------ misc

    def _update_title(self):
        win = self.window()
        if self._state == _ST_RUNNING:
            parts = [f"{'AI' if s.is_ai else s.label.split()[0]}:{s.score}" for s in self._snakes]
            win.setWindowTitle("Snake \u2013 " + "  ".join(parts))
        elif self._state == _ST_GAME_OVER:
            win.setWindowTitle("Snake \u2013 Game Over")
        else:
            win.setWindowTitle("Snake")


class SnakeWindow(QWidget):
    """Top-level window hosting the multi-player snake game."""

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
