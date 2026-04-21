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
COL_FOOD_BONUS = QColor("#fbbf24")  # golden – worth 5 pts
COL_TEXT = QColor("#e2e8f0")

BONUS_FOOD_CHANCE = 0.05  # probability a newly spawned food is golden

TICK_MS = 120  # ms per step (starting speed)
MIN_TICK_MS = 50  # fastest allowed tick
SPEED_UP_MS = 1  # ms shaved off per treat eaten

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


# Logo bounding box in grid coordinates (inclusive) – used for visibility check
_LOGO_ROW_MIN = 13
_LOGO_ROW_MAX = 26
_LOGO_COL_MIN = 6
_LOGO_COL_MAX = 33

# Lobby / game states
_ST_LOBBY_PLAYERS = 0
_ST_LOBBY_AI = 1
_ST_LOBBY_FOOD = 7
_ST_LOBBY_READY = 2
_ST_RUNNING = 3
_ST_GAME_OVER = 4
_ST_COUNTDOWN = 5
_ST_PAUSED = 6


def _bfs_dir(head, targets: set, occupied):
    """BFS from *head* toward any position in *targets* avoiding *occupied* cells.
    Returns the first direction to take, or a safe fallback, or None."""
    if not targets:
        return None
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
            if npos in targets:
                return fd
            visited.add(npos)
            queue.append((npos, fd))
    # No path to any food – return any safe direction
    x, y = head
    for d in (UP, DOWN, LEFT, RIGHT):
        if ((x + d[0]) % COLS, (y + d[1]) % ROWS) not in occupied:
            return d
    return None


class _Snake:
    """State for one snake (human or AI)."""

    __slots__ = ("idx", "is_ai", "body", "direction", "next_dir", "score", "alive", "color_head", "color_body", "label")

    def __init__(self, idx: int, is_ai: bool, sx: int, sy: int, d: tuple):
        self.idx = idx
        self.is_ai = is_ai
        self.score = 0
        self.alive = True
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
        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._countdown_tick)
        self._countdown_val = 0
        self._current_tick_ms = TICK_MS
        self._state = _ST_LOBBY_PLAYERS
        self._num_players = 1
        self._num_ai = 0
        self._num_foods = 2
        self._snakes: list[_Snake] = []
        self._foods: dict = {}  # pos -> points
        self._show_logo = True
        self._update_title()

    # ------------------------------------------------------------------ state

    def _enter_lobby(self):
        self._timer.stop()
        self._countdown_timer.stop()
        self._current_tick_ms = TICK_MS
        self._state = _ST_LOBBY_PLAYERS
        self._num_players = 1
        self._num_ai = 0
        self._num_foods = 2
        self._snakes = []
        self._foods = {}
        self._show_logo = True
        self._update_title()
        self.update()

    def _random_start_positions(self, count: int) -> list:
        """Return list of (sx, sy, direction) for *count* snakes without body overlap."""
        dirs = [UP, DOWN, LEFT, RIGHT]
        result = []
        all_occupied: set = set()
        for _ in range(count):
            for _attempt in range(400):
                d = random.choice(dirs)
                dx, dy = d
                sx = random.randint(3, COLS - 4)
                sy = random.randint(3, ROWS - 4)
                body = [((sx - dx * i) % COLS, (sy - dy * i) % ROWS) for i in range(3)]
                if not any(b in all_occupied for b in body):
                    result.append((sx, sy, d))
                    all_occupied.update(body)
                    break
            else:
                # Fallback: place anywhere (very unlikely to be needed)
                d = random.choice(dirs)
                result.append((random.randint(0, COLS - 1), random.randint(0, ROWS - 1), d))
        return result

    def _start_game(self):
        total = self._num_players + self._num_ai
        starts = self._random_start_positions(total)
        self._snakes = [_Snake(i, is_ai=False, sx=starts[i][0], sy=starts[i][1], d=starts[i][2]) for i in range(self._num_players)] + [
            _Snake(self._num_players + i, is_ai=True, sx=starts[self._num_players + i][0], sy=starts[self._num_players + i][1], d=starts[self._num_players + i][2])
            for i in range(self._num_ai)
        ]
        self._foods = {}
        for _ in range(self._num_foods):
            self._spawn_food()
        self._show_logo = True
        self._current_tick_ms = TICK_MS
        self._countdown_val = 3
        self._state = _ST_COUNTDOWN
        self._countdown_timer.start(1000)
        self._update_title()
        self.update()

    def _countdown_tick(self):
        self._countdown_val -= 1
        if self._countdown_val <= 0:
            self._countdown_timer.stop()
            self._state = _ST_RUNNING
            self._timer.start(self._current_tick_ms)
        self._update_title()
        self.update()

    # ----------------------------------------------------------------- helpers

    def _occupied(self) -> set:
        s: set = set()
        for sn in self._snakes:
            if sn.alive:
                s.update(sn.body)
        return s

    def _spawn_food(self):
        """Add one food item to self._foods at a free position."""
        occ = self._occupied() | set(self._foods.keys())
        for _ in range(2000):
            pos = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
            if pos not in occ:
                pts = 5 if random.random() < BONUS_FOOD_CHANCE else 1
                self._foods[pos] = pts
                return
        # Grid is very full – skip

    # ------------------------------------------------------------------- AI

    def _ai_direction(self, snake: _Snake):
        """BFS toward nearest food; exclude own body (minus head+tail) and all others."""
        occ: set = set()
        for s in self._snakes:
            if not s.alive:
                continue
            if s is snake:
                occ.update(s.body[1:-1])
            else:
                occ.update(s.body)
        targets = set(self._foods.keys())
        d = _bfs_dir(snake.body[0], targets, occ)
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

        # Which snakes land on a food cell?
        eat = {idx: nh if nh in self._foods else None for idx, nh in new_heads.items()}

        # Tentative new bodies
        new_bodies: dict = {}
        for s in self._snakes:
            if s.alive:
                nh = new_heads[s.idx]
                tail = list(s.body) if eat[s.idx] is not None else list(s.body[:-1])
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
        treats_eaten = 0
        for s in self._snakes:
            if not s.alive:
                continue
            if s.idx in dead:
                s.alive = False
            else:
                s.body = new_bodies[s.idx]
                food_pos = eat[s.idx]
                if food_pos is not None and food_pos in self._foods:
                    pts = self._foods.pop(food_pos)
                    s.score += pts
                    treats_eaten += 1
                    self._spawn_food()  # replace the eaten item

        if treats_eaten:
            self._current_tick_ms = max(MIN_TICK_MS, self._current_tick_ms - SPEED_UP_MS * treats_eaten)
            self._timer.setInterval(self._current_tick_ms)

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
                    self._state = _ST_LOBBY_FOOD
                    self.update()

        elif self._state == _ST_LOBBY_FOOD:
            _FNUM = {
                Qt.Key.Key_1: 1,
                Qt.Key.Key_2: 2,
                Qt.Key.Key_3: 3,
                Qt.Key.Key_4: 4,
            }
            if key in _FNUM:
                self._num_foods = _FNUM[key]
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
            if key == Qt.Key.Key_Space:
                self._timer.stop()
                self._state = _ST_PAUSED
                self._update_title()
                self.update()
            else:
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

        elif self._state == _ST_PAUSED:
            if key == Qt.Key.Key_Space:
                self._state = _ST_RUNNING
                self._timer.start(self._current_tick_ms)
                self._update_title()
                self.update()
            elif key == Qt.Key.Key_R:
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

        if self._state in (_ST_LOBBY_PLAYERS, _ST_LOBBY_AI, _ST_LOBBY_FOOD, _ST_LOBBY_READY):
            self._paint_lobby(painter)
        elif self._state == _ST_COUNTDOWN:
            self._paint_game(painter)
            if self._show_logo:
                self._paint_logo(painter)
            self._paint_countdown(painter)
        else:
            self._paint_game(painter)
            if self._show_logo:
                self._paint_logo(painter)
            if self._state == _ST_GAME_OVER:
                self._paint_game_over(painter)
            elif self._state == _ST_PAUSED:
                self._paint_paused(painter)

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

        elif self._state == _ST_LOBBY_FOOD:
            msg = "How many food sources?\n\nPress 1 \u2013 4\n\n\u2605 yellow food = 5 pts"
            painter.drawText(QRect(0, HEIGHT // 2 - 40, WIDTH, 130), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, msg)

        elif self._state == _ST_LOBBY_READY:
            msg = "Press SPACE to start\n\nR \u2013 back to menu"
            painter.drawText(QRect(0, HEIGHT // 2 + 20, WIDTH, 80), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, msg)

    def _paint_game(self, painter: QPainter):
        # Food items
        for (fx, fy), pts in self._foods.items():
            col = COL_FOOD_BONUS if pts > 1 else COL_FOOD
            painter.fillRect(fx * BLOCK, fy * BLOCK, BLOCK, BLOCK, col)

        # Snakes: dead first, then alive on top; within each group reverse so P1 is topmost
        dead_snakes = [s for s in self._snakes if not s.alive]
        alive_snakes = [s for s in self._snakes if s.alive]
        for s in list(reversed(dead_snakes)) + list(reversed(alive_snakes)):
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

    def _paint_countdown(self, painter: QPainter):
        painter.fillRect(0, HEIGHT // 2 - 52, WIDTH, 104, QColor(26, 26, 46, 200))
        f = QFont()
        f.setPointSize(48)
        f.setBold(True)
        painter.setFont(f)
        painter.setPen(COL_TEXT)
        label = str(self._countdown_val) if self._countdown_val > 0 else "GO!"
        painter.drawText(QRect(0, HEIGHT // 2 - 52, WIDTH, 104), Qt.AlignmentFlag.AlignCenter, label)

    def _paint_paused(self, painter: QPainter):
        painter.fillRect(0, HEIGHT // 2 - 44, WIDTH, 90, QColor(26, 26, 46, 210))
        f = QFont()
        f.setPointSize(14)
        f.setBold(True)
        painter.setFont(f)
        painter.setPen(COL_TEXT)
        painter.drawText(QRect(0, HEIGHT // 2 - 44, WIDTH, 90), Qt.AlignmentFlag.AlignCenter, "PAUSED\n\nSPACE – resume   R – menu")

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
        elif self._state == _ST_COUNTDOWN:
            win.setWindowTitle(f"Snake \u2013 {self._countdown_val}" if self._countdown_val > 0 else "Snake \u2013 GO!")
        elif self._state == _ST_PAUSED:
            win.setWindowTitle("Snake \u2013 Paused")
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
