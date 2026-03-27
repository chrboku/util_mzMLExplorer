"""
MzML File Explorer Window — browse all scans in a single mzML file.

Left panel : QTreeWidget showing  File → MS1 scans → MS2 scans
Right panel: up to 4 Qt Chart spectrum plots in a 2×2 grid.

Usage
-----
    win = MzMLFileExplorerWindow(filepath, file_manager, parent=None)
    win.show()
"""

import os
import numpy as np
from collections import deque

from PyQt6.QtWidgets import (
    QApplication,
    QMenu,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QFrame,
    QHeaderView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMargins, QPointF
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPen, QColor, QPainter

from .window_msms import MSMSPopupWindow
from .window_shared import ANNOTATION_COLOR_PRESETS


# ──────────────────────────────────────────────────────────────────────────────
# Background loader
# ──────────────────────────────────────────────────────────────────────────────


class MzMLLoadWorker(QThread):
    """Load a single mzML file off the UI thread."""

    finished = pyqtSignal(dict)  # {"ms1": [...], "ms2": [...]}
    error = pyqtSignal(str)

    def __init__(self, filepath: str, file_manager, parent=None):
        super().__init__(parent)
        self._filepath = filepath
        self._file_manager = file_manager

    def run(self):
        try:
            data = self._file_manager.load_single_file(self._filepath)
            self.finished.emit(data)
        except Exception as exc:
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Small interactive chart view (pan / zoom / wheel / double-click reset)
# ──────────────────────────────────────────────────────────────────────────────


class _SpectrumChartView(QChartView):
    """Minimal interactive QChartView reused for each of the 4 panels."""

    def __init__(self, chart, parent=None):
        super().__init__(chart, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRubberBand(QChartView.RubberBand.NoRubberBand)
        self.setMouseTracking(True)

        self._panning = False
        self._zooming = False
        self._pan_start = None
        self._zoom_start = None
        self._start_x = None
        self._start_y = None
        self._anchor_x = 0.0
        self._anchor_y = 0.0
        self._full_x = None  # (min, max) for double-click reset
        self._full_y = None

        # Hover highlight state (set by _ChartPanel after construction)
        self._spec_mz = None
        self._spec_ints = None
        self._hover_mz = None
        self._hover_norm = 0.0
        self._hover_series = None

        # Pinned annotation state
        self._press_pos = None
        self._pinned_annotations: list = []  # [(mz, norm_int, color), ...]
        self._ann_labels: list = []  # permanent QLabel widgets
        self._spectrum_color = "#1565c0"  # updated by _ChartPanel.set_spectrum
        self._right_click_mz: float | None = None
        self._right_click_norm: float = 0.0

        # MSMS popup support
        self._ms_level = 1
        self._open_msms_fn = None
        self._right_press_start = None  # tracks right-button press for click vs drag

        self._hover_label = QLabel(self)
        self._hover_label.setStyleSheet("""
            QLabel {
                background-color: rgba(60, 64, 67, 220);
                color: #ffffff;
                border: none;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 11px;
            }
        """)
        self._hover_label.hide()
        self._hover_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    # public helper --------------------------------------------------------
    def set_full_range(self, x_min, x_max, y_min, y_max):
        self._full_x = (x_min, x_max)
        self._full_y = (y_min, y_max)

    # private helpers ------------------------------------------------------
    def _xa(self):
        axes = self.chart().axes(Qt.Orientation.Horizontal)
        return axes[0] if axes else None

    def _ya(self):
        axes = self.chart().axes(Qt.Orientation.Vertical)
        return axes[0] if axes else None

    # events ---------------------------------------------------------------
    def mousePressEvent(self, event):
        xa, ya = self._xa(), self._ya()
        if xa is None or ya is None:
            return super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._pan_start = event.position()
            self._press_pos = event.position()
            self._start_x = (xa.min(), xa.max())
            self._start_y = (ya.min(), ya.max())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            # Save hover state at press time for use in the context menu
            self._right_click_mz = self._hover_mz
            self._right_click_norm = self._hover_norm
            # Don't start zooming immediately — wait for a drag threshold
            self._right_press_start = event.position()
            self._zoom_start = event.position()
            self._start_x = (xa.min(), xa.max())
            self._start_y = (ya.min(), ya.max())
            pa = self.chart().plotArea()
            rx = max(0.0, min(1.0, (event.position().x() - pa.left()) / pa.width()))
            ry = max(0.0, min(1.0, (event.position().y() - pa.top()) / pa.height()))
            xr = self._start_x[1] - self._start_x[0]
            yr = self._start_y[1] - self._start_y[0]
            self._anchor_x = self._start_x[0] + rx * xr
            self._anchor_y = self._start_y[0] + (1.0 - ry) * yr
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        xa, ya = self._xa(), self._ya()
        if xa is None or ya is None:
            return
        if self._panning and self._start_x is not None:
            pa = self.chart().plotArea()
            dx = event.position().x() - self._pan_start.x()
            dy = event.position().y() - self._pan_start.y()
            ox = -dx * (self._start_x[1] - self._start_x[0]) / pa.width()
            oy = dy * (self._start_y[1] - self._start_y[0]) / pa.height()
            xa.setRange(self._start_x[0] + ox, self._start_x[1] + ox)
            ya.setRange(self._start_y[0] + oy, self._start_y[1] + oy)
        elif self._right_press_start is not None and self._start_x is not None:
            dp = event.position() - self._right_press_start
            if not self._zooming and dp.x() ** 2 + dp.y() ** 2 > 25.0:
                self._zooming = True
            if self._zooming:
                dx = event.position().x() - self._zoom_start.x()
                dy = event.position().y() - self._zoom_start.y()
                s = 0.005
                xf = max(0.1, min(10.0, 1.0 - dx * s))
                yf = max(0.1, min(10.0, 1.0 + dy * s))
                al = self._anchor_x - self._start_x[0]
                ar = self._start_x[1] - self._anchor_x
                ab = self._anchor_y - self._start_y[0]
                at = self._start_y[1] - self._anchor_y
                xa.setRange(self._anchor_x - al * xf, self._anchor_x + ar * xf)
                ya.setRange(self._anchor_y - ab * yf, self._anchor_y + at * yf)
        if not self._panning and not self._zooming:
            self._update_hover(event)
        if self._pinned_annotations:
            self._redraw_annotation_labels()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Pin annotation if this was a short click (not a pan drag) on a highlighted peak
            if self._press_pos is not None and self._hover_mz is not None:
                dp = event.position() - self._press_pos
                if dp.x() ** 2 + dp.y() ** 2 < 25.0:
                    self._pin_annotation(self._hover_mz, self._hover_norm, self._spectrum_color)
            self._panning = False
            self._press_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            was_dragging = self._zooming
            self._zooming = False
            self._right_press_start = None
            if not was_dragging:
                self._show_context_menu(event)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._full_x and self._full_y:
            xa, ya = self._xa(), self._ya()
            if xa and ya:
                xa.setRange(*self._full_x)
                ya.setRange(*self._full_y)
                if self._pinned_annotations:
                    self._redraw_annotation_labels()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        xa, ya = self._xa(), self._ya()
        if xa is None or ya is None:
            return
        zf = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        pa = self.chart().plotArea()
        rx = (event.position().x() - pa.left()) / pa.width()
        ry = (event.position().y() - pa.top()) / pa.height()
        xr, yr = xa.max() - xa.min(), ya.max() - ya.min()
        cx = xa.min() + rx * xr
        cy = ya.min() + (1.0 - ry) * yr
        nxr, nyr = xr / zf, yr / zf
        xa.setRange(cx - rx * nxr, cx + (1.0 - rx) * nxr)
        ya.setRange(cy - (1.0 - ry) * nyr, cy + ry * nyr)
        if self._pinned_annotations:
            self._redraw_annotation_labels()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pinned_annotations:
            self._redraw_annotation_labels()

    def leaveEvent(self, event):
        self._clear_hover()
        super().leaveEvent(event)

    def _clear_hover(self):
        if self._hover_series is not None:
            self.chart().removeSeries(self._hover_series)
            self._hover_series = None
        self._hover_mz = None
        self._hover_norm = 0.0
        self._hover_label.hide()

    def _pin_annotation(self, mz: float, norm_int: float, color: str = "#1565c0"):
        """Pin a colored m/z label; clicking the same peak again removes it."""
        for i, (m, *_) in enumerate(self._pinned_annotations):
            if abs(m - mz) < 1e-9:
                self._pinned_annotations.pop(i)
                lbl = self._ann_labels.pop(i)
                lbl.deleteLater()
                return
        self._pinned_annotations.append((mz, norm_int, color))
        lbl = QLabel(f"{mz:.5f}", self)
        lbl.setStyleSheet(f"color: {color}; background-color: rgba(255,255,255,200); border: none; border-radius: 2px; font-size: 11px; font-weight: bold; padding: 1px 4px;")
        lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lbl.adjustSize()
        self._ann_labels.append(lbl)
        lbl.show()
        self._redraw_annotation_labels()

    def _show_context_menu(self, event):
        """Right-click context menu: Annotate (with colour submenu) + MSMS viewer."""
        mz = self._right_click_mz
        norm = self._right_click_norm
        menu = QMenu(self)
        if mz is not None:
            ann_menu = menu.addMenu("Annotate")
            for name, color_str in ANNOTATION_COLOR_PRESETS:
                act = ann_menu.addAction(name)
                act.triggered.connect(lambda _c=False, m=mz, n=norm, c=color_str: self._pin_annotation(m, n, c))
        if self._ms_level == 2 and self._open_msms_fn is not None:
            menu.addAction("Open in MSMS Spectrum Viewer").triggered.connect(lambda _checked=False: self._open_msms_fn())
        if not menu.isEmpty():
            menu.exec(event.globalPosition().toPoint())

    def _redraw_annotation_labels(self):
        """Reposition all pinned annotation labels to match current chart coordinates."""
        xa = self._xa()
        for lbl, (mz, norm_int, _color) in zip(self._ann_labels, self._pinned_annotations):
            tip_pos = self.chart().mapToPosition(QPointF(mz, norm_int))
            lx = int(tip_pos.x()) - lbl.width() // 2
            ly = int(tip_pos.y()) - lbl.height() - 6
            lx = max(0, min(lx, self.width() - lbl.width()))
            if ly < 0:
                ly = int(tip_pos.y()) + 6
            lbl.move(lx, ly)
            if xa is not None and xa.min() <= mz <= xa.max():
                lbl.show()
            else:
                lbl.hide()

    def _update_hover(self, event):
        """Highlight the closest peak stick to the cursor with firebrick color and show its m/z."""
        if self._spec_mz is None or len(self._spec_mz) == 0:
            self._hover_label.hide()
            return

        plot_area = self.chart().plotArea()
        if not plot_area.contains(event.position()):
            self._clear_hover()
            return

        max_int = float(np.max(self._spec_ints)) if np.max(self._spec_ints) > 0 else 1.0
        PIXEL_THRESHOLD = 20
        mx, my = event.position().x(), event.position().y()

        best_mz = None
        best_norm = 0.0
        best_dist = float("inf")

        for mz_v, int_v in zip(self._spec_mz, self._spec_ints):
            mz_f = float(mz_v)
            norm_int = float(int_v) / max_int * 100.0
            tip = self.chart().mapToPosition(QPointF(mz_f, norm_int))
            base = self.chart().mapToPosition(QPointF(mz_f, 0.0))
            tx, ty, bx, by = tip.x(), tip.y(), base.x(), base.y()
            dx, dy = bx - tx, by - ty
            seg_sq = dx * dx + dy * dy
            t = max(0.0, min(1.0, ((mx - tx) * dx + (my - ty) * dy) / seg_sq)) if seg_sq > 0 else 0.0
            dist = ((mx - tx - t * dx) ** 2 + (my - ty - t * dy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_mz = mz_f
                best_norm = norm_int

        if best_mz is not None and best_dist <= PIXEL_THRESHOLD:
            if best_mz != self._hover_mz:
                self._hover_mz = best_mz
                self._hover_norm = best_norm
                if self._hover_series is not None:
                    self.chart().removeSeries(self._hover_series)
                    self._hover_series = None

                hs = QLineSeries()
                hs.setPen(QPen(QColor(178, 34, 34), 3))  # firebrick
                hs.append(best_mz, 0.0)
                hs.append(best_mz, best_norm)
                hs.append(best_mz, 0.0)
                self.chart().addSeries(hs)
                x_axes = self.chart().axes(Qt.Orientation.Horizontal)
                y_axes = self.chart().axes(Qt.Orientation.Vertical)
                if x_axes and y_axes:
                    hs.attachAxis(x_axes[0])
                    hs.attachAxis(y_axes[0])
                self._hover_series = hs
                self.chart().legend().setVisible(False)

                self._hover_label.setText(f"{best_mz:.5f}")
                self._hover_label.adjustSize()

            tip_pos = self.chart().mapToPosition(QPointF(best_mz, best_norm))
            lx = int(tip_pos.x()) - self._hover_label.width() // 2
            ly = int(tip_pos.y()) - self._hover_label.height() - 6
            lx = max(0, min(lx, self.width() - self._hover_label.width()))
            if ly < 0:
                ly = int(tip_pos.y()) + 6
            self._hover_label.move(lx, ly)
            self._hover_label.show()
        else:
            self._clear_hover()


# ──────────────────────────────────────────────────────────────────────────────
# One chart panel (title label + chart view)
# ──────────────────────────────────────────────────────────────────────────────


class _ChartPanel(QFrame):
    """A single cell in the 2×2 grid: thin border + title + chart view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._title = QLabel("—")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet("font-size: 11px; font-weight: bold; padding: 2px 4px;")
        self._title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._chart_view: _SpectrumChartView | None = None
        self._current_spec = None  # identity-based skip in _refresh_panels

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(2)
        self._layout.addWidget(self._title)

        # placeholder
        self._placeholder = QLabel("Select a scan from the tree →")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #9aa0a6; font-size: 11px;")
        self._layout.addWidget(self._placeholder, stretch=1)

    def set_spectrum(self, title: str, mz_array, intensity_array, ms_level: int = 1, spec=None, open_msms_fn=None):
        """Replace the chart view with a new stick spectrum."""
        # remove old chart view / placeholder
        if self._chart_view is not None:
            self._layout.removeWidget(self._chart_view)
            self._chart_view.deleteLater()
            self._chart_view = None
        if self._placeholder is not None:
            self._layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None

        self._title.setText(title)

        mz = np.asarray(mz_array, dtype=float)
        ints = np.asarray(intensity_array, dtype=float)

        chart = QChart()
        chart.setMargins(QMargins(0, 0, 0, 0))
        chart.layout().setContentsMargins(0, 0, 0, 0)
        chart.legend().setVisible(False)

        color = QColor("#1565c0") if ms_level == 1 else QColor("#0d7a2f")
        series = QLineSeries()
        series.setPen(QPen(color, 1.2))

        if len(mz) > 0:
            max_int = float(np.max(ints)) if np.max(ints) > 0 else 1.0
            rel = ints / max_int * 100.0
            for mz_v, rel_v in zip(mz, rel):
                series.append(float(mz_v), 0.0)
                series.append(float(mz_v), float(rel_v))
                series.append(float(mz_v), 0.0)

        chart.addSeries(series)

        x_axis = QValueAxis()
        x_axis.setTitleText("m/z")
        x_axis.setLabelFormat("%.1f")
        y_axis = QValueAxis()
        y_axis.setTitleText("Rel. int. (%)")
        y_axis.setRange(0.0, 110.0)
        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)

        if len(mz) > 0:
            margin = max((float(mz[-1]) - float(mz[0])) * 0.05, 1.0)
            x_axis.setRange(float(mz[0]) - margin, float(mz[-1]) + margin)

        cv = _SpectrumChartView(chart)
        if len(mz) > 0:
            cv.set_full_range(x_axis.min(), x_axis.max(), 0.0, 110.0)
        cv._spec_mz = mz
        cv._spec_ints = ints
        cv._ms_level = ms_level
        cv._spectrum_color = "#1565c0" if ms_level == 1 else "#0d7a2f"
        cv._open_msms_fn = open_msms_fn
        self._chart_view = cv
        self._current_spec = spec
        self._layout.addWidget(cv, stretch=1)

    def clear(self):
        """Reset panel to empty placeholder state."""
        if self._chart_view is not None:
            self._layout.removeWidget(self._chart_view)
            self._chart_view.deleteLater()
            self._chart_view = None

        self._current_spec = None
        self._title.setText("—")

        if self._placeholder is None:
            self._placeholder = QLabel("Select a scan from the tree →")
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._placeholder.setStyleSheet("color: #9aa0a6; font-size: 11px;")
            self._layout.addWidget(self._placeholder, stretch=1)


# ──────────────────────────────────────────────────────────────────────────────
# Tree widget column indices
# ──────────────────────────────────────────────────────────────────────────────

_COL_SCAN_ID = 0
_COL_TYPE = 1
_COL_RT = 2
_COL_POLARITY = 3
_COL_PREC_MZ = 4
_COL_PREC_INT = 5
_COL_FILTER = 6
_TREE_HEADERS = ["Scan ID", "Type", "RT (min)", "Polarity", "Precursor m/z", "Prec. Int.", "Filter"]

# UserRole data stored on each tree item
_ROLE_SPEC_DATA = Qt.ItemDataRole.UserRole
_ROLE_MS_LEVEL = Qt.ItemDataRole.UserRole + 1


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────


class MzMLFileExplorerWindow(QWidget):
    """Browse all scans in a single mzML file and view up to 4 at once."""

    MAX_SELECTED = 4  # maximum simultaneous chart panels

    def __init__(self, filepath: str, file_manager, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._filepath = filepath
        self._file_manager = file_manager
        self._filename = os.path.basename(filepath)

        # Ordered list of (tree_item, spectrum_data, ms_level) for selected panels
        self._pinned: deque = deque(maxlen=self.MAX_SELECTED)
        self._pinned_items: set = set()  # id(tree_item) → quick membership test

        self.setWindowTitle(f"File Explorer — {self._filename}")
        self.resize(1400, 900)

        self._init_ui()
        self._start_loading()

    # ── UI construction ───────────────────────────────────────────────────────

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # Status bar row
        status_row = QHBoxLayout()
        self._status_label = QLabel(f"Loading {self._filename}…")
        self._status_label.setStyleSheet("font-size: 11px; color: #5f6368;")
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setFixedHeight(12)
        self._progress.setFixedWidth(160)
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        status_row.addWidget(self._progress)
        root_layout.addLayout(status_row)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        # ── Left: tree ────────────────────────────────────────────────────────
        tree_container = QWidget()
        tree_layout = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.setSpacing(2)

        tree_hint = QLabel("Click a scan to pin it (max 4).  Click again to unpin.  Left-drag to pan, right-drag to zoom, wheel to zoom, double-click to reset.")
        tree_hint.setWordWrap(True)
        tree_hint.setStyleSheet("font-size: 10px; color: #5f6368; padding: 2px 0px;")
        tree_layout.addWidget(tree_hint)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(len(_TREE_HEADERS))
        self._tree.setHeaderLabels(_TREE_HEADERS)
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        self._tree.setRootIsDecorated(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setAnimated(False)
        self._tree.setSortingEnabled(False)

        # Column widths
        hh = self._tree.header()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(True)
        self._tree.setColumnWidth(_COL_SCAN_ID, 40)
        self._tree.setColumnWidth(_COL_TYPE, 40)
        self._tree.setColumnWidth(_COL_RT, 70)
        self._tree.setColumnWidth(_COL_POLARITY, 55)
        self._tree.setColumnWidth(_COL_PREC_MZ, 100)
        self._tree.setColumnWidth(_COL_PREC_INT, 90)

        self._tree.itemClicked.connect(self._on_item_clicked)
        tree_layout.addWidget(self._tree, stretch=1)

        splitter.addWidget(tree_container)

        # ── Right: 2×2 chart grid via nested splitters ────────────────────────
        # Outer vertical splitter splits top row / bottom row
        self._vsplit = QSplitter(Qt.Orientation.Vertical)

        # Top row: panels[0] (left) and panels[1] (right)
        self._hsplit_top = QSplitter(Qt.Orientation.Horizontal)
        # Bottom row: panels[2] (left) and panels[3] (right)
        self._hsplit_bot = QSplitter(Qt.Orientation.Horizontal)

        self._panels: list[_ChartPanel] = []
        for hsplit in (self._hsplit_top, self._hsplit_bot):
            for _ in range(2):
                panel = _ChartPanel()
                hsplit.addWidget(panel)
                self._panels.append(panel)
            hsplit.setSizes([500, 500])
            self._vsplit.addWidget(hsplit)

        self._vsplit.setSizes([450, 450])
        splitter.addWidget(self._vsplit)
        splitter.setSizes([380, 1020])

    # ── Loading ───────────────────────────────────────────────────────────────

    def _start_loading(self):
        self._worker = MzMLLoadWorker(self._filepath, self._file_manager, parent=self)
        self._worker.finished.connect(self._on_data_loaded)
        self._worker.error.connect(self._on_load_error)
        self._worker.start()

    def _on_data_loaded(self, data: dict):
        self._progress.setVisible(False)
        ms1_list = data.get("ms1", [])
        ms2_list = data.get("ms2", [])
        self._status_label.setText(f"{self._filename}  —  {len(ms1_list)} MS1 scans, {len(ms2_list)} MS2 scans")
        self._populate_tree(ms1_list, ms2_list)

    def _on_load_error(self, message: str):
        self._progress.setVisible(False)
        self._status_label.setText(f"Error loading file: {message}")

    # ── Tree population ───────────────────────────────────────────────────────

    def _populate_tree(self, ms1_list: list, ms2_list: list):
        """Build the file → MS1 → MS2 hierarchy."""
        self._tree.clear()

        # Sort everything by RT for proper grouping
        ms1_sorted = sorted(ms1_list, key=lambda s: s.get("scan_time", 0.0))
        ms2_sorted = sorted(ms2_list, key=lambda s: s.get("scan_time", 0.0))

        ms1_rts = [s.get("scan_time", 0.0) for s in ms1_sorted]

        # Root: filename
        root = QTreeWidgetItem(self._tree)
        root.setText(_COL_SCAN_ID, self._filename)
        root.setExpanded(True)
        font = root.font(_COL_SCAN_ID)
        font.setBold(True)
        root.setFont(_COL_SCAN_ID, font)

        # Build MS1 items
        ms1_items = []
        for spec in ms1_sorted:
            item = self._make_tree_item(spec, ms_level=1)
            root.addChild(item)
            ms1_items.append(item)

        # Assign MS2 scans to the nearest preceding MS1
        if ms1_items and ms2_sorted:

            def _parent_idx(ms2_rt):
                """Return index of the MS1 scan that precedes this MS2 RT."""
                idx = 0
                for i, rt in enumerate(ms1_rts):
                    if rt <= ms2_rt:
                        idx = i
                    else:
                        break
                return idx

            for spec in ms2_sorted:
                rt = spec.get("scan_time", 0.0)
                pidx = _parent_idx(rt)
                item = self._make_tree_item(spec, ms_level=2)
                ms1_items[pidx].addChild(item)

        # Resize columns to contents after population
        for col in range(len(_TREE_HEADERS) - 1):
            self._tree.resizeColumnToContents(col)

    @staticmethod
    def _make_tree_item(spec: dict, ms_level: int) -> QTreeWidgetItem:
        item = QTreeWidgetItem()
        scan_id = str(spec.get("scan_id", "?"))
        rt = spec.get("scan_time", 0.0)
        polarity = spec.get("polarity", "")
        filt = spec.get("filter_string", "") or ""

        item.setText(_COL_SCAN_ID, scan_id)
        item.setText(_COL_TYPE, f"MS{ms_level}")
        item.setText(_COL_RT, f"{rt:.4f}")
        item.setText(_COL_POLARITY, polarity)
        item.setText(_COL_FILTER, filt)

        if ms_level == 2:
            prec = spec.get("precursor_mz")
            if prec is not None:
                item.setText(_COL_PREC_MZ, f"{float(prec):.4f}")
            prec_int = spec.get("precursor_intensity")
            if prec_int is not None:
                try:
                    item.setText(_COL_PREC_INT, f"{float(prec_int):.2e}")
                except (ValueError, TypeError):
                    pass

        # Store spectrum data on the item
        item.setData(_COL_SCAN_ID, _ROLE_SPEC_DATA, spec)
        item.setData(_COL_SCAN_ID, _ROLE_MS_LEVEL, ms_level)

        # Subtle colour hint for MS2
        if ms_level == 2:
            for col in range(len(_TREE_HEADERS)):
                item.setForeground(col, QColor("#5f6368"))

        return item

    # ── Item click → pin / unpin ──────────────────────────────────────────────

    def _on_item_clicked(self, item: QTreeWidgetItem, col: int):
        spec = item.data(_COL_SCAN_ID, _ROLE_SPEC_DATA)
        if spec is None:
            return  # root / header item

        ctrl = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)
        iid = id(item)

        if ctrl:
            # Ctrl+click: toggle this item while keeping others
            if iid in self._pinned_items:
                self._pinned = deque((p for p in self._pinned if p[0] is not item), maxlen=self.MAX_SELECTED)
                self._pinned_items.discard(iid)
                self._mark_item(item, selected=False)
            else:
                if len(self._pinned) >= self.MAX_SELECTED:
                    evicted_item, _, _ = self._pinned.popleft()
                    self._pinned_items.discard(id(evicted_item))
                    self._mark_item(evicted_item, selected=False)
                ms_level = item.data(_COL_SCAN_ID, _ROLE_MS_LEVEL) or 1
                self._pinned.append((item, spec, ms_level))
                self._pinned_items.add(iid)
                self._mark_item(item, selected=True)
        else:
            # Plain click: clear all and select only this item
            for old_item, _, _ in list(self._pinned):
                self._mark_item(old_item, selected=False)
            self._pinned.clear()
            self._pinned_items.clear()

            ms_level = item.data(_COL_SCAN_ID, _ROLE_MS_LEVEL) or 1
            self._pinned.append((item, spec, ms_level))
            self._pinned_items.add(iid)
            self._mark_item(item, selected=True)

        self._refresh_panels()

    @staticmethod
    def _mark_item(item: QTreeWidgetItem, selected: bool):
        """Visually distinguish pinned items."""
        font = item.font(_COL_SCAN_ID)
        font.setBold(selected)
        for col in range(len(_TREE_HEADERS)):
            item.setFont(col, font)
            if selected:
                item.setBackground(col, QColor("#e8f0fe"))
            else:
                item.setBackground(col, QColor(0, 0, 0, 0))  # transparent

    # ── Panel refresh ─────────────────────────────────────────────────────────

    def _refresh_panels(self):
        """Update only chart panels whose content has changed, preserving zoom/annotations."""
        pinned_list = list(self._pinned)
        for idx, panel in enumerate(self._panels):
            if idx < len(pinned_list):
                item, spec, ms_level = pinned_list[idx]
                if panel._current_spec is spec:
                    continue  # same scan already shown — preserve zoom/pan/annotations
                scan_id = spec.get("scan_id", "?")
                rt = spec.get("scan_time", 0.0)
                polarity = spec.get("polarity", "")
                title = f"MS{ms_level}  |  Scan: {scan_id}  |  RT: {rt:.4f} min  |  {polarity}"
                if ms_level == 2:
                    prec = spec.get("precursor_mz")
                    if prec is not None:
                        title += f"  |  Precursor: {float(prec):.4f}"
                open_fn = (lambda s=spec, fn=self._filename: self._open_msms_popup(s, fn)) if ms_level == 2 else None
                panel.set_spectrum(
                    title,
                    spec.get("mz", []),
                    spec.get("intensity", []),
                    ms_level,
                    spec=spec,
                    open_msms_fn=open_fn,
                )
            else:
                if panel._current_spec is None:
                    continue  # already empty
                panel.clear()

    def _open_msms_popup(self, spec: dict, filename: str):
        """Open a MSMSPopupWindow for the given MS2 spectrum."""
        adapted = dict(spec)
        adapted.setdefault("rt", spec.get("scan_time", 0.0))
        win = MSMSPopupWindow(
            spectrum_data=adapted,
            filename=filename,
            group="File Explorer",
            parent=None,
            filepath=self._filepath,
            file_manager=self._file_manager,
        )
        win.show()
        if not hasattr(self, "_popup_windows"):
            self._popup_windows = []
        self._popup_windows.append(win)
        win.destroyed.connect(lambda _, w=win: self._popup_windows.remove(w) if hasattr(self, "_popup_windows") and w in self._popup_windows else None)
