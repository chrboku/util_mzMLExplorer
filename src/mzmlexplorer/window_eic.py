"""
EIC (Extracted Ion Chromatogram) window and supporting classes:
InteractiveChartView, EICExtractionWorker, EICWindow,
EmbeddedScatterPlotView, Interactive2DScatterChartView.
"""

import os
import re
import traceback
import pandas as pd
import time
import numpy as np
from typing import Optional
from natsort import natsorted, natsort_keygen
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import seaborn as sns
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QComboBox,
    QMenu,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QTabWidget,
    QApplication,
    QSizePolicy,
    QSlider,
    QDialog,
    QProgressDialog,
    QScrollArea,
    QTextEdit,
    QStyle,
    QLineEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QMargins
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPen, QColor, QPainter, QMouseEvent, QAction, QBrush
from .window_shared import BarDelegate, CenteredBarDelegate, CollapsibleBox
from .window_shared import NumericTableWidgetItem
from .window_shared import NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollComboBox
from .window_ms1 import MS1ViewerWindow
from .window_msms import MSMSViewerWindow
from .utils import (
    format_mz,
    format_retention_time,
    adduct_mass_change,
    calculate_molecular_mass,
    calculate_mz_from_formula,
    parse_molecular_formula,
)
from .compound_manager import CompoundManager


class InteractiveChartView(QChartView):
    """Custom chart view with interactive mouse controls"""

    # Signal emitted when right-clicking for context menu (rt_value, mouse_position)
    contextMenuRequested = pyqtSignal(float, QPointF)

    def __init__(self, chart):
        super().__init__(chart)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Mouse interaction state
        self.is_panning = False
        self.is_zooming = False
        self.last_mouse_pos = QPointF()
        self.pan_start_pos = QPointF()
        self.zoom_start_pos = QPointF()

        # Store chart ranges for interactions
        self.interaction_start_x_range = None
        self.interaction_start_y_range = None

        # Zoom anchor point (where the mouse was clicked)
        self.zoom_anchor_x = 0.0
        self.zoom_anchor_y = 0.0

        # Context menu timing variables
        self.mouse_press_time = 0
        self.mouse_press_pos = None
        self.right_click_pending = False
        self.drag_threshold = 5  # pixels
        self.click_timeout = 0.5  # seconds

        # Hover detection variables
        self.hover_threshold = 0.05  # Normalized distance threshold (0.0 to 1.0)
        self.current_hovered_series = None
        self.series_data_cache = {}  # Cache for series data points

        # Create persistent hover label
        self.hover_label = QLabel(self)
        self.hover_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 200);
                border: 1px solid #333;
                border-radius: 3px;
                padding: 3px 6px;
                font-weight: bold;
            }
        """)
        self.hover_label.hide()
        self.hover_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)  # Don't interfere with mouse events

        # Disable default rubber band
        self.setRubberBand(QChartView.RubberBand.NoRubberBand)

        # Enable mouse tracking for smooth interactions
        self.setMouseTracking(True)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Left click: start panning
            self.is_panning = True
            self.pan_start_pos = event.position()
            self.last_mouse_pos = event.position()

            # Store current ranges
            x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
            y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]
            self.interaction_start_x_range = (x_axis.min(), x_axis.max())
            self.interaction_start_y_range = (y_axis.min(), y_axis.max())

            self.setCursor(Qt.CursorShape.ClosedHandCursor)

        elif event.button() == Qt.MouseButton.RightButton:
            # Check if we're over the plot area for potential context menu
            plot_area = self.chart().plotArea()
            if plot_area.contains(event.position()):
                # Start tracking for context menu
                self.mouse_press_time = time.time()
                self.mouse_press_pos = event.position().toPoint()
                self.right_click_pending = True
                return  # Don't start zooming yet

            # If not over plot area, start zooming
            self.is_zooming = True
            self.zoom_start_pos = event.position()
            self.last_mouse_pos = event.position()

            # Store current ranges
            x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
            y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]
            self.interaction_start_x_range = (x_axis.min(), x_axis.max())
            self.interaction_start_y_range = (y_axis.min(), y_axis.max())

            # Convert mouse position to data coordinates for zoom anchor
            plot_area = self.chart().plotArea()
            # Calculate relative position within plot area (0.0 to 1.0)
            rel_x = (event.position().x() - plot_area.left()) / plot_area.width()
            rel_y = (event.position().y() - plot_area.top()) / plot_area.height()

            # Clamp to plot area bounds
            rel_x = max(0.0, min(1.0, rel_x))
            rel_y = max(0.0, min(1.0, rel_y))

            # Convert to data coordinates
            x_range = self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
            y_range = self.interaction_start_y_range[1] - self.interaction_start_y_range[0]

            self.zoom_anchor_x = self.interaction_start_x_range[0] + rel_x * x_range
            self.zoom_anchor_y = self.interaction_start_y_range[1] - rel_y * y_range

            self.setCursor(Qt.CursorShape.SizeAllCursor)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events"""
        # Check if we're tracking a potential right-click for context menu
        if self.right_click_pending and self.mouse_press_pos is not None and event.buttons() & Qt.MouseButton.RightButton:
            current_pos = event.position().toPoint()
            distance = ((current_pos.x() - self.mouse_press_pos.x()) ** 2 + (current_pos.y() - self.mouse_press_pos.y()) ** 2) ** 0.5

            # If movement exceeds threshold, it's a drag - cancel context menu and start zooming
            if distance > self.drag_threshold:
                self.right_click_pending = False
                self.mouse_press_pos = None

                # Start zooming interaction
                self.is_zooming = True
                self.zoom_start_pos = event.position()

                # Store current ranges
                x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
                y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]
                self.interaction_start_x_range = (x_axis.min(), x_axis.max())
                self.interaction_start_y_range = (y_axis.min(), y_axis.max())

                # Set zoom anchor
                plot_area = self.chart().plotArea()
                rel_x = (event.position().x() - plot_area.left()) / plot_area.width()
                rel_y = (event.position().y() - plot_area.top()) / plot_area.height()
                rel_x = max(0.0, min(1.0, rel_x))
                rel_y = max(0.0, min(1.0, rel_y))

                x_range = self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
                y_range = self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
                self.zoom_anchor_x = self.interaction_start_x_range[0] + rel_x * x_range
                self.zoom_anchor_y = self.interaction_start_y_range[1] - rel_y * y_range

        if self.is_panning:
            self._handle_panning(event)
        elif self.is_zooming:
            self._handle_zooming(event)

        self.last_mouse_pos = event.position()

        # Handle hover detection for tooltips (only when not interacting)
        if not self.is_panning and not self.is_zooming and not self.right_click_pending:
            self._handle_hover(event)

        super().mouseMoveEvent(event)

    def _handle_hover(self, event: QMouseEvent):
        """Handle hover detection for showing sample names"""
        plot_area = self.chart().plotArea()
        if not plot_area.contains(event.position()):
            # Mouse outside plot area, hide any tooltip
            if self.current_hovered_series:
                self.hover_label.hide()
                self.current_hovered_series = None
            return

        # Convert mouse position to chart coordinates
        chart_point = self.chart().mapToValue(event.position())
        mouse_x = chart_point.x()
        mouse_y = chart_point.y()

        closest_series = None
        closest_distance = float("inf")
        closest_sample_name = None

        # Check all series for proximity
        for series in self.chart().series():
            if isinstance(series, QLineSeries):
                # Get series data from cache or create it
                series_id = id(series)
                if series_id not in self.series_data_cache:
                    self._cache_series_data(series)

                series_data = self.series_data_cache.get(series_id, {})
                points = series_data.get("points", [])
                sample_name = series_data.get("name", "Unknown")

                # Find closest point on this series
                min_distance = self._find_closest_distance_to_series(mouse_x, mouse_y, points)

                if min_distance < closest_distance:
                    closest_distance = min_distance
                    closest_series = series
                    closest_sample_name = sample_name

        # Show tooltip if close enough to a series
        if closest_distance <= self.hover_threshold:
            if closest_series != self.current_hovered_series:
                self.current_hovered_series = closest_series

                # Get the series color for the tooltip
                series_color = closest_series.pen().color()
                color_hex = series_color.name()

                # Show persistent label with sample name
                self.hover_label.setText(closest_sample_name)
                self.hover_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: rgba(255, 255, 255, 200);
                        border: 1px solid {color_hex};
                        border-radius: 3px;
                        padding: 3px 6px;
                        font-weight: bold;
                        color: {color_hex};
                    }}
                """)

                # Position the label near the mouse cursor
                label_pos = event.position().toPoint()
                label_pos.setX(label_pos.x() + 10)  # Offset to avoid covering the cursor
                label_pos.setY(label_pos.y() - 10)

                # Ensure label stays within widget bounds
                label_size = self.hover_label.sizeHint()
                max_x = self.width() - label_size.width()
                max_y = self.height() - label_size.height()

                if label_pos.x() > max_x:
                    label_pos.setX(max_x)
                if label_pos.y() < 0:
                    label_pos.setY(label_pos.y() + 30)

                self.hover_label.move(label_pos)
                self.hover_label.show()
        else:
            # Too far from any series, hide label
            if self.current_hovered_series:
                self.hover_label.hide()
                self.current_hovered_series = None

    def _cache_series_data(self, series: QLineSeries):
        """Cache series data points and metadata for hover detection"""
        series_id = id(series)
        points = []

        # Extract all points from the series
        for i in range(series.count()):
            point = series.at(i)
            points.append((point.x(), point.y()))

        # Get sample name from stored property or fall back to series name
        sample_name = series.property("sample_filename")
        if not sample_name:
            # Fallback to series name if property not set
            series_name = series.name()
            sample_name = series_name if series_name else "Unknown Sample"

        # Store in cache
        self.series_data_cache[series_id] = {"points": points, "name": sample_name}

    def _find_closest_distance_to_series(self, mouse_x, mouse_y, points):
        """Find the closest distance from mouse to the series line"""
        if len(points) < 2:
            return float("inf")

        # Get axes to normalize distances
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0] if self.chart().axes(Qt.Orientation.Horizontal) else None
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0] if self.chart().axes(Qt.Orientation.Vertical) else None

        if not x_axis or not y_axis:
            return float("inf")

        # Get axis ranges for normalization
        x_range = x_axis.max() - x_axis.min()
        y_range = y_axis.max() - y_axis.min()

        # Avoid division by zero
        if x_range <= 0 or y_range <= 0:
            return float("inf")

        # Normalize mouse coordinates
        norm_mouse_x = (mouse_x - x_axis.min()) / x_range
        norm_mouse_y = (mouse_y - y_axis.min()) / y_range

        min_distance = float("inf")

        # Check distance to each line segment
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            # Normalize line segment coordinates
            norm_x1 = (x1 - x_axis.min()) / x_range
            norm_y1 = (y1 - y_axis.min()) / y_range
            norm_x2 = (x2 - x_axis.min()) / x_range
            norm_y2 = (y2 - y_axis.min()) / y_range

            # Calculate distance from point to line segment
            distance = self._point_to_line_segment_distance(norm_mouse_x, norm_mouse_y, norm_x1, norm_y1, norm_x2, norm_y2)
            min_distance = min(min_distance, distance)

        return min_distance

    def _point_to_line_segment_distance(self, px, py, x1, y1, x2, y2):
        """Calculate the distance from a point to a line segment"""
        # Vector from line start to line end
        line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2

        if line_length_sq == 0:
            # Line segment is a point
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        # Calculate projection parameter t
        # t = dot((P - A), (B - A)) / |B - A|^2
        t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq

        # Clamp t to [0, 1] to stay within the line segment
        t = max(0, min(1, t))

        # Find the closest point on the line segment
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)

        # Return distance from point to closest point on line segment
        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5

    def update_series_cache(self):
        """Update the series data cache when plot data changes"""
        self.series_data_cache.clear()
        for series in self.chart().series():
            if isinstance(series, QLineSeries):
                self._cache_series_data(series)

    def leaveEvent(self, event):
        """Handle mouse leave events to hide tooltips"""
        if self.current_hovered_series:
            self.hover_label.hide()
            self.current_hovered_series = None
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
        elif event.button() == Qt.MouseButton.RightButton:
            # Check if this was a pending right-click for context menu
            if self.right_click_pending and self.mouse_press_pos is not None:
                current_time = time.time()
                time_diff = current_time - self.mouse_press_time

                release_pos = event.position().toPoint()
                distance = ((release_pos.x() - self.mouse_press_pos.x()) ** 2 + (release_pos.y() - self.mouse_press_pos.y()) ** 2) ** 0.5

                # Show context menu only if:
                # 1. Time between press and release is less than timeout
                # 2. Mouse didn't move more than threshold (not a drag)
                if time_diff <= self.click_timeout and distance <= self.drag_threshold:
                    plot_area = self.chart().plotArea()
                    if plot_area.contains(event.position()):
                        # Convert mouse position to data coordinates
                        rel_x = (event.position().x() - plot_area.left()) / plot_area.width()

                        # Get X-axis range and calculate RT value
                        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
                        x_range = x_axis.max() - x_axis.min()
                        rt_value = x_axis.min() + rel_x * x_range

                        # Emit signal for context menu
                        self.contextMenuRequested.emit(rt_value, event.position())

                # Reset context menu tracking
                self.right_click_pending = False
                self.mouse_press_pos = None
                self.mouse_press_time = 0
            else:
                self.is_zooming = False

        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def _handle_panning(self, event: QMouseEvent):
        """Handle panning interaction"""
        if not self.interaction_start_x_range or not self.interaction_start_y_range:
            return

        # Calculate movement delta
        delta_x = event.position().x() - self.pan_start_pos.x()
        delta_y = event.position().y() - self.pan_start_pos.y()

        # Get chart plot area
        plot_area = self.chart().plotArea()

        # Convert pixel movement to data coordinates
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]

        # Calculate data range per pixel
        x_range = self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
        y_range = self.interaction_start_y_range[1] - self.interaction_start_y_range[0]

        x_per_pixel = x_range / plot_area.width()
        y_per_pixel = y_range / plot_area.height()

        # Calculate new ranges (negative because we want to move the data, not the view)
        x_offset = -delta_x * x_per_pixel
        y_offset = delta_y * y_per_pixel  # Positive because Y axis is inverted in screen coordinates

        new_x_min = self.interaction_start_x_range[0] + x_offset
        new_x_max = self.interaction_start_x_range[1] + x_offset
        new_y_min = self.interaction_start_y_range[0] + y_offset
        new_y_max = self.interaction_start_y_range[1] + y_offset

        # Apply new ranges
        x_axis.setRange(new_x_min, new_x_max)
        y_axis.setRange(new_y_min, new_y_max)  # Allow negative Y values during panning

    def _handle_zooming(self, event: QMouseEvent):
        """Handle zooming interaction"""
        if not self.interaction_start_x_range or not self.interaction_start_y_range:
            return

        # Calculate movement delta
        delta_x = event.position().x() - self.zoom_start_pos.x()
        delta_y = event.position().y() - self.zoom_start_pos.y()

        # Calculate zoom factors (more sensitive)
        # Moving right/up = zoom in (smaller range), moving left/down = zoom out (larger range)
        zoom_sensitivity = 0.005  # Adjust this to change zoom sensitivity

        x_zoom_factor = 1.0 - (delta_x * zoom_sensitivity)
        y_zoom_factor = 1.0 + (delta_y * zoom_sensitivity)  # Inverted for intuitive behavior

        # Clamp zoom factors to reasonable limits
        x_zoom_factor = max(0.1, min(10.0, x_zoom_factor))
        y_zoom_factor = max(0.1, min(10.0, y_zoom_factor))

        # Calculate new ranges anchored at the mouse click position
        original_x_range = self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
        original_y_range = self.interaction_start_y_range[1] - self.interaction_start_y_range[0]

        new_x_range = original_x_range * x_zoom_factor
        new_y_range = original_y_range * y_zoom_factor

        # Calculate distances from anchor to original boundaries
        anchor_to_left = self.zoom_anchor_x - self.interaction_start_x_range[0]
        anchor_to_right = self.interaction_start_x_range[1] - self.zoom_anchor_x
        anchor_to_bottom = self.zoom_anchor_y - self.interaction_start_y_range[0]
        anchor_to_top = self.interaction_start_y_range[1] - self.zoom_anchor_y

        # Calculate scale factors for each direction
        left_scale = anchor_to_left / original_x_range
        right_scale = anchor_to_right / original_x_range
        bottom_scale = anchor_to_bottom / original_y_range
        top_scale = anchor_to_top / original_y_range

        # Calculate new boundaries maintaining anchor position
        new_x_min = self.zoom_anchor_x - (new_x_range * left_scale)
        new_x_max = self.zoom_anchor_x + (new_x_range * right_scale)
        new_y_min = self.zoom_anchor_y - (new_y_range * bottom_scale)
        new_y_max = self.zoom_anchor_y + (new_y_range * top_scale)

        # Apply new ranges
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]

        x_axis.setRange(new_x_min, new_x_max)
        y_axis.setRange(new_y_min, new_y_max)  # Allow negative values during zooming


class EICExtractionWorker(QThread):
    """Worker thread for extracting EIC data"""

    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        file_manager,
        target_mz,
        mz_tolerance,
        rt_start,
        rt_end,
        eic_method="Sum of all signals",
        adduct=None,
        polarity=None,
    ):
        super().__init__()
        self.file_manager = file_manager
        self.target_mz = target_mz
        self.mz_tolerance = mz_tolerance
        self.rt_start = rt_start
        self.rt_end = rt_end
        self.eic_method = eic_method
        self.adduct = adduct
        self.polarity = self._normalize_polarity(polarity)

    def _normalize_polarity(self, polarity):
        """Convert different polarity representations to '+', '-', or None."""
        if polarity is None:
            return None

        if isinstance(polarity, str):
            polarity = polarity.strip().lower()
            if polarity in {"+", "positive", "pos", "pos."}:
                return "+"
            if polarity in {"-", "negative", "neg", "neg."}:
                return "-"

        return None

    def run(self):
        try:
            files_data = self.file_manager.get_files_data()
            total_files = len(files_data)
            eic_data = {}

            for i, (_, row) in enumerate(files_data.iterrows()):
                filepath = row["Filepath"]

                # Extract EIC with polarity consideration
                rt, intensity = self.file_manager.extract_eic(
                    filepath,
                    self.target_mz,
                    self.mz_tolerance,
                    self.rt_start,
                    self.rt_end,
                    self.eic_method,
                    self.polarity,
                )

                eic_data[filepath] = {
                    "rt": rt,
                    "intensity": intensity,
                    "metadata": row.to_dict(),
                }

                # Update progress
                progress_value = int((i + 1) / total_files * 100)
                self.progress.emit(progress_value)

            self.finished.emit(eic_data)

        except Exception as e:
            self.error.emit(str(e))


class EICWindow(QWidget):
    """Window for displaying extracted ion chromatograms"""

    def __init__(
        self,
        compound_data: dict,
        adduct: str,
        file_manager,
        mz_value=None,
        polarity=None,
        defaults=None,
        parent=None,
        integration_callback=None,
        settings_callback=None,
        adducts_data=None,
    ):
        super().__init__(parent)

        # Configure as independent window
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.compound_data = compound_data
        self.adduct = adduct
        self.file_manager = file_manager
        self._adducts_data = adducts_data  # optional adducts DataFrame for annotation
        self._msms_windows = []  # keep references so windows are not garbage-collected
        self.eic_data = {}
        self.group_shifts = {}
        self.file_shifts = {}
        self.integration_callback = integration_callback
        self.settings_callback = settings_callback
        self.grouping_column = "group"  # Default grouping column

        # Store defaults (use application defaults if none provided)
        self.defaults = (
            defaults
            if defaults is not None
            else {
                "mz_tolerance_ppm": 5.0,
                "separation_mode": "By group",
                "rt_shift_min": 1.0,
                "crop_rt_window": False,
                "normalize_samples": False,
            }
        )

        # Use pre-calculated m/z value if provided, otherwise calculate
        if mz_value is not None:
            self.target_mz = mz_value
            # Normalise "positive"/"negative" → "+"/"-" so comparisons are consistent
            if isinstance(polarity, str):
                _p = polarity.strip().lower()
                if _p in {"+", "positive", "pos", "pos."}:
                    polarity = "+"
                elif _p in {"-", "negative", "neg", "neg."}:
                    polarity = "-"
                else:
                    polarity = None
            self.polarity = polarity
        else:
            # Fallback: Calculate target m/z using compound manager
            try:
                # Use the compound manager to calculate m/z properly
                temp_manager = CompoundManager()

                # Create a temporary DataFrame with just this compound
                temp_compound_data = pd.DataFrame([compound_data])
                temp_manager.compounds_data = temp_compound_data

                self.target_mz = temp_manager.calculate_compound_mz(compound_data["Name"], adduct)

                if self.target_mz is None:
                    raise ValueError("Could not calculate m/z value")

                self.polarity = None  # Polarity not available in fallback mode

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to calculate m/z: {str(e)}")
                self.target_mz = 0.0
                self.polarity = None

        # Initialize scatter plot attributes
        self.scatter_plot_view = None
        self.scatter_separator_label = None
        self.scatter_plot_menu_text = "View 2D scatter plot (RT vs m/z)"
        self._syncing_scatter_x_axis = False

        # Extra EIC trace state (additional subplots below the main EIC)
        # Each entry: {"label": str, "mz": float, "ppm": float, "polarity": str,
        #              "eic_data": dict, "chart": QChart, "chart_view": InteractiveChartView,
        #              "x_axis": QValueAxis, "y_axis": QValueAxis, "color": str}
        self._extra_eic_traces = []
        self._syncing_x_axes = False  # Guard against recursive axis-sync loops
        self._equalizing_y_widths = False  # Guard against recursive width equalization

        # Group name annotation items drawn on the chart scene (cleared/rebuilt on each update_plot)
        self._group_annotations = []  # QGraphicsSimpleTextItem references
        self._group_annotation_data = []  # (text_item, x_rt_data) for live repositioning

        # Initialize peak boundary line attributes
        self.peak_boundary_lines = []  # List of QLineSeries for boundary lines
        self.peak_start_rt = None  # Start RT of peak boundary
        self.peak_end_rt = None  # End RT of peak boundary
        self.dragging_line = None  # Reference to line being dragged
        self.drag_offset = 0.0  # Offset for smooth dragging

        # Initialize boxplot widget
        self.boxplot_widget = None
        self.boxplot_canvas = None

        # Initialize group settings for EIC plotting
        self.group_settings = {}  # Will be populated when EIC data is loaded
        self._group_update_in_progress = False  # Guard flag to avoid re-entrant redraws
        self._rt_unit = "min"  # "min" or "s"

        # Initialize sample settings for EIC plotting (persisted across windows via defaults)
        self.sample_settings = dict(defaults.get("sample_settings", {})) if defaults else {}

        self.init_ui()
        self._load_stylesheet()
        self.extract_eic_data()

    def _load_stylesheet(self):
        """Apply the shared application stylesheet so table styles are consistent."""
        stylesheet_path = os.path.join(os.path.dirname(__file__), "style.css")
        if os.path.exists(stylesheet_path):
            with open(stylesheet_path, "r") as f:
                self.setStyleSheet(f.read())

    def _lookup_adduct_info(self):
        """Return the adduct row as a dict for the current adduct, or None.

        Looks up from *self._adducts_data* when available; otherwise returns
        None so the fragment annotator falls back to a proton offset.
        """
        if self._adducts_data is None or self._adducts_data.empty:
            return None
        row = self._adducts_data[self._adducts_data["Adduct"] == self.adduct]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def init_ui(self):
        """Initialize the user interface"""
        _pol_suffix = f" [{self.polarity}]" if self.polarity else ""
        _mz_suffix = f"  m/z {format_mz(self.target_mz)}{_pol_suffix}" if self.target_mz else ""
        self.setWindowTitle(f"EIC: {self.compound_data['Name']} - {self.adduct}{_mz_suffix}")
        self.setGeometry(200, 200, 1400, 800)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Toggle button – always visible on the left edge
        self.panel_toggle_btn = QPushButton()
        self.panel_toggle_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView))
        self.panel_toggle_btn.setToolTip("Show / Hide Properties Panel")
        self.panel_toggle_btn.setFixedWidth(32)
        self.panel_toggle_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.panel_toggle_btn.clicked.connect(self._toggle_left_panel)
        layout.addWidget(self.panel_toggle_btn)

        # Left panel wrapped in a vertically-scrollable area (hidden by default)
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.left_scroll.setFixedWidth(400)
        left_panel = self.create_left_panel()
        self.left_scroll.setWidget(left_panel)
        self.left_scroll.setVisible(False)  # Hidden by default
        layout.addWidget(self.left_scroll)

        # Right panel for chart – expands to fill all remaining space
        right_panel = self.create_right_panel()
        layout.addWidget(right_panel, stretch=1)

    def _toggle_left_panel(self) -> None:
        """Show or hide the left properties panel."""
        self.left_scroll.setVisible(not self.left_scroll.isVisible())

    def create_left_panel(self) -> QWidget:
        """Create the left panel with compound info and controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Compound information
        info_group = self.create_compound_info_group()
        layout.addWidget(info_group)

        # Extraction parameters (contains Extract EIC button at bottom)
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)

        # Group settings table
        self.create_group_settings_table()
        layout.addWidget(self.group_settings_box)

        # Sample settings table (collapsible, placed below group settings)
        self.create_sample_settings_table()
        layout.addWidget(self.sample_settings_box)

        # Settings template button
        template_btn = QPushButton("💾  Save current settings as template")
        template_btn.setToolTip("Save current EIC extraction, group, and sample settings as a reusable template")
        template_btn.clicked.connect(self._save_settings_template)
        layout.addWidget(template_btn)

        layout.addStretch()
        return panel

    def create_right_panel(self) -> QWidget:
        """Create the right panel with the chart and boxplot"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create a vertical splitter for EIC chart and boxplot
        self.eic_boxplot_splitter = QSplitter(Qt.Orientation.Vertical)
        self.eic_boxplot_splitter.setChildrenCollapsible(False)  # Prevent complete collapse

        # Inner vertical splitter that holds the main EIC chart + any extra trace charts
        self._eic_charts_splitter = QSplitter(Qt.Orientation.Vertical)
        self._eic_charts_splitter.setChildrenCollapsible(False)

        # Main chart (EIC)
        self.chart_view = self.create_chart()
        self._eic_charts_splitter.addWidget(self.chart_view)

        self.eic_boxplot_splitter.addWidget(self._eic_charts_splitter)

        # Boxplot widget
        self.create_boxplot_widget()
        self.eic_boxplot_splitter.addWidget(self.boxplot_widget)

        # Set initial sizes (75% EIC, 25% boxplot)
        self.eic_boxplot_splitter.setSizes([750, 250])
        self.eic_boxplot_splitter.setStretchFactor(0, 1)  # EIC chart is stretchable
        self.eic_boxplot_splitter.setStretchFactor(1, 1)  # Boxplot adapts with available space

        # Add the splitter to the main layout
        layout.addWidget(self.eic_boxplot_splitter)

        return panel

    def create_compound_info_group(self) -> QGroupBox:
        """Create the compound information group"""
        group = QGroupBox("Compound Information")
        layout = QVBoxLayout(group)

        # Determine compound info display
        formula_info = ""
        if "ChemicalFormula" in self.compound_data and self.compound_data["ChemicalFormula"]:
            formula_info = f"<b>Formula:</b> {self.compound_data['ChemicalFormula']}<br>"
        elif "Mass" in self.compound_data and self.compound_data["Mass"]:
            formula_info = f"<b>Mass:</b> {self.compound_data['Mass']} Da<br>"

        # Compound info
        _pol_str = f"<br><b>Polarity:</b> {self.polarity}" if self.polarity else ""
        compound_info = QLabel(f"<b>Compound:</b> {self.compound_data['Name']}<br>{formula_info}<b>Adduct:</b> {self.adduct}<br><b>m/z:</b> {format_mz(self.target_mz)}{_pol_str}")
        layout.addWidget(compound_info)

        # RT info — only shown when the compound data contains RT fields
        rt_min = self.compound_data.get("RT_min")
        rt_start = self.compound_data.get("RT_start_min")
        rt_end = self.compound_data.get("RT_end_min")
        if rt_min is not None:
            rt_text = f"<b>RT:</b> {format_retention_time(rt_min)}"
            if rt_start is not None and rt_end is not None:
                rt_text += f"<br><b>RT Window:</b> {format_retention_time(rt_start)} - {format_retention_time(rt_end)}"
            rt_info = QLabel(rt_text)
            layout.addWidget(rt_info)

        # SMILES structure display
        smiles = self.compound_data.get("SMILES") or self.compound_data.get("smiles")
        if smiles and pd.notna(smiles) and str(smiles).strip():
            smiles = str(smiles).strip()
            struct_widget = self._create_structure_widget(smiles)
            if struct_widget is not None:
                layout.addWidget(struct_widget)

        return group

    def _create_structure_widget(self, smiles: str):
        """Render a SMILES string as a widget showing the SMILES text and, if rdkit
        is available, a 2-D structure image below it.
        Clicking on the SMILES text or the image opens a high-resolution popup.
        """
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)

        # SMILES text – clickable
        smiles_label = QLabel(f"<b>SMILES:</b><br><small>{smiles}</small>")
        smiles_label.setWordWrap(True)
        smiles_label.setToolTip("Click to view full details")
        smiles_label.setCursor(Qt.CursorShape.PointingHandCursor)
        smiles_label.mousePressEvent = lambda _ev, s=smiles: self._show_structure_detail_dialog(s)
        container_layout.addWidget(smiles_label)

        # Try to render the 2-D structure with rdkit
        try:
            from rdkit import Chem
            from rdkit.Chem.Draw import rdMolDraw2D
            from PyQt6.QtGui import QPixmap

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")

            drawer = rdMolDraw2D.MolDraw2DCairo(220, 160)
            drawer.drawOptions().clearBackground = False
            drawer.drawOptions().addStereoAnnotation = True
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            png_bytes = drawer.GetDrawingText()

            pixmap = QPixmap()
            pixmap.loadFromData(png_bytes, "PNG")

            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setToolTip("Click to view high-resolution structure")
            img_label.setCursor(Qt.CursorShape.PointingHandCursor)
            img_label.mousePressEvent = lambda _ev, s=smiles: self._show_structure_detail_dialog(s)
            container_layout.addWidget(img_label)

        except ImportError:
            pass  # rdkit not installed — SMILES text already added above
        except Exception:
            pass  # Invalid SMILES or rendering error — SMILES text already added above

        return container

    def _show_structure_detail_dialog(self, smiles: str) -> None:
        """Open a modal popup with a high-resolution structure image, SMILES text,
        and computed compound properties."""
        import math

        def _val_ok(v):
            """Return True when v is a non-empty, non-NaN usable scalar."""
            if v is None:
                return False
            try:
                if math.isnan(float(v)):
                    return False
            except (TypeError, ValueError):
                pass
            return str(v).strip().lower() not in ("", "nan", "none", "na")

        # Inner helper: QLabel that rescales its pixmap whenever the label is resized.
        class _ResizableImageLabel(QLabel):
            def __init__(self, orig_pixmap, parent=None):
                super().__init__(parent)
                self._orig = orig_pixmap
                self.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setMinimumSize(300, 300)
                self.setAutoFillBackground(False)

            def resizeEvent(self, event):
                super().resizeEvent(event)
                if self._orig and not self._orig.isNull():
                    self.setPixmap(
                        self._orig.scaled(
                            self.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Structure \u2013 {self.compound_data.get('Name', '')}")
        dialog.setMinimumSize(700, 500)
        dialog.resize(820, 580)

        outer = QHBoxLayout(dialog)
        outer.setContentsMargins(8, 8, 8, 8)

        # ---- Left: high-res structure image (stretches when dialog is resized) ----
        orig_pixmap = None
        try:
            from rdkit import Chem
            from rdkit.Chem.Draw import rdMolDraw2D
            from PyQt6.QtGui import QPixmap

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
                drawer.drawOptions().clearBackground = False  # transparent background
                drawer.drawOptions().addStereoAnnotation = True
                drawer.drawOptions().baseFontSize = 0.8
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                png_bytes = drawer.GetDrawingText()
                orig_pixmap = QPixmap()
                orig_pixmap.loadFromData(png_bytes, "PNG")
        except Exception:
            pass

        img_widget = _ResizableImageLabel(orig_pixmap)
        if orig_pixmap is None:
            img_widget.setText("(structure unavailable)")
        # Image side stretches on resize; right panel stays at fixed width
        outer.addWidget(img_widget, stretch=1)

        # ---- Right: SMILES + properties (fixed width, does not grow on resize) ----
        right_widget = QWidget()
        right_widget.setFixedWidth(290)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)

        # SMILES (selectable)
        smiles_header = QLabel("<b>SMILES:</b>")
        smiles_header.setStyleSheet("font-size: 13px;")
        right_layout.addWidget(smiles_header)
        smiles_text = QTextEdit()
        smiles_text.setPlainText(smiles)
        smiles_text.setReadOnly(True)
        smiles_text.setMaximumHeight(70)
        smiles_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        right_layout.addWidget(smiles_text)

        # Properties
        props_header = QLabel("<b>Properties:</b>")
        props_header.setStyleSheet("font-size: 13px; margin-top: 8px;")
        right_layout.addWidget(props_header)

        props: list[tuple[str, str]] = []
        cd = self.compound_data
        if _val_ok(cd.get("ChemicalFormula")):
            props.append(("Formula", str(cd["ChemicalFormula"])))
        if _val_ok(cd.get("Mass")):
            props.append(("Monoisotopic mass", f"{float(cd['Mass']):.4f} Da"))
        if _val_ok(cd.get("RT_min")):
            props.append(("RT (center)", f"{float(cd['RT_min']):.2f} min"))

        # From rdkit
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                props.append(("Molecular weight", f"{Descriptors.MolWt(mol):.4f}"))
                props.append(("Exact mass", f"{Descriptors.ExactMolWt(mol):.4f}"))
                props.append(("LogP", f"{Descriptors.MolLogP(mol):.3f}"))
                props.append(("H-bond donors", str(rdMolDescriptors.CalcNumHBD(mol))))
                props.append(("H-bond acceptors", str(rdMolDescriptors.CalcNumHBA(mol))))
                props.append(("TPSA", f"{rdMolDescriptors.CalcTPSA(mol):.2f} \u00c5\u00b2"))
                props.append(("Rotatable bonds", str(rdMolDescriptors.CalcNumRotatableBonds(mol))))
                props.append(("Aromatic rings", str(rdMolDescriptors.CalcNumAromaticRings(mol))))
                props.append(("Heavy atoms", str(mol.GetNumHeavyAtoms())))
        except Exception:
            pass

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        props_container = QWidget()
        props_layout = QVBoxLayout(props_container)
        props_layout.setSpacing(2)
        for key, val in props:
            row_label = QLabel(f"<b>{key}:</b>&nbsp;&nbsp;{val}")
            row_label.setStyleSheet("font-size: 12px;")
            props_layout.addWidget(row_label)
        props_layout.addStretch()
        scroll.setWidget(props_container)
        right_layout.addWidget(scroll)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        right_layout.addWidget(close_btn)

        outer.addWidget(right_widget, stretch=0)
        dialog.exec()

    def create_control_panel(self) -> CollapsibleBox:
        """Create the control panel"""
        group = CollapsibleBox("View Settings")
        inner = QWidget()
        layout = QFormLayout(inner)
        group.add_widget(inner)
        group.set_expanded(False)

        # EIC calculation method
        self.eic_method_combo = QComboBox()
        self.eic_method_combo.addItems(["Sum of all signals", "Most intensive signal"])
        default_eic_method = self.defaults.get("eic_method", "Sum of all signals")
        idx = self.eic_method_combo.findText(default_eic_method)
        self.eic_method_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.eic_method_combo.currentTextChanged.connect(self.update_plot)
        self.eic_method_combo.currentTextChanged.connect(lambda v: self._notify_setting("eic_method", v))
        layout.addRow("EIC Method:", self.eic_method_combo)

        # m/z tolerance in ppm (primary)
        self.mz_tolerance_ppm_spin = NoScrollDoubleSpinBox()
        self.mz_tolerance_ppm_spin.setRange(0.1, 10000.0)
        self.mz_tolerance_ppm_spin.setValue(self.defaults["mz_tolerance_ppm"])  # Use default
        self.mz_tolerance_ppm_spin.setSuffix(" ppm")
        self.mz_tolerance_ppm_spin.setDecimals(1)
        self.mz_tolerance_ppm_spin.setSingleStep(1.0)
        self.mz_tolerance_ppm_spin.valueChanged.connect(self.update_mz_tolerance_da)
        self.mz_tolerance_ppm_spin.valueChanged.connect(lambda v: self._notify_setting("mz_tolerance_ppm", v))
        layout.addRow("m/z Tolerance (ppm):", self.mz_tolerance_ppm_spin)

        # m/z tolerance in Da (linked to ppm)
        self.mz_tolerance_da_spin = NoScrollDoubleSpinBox()
        self.mz_tolerance_da_spin.setRange(0.0001, 1.0)
        self.mz_tolerance_da_spin.setSuffix(" Da")
        self.mz_tolerance_da_spin.setDecimals(4)
        self.mz_tolerance_da_spin.setSingleStep(0.005)
        self.mz_tolerance_da_spin.valueChanged.connect(self.update_mz_tolerance_ppm)
        layout.addRow("m/z Tolerance (Da):", self.mz_tolerance_da_spin)

        # Initialize Da value based on default ppm
        self.update_mz_tolerance_da()

        # Grouping column selector
        self.grouping_column_combo = QComboBox()
        self._populate_grouping_columns()
        self.grouping_column_combo.currentTextChanged.connect(self.on_grouping_column_changed)
        layout.addRow("Group by Column:", self.grouping_column_combo)
        # Hide the grouping-column row — it's an internal setting; users should not
        # change it in this panel.
        self.grouping_column_combo.setVisible(False)
        _gcol_label = layout.labelForField(self.grouping_column_combo)
        if _gcol_label:
            _gcol_label.setVisible(False)

        # Separation mode
        self.separation_mode_combo = QComboBox()
        self.separation_mode_combo.addItems(
            [
                "None",
                "By group",
                "By injection order",
                "By group, then injection order",
            ]
        )
        default_mode = self.defaults.get("separation_mode", "By group")
        # Back-compat: translate old boolean defaults if present
        if "separate_groups" in self.defaults and "separation_mode" not in self.defaults:
            default_mode = "By group" if self.defaults["separate_groups"] else "None"
        idx = self.separation_mode_combo.findText(default_mode)
        if idx >= 0:
            self.separation_mode_combo.setCurrentIndex(idx)
        self.separation_mode_combo.currentTextChanged.connect(self.update_plot)
        layout.addRow("Separation:", self.separation_mode_combo)

        # RT shift for group separation (more flexible range)
        self.rt_shift_spin = NoScrollDoubleSpinBox()
        self.rt_shift_spin.setRange(0.0, 60.0)  # Allow up to 60 minutes
        self.rt_shift_spin.setValue(self.defaults["rt_shift_min"])  # Use default
        self.rt_shift_spin.setSuffix(" min")
        self.rt_shift_spin.setDecimals(1)
        self.rt_shift_spin.setEnabled(True)  # Always enabled
        self.rt_shift_spin.valueChanged.connect(self.update_plot)
        self.rt_shift_spin.valueChanged.connect(lambda v: self._notify_setting("rt_shift_min", v))
        layout.addRow("Group RT Shift:", self.rt_shift_spin)

        # RT cropping option
        self.crop_rt_cb = QCheckBox("Crop to RT Window")
        self.crop_rt_cb.setChecked(self.defaults["crop_rt_window"])  # Use default
        self.crop_rt_cb.stateChanged.connect(self.update_plot)
        layout.addRow(self.crop_rt_cb)

        # Normalization option
        self.normalize_cb = QCheckBox("Normalize to Max per Sample")
        self.normalize_cb.setChecked(self.defaults["normalize_samples"])  # Use default
        self.normalize_cb.stateChanged.connect(self.update_plot)
        layout.addRow(self.normalize_cb)

        # Legend position
        self.legend_position_combo = QComboBox()
        self.legend_position_combo.addItems(["Right", "Top", "Off"])
        default_legend = self.defaults.get("legend_position", "Right")
        idx = self.legend_position_combo.findText(default_legend)
        self.legend_position_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.legend_position_combo.currentTextChanged.connect(self.update_plot)
        self.legend_position_combo.currentTextChanged.connect(lambda v: self._notify_setting("legend_position", v))
        layout.addRow("Legend:", self.legend_position_combo)

        # RT unit selector
        self.rt_unit_combo = NoScrollComboBox()
        self.rt_unit_combo.addItems(["min", "s"])
        self.rt_unit_combo.setCurrentText("min")
        self.rt_unit_combo.currentTextChanged.connect(self._on_rt_unit_changed)
        layout.addRow("RT unit:", self.rt_unit_combo)

        # Logarithmic Y-axis option
        self.log_y_cb = QCheckBox("Log\u2081\u2080 Y-axis")
        self.log_y_cb.setChecked(False)
        self.log_y_cb.stateChanged.connect(lambda _: self.update_plot())
        layout.addRow(self.log_y_cb)

        # Ridge plot option
        self.ridge_plot_cb = QCheckBox("Ridge plot")
        self.ridge_plot_cb.setChecked(False)
        self.ridge_plot_cb.stateChanged.connect(self._on_ridge_plot_toggled)
        layout.addRow(self.ridge_plot_cb)

        # Ridge increment slider (hidden until ridge plot is enabled)
        self._ridge_increment_max = 1.0
        self.ridge_increment_widget = QWidget()
        ridge_layout = QHBoxLayout(self.ridge_increment_widget)
        ridge_layout.setContentsMargins(0, 0, 0, 0)
        self.ridge_increment_slider = QSlider(Qt.Orientation.Horizontal)
        self.ridge_increment_slider.setRange(0, 10000)
        self.ridge_increment_slider.setValue(1000)  # 10% default; updated when data loads
        self.ridge_increment_slider.valueChanged.connect(self._on_ridge_slider_changed)
        ridge_layout.addWidget(self.ridge_increment_slider)
        self.ridge_increment_label = QLabel("0.00e+00")
        self.ridge_increment_label.setMinimumWidth(72)
        ridge_layout.addWidget(self.ridge_increment_label)
        self.ridge_increment_widget.setVisible(False)
        layout.addRow("Increment:", self.ridge_increment_widget)

        # Extract EIC button at the bottom of the extraction parameters
        self.extract_btn = QPushButton("Extract EIC")
        self.extract_btn.clicked.connect(self.extract_eic_data)
        layout.addRow(self.extract_btn)

        return group

    def _on_rt_unit_changed(self, unit: str) -> None:
        """Handle RT unit change. Triggers a full redraw with the new unit."""
        self._rt_unit = unit
        self.update_plot()

    @property
    def _rt_factor(self) -> float:
        """Multiplier to convert stored RT values (minutes) to the current display unit."""
        return 60.0 if getattr(self, "_rt_unit", "min") == "s" else 1.0

    @property
    def _rt_label(self) -> str:
        """Unit abbreviation for x-axis titles based on the current RT unit selection."""
        return "s" if getattr(self, "_rt_unit", "min") == "s" else "min"

    def create_group_settings_table(self):
        """Create the group settings table for EIC display controls"""
        # Create a collapsible box for the table
        self.group_settings_box = CollapsibleBox("Group Display Settings")

        # Create the table
        self.group_settings_table = QTableWidget()
        self.group_settings_table.setColumnCount(5)
        self.group_settings_table.setHorizontalHeaderLabels(["Group", "Scaling", "Plot", "Neg.", "Line Width"])

        # Configure table appearance
        self.group_settings_table.setAlternatingRowColors(True)
        self.group_settings_table.setSortingEnabled(True)
        self.group_settings_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.group_settings_table.horizontalHeader().setStretchLastSection(False)
        self.group_settings_table.verticalHeader().setVisible(False)
        self.group_settings_table.setMaximumHeight(800)
        self.group_settings_table.setMinimumHeight(340)

        # Set initial column widths
        self.group_settings_table.setColumnWidth(0, 110)  # Group name column
        self.group_settings_table.setColumnWidth(1, 95)  # Scaling column
        self.group_settings_table.setColumnWidth(2, 30)  # Plot checkbox column
        self.group_settings_table.setColumnWidth(3, 30)  # Negative checkbox column
        self.group_settings_table.setColumnWidth(4, 80)  # Line Width column

        # Enable context menu
        self.group_settings_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.group_settings_table.customContextMenuRequested.connect(self._group_settings_context_menu)

        # Add table to collapsible box
        self.group_settings_box.add_widget(self.group_settings_table)

    def populate_group_settings_table(self):
        """Populate the group settings table with current groups

        Shows all groups from the entire sample matrix, not just those
        with data in the current compound.
        """
        table = self.group_settings_table

        if table is None:
            return

        # Get unique groups from the entire sample matrix (files_data)
        groups = set()

        if hasattr(self.file_manager, "files_data") and self.file_manager.files_data is not None:
            # Get all possible group values from the entire sample matrix
            if self.grouping_column in self.file_manager.files_data.columns:
                group_values = self.file_manager.files_data[self.grouping_column].dropna().unique()
                for value in group_values:
                    groups.add(str(value))

        # Also include groups from current EIC data (in case some aren't in files_data)
        for data in self.eic_data.values():
            if self.grouping_column in data["metadata"]:
                group_value = data["metadata"][self.grouping_column]
                # Convert to string to ensure consistency
                groups.add(str(group_value) if group_value is not None else "Unknown")

        # Sort groups naturally
        sorted_groups = natsorted(groups)

        # Set row count
        table.setRowCount(len(sorted_groups))

        # Initialize group settings if not already done
        for group in sorted_groups:
            if group not in self.group_settings:
                self.group_settings[group] = {
                    "scaling": 1.0,
                    "plot": True,
                    "negative": False,
                    "line_width": 1.0,
                }

        # Populate table rows
        for row, group in enumerate(sorted_groups):
            # Column 0: Group name as a regular cell so setForeground() works
            group_item = QTableWidgetItem(group)
            group_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            group_color = self._get_group_color(group)
            if group_color:
                group_item.setForeground(QColor(group_color))
                _gf = group_item.font()
                _gf.setBold(True)
                group_item.setFont(_gf)
            table.setItem(row, 0, group_item)

            # Column 1: Scaling factor (QDoubleSpinBox)
            scaling_spin = NoScrollDoubleSpinBox()
            scaling_spin.setRange(0.00001, 100000.0)
            scaling_spin.setValue(self.group_settings[group]["scaling"])
            scaling_spin.setDecimals(5)
            scaling_spin.setSingleStep(0.1)
            scaling_spin.valueChanged.connect(lambda value, g=group: self.on_group_setting_changed(g, "scaling", value))
            table.setCellWidget(row, 1, scaling_spin)

            # Column 2: Plot checkbox
            plot_checkbox = QCheckBox()
            plot_checkbox.setChecked(self.group_settings[group]["plot"])
            plot_checkbox.stateChanged.connect(lambda state, g=group: self.on_group_setting_changed(g, "plot", state == Qt.CheckState.Checked.value))
            # Center the checkbox
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(plot_checkbox)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            table.setCellWidget(row, 2, checkbox_widget)

            # Column 3: Negative intensities checkbox
            negative_checkbox = QCheckBox()
            negative_checkbox.setChecked(self.group_settings[group]["negative"])
            negative_checkbox.stateChanged.connect(lambda state, g=group: self.on_group_setting_changed(g, "negative", state == Qt.CheckState.Checked.value))
            # Center the checkbox
            neg_checkbox_widget = QWidget()
            neg_checkbox_layout = QHBoxLayout(neg_checkbox_widget)
            neg_checkbox_layout.addWidget(negative_checkbox)
            neg_checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            neg_checkbox_layout.setContentsMargins(0, 0, 0, 0)
            table.setCellWidget(row, 3, neg_checkbox_widget)

            # Column 4: Line width (QDoubleSpinBox)
            width_spin = NoScrollDoubleSpinBox()
            width_spin.setRange(0.5, 10.0)
            width_spin.setValue(self.group_settings[group]["line_width"])
            width_spin.setDecimals(1)
            width_spin.setSingleStep(0.5)
            width_spin.valueChanged.connect(lambda value, g=group: self.on_group_setting_changed(g, "line_width", value))
            table.setCellWidget(row, 4, width_spin)

    def _group_settings_context_menu(self, pos) -> None:
        """Show context menu on the group settings table."""
        menu = QMenu(self)

        def _set_all_groups(visible: bool):
            table = self.group_settings_table
            self._group_update_in_progress = True
            try:
                for row in range(table.rowCount()):
                    group_item = table.item(row, 0)
                    if group_item is None:
                        continue
                    grp = group_item.text()
                    if grp not in self.group_settings:
                        continue
                    self.group_settings[grp]["plot"] = visible
                    if hasattr(self.file_manager, "files_data") and self.file_manager.files_data is not None:
                        df = self.file_manager.files_data
                        group_col = self.grouping_column if self.grouping_column in df.columns else None
                        if group_col is not None:
                            matching_files = df[df[group_col].astype(str) == grp]
                            for _, frow in matching_files.iterrows():
                                fn = str(frow.get("filename", ""))
                                if fn and fn in self.sample_settings:
                                    self.sample_settings[fn]["plot"] = visible
                    cell_widget = table.cellWidget(row, 2)
                    if cell_widget is not None:
                        cb = cell_widget.findChild(QCheckBox)
                        if cb is not None:
                            cb.blockSignals(True)
                            cb.setChecked(visible)
                            cb.blockSignals(False)
            finally:
                self._group_update_in_progress = False
            self._sync_sample_table_from_settings()
            self._notify_setting("sample_settings", dict(self.sample_settings))
            self.update_plot(preserve_view=True)

        def _reset_all_group_views():
            table = self.group_settings_table
            self._group_update_in_progress = True
            try:
                for row in range(table.rowCount()):
                    group_item = table.item(row, 0)
                    if group_item is None:
                        continue
                    grp = group_item.text()
                    defaults = {"scaling": 1.0, "plot": True, "negative": False, "line_width": 1.0}
                    self.group_settings[grp] = dict(defaults)
                    if hasattr(self.file_manager, "files_data") and self.file_manager.files_data is not None:
                        df = self.file_manager.files_data
                        group_col = self.grouping_column if self.grouping_column in df.columns else None
                        if group_col is not None:
                            matching_files = df[df[group_col].astype(str) == grp]
                            for _, frow in matching_files.iterrows():
                                fn = str(frow.get("filename", ""))
                                if fn:
                                    self.sample_settings[fn] = dict(defaults)
                    scaling_widget = table.cellWidget(row, 1)
                    if scaling_widget is not None and hasattr(scaling_widget, "setValue"):
                        scaling_widget.blockSignals(True)
                        scaling_widget.setValue(1.0)
                        scaling_widget.blockSignals(False)
                    plot_widget = table.cellWidget(row, 2)
                    if plot_widget is not None:
                        cb = plot_widget.findChild(QCheckBox)
                        if cb is not None:
                            cb.blockSignals(True)
                            cb.setChecked(True)
                            cb.blockSignals(False)
                    neg_widget = table.cellWidget(row, 3)
                    if neg_widget is not None:
                        cb = neg_widget.findChild(QCheckBox)
                        if cb is not None:
                            cb.blockSignals(True)
                            cb.setChecked(False)
                            cb.blockSignals(False)
                    width_widget = table.cellWidget(row, 4)
                    if width_widget is not None and hasattr(width_widget, "setValue"):
                        width_widget.blockSignals(True)
                        width_widget.setValue(1.0)
                        width_widget.blockSignals(False)
            finally:
                self._group_update_in_progress = False
            self._sync_sample_table_from_settings()
            self._notify_setting("sample_settings", dict(self.sample_settings))
            self.update_plot(preserve_view=True)

        show_all_action = QAction("Show all", self)
        hide_all_action = QAction("Hide all", self)
        reset_action = QAction("Reset all views", self)

        show_all_action.triggered.connect(lambda: _set_all_groups(True))
        hide_all_action.triggered.connect(lambda: _set_all_groups(False))
        reset_action.triggered.connect(_reset_all_group_views)

        menu.addAction(show_all_action)
        menu.addAction(hide_all_action)
        menu.addSeparator()
        menu.addAction(reset_action)
        menu.exec(self.group_settings_table.viewport().mapToGlobal(pos))

    def _populate_grouping_columns(self):
        """Populate the grouping column selector with available columns"""
        self.grouping_column_combo.clear()

        # Get available columns from files_data
        files_data = self.file_manager.get_files_data()
        if not files_data.empty:
            # Exclude system columns that shouldn't be used for grouping
            exclude_columns = {"Filepath", "filename"}
            available_columns = [col for col in files_data.columns if col not in exclude_columns]

            # Sort columns with 'group' and 'color' first if they exist
            priority_columns = []
            if "group" in available_columns:
                priority_columns.append("group")
                available_columns.remove("group")
            if "color" in available_columns:
                priority_columns.append("color")
                available_columns.remove("color")

            # Combine priority columns with the rest
            all_columns = priority_columns + sorted(available_columns)

            self.grouping_column_combo.addItems(all_columns)

            # Set default to 'group' if available
            if "group" in all_columns:
                self.grouping_column_combo.setCurrentText("group")

    def on_grouping_column_changed(self, column_name):
        """Handle changes to the grouping column selection"""
        if column_name:
            self.grouping_column = column_name
            # Recalculate group shifts and update the plot
            if self.eic_data:
                self.calculate_group_shifts()
                self.calculate_file_shifts()
                self.populate_group_settings_table()
                self.update_plot()

    def on_group_setting_changed(self, group, setting, value):
        """Handle changes to group settings.

        The new value is stored in ``group_settings``, then propagated to every
        matching sample in ``sample_settings`` so that rendering (which uses
        sample_settings exclusively) picks up the change immediately.
        Table widgets are updated with signals blocked to avoid per-row redraws.
        """
        if group not in self.group_settings:
            return
        self.group_settings[group][setting] = value

        # Temporarily block sample-setting change handler to avoid redundant redraws
        self._group_update_in_progress = True
        try:
            # Propagate to sample settings for every file belonging to this group
            if hasattr(self.file_manager, "files_data") and self.file_manager.files_data is not None:
                df = self.file_manager.files_data
                group_col = self.grouping_column if self.grouping_column in df.columns else None
                if group_col is not None:
                    matching_files = df[df[group_col].astype(str) == group]
                    for _, frow in matching_files.iterrows():
                        fn = str(frow.get("filename", ""))
                        if not fn:
                            continue
                        if fn not in self.sample_settings:
                            self.sample_settings[fn] = {"scaling": 1.0, "plot": True, "negative": False, "line_width": 1.0}
                        self.sample_settings[fn][setting] = value

            # Silently update all sample table widgets (signals blocked → no cascade redraws)
            self._sync_sample_table_from_settings()
        finally:
            self._group_update_in_progress = False

        # Persist the updated sample settings
        self._notify_setting("sample_settings", dict(self.sample_settings))

        # Trigger a single redraw now that all control values have been updated
        self.update_plot(preserve_view=True)

    def _sync_sample_table_from_settings(self):
        """Refresh sample table widgets from self.sample_settings without triggering per-row plot updates."""
        table = self.sample_settings_table
        if table is None:
            return
        for row in range(table.rowCount()):
            name_item = table.item(row, 0)
            if name_item is None:
                continue
            fn = name_item.data(Qt.ItemDataRole.UserRole)
            if fn is None or fn not in self.sample_settings:
                continue
            ss = self.sample_settings[fn]
            # Column 1: Scaling spinbox
            scale_widget = table.cellWidget(row, 1)
            if scale_widget is not None and isinstance(scale_widget, QDoubleSpinBox):
                scale_widget.blockSignals(True)
                scale_widget.setValue(ss.get("scaling", 1.0))
                scale_widget.blockSignals(False)
            # Column 2: Plot checkbox
            cb_widget = table.cellWidget(row, 2)
            if cb_widget is not None:
                cb = cb_widget.findChild(QCheckBox)
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setChecked(ss.get("plot", True))
                    cb.blockSignals(False)
            # Column 3: Neg. checkbox
            neg_widget = table.cellWidget(row, 3)
            if neg_widget is not None:
                neg_cb = neg_widget.findChild(QCheckBox)
                if neg_cb is not None:
                    neg_cb.blockSignals(True)
                    neg_cb.setChecked(ss.get("negative", False))
                    neg_cb.blockSignals(False)
            # Column 4: Line width spinbox
            lw_widget = table.cellWidget(row, 4)
            if lw_widget is not None and isinstance(lw_widget, QDoubleSpinBox):
                lw_widget.blockSignals(True)
                lw_widget.setValue(ss.get("line_width", 1.0))
                lw_widget.blockSignals(False)

    def _sync_group_table_from_settings(self):
        """Refresh group table widgets from self.group_settings without triggering per-row plot updates."""
        table = self.group_settings_table
        if table is None:
            return
        for row in range(table.rowCount()):
            name_item = table.item(row, 0)
            if name_item is None:
                continue
            grp = name_item.text()
            if grp not in self.group_settings:
                continue
            gs = self.group_settings[grp]
            # Column 1: Scaling spinbox
            scale_widget = table.cellWidget(row, 1)
            if scale_widget is not None and isinstance(scale_widget, QDoubleSpinBox):
                scale_widget.blockSignals(True)
                scale_widget.setValue(gs.get("scaling", 1.0))
                scale_widget.blockSignals(False)
            # Column 2: Plot checkbox
            cb_widget = table.cellWidget(row, 2)
            if cb_widget is not None:
                cb = cb_widget.findChild(QCheckBox)
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setChecked(gs.get("plot", True))
                    cb.blockSignals(False)
            # Column 3: Neg. checkbox
            neg_widget = table.cellWidget(row, 3)
            if neg_widget is not None:
                neg_cb = neg_widget.findChild(QCheckBox)
                if neg_cb is not None:
                    neg_cb.blockSignals(True)
                    neg_cb.setChecked(gs.get("negative", False))
                    neg_cb.blockSignals(False)
            # Column 4: Line width spinbox
            lw_widget = table.cellWidget(row, 4)
            if lw_widget is not None and isinstance(lw_widget, QDoubleSpinBox):
                lw_widget.blockSignals(True)
                lw_widget.setValue(gs.get("line_width", 1.0))
                lw_widget.blockSignals(False)

    def _save_settings_template(self):
        """Open a dialog to save current settings as a named template."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QPushButton, QMessageBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Save Settings Template")
        dialog.setMinimumWidth(350)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Template name:"))
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("Enter a name for this template")
        layout.addWidget(name_edit)
        layout.addWidget(QLabel("Include in template:"))
        cb_extraction = QCheckBox("EIC extraction settings (m/z tolerance, method, separation, RT shift…)")
        cb_extraction.setChecked(True)
        layout.addWidget(cb_extraction)
        cb_group = QCheckBox("Group display settings (scaling, visibility, line width)")
        cb_group.setChecked(True)
        layout.addWidget(cb_group)
        cb_sample = QCheckBox("Sample display settings (scaling, visibility, line width)")
        cb_sample.setChecked(True)
        layout.addWidget(cb_sample)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        tname = name_edit.text().strip()
        if not tname:
            QMessageBox.warning(self, "Invalid Name", "Please enter a template name.")
            return
        template = {"name": tname}
        if cb_extraction.isChecked():
            template["eic_extraction"] = {
                "mz_tolerance_ppm": self.mz_tolerance_ppm_spin.value() if hasattr(self, "mz_tolerance_ppm_spin") else self.defaults.get("mz_tolerance_ppm", 5.0),
                "eic_method": self.eic_method_combo.currentText() if hasattr(self, "eic_method_combo") else self.defaults.get("eic_method", "Sum of all signals"),
                "separation_mode": self.separation_mode_combo.currentText() if hasattr(self, "separation_mode_combo") else self.defaults.get("separation_mode", "None"),
                "rt_shift_min": self.rt_shift_spin.value() if hasattr(self, "rt_shift_spin") else self.defaults.get("rt_shift_min", 1.0),
                "crop_rt_window": self.crop_rt_cb.isChecked() if hasattr(self, "crop_rt_cb") else self.defaults.get("crop_rt_window", False),
                "normalize_samples": self.normalize_cb.isChecked() if hasattr(self, "normalize_cb") else self.defaults.get("normalize_samples", False),
                "legend_position": self.legend_position_combo.currentText() if hasattr(self, "legend_position_combo") else self.defaults.get("legend_position", "Right"),
            }
        if cb_group.isChecked():
            template["group_settings"] = dict(self.group_settings)
        if cb_sample.isChecked():
            template["sample_settings"] = dict(self.sample_settings)
        # Store in defaults
        templates = list(self.defaults.get("settings_templates", []))
        # Replace if same name already exists
        templates = [t for t in templates if t.get("name") != tname]
        templates.append(template)
        self.defaults["settings_templates"] = templates
        self._notify_setting("settings_templates", templates)
        QMessageBox.information(self, "Template Saved", f"Template '{tname}' has been saved.")

    def _apply_settings_template(self, template: dict):
        """Apply a settings template to the current EIC window."""
        extraction = template.get("eic_extraction")
        if extraction:
            if hasattr(self, "mz_tolerance_ppm_spin") and "mz_tolerance_ppm" in extraction:
                self.mz_tolerance_ppm_spin.setValue(extraction["mz_tolerance_ppm"])
            if hasattr(self, "eic_method_combo") and "eic_method" in extraction:
                idx = self.eic_method_combo.findText(extraction["eic_method"])
                if idx >= 0:
                    self.eic_method_combo.setCurrentIndex(idx)
            if hasattr(self, "separation_mode_combo") and "separation_mode" in extraction:
                idx = self.separation_mode_combo.findText(extraction["separation_mode"])
                if idx >= 0:
                    self.separation_mode_combo.setCurrentIndex(idx)
            if hasattr(self, "rt_shift_spin") and "rt_shift_min" in extraction:
                self.rt_shift_spin.setValue(extraction["rt_shift_min"])
            if hasattr(self, "crop_rt_cb") and "crop_rt_window" in extraction:
                self.crop_rt_cb.setChecked(extraction["crop_rt_window"])
            if hasattr(self, "normalize_cb") and "normalize_samples" in extraction:
                self.normalize_cb.setChecked(extraction["normalize_samples"])
            if hasattr(self, "legend_position_combo") and "legend_position" in extraction:
                idx = self.legend_position_combo.findText(extraction["legend_position"])
                if idx >= 0:
                    self.legend_position_combo.setCurrentIndex(idx)
        group_settings = template.get("group_settings")
        if group_settings:
            for grp, settings in group_settings.items():
                if grp in self.group_settings:
                    self.group_settings[grp].update(settings)
            # Sync group table
            self._sync_group_table_from_settings()
        sample_settings = template.get("sample_settings")
        if sample_settings:
            for fn, settings in sample_settings.items():
                if fn in self.sample_settings:
                    self.sample_settings[fn].update(settings)
            self._sync_sample_table_from_settings()
        self.update_plot(preserve_view=True)

    def create_sample_settings_table(self):
        """Create the sample settings table for per-sample display controls"""
        self.sample_settings_box = CollapsibleBox("Sample Display Settings")

        self.sample_settings_table = QTableWidget()
        # Column 0: Sample  |  Col 1: Scaling  |  Col 2: Plot  |  Col 3: Neg.  |  Col 4: Line Width
        self.sample_settings_table.setColumnCount(5)
        self.sample_settings_table.setHorizontalHeaderLabels(["Sample", "Scaling", "Plot", "Neg.", "Line Width"])

        self.sample_settings_table.setAlternatingRowColors(True)
        self.sample_settings_table.setSortingEnabled(True)
        self.sample_settings_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.sample_settings_table.horizontalHeader().setStretchLastSection(False)
        self.sample_settings_table.verticalHeader().setVisible(False)
        self.sample_settings_table.setColumnWidth(0, 130)  # Sample name column
        self.sample_settings_table.setColumnWidth(1, 95)  # Scaling column
        self.sample_settings_table.setColumnWidth(2, 30)  # Plot checkbox column
        self.sample_settings_table.setColumnWidth(3, 30)  # Neg. checkbox column
        self.sample_settings_table.setColumnWidth(4, 80)  # Line Width column
        self.sample_settings_table.setMaximumHeight(800)
        self.sample_settings_table.setMinimumHeight(340)

        # Context menu for show/hide all
        self.sample_settings_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sample_settings_table.customContextMenuRequested.connect(self._sample_settings_context_menu)

        self.sample_settings_box.add_widget(self.sample_settings_table)

    def populate_sample_settings_table(self):
        """Populate the sample settings table with all samples from the file manager"""
        table = self.sample_settings_table
        if table is None:
            return

        # Collect all filenames from the sample matrix
        samples = []
        if hasattr(self.file_manager, "files_data") and self.file_manager.files_data is not None:
            df = self.file_manager.files_data
            if "filename" in df.columns:
                for filename in df["filename"].dropna().unique():
                    if filename:
                        samples.append(str(filename))

        # Sort by group first, then by filename within each group
        if hasattr(self.file_manager, "files_data") and self.file_manager.files_data is not None:
            df = self.file_manager.files_data
            if "filename" in df.columns and "group" in df.columns:
                sample_groups = {str(r["filename"]): str(r.get("group", "")) for _, r in df.iterrows()}
            elif "filename" in df.columns:
                sample_groups = {str(r["filename"]): "" for _, r in df.iterrows()}
            else:
                sample_groups = {}
        else:
            sample_groups = {}
        natsort_key_fn = natsort_keygen()
        precomputed_keys = {fn: (natsort_key_fn(sample_groups.get(fn, "")), natsort_key_fn(fn)) for fn in samples}
        sorted_samples = sorted(samples, key=lambda fn: precomputed_keys[fn])

        table.setSortingEnabled(False)
        table.setRowCount(len(sorted_samples))

        # Initialise settings for samples that haven't been seen before
        for filename in sorted_samples:
            if filename not in self.sample_settings:
                self.sample_settings[filename] = {"scaling": 1.0, "plot": True, "negative": False, "line_width": 1.0}
            else:
                self.sample_settings[filename].setdefault("scaling", 1.0)
                self.sample_settings[filename].setdefault("negative", False)
                self.sample_settings[filename].setdefault("line_width", 1.0)

        for row, filename in enumerate(sorted_samples):
            display_name = filename.rsplit(".", 1)[0] if "." in filename else filename

            # Column 0: Sample name (stores full filename in UserRole for later lookup)
            name_item = QTableWidgetItem(display_name)
            name_item.setData(Qt.ItemDataRole.UserRole, filename)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            group_color = self._get_sample_group_color(filename)
            if group_color:
                name_item.setForeground(QColor(group_color))
            table.setItem(row, 0, name_item)

            # Column 1: Scaling spinbox
            scaling_spin = NoScrollDoubleSpinBox()
            scaling_spin.setRange(0.00001, 100000.0)
            scaling_spin.setValue(self.sample_settings[filename]["scaling"])
            scaling_spin.setDecimals(5)
            scaling_spin.setSingleStep(0.1)
            scaling_spin.valueChanged.connect(lambda value, fn=filename: self.on_sample_setting_changed(fn, "scaling", value))
            table.setCellWidget(row, 1, scaling_spin)

            # Column 2: Plot checkbox
            plot_checkbox = QCheckBox()
            plot_checkbox.setChecked(self.sample_settings[filename]["plot"])
            plot_checkbox.stateChanged.connect(lambda state, fn=filename: self.on_sample_setting_changed(fn, "plot", state == Qt.CheckState.Checked.value))
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(plot_checkbox)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            table.setCellWidget(row, 2, checkbox_widget)

            # Column 3: Neg. checkbox
            neg_checkbox = QCheckBox()
            neg_checkbox.setChecked(self.sample_settings[filename]["negative"])
            neg_checkbox.stateChanged.connect(lambda state, fn=filename: self.on_sample_setting_changed(fn, "negative", state == Qt.CheckState.Checked.value))
            neg_checkbox_widget = QWidget()
            neg_checkbox_layout = QHBoxLayout(neg_checkbox_widget)
            neg_checkbox_layout.addWidget(neg_checkbox)
            neg_checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            neg_checkbox_layout.setContentsMargins(0, 0, 0, 0)
            table.setCellWidget(row, 3, neg_checkbox_widget)

            # Column 4: Line width spinbox
            width_spin = NoScrollDoubleSpinBox()
            width_spin.setRange(0.5, 10.0)
            width_spin.setValue(self.sample_settings[filename]["line_width"])
            width_spin.setDecimals(1)
            width_spin.setSingleStep(0.5)
            width_spin.valueChanged.connect(lambda value, fn=filename: self.on_sample_setting_changed(fn, "line_width", value))
            table.setCellWidget(row, 4, width_spin)

        table.setSortingEnabled(True)

    def _get_sample_group_color(self, filename: str):
        """Return the group colour for *filename*, or None if unavailable."""
        if not hasattr(self.file_manager, "files_data") or self.file_manager.files_data is None:
            return None
        df = self.file_manager.files_data
        if "filename" not in df.columns or "group" not in df.columns:
            return None
        matching = df[df["filename"] == filename]
        if matching.empty:
            return None
        group = str(matching.iloc[0]["group"])
        return self.file_manager.get_group_color(group)

    def on_sample_setting_changed(self, filename: str, setting: str, value) -> None:
        """Handle changes to per-sample display settings"""
        if self._group_update_in_progress:
            return
        if filename not in self.sample_settings:
            self.sample_settings[filename] = {"scaling": 1.0, "plot": True, "negative": False, "line_width": 1.0}
        self.sample_settings[filename][setting] = value
        # Persist across future EIC windows
        self._notify_setting("sample_settings", dict(self.sample_settings))
        self.update_plot(preserve_view=True)

    def _sample_settings_context_menu(self, pos) -> None:
        """Show context menu on the sample settings table."""
        menu = QMenu(self)
        show_all_action = QAction("Show all", self)
        hide_all_action = QAction("Hide all", self)

        def _set_all(visible: bool):
            table = self.sample_settings_table
            for row in range(table.rowCount()):
                # Retrieve the full filename stored in column 0's UserRole
                name_item = table.item(row, 0)
                if name_item is None:
                    continue
                fn = name_item.data(Qt.ItemDataRole.UserRole)
                if fn is None:
                    continue
                # Update the settings dict directly (no signal → no individual plot update)
                if fn not in self.sample_settings:
                    self.sample_settings[fn] = {"scaling": 1.0, "plot": True, "negative": False, "line_width": 1.0}
                self.sample_settings[fn]["plot"] = visible
                # Silence the checkbox signal so it doesn't call update_plot per row
                # Plot is now column 2
                cell_widget = table.cellWidget(row, 2)
                if cell_widget is not None:
                    cb = cell_widget.findChild(QCheckBox)
                    if cb is not None:
                        cb.blockSignals(True)
                        cb.setChecked(visible)
                        cb.blockSignals(False)
            # Persist and redraw once
            self._notify_setting("sample_settings", dict(self.sample_settings))
            self.update_plot(preserve_view=True)

        show_all_action.triggered.connect(lambda: _set_all(True))
        hide_all_action.triggered.connect(lambda: _set_all(False))

        def _reset_all_samples():
            table = self.sample_settings_table
            for row in range(table.rowCount()):
                name_item = table.item(row, 0)
                if name_item is None:
                    continue
                fn = name_item.data(Qt.ItemDataRole.UserRole)
                if fn is None:
                    continue
                defaults = {"scaling": 1.0, "plot": True, "negative": False, "line_width": 1.0}
                self.sample_settings[fn] = dict(defaults)
                scaling_widget = table.cellWidget(row, 1)
                if scaling_widget is not None and hasattr(scaling_widget, "setValue"):
                    scaling_widget.blockSignals(True)
                    scaling_widget.setValue(1.0)
                    scaling_widget.blockSignals(False)
                plot_widget = table.cellWidget(row, 2)
                if plot_widget is not None:
                    cb = plot_widget.findChild(QCheckBox)
                    if cb is not None:
                        cb.blockSignals(True)
                        cb.setChecked(True)
                        cb.blockSignals(False)
                neg_widget = table.cellWidget(row, 3)
                if neg_widget is not None:
                    cb = neg_widget.findChild(QCheckBox)
                    if cb is not None:
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)
                width_widget = table.cellWidget(row, 4)
                if width_widget is not None and hasattr(width_widget, "setValue"):
                    width_widget.blockSignals(True)
                    width_widget.setValue(1.0)
                    width_widget.blockSignals(False)
            self._notify_setting("sample_settings", dict(self.sample_settings))
            self.update_plot(preserve_view=True)

        reset_action = QAction("Reset all views", self)
        reset_action.triggered.connect(_reset_all_samples)
        menu.addAction(show_all_action)
        menu.addAction(hide_all_action)
        menu.addSeparator()
        menu.addAction(reset_action)
        menu.exec(self.sample_settings_table.viewport().mapToGlobal(pos))

    def create_boxplot_widget(self):
        """Create the tabbed widget for boxplots and peak area table"""
        # Create the main tabbed widget
        self.boxplot_widget = QTabWidget()

        # Tab 1: Boxplot
        self.boxplot_figure = Figure(figsize=(10, 3))
        self.boxplot_canvas = FigureCanvas(self.boxplot_figure)
        self.boxplot_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.boxplot_canvas.setMinimumSize(0, 0)
        self.boxplot_widget.addTab(self.boxplot_canvas, "Boxplot peak areas")

        self.boxplot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Tab 2: Peak Area Table
        peak_area_tab = QWidget()
        peak_area_layout = QVBoxLayout(peak_area_tab)

        # Buttons for peak area table
        peak_area_buttons_layout = QHBoxLayout()

        self.copy_peak_area_excel_btn = QPushButton("Copy as Excel Tab-delimited")
        self.copy_peak_area_excel_btn.clicked.connect(self._copy_peak_area_table_excel)
        peak_area_buttons_layout.addWidget(self.copy_peak_area_excel_btn)

        self.copy_peak_area_r_btn = QPushButton("Copy as R dataframe")
        self.copy_peak_area_r_btn.clicked.connect(self._copy_peak_area_table_r)
        peak_area_buttons_layout.addWidget(self.copy_peak_area_r_btn)

        peak_area_buttons_layout.addStretch()  # Push buttons to the left
        peak_area_layout.addLayout(peak_area_buttons_layout)

        # Peak area table
        self.peak_area_table = QTableWidget()
        self.peak_area_table.setColumnCount(3)
        self.peak_area_table.setHorizontalHeaderLabels(["Group", "Sample Name", "Peak Area"])

        # Configure table appearance
        self.peak_area_table.setAlternatingRowColors(True)
        self.peak_area_table.setSortingEnabled(True)
        self.peak_area_table.horizontalHeader().setStretchLastSection(False)
        self.peak_area_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.peak_area_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        peak_area_layout.addWidget(self.peak_area_table)

        # Delegate for in-cell bar on the Peak Area column
        self._peak_area_bar_delegate = BarDelegate(self.peak_area_table)
        self.peak_area_table.setItemDelegateForColumn(2, self._peak_area_bar_delegate)

        self.peak_area_table.verticalHeader().setDefaultSectionSize(20)
        self.peak_area_table.verticalHeader().setMinimumSectionSize(16)

        self.boxplot_widget.addTab(peak_area_tab, "Peak area table")

        # Tab 3: Summary Statistics Table
        summary_stats_tab = QWidget()
        summary_stats_layout = QVBoxLayout(summary_stats_tab)

        # Buttons for summary statistics table
        summary_buttons_layout = QHBoxLayout()

        self.copy_summary_excel_btn = QPushButton("Copy as Excel Tab-delimited")
        self.copy_summary_excel_btn.clicked.connect(self._copy_summary_stats_table_excel)
        summary_buttons_layout.addWidget(self.copy_summary_excel_btn)

        self.copy_summary_r_btn = QPushButton("Copy as R dataframe")
        self.copy_summary_r_btn.clicked.connect(self._copy_summary_stats_table_r)
        summary_buttons_layout.addWidget(self.copy_summary_r_btn)

        summary_buttons_layout.addStretch()  # Push buttons to the left
        summary_stats_layout.addLayout(summary_buttons_layout)

        # Summary statistics table
        self.summary_stats_table = QTableWidget()
        self.summary_stats_table.setColumnCount(11)
        self.summary_stats_table.setHorizontalHeaderLabels(
            [
                "Group",
                "Min",
                "Perc_10",
                "Perc_25",
                "Perc_50_Median",
                "Mean",
                "Perc_75",
                "Perc_90",
                "Max",
                "Std_Dev",
                "RSD_%",
            ]
        )

        # Configure summary table appearance
        self.summary_stats_table.setAlternatingRowColors(True)
        self.summary_stats_table.setSortingEnabled(True)
        self.summary_stats_table.horizontalHeader().setStretchLastSection(False)
        self.summary_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.summary_stats_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        summary_stats_layout.addWidget(self.summary_stats_table)

        # Delegates for in-cell bars on the numeric columns (cols 1-9, i.e. all except Group and RSD)
        self._summary_bar_delegates = []
        for _col in range(1, 9):  # Min, P10, P25, Median, Mean, P75, P90, Max
            _d = BarDelegate(self.summary_stats_table)
            self.summary_stats_table.setItemDelegateForColumn(_col, _d)
            self._summary_bar_delegates.append(_d)

        # Compact appearance — stylesheet forces pixel font on every cell
        self.summary_stats_table.verticalHeader().setDefaultSectionSize(20)
        self.summary_stats_table.verticalHeader().setMinimumSectionSize(16)

        self.boxplot_widget.addTab(summary_stats_tab, "Peak area group summaries")

        # Tab 4: m/z statistics per sample
        self._mz_ppm_mode = False
        self._mz_stats_sample_rows = []
        self._mz_stats_group_rows = []

        mz_sample_tab = QWidget()
        mz_sample_layout = QVBoxLayout(mz_sample_tab)

        mz_sample_buttons = QHBoxLayout()
        self.copy_mz_sample_excel_btn = QPushButton("Copy as Excel Tab-delimited")
        self.copy_mz_sample_excel_btn.clicked.connect(self._copy_mz_sample_table_excel)
        mz_sample_buttons.addWidget(self.copy_mz_sample_excel_btn)
        self.copy_mz_sample_r_btn = QPushButton("Copy as R dataframe")
        self.copy_mz_sample_r_btn.clicked.connect(self._copy_mz_sample_table_r)
        mz_sample_buttons.addWidget(self.copy_mz_sample_r_btn)
        self.mz_ppm_toggle = QCheckBox("Show ppm deviation from theoretical m/z")
        self.mz_ppm_toggle.toggled.connect(self._toggle_mz_ppm_mode)
        mz_sample_buttons.addWidget(self.mz_ppm_toggle)
        mz_sample_buttons.addStretch()
        mz_sample_layout.addLayout(mz_sample_buttons)

        self.mz_sample_table = QTableWidget()
        self.mz_sample_table.setColumnCount(6)
        self.mz_sample_table.setHorizontalHeaderLabels(["Group", "Sample Name", "Mean m/z", "m/z P10", "m/z P90", "EIC Width (Da)"])
        self.mz_sample_table.setAlternatingRowColors(True)
        self.mz_sample_table.setSortingEnabled(True)
        self.mz_sample_table.horizontalHeader().setStretchLastSection(False)
        self.mz_sample_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.mz_sample_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        # Compact appearance — stylesheet forces pixel font on every cell
        self.mz_sample_table.verticalHeader().setDefaultSectionSize(20)
        self.mz_sample_table.verticalHeader().setMinimumSectionSize(16)

        mz_sample_layout.addWidget(self.mz_sample_table)
        # Centred bar delegate for ppm deviation visualisation (always shown)
        self._mz_ppm_bar_delegate_sample = CenteredBarDelegate(self.mz_sample_table)
        for _col in (2, 3, 4):
            self.mz_sample_table.setItemDelegateForColumn(_col, self._mz_ppm_bar_delegate_sample)
        self.boxplot_widget.addTab(mz_sample_tab, "m/z statistics per sample")

        # Tab 5: m/z statistics per group
        mz_group_tab = QWidget()
        mz_group_layout = QVBoxLayout(mz_group_tab)

        mz_group_buttons = QHBoxLayout()
        self.copy_mz_group_excel_btn = QPushButton("Copy as Excel Tab-delimited")
        self.copy_mz_group_excel_btn.clicked.connect(self._copy_mz_group_table_excel)
        mz_group_buttons.addWidget(self.copy_mz_group_excel_btn)
        self.copy_mz_group_r_btn = QPushButton("Copy as R dataframe")
        self.copy_mz_group_r_btn.clicked.connect(self._copy_mz_group_table_r)
        mz_group_buttons.addWidget(self.copy_mz_group_r_btn)
        self.mz_ppm_toggle_group = QCheckBox("Show ppm deviation from theoretical m/z")
        self.mz_ppm_toggle_group.toggled.connect(self._toggle_mz_ppm_mode)
        mz_group_buttons.addWidget(self.mz_ppm_toggle_group)
        mz_group_buttons.addStretch()
        mz_group_layout.addLayout(mz_group_buttons)

        self.mz_group_table = QTableWidget()
        self.mz_group_table.setColumnCount(7)
        self.mz_group_table.setHorizontalHeaderLabels(
            [
                "Group",
                "Mean of Mean m/z",
                "Mean of P10 m/z",
                "Mean of P90 m/z",
                "Std Mean m/z",
                "Min m/z",
                "Max m/z",
            ]
        )
        self.mz_group_table.setAlternatingRowColors(True)
        self.mz_group_table.setSortingEnabled(True)
        self.mz_group_table.horizontalHeader().setStretchLastSection(False)
        self.mz_group_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.mz_group_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        # Compact appearance — stylesheet forces pixel font on every cell
        self.mz_group_table.verticalHeader().setDefaultSectionSize(20)
        self.mz_group_table.verticalHeader().setMinimumSectionSize(16)

        mz_group_layout.addWidget(self.mz_group_table)
        # Centred bar delegate for ppm deviation visualisation (always shown)
        self._mz_ppm_bar_delegate_group = CenteredBarDelegate(self.mz_group_table)
        for _col in (1, 2, 3, 5, 6):
            self.mz_group_table.setItemDelegateForColumn(_col, self._mz_ppm_bar_delegate_group)
        self.boxplot_widget.addTab(mz_group_tab, "m/z statistics per group")

        # RT statistics per sample
        self._rt_stats_sample_rows = []
        self._rt_stats_group_rows = []

        rt_sample_tab = QWidget()
        rt_sample_layout = QVBoxLayout(rt_sample_tab)

        rt_sample_buttons = QHBoxLayout()
        self.copy_rt_sample_excel_btn = QPushButton("Copy as Excel Tab-delimited")
        self.copy_rt_sample_excel_btn.clicked.connect(self._copy_rt_sample_table_excel)
        rt_sample_buttons.addWidget(self.copy_rt_sample_excel_btn)
        self.copy_rt_sample_r_btn = QPushButton("Copy as R dataframe")
        self.copy_rt_sample_r_btn.clicked.connect(self._copy_rt_sample_table_r)
        rt_sample_buttons.addWidget(self.copy_rt_sample_r_btn)
        rt_sample_buttons.addStretch()
        rt_sample_layout.addLayout(rt_sample_buttons)

        self.rt_sample_table = QTableWidget()
        self.rt_sample_table.setColumnCount(6)
        self.rt_sample_table.setHorizontalHeaderLabels(
            [
                "Group",
                "Sample Name",
                "FWHM Left RT",
                "Apex RT",
                "FWHM Right RT",
                "FWHM Width",
            ]
        )
        self.rt_sample_table.setAlternatingRowColors(True)
        self.rt_sample_table.setSortingEnabled(True)
        self.rt_sample_table.horizontalHeader().setStretchLastSection(False)
        self.rt_sample_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.rt_sample_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rt_sample_table.verticalHeader().setDefaultSectionSize(20)
        self.rt_sample_table.verticalHeader().setMinimumSectionSize(16)
        rt_sample_layout.addWidget(self.rt_sample_table)

        # CenteredBarDelegate for RT-position columns (apex, left, right); BarDelegate for width
        self._rt_sample_centered_delegate = CenteredBarDelegate(self.rt_sample_table)
        for _col in (2, 3, 4):
            self.rt_sample_table.setItemDelegateForColumn(_col, self._rt_sample_centered_delegate)
        self._rt_sample_width_delegate = BarDelegate(self.rt_sample_table)
        self.rt_sample_table.setItemDelegateForColumn(5, self._rt_sample_width_delegate)

        self.boxplot_widget.addTab(rt_sample_tab, "RT statistics per sample")

        # RT statistics per group
        rt_group_tab = QWidget()
        rt_group_layout = QVBoxLayout(rt_group_tab)

        rt_group_buttons = QHBoxLayout()
        self.copy_rt_group_excel_btn = QPushButton("Copy as Excel Tab-delimited")
        self.copy_rt_group_excel_btn.clicked.connect(self._copy_rt_group_table_excel)
        rt_group_buttons.addWidget(self.copy_rt_group_excel_btn)
        self.copy_rt_group_r_btn = QPushButton("Copy as R dataframe")
        self.copy_rt_group_r_btn.clicked.connect(self._copy_rt_group_table_r)
        rt_group_buttons.addWidget(self.copy_rt_group_r_btn)
        rt_group_buttons.addStretch()
        rt_group_layout.addLayout(rt_group_buttons)

        self.rt_group_table = QTableWidget()
        self.rt_group_table.setColumnCount(7)
        self.rt_group_table.setHorizontalHeaderLabels(
            [
                "Group",
                "Mean Apex RT",
                "Std Apex RT",
                "Mean FWHM Width",
                "Std FWHM Width",
                "Min FWHM Width",
                "Max FWHM Width",
            ]
        )
        self.rt_group_table.setAlternatingRowColors(True)
        self.rt_group_table.setSortingEnabled(True)
        self.rt_group_table.horizontalHeader().setStretchLastSection(False)
        self.rt_group_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.rt_group_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rt_group_table.verticalHeader().setDefaultSectionSize(20)
        self.rt_group_table.verticalHeader().setMinimumSectionSize(16)
        rt_group_layout.addWidget(self.rt_group_table)

        # CenteredBarDelegate for Mean Apex RT (col 1); BarDelegate for the rest (cols 2–6)
        self._rt_group_centered_delegate = CenteredBarDelegate(self.rt_group_table)
        self.rt_group_table.setItemDelegateForColumn(1, self._rt_group_centered_delegate)
        self._rt_group_bar_delegates = []
        for _col in range(2, 7):
            _d = BarDelegate(self.rt_group_table)
            self.rt_group_table.setItemDelegateForColumn(_col, _d)
            self._rt_group_bar_delegates.append(_d)

        self.boxplot_widget.addTab(rt_group_tab, "RT statistics per group")

        # Tab 6: Peak Shape Similarity (Matrix)
        similarity_tab = QWidget()
        similarity_layout = QVBoxLayout(similarity_tab)

        # Buttons for similarity table
        similarity_buttons_layout = QHBoxLayout()

        self.copy_similarity_excel_btn = QPushButton("Copy as Excel Tab-delimited")
        self.copy_similarity_excel_btn.clicked.connect(self._copy_similarity_table_excel)
        similarity_buttons_layout.addWidget(self.copy_similarity_excel_btn)

        self.copy_similarity_r_btn = QPushButton("Copy as R matrix")
        self.copy_similarity_r_btn.clicked.connect(self._copy_similarity_table_r)
        similarity_buttons_layout.addWidget(self.copy_similarity_r_btn)

        similarity_buttons_layout.addStretch()
        similarity_layout.addLayout(similarity_buttons_layout)

        # Peak shape similarity table (as matrix)
        self.similarity_table = QTableWidget()
        self.similarity_table.setAlternatingRowColors(True)
        self.similarity_table.setSortingEnabled(False)  # Disable sorting for matrix view
        self.similarity_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.similarity_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        # Compact appearance — stylesheet forces pixel font on every cell
        self.similarity_table.verticalHeader().setDefaultSectionSize(20)
        self.similarity_table.verticalHeader().setMinimumSectionSize(16)

        similarity_layout.addWidget(self.similarity_table)
        self.boxplot_widget.addTab(similarity_tab, "Peak shape similarity")

        # Tab 7: PCA of correlation coefficients
        pca_tab = QWidget()
        pca_layout = QVBoxLayout(pca_tab)

        # PCA plot
        self.pca_figure = Figure(figsize=(8, 6))
        self.pca_canvas = FigureCanvas(self.pca_figure)
        self.pca_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.pca_canvas.setMinimumSize(0, 0)

        # Add navigation toolbar for zoom and pan
        self.pca_toolbar = NavigationToolbar(self.pca_canvas, pca_tab)

        pca_layout.addWidget(self.pca_toolbar)
        pca_layout.addWidget(self.pca_canvas)
        self.boxplot_widget.addTab(pca_tab, "Peak shape similarity PCA")

        # Tab 8: Quantification Calibration
        calibration_tab = QWidget()
        calibration_layout = QVBoxLayout(calibration_tab)

        # Controls for calibration
        calibration_controls = QHBoxLayout()

        # Model selection
        calibration_controls.addWidget(QLabel("Regression Model:"))
        self.regression_model_combo = QComboBox()
        self.regression_model_combo.addItems(["Linear", "Quadratic"])
        self.regression_model_combo.currentTextChanged.connect(self._update_calibration_plot)
        calibration_controls.addWidget(self.regression_model_combo)

        # Axis transformation selection
        calibration_controls.addWidget(QLabel("Axis Scale:"))
        self.axis_transform_combo = QComboBox()
        self.axis_transform_combo.addItems(["Linear", "Log2/Log2", "Log10/Log10"])
        self.axis_transform_combo.currentTextChanged.connect(self._update_calibration_plot)
        calibration_controls.addWidget(self.axis_transform_combo)

        # Injection volume / dilution normalization
        self.normalize_peak_area_checkbox = QCheckBox("Normalize peak areas by injection volume × dilution")
        self.normalize_peak_area_checkbox.setChecked(False)
        self.normalize_peak_area_checkbox.setToolTip(
            "When checked, each peak area is multiplied by the sample's injection volume and dilution factor before being used in regression fitting and prediction."
        )
        self.normalize_peak_area_checkbox.stateChanged.connect(lambda _: self._update_calibration_table(self._all_peak_data) if hasattr(self, "_all_peak_data") else None)
        calibration_controls.addWidget(self.normalize_peak_area_checkbox)

        calibration_controls.addStretch()
        calibration_layout.addLayout(calibration_controls)

        # Splitter for table and plot
        calibration_splitter = QSplitter(Qt.Orientation.Vertical)

        # Calibration table
        self.calibration_table = QTableWidget()
        self.calibration_table.setColumnCount(8)
        self.calibration_table.setHorizontalHeaderLabels(
            [
                "Use",
                "Sample Name",
                "Peak Area",
                "Abundance",
                "Unit",
                "Dilution",
                "Injection Volume",
                "Correction Factor",
            ]
        )
        self.calibration_table.setAlternatingRowColors(True)
        self.calibration_table.horizontalHeader().setStretchLastSection(False)
        self.calibration_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.calibration_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.calibration_table.verticalHeader().setDefaultSectionSize(20)
        self.calibration_table.verticalHeader().setMinimumSectionSize(16)
        self.calibration_table.itemChanged.connect(self._on_calibration_table_changed)
        calibration_splitter.addWidget(self.calibration_table)

        # Calibration plot
        self.calibration_figure = Figure(figsize=(8, 6))
        self.calibration_canvas = FigureCanvas(self.calibration_figure)
        self.calibration_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.calibration_toolbar = NavigationToolbar(self.calibration_canvas, calibration_tab)

        calibration_plot_widget = QWidget()
        calibration_plot_layout = QVBoxLayout(calibration_plot_widget)
        calibration_plot_layout.addWidget(self.calibration_toolbar)
        calibration_plot_layout.addWidget(self.calibration_canvas)
        calibration_splitter.addWidget(calibration_plot_widget)

        calibration_layout.addWidget(calibration_splitter)
        self.boxplot_widget.addTab(calibration_tab, "Quantification Calibration")

        # Tab 9: Calculated Abundances
        calculated_abundances_tab = QWidget()
        calculated_abundances_layout = QVBoxLayout(calculated_abundances_tab)

        # Buttons for calculated abundances
        abundances_buttons_layout = QHBoxLayout()
        self.copy_abundances_excel_btn = QPushButton("Copy as Excel Tab-delimited")
        self.copy_abundances_excel_btn.clicked.connect(self._copy_abundances_table_excel)
        abundances_buttons_layout.addWidget(self.copy_abundances_excel_btn)

        self.copy_abundances_r_btn = QPushButton("Copy as R dataframe")
        self.copy_abundances_r_btn.clicked.connect(self._copy_abundances_table_r)
        abundances_buttons_layout.addWidget(self.copy_abundances_r_btn)

        abundances_buttons_layout.addStretch()
        calculated_abundances_layout.addLayout(abundances_buttons_layout)

        # Calculated abundances table
        self.calculated_abundances_table = QTableWidget()
        self.calculated_abundances_table.setColumnCount(10)
        self.calculated_abundances_table.setHorizontalHeaderLabels(
            [
                "Type",
                "Group",
                "Sample Name",
                "Peak Area",
                "Actual Abundance",
                "Predicted Abundance",
                "Unit",
                "Dilution",
                "Injection Volume",
                "Correction Factor",
            ]
        )
        self.calculated_abundances_table.setAlternatingRowColors(True)
        self.calculated_abundances_table.setSortingEnabled(True)
        self.calculated_abundances_table.horizontalHeader().setStretchLastSection(False)
        self.calculated_abundances_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.calculated_abundances_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.calculated_abundances_table.verticalHeader().setDefaultSectionSize(20)
        self.calculated_abundances_table.verticalHeader().setMinimumSectionSize(16)
        calculated_abundances_layout.addWidget(self.calculated_abundances_table)
        self.boxplot_widget.addTab(calculated_abundances_tab, "Calculated Abundances")

        # Tab 10: Quantification Group Summaries
        quant_summary_tab = QWidget()
        quant_summary_layout = QVBoxLayout(quant_summary_tab)

        quant_summary_buttons = QHBoxLayout()
        self.copy_quant_summary_excel_btn = QPushButton("Copy as Excel Tab-delimited")
        self.copy_quant_summary_excel_btn.clicked.connect(self._copy_quant_summary_table_excel)
        quant_summary_buttons.addWidget(self.copy_quant_summary_excel_btn)
        self.copy_quant_summary_r_btn = QPushButton("Copy as R dataframe")
        self.copy_quant_summary_r_btn.clicked.connect(self._copy_quant_summary_table_r)
        quant_summary_buttons.addWidget(self.copy_quant_summary_r_btn)
        quant_summary_buttons.addStretch()
        quant_summary_layout.addLayout(quant_summary_buttons)

        self.quant_summary_table = QTableWidget()
        self.quant_summary_table.setColumnCount(7)
        self.quant_summary_table.setHorizontalHeaderLabels(["Group", "Min", "Perc_10", "Median", "Mean", "Perc_90", "Max"])
        self.quant_summary_table.setAlternatingRowColors(True)
        self.quant_summary_table.setSortingEnabled(True)
        self.quant_summary_table.horizontalHeader().setStretchLastSection(False)
        self.quant_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.quant_summary_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.quant_summary_table.verticalHeader().setDefaultSectionSize(20)
        self.quant_summary_table.verticalHeader().setMinimumSectionSize(16)
        quant_summary_layout.addWidget(self.quant_summary_table)

        # Bar delegates for numeric columns (1–6)
        self._quant_summary_bar_delegates = []
        for _col in range(1, 7):
            _d = BarDelegate(self.quant_summary_table)
            self.quant_summary_table.setItemDelegateForColumn(_col, _d)
            self._quant_summary_bar_delegates.append(_d)

        self.boxplot_widget.addTab(quant_summary_tab, "Quantification group summaries")

        # Initially hide the boxplot widget
        self.boxplot_widget.setVisible(False)

    def add_peak_boundary(self, rt_value: float):
        """Add a single peak boundary line at the specified RT"""
        if len(self.peak_boundary_lines) >= 2:
            return  # Already have 2 boundary lines

        # Get current axis ranges
        x_axis = self.chart.axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart.axes(Qt.Orientation.Vertical)[0]

        x_min, x_max = x_axis.min(), x_axis.max()
        y_min, y_max = y_axis.min(), y_axis.max()

        # Ensure the RT is within the visible range
        line_x = max(x_min, min(x_max, rt_value))

        # Create vertical line
        line_series = QLineSeries()
        line_series.setName("")  # No legend entry
        line_series.setProperty("is_decoration", True)
        line_series.append(line_x, y_min)
        line_series.append(line_x, y_max)

        # Style: solid red line for peak boundaries
        pen = QPen(QColor(255, 0, 0))  # Red
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.SolidLine)
        line_series.setPen(pen)

        # Add to chart
        self.chart.addSeries(line_series)
        line_series.attachAxis(x_axis)
        line_series.attachAxis(y_axis)

        # Store reference and RT value
        self.peak_boundary_lines.append(line_series)

        if len(self.peak_boundary_lines) == 1:
            self.peak_start_rt = line_x
        else:  # len == 2
            self.peak_end_rt = line_x
            # Ensure start_rt <= end_rt
            if self.peak_start_rt > self.peak_end_rt:
                self.peak_start_rt, self.peak_end_rt = (
                    self.peak_end_rt,
                    self.peak_start_rt,
                )

        # Update info and show boxplot only when we have 2 boundaries
        if len(self.peak_boundary_lines) == 2:
            self.update_boundary_info()
            self.update_boxplot()
        elif hasattr(self, "boundary_info_label"):
            self.boundary_info_label.setText(f"Peak boundary: {line_x:.2f} min (add second boundary)")

    def remove_peak_boundaries(self):
        """Remove all peak boundary lines"""
        for line in self.peak_boundary_lines:
            self.chart.removeSeries(line)

        self.peak_boundary_lines.clear()
        self.peak_start_rt = None
        self.peak_end_rt = None

        # Hide boxplot and clear info
        self.boxplot_widget.setVisible(False)

        # Clear the peak area table
        if hasattr(self, "peak_area_table"):
            self.peak_area_table.setRowCount(0)

        # Clear the similarity table
        if hasattr(self, "similarity_table"):
            self.similarity_table.setRowCount(0)

        # Clear the m/z statistics tables
        if hasattr(self, "mz_sample_table"):
            self.mz_sample_table.setRowCount(0)
        if hasattr(self, "mz_group_table"):
            self.mz_group_table.setRowCount(0)

        # Clear the RT statistics tables
        if hasattr(self, "rt_sample_table"):
            self.rt_sample_table.setRowCount(0)
        if hasattr(self, "rt_group_table"):
            self.rt_group_table.setRowCount(0)

        if hasattr(self, "boundary_info_label"):
            self.boundary_info_label.setText("")

    def update_boundary_info(self):
        """Update the boundary info (no longer displays label since UI was removed)"""
        # This method is kept for compatibility but no longer updates any UI element
        # The peak boundaries are now managed only through the context menu
        pass

    def update_boxplot(self):
        """Update the boxplot with integrated peak areas"""
        if len(self.peak_boundary_lines) != 2 or not self.eic_data:
            self.boxplot_widget.setVisible(False)
            # Clear the peak area table
            if hasattr(self, "peak_area_table"):
                self.peak_area_table.setRowCount(0)
            return

        # Calculate integrated areas for each sample
        start_rt = min(self.peak_start_rt, self.peak_end_rt) if self.peak_start_rt and self.peak_end_rt else 0
        end_rt = max(self.peak_start_rt, self.peak_end_rt) if self.peak_start_rt and self.peak_end_rt else 0

        if start_rt >= end_rt:
            return

        # Collect data for boxplot, table, and apex detection
        boxplot_data = {}  # group_name -> list of integrated areas
        table_data = []  # list of tuples (group, sample_name, peak_area)
        rt_data = []  # list of tuples (group, sample_name, apex_rt, fwhm_left, fwhm_right, fwhm_width)
        apex_rt = None
        apex_intensity = float("-inf")

        sep_mode = self._separation_mode()
        separate_groups = sep_mode == "By group"

        for filepath, data in self.eic_data.items():
            rt = data["rt"]
            intensity = data["intensity"]
            metadata = data["metadata"]

            if len(rt) == 0 or len(intensity) == 0:
                continue

            # Get group value from the selected grouping column
            group_value = metadata.get(self.grouping_column, "Unknown")
            group = str(group_value) if group_value is not None else "Unknown"
            sample_name = metadata.get("filename", "Unknown")

            # Remove file extension from sample name for cleaner display
            if "." in sample_name:
                sample_name = sample_name.rsplit(".", 1)[0]

            # Use original RT values for integration (not shifted)
            # This ensures we always use the same RT scale regardless of visualization
            original_rt = rt.copy()

            # Calculate integrated area with proper boundary handling
            integrated_area = self._calculate_peak_area_with_boundaries(original_rt, intensity, start_rt, end_rt)

            # Per-sample apex and FWHM computation within the integration window
            sample_apex_rt = None
            sample_apex_intensity = float("-inf")
            sample_fwhm_left = None
            sample_fwhm_right = None
            sample_fwhm_width = None

            # Build arrays confined to the integration window
            win_rt = []
            win_int = []
            for rt_val, intensity_val in zip(original_rt, intensity):
                try:
                    rt_float = float(rt_val)
                    intensity_float = float(intensity_val)
                except (TypeError, ValueError):
                    continue
                if start_rt <= rt_float <= end_rt and not np.isnan(intensity_float):
                    win_rt.append(rt_float)
                    win_int.append(intensity_float)
                    # Also track global apex for the plot title
                    if intensity_float > apex_intensity:
                        apex_intensity = intensity_float
                        apex_rt = rt_float

            if win_rt:
                apex_idx = int(np.argmax(win_int))
                sample_apex_rt = win_rt[apex_idx]
                sample_apex_intensity = win_int[apex_idx]
                half_max = sample_apex_intensity / 2.0

                # Find left FWHM crossing by scanning leftward from apex
                fwhm_left = None
                for i in range(apex_idx - 1, -1, -1):
                    if win_int[i] <= half_max:
                        # Linear interpolation between point i and i+1
                        dI = win_int[i + 1] - win_int[i]
                        if dI != 0:
                            fwhm_left = win_rt[i] + (half_max - win_int[i]) / dI * (win_rt[i + 1] - win_rt[i])
                        else:
                            fwhm_left = win_rt[i]
                        break
                if fwhm_left is None:
                    fwhm_left = win_rt[0]  # clamp to window start

                # Find right FWHM crossing by scanning rightward from apex
                fwhm_right = None
                for i in range(apex_idx + 1, len(win_rt)):
                    if win_int[i] <= half_max:
                        # Linear interpolation between point i-1 and i
                        dI = win_int[i] - win_int[i - 1]
                        if dI != 0:
                            fwhm_right = win_rt[i - 1] + (half_max - win_int[i - 1]) / dI * (win_rt[i] - win_rt[i - 1])
                        else:
                            fwhm_right = win_rt[i]
                        break
                if fwhm_right is None:
                    fwhm_right = win_rt[-1]  # clamp to window end

                sample_fwhm_left = fwhm_left
                sample_fwhm_right = fwhm_right
                sample_fwhm_width = fwhm_right - fwhm_left

            rt_data.append(
                (
                    group,
                    sample_name,
                    sample_apex_rt,
                    sample_fwhm_left,
                    sample_fwhm_right,
                    sample_fwhm_width,
                )
            )

            if integrated_area > 0:  # Only add non-zero areas
                if group not in boxplot_data:
                    boxplot_data[group] = []
                boxplot_data[group].append(integrated_area)

                # Add to table data
                table_data.append((group, sample_name, integrated_area))

        if not boxplot_data:
            self.boxplot_widget.setVisible(False)
            # Clear the peak area table
            if hasattr(self, "peak_area_table"):
                self.peak_area_table.setRowCount(0)
            return

        # Create boxplot
        self.boxplot_figure.clear()
        ax = self.boxplot_figure.add_subplot(111)

        # Prepare data for plotting
        groups = list(boxplot_data.keys())
        data_lists = [boxplot_data[group] for group in groups]

        # Create horizontal boxplot so groups appear on the y-axis
        bp = ax.boxplot(
            data_lists,
            labels=groups,
            patch_artist=True,
            vert=False,
        )

        # Color the boxes according to group colors
        for group, patch in zip(groups, bp["boxes"]):
            group_color = self._get_group_color(group)
            if group_color:
                color = QColor(group_color)
                patch.set_facecolor((color.red() / 255, color.green() / 255, color.blue() / 255, 0.7))

        # Add jitter points
        for i, (group, values) in enumerate(zip(groups, data_lists)):
            # Add some jitter to y positions around the boxplot line index
            y_pos = i + 1
            jitter_y = np.random.normal(y_pos, 0.05, len(values))  # Small jitter

            group_color = self._get_group_color(group)
            if group_color:
                color = QColor(group_color)
                color_rgb = (color.red() / 255, color.green() / 255, color.blue() / 255)
            else:
                color_rgb = (0.5, 0.5, 0.5)  # Default gray

            ax.scatter(
                values,
                jitter_y,
                alpha=0.6,
                color=color_rgb,
                s=30,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Integrated Peak Area")
        ax.set_ylabel("Group")
        if apex_rt is not None:
            ax.set_title(f"Peak Integration ({start_rt:.2f} - {end_rt:.2f} min, apex {apex_rt:.2f} min)")
        else:
            ax.set_title(f"Peak Integration ({start_rt:.2f} - {end_rt:.2f} min)")
        ax.grid(True, alpha=0.3)

        # Always set x-axis to start from 0 and adapt upper bound to data range
        data_max = max((max(values) for values in data_lists if values), default=0)
        upper_bound = data_max * 1.05 if data_max > 0 else 1.0
        ax.set_xlim(left=0, right=upper_bound)

        self.boxplot_figure.tight_layout()
        self.boxplot_canvas.draw_idle()

        # Update the peak area table
        self._update_peak_area_table(table_data)

        # Update the summary statistics table
        self._update_summary_stats_table(table_data)

        # Update the peak shape similarity table
        self._update_similarity_table(start_rt, end_rt)

        # Update the m/z statistics tables (per sample and per group)
        self._update_mz_stats_tables(start_rt, end_rt)

        # Update the RT statistics tables (per sample and per group)
        self._update_rt_stats_tables(rt_data)

        # Update the PCA plot
        self._update_pca_plot(start_rt, end_rt)

        # Store all peak data for quantification (calibration + unknowns)
        self._all_peak_data = list(table_data)

        # Update the quantification calibration table and plot
        self._update_calibration_table(table_data)

        # Record peak integration data if callback is provided
        if self.integration_callback:
            self._record_integration_data(table_data, start_rt, end_rt)

        # Show the boxplot widget
        self.boxplot_widget.setVisible(True)

    def _record_integration_data(self, table_data, start_rt, end_rt):
        """Record integration data using the callback"""
        try:
            # Prepare the data for the callback
            compound_name = self.compound_data.get("Name", "Unknown")
            ion_name = self.adduct
            mz_value = self.target_mz
            rt_value = self.compound_data.get("RT_min", 0)

            # Determine ion mode from polarity
            if self.polarity == "+":
                ion_mode = "positive"
            elif self.polarity == "-":
                ion_mode = "negative"
            else:
                ion_mode = "unknown"

            # Prepare sample data list for the callback
            sample_data_list = []
            for group, sample_name, peak_area in table_data:
                sample_data_list.append((sample_name, group, peak_area))

            # Call the integration callback to record the first sample
            if sample_data_list:
                first_sample = sample_data_list[0]
                self.integration_callback(
                    compound_name=compound_name,
                    ion_name=ion_name,
                    mz_value=mz_value,
                    rt_value=rt_value,
                    ion_mode=ion_mode,
                    sample_name=first_sample[0],
                    group_name=first_sample[1],
                    peak_start_rt=start_rt,
                    peak_end_rt=end_rt,
                    peak_area=first_sample[2],
                )

                # Update with all sample data
                self.integration_callback.__self__.update_peak_integration_samples(compound_name, ion_name, sample_data_list)

        except Exception as e:
            print(f"Error recording integration data: {e}")

    def _update_peak_area_table(self, table_data):
        """Update the peak area table with the calculated data"""
        # Clear existing data
        self.peak_area_table.setRowCount(0)

        if not table_data:
            return

        # Sort data by group, then by sample name (using natural sort)
        natsort_key = natsort_keygen()
        table_data.sort(key=lambda x: (natsort_key(x[0]), natsort_key(x[1])))

        # Global maximum for bar scaling
        global_max = max((pa for _, _, pa in table_data), default=0)

        # Set number of rows
        self.peak_area_table.setSortingEnabled(False)
        self.peak_area_table.verticalHeader().setDefaultSectionSize(20)
        self.peak_area_table.setRowCount(len(table_data))

        # Populate the table
        for row, (group, sample_name, peak_area) in enumerate(table_data):
            # Group column
            group_item = QTableWidgetItem(str(group))
            group_color = self._get_group_color(group)
            if group_color:
                gc = QColor(group_color)
                gc.setAlphaF(0.5)
                group_item.setBackground(gc)
            self.peak_area_table.setItem(row, 0, group_item)

            # Sample name column
            sample_item = QTableWidgetItem(str(sample_name))
            if group_color:
                sample_item.setBackground(gc)
            self.peak_area_table.setItem(row, 1, sample_item)

            # Peak area column (formatted to scientific notation) with bar
            area_item = NumericTableWidgetItem(f"{peak_area:.2e}")
            area_item.setData(Qt.ItemDataRole.UserRole, peak_area)

            # Bar fraction and colour
            frac = (peak_area / global_max) if global_max > 0 else 0.0
            area_item.setData(BarDelegate.BAR_FRAC_ROLE, frac)
            group_color = self._get_group_color(group)
            if group_color:
                area_item.setData(BarDelegate.BAR_COLOR_ROLE, QColor(group_color))

            self.peak_area_table.setItem(row, 2, area_item)

        self.peak_area_table.setSortingEnabled(True)
        # Auto-resize columns to fit content
        self.peak_area_table.resizeColumnsToContents()

    def _update_summary_stats_table(self, table_data):
        """Update the summary statistics table with group-level statistics"""
        import numpy as np

        # Clear existing data
        self.summary_stats_table.setRowCount(0)

        if not table_data:
            return

        # Group data by group name
        group_data = {}
        for group, sample_name, peak_area in table_data:
            if group not in group_data:
                group_data[group] = []
            group_data[group].append(peak_area)

        # Calculate statistics for each group
        stats_data = []
        for group, areas in group_data.items():
            areas_array = np.array(areas)

            # Calculate all required statistics
            stats = {
                "group": str(group),
                "min": np.min(areas_array),
                "p10": np.percentile(areas_array, 10),
                "p25": np.percentile(areas_array, 25),
                "median": np.percentile(areas_array, 50),
                "mean": np.mean(areas_array),
                "p75": np.percentile(areas_array, 75),
                "p90": np.percentile(areas_array, 90),
                "max": np.max(areas_array),
                "std": np.std(areas_array, ddof=1),  # Sample standard deviation
            }
            # Calculate RSD (Relative Standard Deviation) as percentage
            if stats["mean"] != 0:
                stats["rsd"] = (stats["std"] / stats["mean"]) * 100
            else:
                stats["rsd"] = 0.0
            stats_data.append(stats)

        # Sort by group name (using natural sort)
        natsort_key = natsort_keygen()
        stats_data.sort(key=lambda x: natsort_key(x["group"]))

        # Per-column max for bar scaling (cols 1-8: min,p10,p25,median,mean,p75,p90,max)
        stat_keys_for_bar = ["min", "p10", "p25", "median", "mean", "p75", "p90", "max"]
        col_maxima = {key: max((s[key] for s in stats_data), default=0) for key in stat_keys_for_bar}

        # Set number of rows
        self.summary_stats_table.setSortingEnabled(False)
        self.summary_stats_table.verticalHeader().setDefaultSectionSize(20)
        self.summary_stats_table.setRowCount(len(stats_data))

        # Populate the table
        for row, stats in enumerate(stats_data):
            # Group name
            group_item = QTableWidgetItem(stats["group"])
            group_color = self._get_group_color(stats["group"])
            if group_color:
                group_item.setForeground(QColor(group_color))
                _sf = group_item.font()
                _sf.setBold(True)
                group_item.setFont(_sf)
            self.summary_stats_table.setItem(row, 0, group_item)

            # Statistical values (formatted to scientific notation)
            stat_values = [
                stats["min"],
                stats["p10"],
                stats["p25"],
                stats["median"],
                stats["mean"],
                stats["p75"],
                stats["p90"],
                stats["max"],
                stats["std"],
                stats["rsd"],
            ]

            for col, value in enumerate(stat_values, 1):
                # Format RSD as percentage with 2 decimal places
                if col == 10:  # RSD column
                    item = NumericTableWidgetItem(f"{value:.2f}")
                else:
                    item = NumericTableWidgetItem(f"{value:.2e}")
                item.setData(Qt.ItemDataRole.UserRole, value)

                # Add bar data for cols 1-8
                if col <= 8:
                    bar_key = stat_keys_for_bar[col - 1]
                    col_max = col_maxima[bar_key]
                    frac = (value / col_max) if col_max > 0 else 0.0
                    item.setData(BarDelegate.BAR_FRAC_ROLE, frac)
                    if group_color:
                        item.setData(BarDelegate.BAR_COLOR_ROLE, QColor(group_color))

                self.summary_stats_table.setItem(row, col, item)

        self.summary_stats_table.setSortingEnabled(True)
        # Auto-resize columns to fit content
        self.summary_stats_table.resizeColumnsToContents()

    def _update_mz_stats_tables(self, start_rt, end_rt):
        """Compute and populate the m/z statistics tables for both per-sample and per-group views."""
        self.mz_sample_table.setRowCount(0)
        self.mz_group_table.setRowCount(0)

        if not self.eic_data:
            return

        mz_tolerance = self.mz_tolerance_da_spin.value()
        polarity = self.polarity

        sample_rows = []  # list of (group, sample_name, mean_mz, p10, p90, eic_width)

        for filepath, data in self.eic_data.items():
            metadata = data["metadata"]
            group_value = metadata.get(self.grouping_column, "Unknown")
            group = str(group_value) if group_value is not None else "Unknown"
            sample_name = metadata.get("filename", "Unknown")
            if "." in sample_name:
                sample_name = sample_name.rsplit(".", 1)[0]

            stats = self.file_manager.get_mz_stats_in_rt_window(filepath, self.target_mz, mz_tolerance, start_rt, end_rt, polarity)
            if stats is None:
                continue

            eic_width = mz_tolerance * 2
            sample_rows.append(
                (
                    group,
                    sample_name,
                    stats["mean_mz"],
                    stats["p10_mz"],
                    stats["p90_mz"],
                    eic_width,
                )
            )

        if not sample_rows:
            return

        # Sort by group then sample name
        natsort_key = natsort_keygen()
        sample_rows.sort(key=lambda x: (natsort_key(x[0]), natsort_key(x[1])))

        # --- Per-group aggregation ---
        group_data: dict = {}
        for group, _, mean_mz, p10, p90, _ in sample_rows:
            group_data.setdefault(group, {"mean": [], "p10": [], "p90": []})
            group_data[group]["mean"].append(mean_mz)
            group_data[group]["p10"].append(p10)
            group_data[group]["p90"].append(p90)

        group_rows = []
        for grp, vals in group_data.items():
            m = np.array(vals["mean"])
            group_rows.append(
                {
                    "group": grp,
                    "mean_mean": float(np.mean(m)),
                    "mean_p10": float(np.mean(vals["p10"])),
                    "mean_p90": float(np.mean(vals["p90"])),
                    "std_mean": float(np.std(m, ddof=1)) if len(m) > 1 else 0.0,
                    "min_mz": float(np.min(m)),
                    "max_mz": float(np.max(m)),
                }
            )
        group_rows.sort(key=lambda x: natsort_key(x["group"]))

        # Store raw data and delegate rendering to _refresh_mz_stats_display
        self._mz_stats_sample_rows = sample_rows
        self._mz_stats_group_rows = group_rows
        self._refresh_mz_stats_display()

    # ------------------------------------------------------------------
    # Copy helpers for m/z statistics tables
    # ------------------------------------------------------------------

    def _copy_mz_sample_table_excel(self):
        """Copy m/z per-sample table as Excel tab-delimited text."""
        tbl = self.mz_sample_table
        if tbl.rowCount() == 0:
            return
        headers = [tbl.horizontalHeaderItem(c).text() for c in range(tbl.columnCount())]
        lines = ["\t".join(headers)]
        for r in range(tbl.rowCount()):
            row_vals = []
            for c in range(tbl.columnCount()):
                item = tbl.item(r, c)
                row_vals.append(item.text() if item else "")
            lines.append("\t".join(row_vals))
        QApplication.clipboard().setText("\n".join(lines))

    def _copy_mz_sample_table_r(self):
        """Copy m/z per-sample table as R dataframe code."""
        tbl = self.mz_sample_table
        if tbl.rowCount() == 0:
            return
        headers = [tbl.horizontalHeaderItem(c).text().replace(" ", "_").replace("/", "_") for c in range(tbl.columnCount())]
        lines = [
            f"df <- data.frame(",
            "  " + ",\n  ".join(f"{h} = c()" for h in headers),
            ")",
        ]
        rows_data = []
        for r in range(tbl.rowCount()):
            row_vals = [tbl.item(r, c).text() if tbl.item(r, c) else "" for c in range(tbl.columnCount())]
            rows_data.append(row_vals)
        col_lines = []
        for ci, h in enumerate(headers):
            vals = [f'"{rows_data[ri][ci]}"' if ci < 2 else rows_data[ri][ci] for ri in range(len(rows_data))]
            col_lines.append(f"  {h} = c({', '.join(vals)})")
        r_code = "df <- data.frame(\n" + ",\n".join(col_lines) + "\n)"
        QApplication.clipboard().setText(r_code)

    def _copy_mz_group_table_excel(self):
        """Copy m/z per-group table as Excel tab-delimited text."""
        tbl = self.mz_group_table
        if tbl.rowCount() == 0:
            return
        headers = [tbl.horizontalHeaderItem(c).text() for c in range(tbl.columnCount())]
        lines = ["\t".join(headers)]
        for r in range(tbl.rowCount()):
            row_vals = []
            for c in range(tbl.columnCount()):
                item = tbl.item(r, c)
                row_vals.append(item.text() if item else "")
            lines.append("\t".join(row_vals))
        QApplication.clipboard().setText("\n".join(lines))

    def _copy_mz_group_table_r(self):
        """Copy m/z per-group table as R dataframe code."""
        tbl = self.mz_group_table
        if tbl.rowCount() == 0:
            return
        headers = [tbl.horizontalHeaderItem(c).text().replace(" ", "_").replace("/", "_") for c in range(tbl.columnCount())]
        rows_data = []
        for r in range(tbl.rowCount()):
            rows_data.append([tbl.item(r, c).text() if tbl.item(r, c) else "" for c in range(tbl.columnCount())])
        col_lines = []
        for ci, h in enumerate(headers):
            vals = [f'"{rows_data[ri][ci]}"' if ci == 0 else rows_data[ri][ci] for ri in range(len(rows_data))]
            col_lines.append(f"  {h} = c({', '.join(vals)})")
        r_code = "df <- data.frame(\n" + ",\n".join(col_lines) + "\n)"
        QApplication.clipboard().setText(r_code)

    # ------------------------------------------------------------------
    # RT apex / FWHM statistics tables
    # ------------------------------------------------------------------

    def _update_rt_stats_tables(self, rt_data):
        """Compute and populate the RT apex/FWHM statistics tables for both per-sample and per-group views.

        Parameters
        ----------
        rt_data : list of tuples
            Each tuple: (group, sample_name, apex_rt, fwhm_left, fwhm_right, fwhm_width)
            Any of the float fields can be None if not computable.
        """
        self.rt_sample_table.setRowCount(0)
        self.rt_group_table.setRowCount(0)

        if not rt_data:
            return

        # Sort by group then sample name
        natsort_key = natsort_keygen()
        sample_rows = sorted(rt_data, key=lambda x: (natsort_key(x[0]), natsort_key(x[1])))

        # Store for potential later refresh
        self._rt_stats_sample_rows = sample_rows

        # Reference RT and range for centred-bar visualisation (same principle as m/z ppm offset)
        ref_rt = float(self.compound_data.get("RT_min", 0))
        _int_start = getattr(self, "peak_start_rt", None) or ref_rt
        _int_end = getattr(self, "peak_end_rt", None) or ref_rt
        rt_range = max(abs(_int_start - ref_rt), abs(_int_end - ref_rt))
        if rt_range < 1e-6:
            rt_range = 0.5  # fallback ±0.5 min

        # Max for FWHM Width BarDelegate (col 5 only)
        fwhm_width_vals = [row[5] for row in sample_rows if row[5] is not None]
        fwhm_width_max = max(fwhm_width_vals) if fwhm_width_vals else 1.0

        # --- Per-sample table ---
        self.rt_sample_table.setSortingEnabled(False)
        self.rt_sample_table.verticalHeader().setDefaultSectionSize(20)
        self.rt_sample_table.setRowCount(len(sample_rows))
        for row_idx, (
            group,
            sample_name,
            apex_rt,
            fwhm_left,
            fwhm_right,
            fwhm_width,
        ) in enumerate(sample_rows):
            grp_color = self._get_group_color(group)

            def _colored_item(text, color=grp_color):
                it = QTableWidgetItem(text)
                if color:
                    c = QColor(color)
                    c.setAlphaF(0.5)
                    it.setBackground(c)
                return it

            self.rt_sample_table.setItem(row_idx, 0, _colored_item(group))
            self.rt_sample_table.setItem(row_idx, 1, _colored_item(sample_name))

            # Cols 2–4: RT positions — display actual value, centred bar shows offset from RT_min
            for col_idx, val in enumerate([fwhm_left, apex_rt, fwhm_right], start=2):
                if val is not None:
                    item = NumericTableWidgetItem(f"{val:.4f}")
                    item.setData(Qt.ItemDataRole.UserRole, val)
                    item.setData(CenteredBarDelegate.PPM_DEVIATION_ROLE, val - ref_rt)
                    item.setData(CenteredBarDelegate.PPM_RANGE_ROLE, rt_range)
                else:
                    item = QTableWidgetItem("N/A")
                    item.setData(Qt.ItemDataRole.UserRole, -1)
                self.rt_sample_table.setItem(row_idx, col_idx, item)

            # Col 5: FWHM Width — proportional bar
            if fwhm_width is not None:
                item = NumericTableWidgetItem(f"{fwhm_width:.4f}")
                item.setData(Qt.ItemDataRole.UserRole, fwhm_width)
                frac = (fwhm_width / fwhm_width_max) if fwhm_width_max > 0 else 0.0
                item.setData(BarDelegate.BAR_FRAC_ROLE, frac)
                if grp_color:
                    item.setData(BarDelegate.BAR_COLOR_ROLE, QColor(grp_color))
            else:
                item = QTableWidgetItem("N/A")
                item.setData(Qt.ItemDataRole.UserRole, -1)
            self.rt_sample_table.setItem(row_idx, 5, item)

        self.rt_sample_table.setSortingEnabled(True)
        self.rt_sample_table.resizeColumnsToContents()

        # --- Per-group aggregation ---
        group_data: dict = {}
        for group, _, apex_rt, _fl, _fr, fwhm_width in sample_rows:
            entry = group_data.setdefault(group, {"apex": [], "fwhm": []})
            if apex_rt is not None:
                entry["apex"].append(apex_rt)
            if fwhm_width is not None:
                entry["fwhm"].append(fwhm_width)

        group_rows = []
        for grp, vals in group_data.items():
            a = np.array(vals["apex"]) if vals["apex"] else np.array([])
            f = np.array(vals["fwhm"]) if vals["fwhm"] else np.array([])
            group_rows.append(
                {
                    "group": grp,
                    "mean_apex": float(np.mean(a)) if len(a) > 0 else None,
                    "std_apex": float(np.std(a, ddof=1)) if len(a) > 1 else (0.0 if len(a) == 1 else None),
                    "mean_fwhm": float(np.mean(f)) if len(f) > 0 else None,
                    "std_fwhm": float(np.std(f, ddof=1)) if len(f) > 1 else (0.0 if len(f) == 1 else None),
                    "min_fwhm": float(np.min(f)) if len(f) > 0 else None,
                    "max_fwhm": float(np.max(f)) if len(f) > 0 else None,
                }
            )
        group_rows.sort(key=lambda x: natsort_key(x["group"]))
        self._rt_stats_group_rows = group_rows

        # Compute per-column maxima for the non-apex BarDelegate columns
        grp_col_keys = [
            "mean_apex",
            "std_apex",
            "mean_fwhm",
            "std_fwhm",
            "min_fwhm",
            "max_fwhm",
        ]
        grp_col_maxima = {}
        for key in ["std_apex", "mean_fwhm", "std_fwhm", "min_fwhm", "max_fwhm"]:
            vals = [r[key] for r in group_rows if r[key] is not None]
            grp_col_maxima[key] = max(vals) if vals else 1.0

        # --- Per-group table ---
        self.rt_group_table.setSortingEnabled(False)
        self.rt_group_table.verticalHeader().setDefaultSectionSize(20)
        self.rt_group_table.setRowCount(len(group_rows))
        for row_idx, gr in enumerate(group_rows):
            _grp_color = self._get_group_color(gr["group"])
            grp_cell = QTableWidgetItem(gr["group"])
            if _grp_color:
                grp_cell.setForeground(QColor(_grp_color))
                _gf3 = grp_cell.font()
                _gf3.setBold(True)
                grp_cell.setFont(_gf3)
            self.rt_group_table.setItem(row_idx, 0, grp_cell)

            for col_idx, key in enumerate(grp_col_keys, start=1):
                val = gr[key]
                if val is not None:
                    item = NumericTableWidgetItem(f"{val:.4f}")
                    item.setData(Qt.ItemDataRole.UserRole, val)
                    if key == "mean_apex":
                        # Centred bar: offset from reference RT
                        item.setData(CenteredBarDelegate.PPM_DEVIATION_ROLE, val - ref_rt)
                        item.setData(CenteredBarDelegate.PPM_RANGE_ROLE, rt_range)
                    else:
                        col_max = grp_col_maxima.get(key, 1.0)
                        frac = (val / col_max) if col_max > 0 else 0.0
                        item.setData(BarDelegate.BAR_FRAC_ROLE, frac)
                        if _grp_color:
                            item.setData(BarDelegate.BAR_COLOR_ROLE, QColor(_grp_color))
                else:
                    item = QTableWidgetItem("N/A")
                    item.setData(Qt.ItemDataRole.UserRole, -1)
                self.rt_group_table.setItem(row_idx, col_idx, item)

        self.rt_group_table.setSortingEnabled(True)
        self.rt_group_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Copy helpers for RT statistics tables
    # ------------------------------------------------------------------

    def _copy_rt_sample_table_excel(self):
        """Copy RT per-sample table as Excel tab-delimited text."""
        tbl = self.rt_sample_table
        if tbl.rowCount() == 0:
            return
        headers = [tbl.horizontalHeaderItem(c).text() for c in range(tbl.columnCount())]
        lines = ["\t".join(headers)]
        for r in range(tbl.rowCount()):
            row_vals = [tbl.item(r, c).text() if tbl.item(r, c) else "" for c in range(tbl.columnCount())]
            lines.append("\t".join(row_vals))
        QApplication.clipboard().setText("\n".join(lines))

    def _copy_rt_sample_table_r(self):
        """Copy RT per-sample table as R dataframe code."""
        tbl = self.rt_sample_table
        if tbl.rowCount() == 0:
            return
        headers = [tbl.horizontalHeaderItem(c).text().replace(" ", "_").replace("/", "_") for c in range(tbl.columnCount())]
        rows_data = [[tbl.item(r, c).text() if tbl.item(r, c) else "" for c in range(tbl.columnCount())] for r in range(tbl.rowCount())]
        text_cols = {0, 1}
        col_lines = []
        for ci, h in enumerate(headers):
            vals = [f'"{rows_data[ri][ci]}"' if ci in text_cols else rows_data[ri][ci] for ri in range(len(rows_data))]
            col_lines.append(f"  {h} = c({', '.join(vals)})")
        r_code = "df <- data.frame(\n" + ",\n".join(col_lines) + "\n)"
        QApplication.clipboard().setText(r_code)

    def _copy_rt_group_table_excel(self):
        """Copy RT per-group table as Excel tab-delimited text."""
        tbl = self.rt_group_table
        if tbl.rowCount() == 0:
            return
        headers = [tbl.horizontalHeaderItem(c).text() for c in range(tbl.columnCount())]
        lines = ["\t".join(headers)]
        for r in range(tbl.rowCount()):
            row_vals = [tbl.item(r, c).text() if tbl.item(r, c) else "" for c in range(tbl.columnCount())]
            lines.append("\t".join(row_vals))
        QApplication.clipboard().setText("\n".join(lines))

    def _copy_rt_group_table_r(self):
        """Copy RT per-group table as R dataframe code."""
        tbl = self.rt_group_table
        if tbl.rowCount() == 0:
            return
        headers = [tbl.horizontalHeaderItem(c).text().replace(" ", "_").replace("/", "_") for c in range(tbl.columnCount())]
        rows_data = [[tbl.item(r, c).text() if tbl.item(r, c) else "" for c in range(tbl.columnCount())] for r in range(tbl.rowCount())]
        col_lines = []
        for ci, h in enumerate(headers):
            vals = [f'"{rows_data[ri][ci]}"' if ci == 0 else rows_data[ri][ci] for ri in range(len(rows_data))]
            col_lines.append(f"  {h} = c({', '.join(vals)})")
        r_code = "df <- data.frame(\n" + ",\n".join(col_lines) + "\n)"
        QApplication.clipboard().setText(r_code)

    # ------------------------------------------------------------------
    # ppm / absolute m/z display toggle
    # ------------------------------------------------------------------

    def _toggle_mz_ppm_mode(self, checked: bool):
        """Toggle between actual m/z values and ppm deviation from theoretical."""
        self._mz_ppm_mode = checked
        # Keep both checkboxes in sync
        for cb in (self.mz_ppm_toggle, self.mz_ppm_toggle_group):
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._refresh_mz_stats_display()

    def _refresh_mz_stats_display(self):
        """Repopulate m/z statistics tables in absolute or ppm-deviation mode."""
        if not getattr(self, "_mz_stats_sample_rows", None):
            return

        ppm_mode = getattr(self, "_mz_ppm_mode", False)
        theo = self.target_mz

        def to_disp(val):
            return (val - theo) / theo * 1e6 if ppm_mode else val

        def fmt(val):
            return f"{val:.2f}" if ppm_mode else f"{val:.6f}"

        # Update column headers
        if ppm_mode:
            self.mz_sample_table.setHorizontalHeaderLabels(
                [
                    "Group",
                    "Sample Name",
                    "Mean \u0394m/z (ppm)",
                    "\u0394m/z P10 (ppm)",
                    "\u0394m/z P90 (ppm)",
                    "EIC Width (Da)",
                ]
            )
            self.mz_group_table.setHorizontalHeaderLabels(
                [
                    "Group",
                    "Mean \u0394m/z (ppm)",
                    "P10 \u0394m/z (ppm)",
                    "P90 \u0394m/z (ppm)",
                    "Std \u0394m/z (ppm)",
                    "Min \u0394m/z (ppm)",
                    "Max \u0394m/z (ppm)",
                ]
            )
        else:
            self.mz_sample_table.setHorizontalHeaderLabels(
                [
                    "Group",
                    "Sample Name",
                    "Mean m/z",
                    "m/z P10",
                    "m/z P90",
                    "EIC Width (Da)",
                ]
            )
            self.mz_group_table.setHorizontalHeaderLabels(
                [
                    "Group",
                    "Mean of Mean m/z",
                    "Mean of P10 m/z",
                    "Mean of P90 m/z",
                    "Std Mean m/z",
                    "Min m/z",
                    "Max m/z",
                ]
            )

        # --- Per-sample table ---
        self.mz_sample_table.setSortingEnabled(False)
        self.mz_sample_table.verticalHeader().setDefaultSectionSize(20)
        self.mz_sample_table.setRowCount(len(self._mz_stats_sample_rows))
        for row_idx, (group, sample_name, mean_mz, p10, p90, eic_w) in enumerate(self._mz_stats_sample_rows):
            grp_color = self._get_group_color(group)

            def _colored_item(text, color=grp_color):
                it = QTableWidgetItem(text)
                if color:
                    c = QColor(color)
                    c.setAlphaF(0.5)
                    it.setBackground(c)
                return it

            self.mz_sample_table.setItem(row_idx, 0, _colored_item(group))
            self.mz_sample_table.setItem(row_idx, 1, _colored_item(sample_name))
            _ppm_range = self.mz_tolerance_ppm_spin.value()
            for col_idx, val in enumerate([mean_mz, p10, p90], start=2):
                disp = to_disp(val)
                item = QTableWidgetItem(fmt(disp))
                item.setData(Qt.ItemDataRole.UserRole, disp)
                # Always store ppm deviation for the bar regardless of display mode
                _ppm_dev = (val - theo) / theo * 1e6
                item.setData(CenteredBarDelegate.PPM_DEVIATION_ROLE, _ppm_dev)
                item.setData(CenteredBarDelegate.PPM_RANGE_ROLE, _ppm_range)
                if grp_color:
                    item.setData(CenteredBarDelegate.PPM_BAR_COLOR_ROLE, QColor(grp_color))
                self.mz_sample_table.setItem(row_idx, col_idx, item)
            eic_item = QTableWidgetItem(f"{eic_w:.6f}")
            eic_item.setData(Qt.ItemDataRole.UserRole, eic_w)
            self.mz_sample_table.setItem(row_idx, 5, eic_item)
        self.mz_sample_table.setSortingEnabled(True)
        self.mz_sample_table.resizeColumnsToContents()

        # --- Per-group table ---
        self.mz_group_table.setSortingEnabled(False)
        self.mz_group_table.verticalHeader().setDefaultSectionSize(20)
        self.mz_group_table.setRowCount(len(self._mz_stats_group_rows))
        _PPM_BAR_KEYS = {"mean_mean", "mean_p10", "mean_p90", "min_mz", "max_mz"}
        for row_idx, gr in enumerate(self._mz_stats_group_rows):
            _grp_color_g = self._get_group_color(gr["group"])
            grp_cell = QTableWidgetItem(gr["group"])
            if _grp_color_g:
                grp_cell.setForeground(QColor(_grp_color_g))
                _gf2 = grp_cell.font()
                _gf2.setBold(True)
                grp_cell.setFont(_gf2)
            self.mz_group_table.setItem(row_idx, 0, grp_cell)
            _ppm_range_g = self.mz_tolerance_ppm_spin.value()
            for col_idx, key in enumerate(
                ["mean_mean", "mean_p10", "mean_p90", "std_mean", "min_mz", "max_mz"],
                start=1,
            ):
                raw = gr[key]
                if ppm_mode and key == "std_mean":
                    disp = raw / theo * 1e6  # std is already a Da difference
                else:
                    disp = to_disp(raw)
                item = QTableWidgetItem(fmt(disp))
                item.setData(Qt.ItemDataRole.UserRole, disp)
                if key in _PPM_BAR_KEYS:
                    _ppm_dev_g = (raw - theo) / theo * 1e6
                    item.setData(CenteredBarDelegate.PPM_DEVIATION_ROLE, _ppm_dev_g)
                    item.setData(CenteredBarDelegate.PPM_RANGE_ROLE, _ppm_range_g)
                    if _grp_color_g:
                        item.setData(CenteredBarDelegate.PPM_BAR_COLOR_ROLE, QColor(_grp_color_g))
                self.mz_group_table.setItem(row_idx, col_idx, item)
        self.mz_group_table.setSortingEnabled(True)
        self.mz_group_table.resizeColumnsToContents()

    def _copy_peak_area_table_excel(self):
        """Copy peak area table data in Excel tab-delimited format"""
        if self.peak_area_table.rowCount() == 0:
            return

        # Get headers
        headers = []
        for col in range(self.peak_area_table.columnCount()):
            headers.append(self.peak_area_table.horizontalHeaderItem(col).text())

        # Get data
        data_lines = ["\t".join(headers)]
        for row in range(self.peak_area_table.rowCount()):
            row_data = []
            for col in range(self.peak_area_table.columnCount()):
                item = self.peak_area_table.item(row, col)
                if item:
                    if col == 2:  # Peak area column - use actual numeric value
                        value = item.data(Qt.ItemDataRole.UserRole)
                        row_data.append(str(value))
                    else:
                        row_data.append(item.text())
                else:
                    row_data.append("")
            data_lines.append("\t".join(row_data))

        # Copy to clipboard
        clipboard_text = "\n".join(data_lines)
        QApplication.clipboard().setText(clipboard_text)

    def _copy_peak_area_table_r(self):
        """Copy peak area table data as R dataframe code"""
        if self.peak_area_table.rowCount() == 0:
            return

        # Get headers (make them R-compatible)
        headers = []
        for col in range(self.peak_area_table.columnCount()):
            header = self.peak_area_table.horizontalHeaderItem(col).text()
            # Replace spaces and special characters for R compatibility
            r_header = header.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
            headers.append(r_header)

        # Collect data by columns
        columns_data = {header: [] for header in headers}

        for row in range(self.peak_area_table.rowCount()):
            for col in range(self.peak_area_table.columnCount()):
                item = self.peak_area_table.item(row, col)
                header = headers[col]
                if item:
                    if col == 2:  # Peak area column - use actual numeric value
                        value = item.data(Qt.ItemDataRole.UserRole)
                        columns_data[header].append(str(value))
                    else:
                        # Escape quotes and wrap strings in quotes
                        text = item.text().replace('"', '\\"')
                        columns_data[header].append(f'"{text}"')
                else:
                    columns_data[header].append('""')

        # Build R dataframe code
        r_code_lines = ["peak_area_data <- data.frame("]
        for i, (header, values) in enumerate(columns_data.items()):
            values_str = ", ".join(values)
            line = f"  {header} = c({values_str})"
            if i < len(columns_data) - 1:
                line += ","
            r_code_lines.append(line)
        r_code_lines.append(")")

        # Copy to clipboard
        clipboard_text = "\n".join(r_code_lines)
        QApplication.clipboard().setText(clipboard_text)

    def _copy_summary_stats_table_excel(self):
        """Copy summary statistics table data in Excel tab-delimited format"""
        if self.summary_stats_table.rowCount() == 0:
            return

        # Get headers
        headers = []
        for col in range(self.summary_stats_table.columnCount()):
            headers.append(self.summary_stats_table.horizontalHeaderItem(col).text())

        # Get data
        data_lines = ["\t".join(headers)]
        for row in range(self.summary_stats_table.rowCount()):
            row_data = []
            for col in range(self.summary_stats_table.columnCount()):
                item = self.summary_stats_table.item(row, col)
                if item:
                    if col > 0:  # Numeric columns - use actual numeric value
                        value = item.data(Qt.ItemDataRole.UserRole)
                        row_data.append(str(value))
                    else:
                        row_data.append(item.text())
                else:
                    row_data.append("")
            data_lines.append("\t".join(row_data))

        # Copy to clipboard
        clipboard_text = "\n".join(data_lines)
        QApplication.clipboard().setText(clipboard_text)

    def _copy_summary_stats_table_r(self):
        """Copy summary statistics table data as R dataframe code"""
        if self.summary_stats_table.rowCount() == 0:
            return

        # Get headers (make them R-compatible)
        headers = []
        for col in range(self.summary_stats_table.columnCount()):
            header = self.summary_stats_table.horizontalHeaderItem(col).text()
            # Replace spaces and special characters for R compatibility
            r_header = header.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
            headers.append(r_header)

        # Collect data by columns
        columns_data = {header: [] for header in headers}

        for row in range(self.summary_stats_table.rowCount()):
            for col in range(self.summary_stats_table.columnCount()):
                item = self.summary_stats_table.item(row, col)
                header = headers[col]
                if item:
                    if col == 0:  # Group column - wrap in quotes
                        text = item.text().replace('"', '\\"')
                        columns_data[header].append(f'"{text}"')
                    else:  # Numeric columns - use actual numeric value
                        value = item.data(Qt.ItemDataRole.UserRole)
                        columns_data[header].append(str(value))
                else:
                    if col == 0:
                        columns_data[header].append('""')
                    else:
                        columns_data[header].append("NA")

        # Build R dataframe code
        r_code_lines = ["summary_stats_data <- data.frame("]
        for i, (header, values) in enumerate(columns_data.items()):
            values_str = ", ".join(values)
            line = f"  {header} = c({values_str})"
            if i < len(columns_data) - 1:
                line += ","
            r_code_lines.append(line)
        r_code_lines.append(")")

        # Copy to clipboard
        clipboard_text = "\n".join(r_code_lines)
        QApplication.clipboard().setText(clipboard_text)

    def _copy_similarity_table_excel(self):
        """Copy peak shape similarity matrix data in Excel tab-delimited format"""
        if self.similarity_table.rowCount() == 0:
            return

        # Get column headers (sample names)
        col_headers = []
        for col in range(self.similarity_table.columnCount()):
            header_item = self.similarity_table.horizontalHeaderItem(col)
            if header_item:
                col_headers.append(header_item.text())

        # Get row headers (sample names)
        row_headers = []
        for row in range(self.similarity_table.rowCount()):
            header_item = self.similarity_table.verticalHeaderItem(row)
            if header_item:
                row_headers.append(header_item.text())

        # Build data lines
        data_lines = []

        # First line: empty cell + column headers
        data_lines.append("\t" + "\t".join(col_headers))

        # Data rows: row header + values
        for row in range(self.similarity_table.rowCount()):
            row_data = [row_headers[row] if row < len(row_headers) else ""]
            for col in range(self.similarity_table.columnCount()):
                item = self.similarity_table.item(row, col)
                if item:
                    value = item.data(Qt.ItemDataRole.UserRole)
                    if value is not None:
                        row_data.append(str(value))
                    else:
                        row_data.append(item.text())
                else:
                    row_data.append("")
            data_lines.append("\t".join(row_data))

        # Copy to clipboard
        clipboard_text = "\n".join(data_lines)
        QApplication.clipboard().setText(clipboard_text)

    def _copy_similarity_table_r(self):
        """Copy peak shape similarity matrix data as R matrix code"""
        if self.similarity_table.rowCount() == 0:
            return

        # Get sample names from headers
        sample_names = []
        for col in range(self.similarity_table.columnCount()):
            header_item = self.similarity_table.horizontalHeaderItem(col)
            if header_item:
                sample_names.append(header_item.text())

        # Collect matrix values
        matrix_values = []
        for row in range(self.similarity_table.rowCount()):
            row_values = []
            for col in range(self.similarity_table.columnCount()):
                item = self.similarity_table.item(row, col)
                if item:
                    value = item.data(Qt.ItemDataRole.UserRole)
                    if value is not None:
                        row_values.append(str(value))
                    else:
                        row_values.append("NA")
                else:
                    row_values.append("NA")
            matrix_values.append(", ".join(row_values))

        # Build R matrix code
        r_code_lines = ["# Peak shape similarity correlation matrix"]
        r_code_lines.append("similarity_matrix <- matrix(")
        r_code_lines.append("  c(" + ",\n    ".join(matrix_values) + "),")
        r_code_lines.append(f"  nrow = {self.similarity_table.rowCount()},")
        r_code_lines.append(f"  ncol = {self.similarity_table.columnCount()},")
        r_code_lines.append("  byrow = TRUE")
        r_code_lines.append(")")

        # Add row and column names
        sample_names_r = [f'"{name}"' for name in sample_names]
        r_code_lines.append(f"rownames(similarity_matrix) <- c({', '.join(sample_names_r)})")
        r_code_lines.append(f"colnames(similarity_matrix) <- c({', '.join(sample_names_r)})")

        # Copy to clipboard
        clipboard_text = "\n".join(r_code_lines)
        QApplication.clipboard().setText(clipboard_text)

    def _supersample_and_correlate(self, rt1, intensity1, rt2, intensity2):
        """
        Calculate Pearson correlation between two EIC segments using supersampling.

        This method handles different retention times between samples by:
        1. Creating a union of all unique RT values from both samples
        2. Interpolating intensities for missing RT points using linear interpolation
        3. Calculating Pearson correlation coefficient on the supersampled data

        Args:
            rt1: Retention time array for sample 1
            rt2: Retention time array for sample 2
            intensity1: Intensity array for sample 1
            intensity2: Intensity array for sample 2

        Returns:
            Pearson correlation coefficient (float)
        """
        if len(rt1) < 2 or len(rt2) < 2:
            return np.nan

        # Convert to numpy arrays
        rt1 = np.array(rt1)
        intensity1 = np.array(intensity1)
        rt2 = np.array(rt2)
        intensity2 = np.array(intensity2)

        # Get union of all unique RT values
        all_rt = np.unique(np.concatenate([rt1, rt2]))
        all_rt = np.sort(all_rt)

        # Interpolate intensities for both samples at all RT points
        # Use linear interpolation between points
        interp_intensity1 = np.interp(all_rt, rt1, intensity1)
        interp_intensity2 = np.interp(all_rt, rt2, intensity2)

        # Calculate Pearson correlation coefficient
        # Remove any NaN values
        valid_mask = ~(np.isnan(interp_intensity1) | np.isnan(interp_intensity2))
        if np.sum(valid_mask) < 2:
            return np.nan

        clean_intensity1 = interp_intensity1[valid_mask]
        clean_intensity2 = interp_intensity2[valid_mask]

        # Calculate correlation
        if len(clean_intensity1) < 2 or np.std(clean_intensity1) == 0 or np.std(clean_intensity2) == 0:
            return np.nan

        correlation = np.corrcoef(clean_intensity1, clean_intensity2)[0, 1]
        return correlation

    def _get_color_for_correlation(self, correlation):
        """
        Get a color for a correlation coefficient value.
        Green for high correlation (close to 1), red for low correlation (close to 0 or negative).

        Args:
            correlation: Correlation coefficient value (-1 to 1)

        Returns:
            QColor object
        """
        if np.isnan(correlation):
            return QColor(200, 200, 200)  # Gray for NaN

        # Clamp correlation to [-1, 1]
        correlation = max(-1, min(1, correlation))

        # Map correlation to color
        # High correlation (0.8 to 1.0) -> Green
        # Medium correlation (0.4 to 0.8) -> Yellow/Orange
        # Low correlation (-1.0 to 0.4) -> Red

        if correlation >= 0.8:
            # Green for high correlation
            # Interpolate from light green (0.8) to dark green (1.0)
            t = (correlation - 0.8) / 0.2
            green_val = int(200 + 55 * t)  # 200 to 255
            return QColor(50, green_val, 50)
        elif correlation >= 0.4:
            # Yellow/Orange for medium correlation
            # Interpolate from orange (0.4) to light green (0.8)
            t = (correlation - 0.4) / 0.4
            red_val = int(255 - 205 * t)  # 255 to 50
            green_val = int(150 + 50 * t)  # 150 to 200
            return QColor(red_val, green_val, 50)
        else:
            # Red for low correlation
            # Interpolate from dark red (-1 or 0) to orange (0.4)
            if correlation < 0:
                correlation = 0  # Treat negative correlations as very low
            t = correlation / 0.4
            red_val = int(150 + 105 * t)  # 150 to 255
            green_val = int(50 + 100 * t)  # 50 to 150
            return QColor(red_val, green_val, 50)

    def _update_similarity_table(self, start_rt, end_rt):
        """
        Update the peak shape similarity table with pairwise correlations as an n×n matrix.
        Uses hierarchical clustering to reorder rows and columns for better visualization.

        Args:
            start_rt: Start retention time for the peak region
            end_rt: End retention time for the peak region
        """
        if not self.eic_data:
            return

        # Extract peak regions for all samples
        sample_data = {}
        sample_group = {}  # sample_name -> group
        for filepath, data in self.eic_data.items():
            rt = np.array(data["rt"])
            intensity = np.array(data["intensity"])
            metadata = data["metadata"]

            if len(rt) == 0:
                continue

            # Get sample name
            sample_name = metadata.get("filename", "Unknown")
            if "." in sample_name:
                sample_name = sample_name.rsplit(".", 1)[0]

            # Track group for header colouring
            group_value = metadata.get(self.grouping_column, "Unknown")
            sample_group[sample_name] = str(group_value) if group_value is not None else "Unknown"

            # Extract peak region
            mask = (rt >= start_rt) & (rt <= end_rt)
            peak_rt = rt[mask]
            peak_intensity = intensity[mask]

            if len(peak_rt) >= 2:  # Need at least 2 points
                sample_data[sample_name] = {"rt": peak_rt, "intensity": peak_intensity}

        if len(sample_data) < 2:
            # Need at least 2 samples for comparison
            self.similarity_table.setRowCount(0)
            return

        # Calculate pairwise correlations and store in a matrix
        sample_names = natsorted(list(sample_data.keys()), key=natsort_keygen())
        n_samples = len(sample_names)

        # Create correlation matrix (n x n)
        corr_matrix = np.ones((n_samples, n_samples))  # Diagonal is 1.0 (self-correlation)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                sample1 = sample_names[i]
                sample2 = sample_names[j]

                corr = self._supersample_and_correlate(
                    sample_data[sample1]["rt"],
                    sample_data[sample1]["intensity"],
                    sample_data[sample2]["rt"],
                    sample_data[sample2]["intensity"],
                )

                # Store in both positions (matrix is symmetric)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        # Sort samples by group then sample name (within-group order)
        natsort_key = natsort_keygen()
        sorted_names = sorted(
            sample_names,
            key=lambda n: (
                natsort_key(sample_group.get(n, "Unknown")),
                natsort_key(n),
            ),
        )
        if sorted_names != sample_names:
            sort_order = [sample_names.index(n) for n in sorted_names]
            sample_names = sorted_names
            corr_matrix = corr_matrix[sort_order, :][:, sort_order]

        # Update similarity table as n×n matrix
        # Rows = samples, Columns = samples
        self.similarity_table.verticalHeader().setDefaultSectionSize(20)
        self.similarity_table.setRowCount(n_samples)
        self.similarity_table.setColumnCount(n_samples)

        # Set header labels with group colour tinting
        for idx, name in enumerate(sample_names):
            grp = sample_group.get(name, "Unknown")
            grp_color = self._get_group_color(grp)

            h_item = QTableWidgetItem(name)
            v_item = QTableWidgetItem(name)
            if grp_color:
                c = QColor(grp_color)
                c.setAlphaF(0.5)
                h_item.setBackground(c)
                v_item.setBackground(c)
            f = h_item.font()
            # f.setPointSize(6)
            h_item.setFont(f)
            v_fnt = v_item.font()
            # v_fnt.setPointSize(6)
            v_item.setFont(v_fnt)
            self.similarity_table.setHorizontalHeaderItem(idx, h_item)
            self.similarity_table.setVerticalHeaderItem(idx, v_item)

        # Compact column/row sizes — vertical header (sample names) auto-fits;
        # data columns use the same fixed width as the MSMS inter-file table (70 px)
        self.similarity_table.horizontalHeader().setDefaultSectionSize(70)
        self.similarity_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        # Disable sorting for matrix view
        self.similarity_table.setSortingEnabled(False)

        # Fill the matrix
        for i in range(n_samples):
            for j in range(n_samples):
                corr_val = corr_matrix[i, j]

                if np.isnan(corr_val):
                    item = QTableWidgetItem("NaN")
                else:
                    item = QTableWidgetItem(f"{corr_val:.4f}")
                    item.setData(Qt.ItemDataRole.UserRole, corr_val)

                    # Set background color
                    color = self._get_color_for_correlation(corr_val)
                    item.setBackground(QBrush(color))

                # Center align the text; small font for density
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                fnt = item.font()
                # fnt.setPointSize(6)
                item.setFont(fnt)
                self.similarity_table.setItem(i, j, item)

        # Fix all column widths to 70 px (same as MSMS inter-file table);
        # leave the vertical header (sample name column) to auto-size
        self.similarity_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        for col in range(n_samples):
            self.similarity_table.setColumnWidth(col, 70)
        # Restore Interactive so user can still resize manually afterwards
        self.similarity_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

    def _update_pca_plot(self, start_rt, end_rt):
        """
        Update the PCA plot using pairwise correlation coefficients as features.
        Each sample is a point, and the features are correlation coefficients with all other samples.

        Args:
            start_rt: Start retention time for the peak region
            end_rt: End retention time for the peak region
        """
        if not self.eic_data:
            return

        # Extract peak regions for all samples
        sample_data = {}
        sample_metadata = {}

        for filepath, data in self.eic_data.items():
            rt = np.array(data["rt"])
            intensity = np.array(data["intensity"])
            metadata = data["metadata"]

            if len(rt) == 0:
                continue

            # Get sample name and group
            sample_name = metadata.get("filename", "Unknown")
            if "." in sample_name:
                sample_name = sample_name.rsplit(".", 1)[0]

            group_value = metadata.get(self.grouping_column, "Unknown")
            group = str(group_value) if group_value is not None else "Unknown"

            # Extract peak region
            mask = (rt >= start_rt) & (rt <= end_rt)
            peak_rt = rt[mask]
            peak_intensity = intensity[mask]

            if len(peak_rt) >= 2:  # Need at least 2 points
                sample_data[sample_name] = {"rt": peak_rt, "intensity": peak_intensity}
                sample_metadata[sample_name] = {"group": group}

        if len(sample_data) < 3:
            # Need at least 3 samples for meaningful PCA
            self.pca_figure.clear()
            ax = self.pca_figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Need at least 3 samples for PCA",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.axis("off")
            self.pca_canvas.draw_idle()
            return

        # Calculate pairwise correlations
        sample_names = natsorted(list(sample_data.keys()), key=natsort_keygen())
        n_samples = len(sample_names)

        # Create correlation matrix (n x n)
        corr_matrix = np.ones((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                sample1 = sample_names[i]
                sample2 = sample_names[j]

                corr = self._supersample_and_correlate(
                    sample_data[sample1]["rt"],
                    sample_data[sample1]["intensity"],
                    sample_data[sample2]["rt"],
                    sample_data[sample2]["intensity"],
                )

                # Store in both positions (matrix is symmetric)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        # Handle NaN values by replacing with 0 (no correlation)
        corr_matrix_clean = np.nan_to_num(corr_matrix, nan=0.0)

        # Perform PCA on the correlation matrix
        # Each row represents a sample, each column represents correlation with another sample
        try:
            # Standardize the features (correlation coefficients)
            scaler = StandardScaler()
            corr_matrix_scaled = scaler.fit_transform(corr_matrix_clean)

            # Apply PCA
            pca = PCA(n_components=min(2, n_samples - 1))  # Get first 2 components
            pca_scores = pca.fit_transform(corr_matrix_scaled)

            # Get explained variance
            explained_var = pca.explained_variance_ratio_

            # Create the PCA plot
            self.pca_figure.clear()
            ax = self.pca_figure.add_subplot(111)

            # Get colors for each sample based on group
            colors = []
            group_color_map = {}

            # Reset annotations list for this update
            self._pca_annotations = []

            for sample_name in sample_names:
                group = sample_metadata[sample_name]["group"]

                # Get or assign color for this group
                if group not in group_color_map:
                    group_color = self._get_group_color(group)
                    if group_color:
                        color = QColor(group_color)
                        group_color_map[group] = (
                            color.red() / 255,
                            color.green() / 255,
                            color.blue() / 255,
                        )
                    else:
                        # Default color if not defined
                        group_color_map[group] = (0.5, 0.5, 0.5)

                colors.append(group_color_map[group])

            # Plot the points
            for i, (sample_name, color) in enumerate(zip(sample_names, colors)):
                if pca_scores.shape[1] >= 2:
                    x, y = pca_scores[i, 0], pca_scores[i, 1]
                else:
                    x, y = pca_scores[i, 0], 0

                # Plot the point
                scatter = ax.scatter(x, y, c=[color], s=100, alpha=0.7, edgecolors="black", linewidth=1.5)

                # Add sample name as annotation (initially invisible, shown on hover)
                # Position will be adjusted dynamically in hover handler
                annotation = ax.annotate(
                    sample_name,
                    xy=(x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                    visible=False,
                )

                # Store annotation for hover functionality
                self._pca_annotations.append((scatter, annotation, (x, y), sample_name))

            # Set labels with explained variance
            if len(explained_var) >= 2:
                ax.set_xlabel(f"PC1 ({explained_var[0] * 100:.1f}%)", fontsize=11)
                ax.set_ylabel(f"PC2 ({explained_var[1] * 100:.1f}%)", fontsize=11)
            elif len(explained_var) == 1:
                ax.set_xlabel(f"PC1 ({explained_var[0] * 100:.1f}%)", fontsize=11)
                ax.set_ylabel("PC2 (0.0%)", fontsize=11)
            else:
                ax.set_xlabel("PC1", fontsize=11)
                ax.set_ylabel("PC2", fontsize=11)

            ax.set_title("PCA of Sample Correlations", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
            ax.axvline(x=0, color="k", linestyle="--", alpha=0.3, linewidth=0.5)

            self.pca_figure.tight_layout()

            # Set up hover functionality
            self._setup_pca_hover()

            self.pca_canvas.draw_idle()

        except Exception as e:
            print(f"Error performing PCA: {e}")
            self.pca_figure.clear()
            ax = self.pca_figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Error performing PCA: {str(e)}",
                ha="center",
                va="center",
                fontsize=10,
                color="red",
            )
            ax.axis("off")
            self.pca_canvas.draw_idle()

    def _setup_pca_hover(self):
        """Set up hover functionality for PCA plot to show sample names"""
        if not hasattr(self, "_pca_annotations"):
            return

        def on_hover(event):
            if event.inaxes is None:
                return

            # Check if mouse is near any point
            for scatter, annotation, (x, y), sample_name in self._pca_annotations:
                # Calculate distance from mouse to point
                try:
                    if event.xdata is not None and event.ydata is not None:
                        distance = np.sqrt((event.xdata - x) ** 2 + (event.ydata - y) ** 2)

                        # Get axis ranges to determine relative distance
                        ax = event.inaxes
                        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        relative_distance = distance / np.sqrt(x_range**2 + y_range**2)

                        # Show annotation if close enough (within 2% of axis diagonal)
                        if relative_distance < 0.02:
                            # Determine annotation position based on point location
                            # to prevent clipping at edges
                            x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2

                            # If point is on the right half, position annotation to the left
                            # If point is on the left half, position annotation to the right
                            if x > x_center:
                                # Right side - position label to the left of point
                                annotation.set_position((x, y))
                                annotation.xyann = (-5, 5)
                                annotation.set_horizontalalignment("right")
                            else:
                                # Left side - position label to the right of point
                                annotation.set_position((x, y))
                                annotation.xyann = (5, 5)
                                annotation.set_horizontalalignment("left")

                            annotation.set_visible(True)
                        else:
                            annotation.set_visible(False)
                except (TypeError, AttributeError):
                    pass

            self.pca_canvas.draw_idle()

        # Store the connection ID to avoid multiple connections
        if hasattr(self, "_pca_hover_cid"):
            self.pca_canvas.mpl_disconnect(self._pca_hover_cid)

        self._pca_hover_cid = self.pca_canvas.mpl_connect("motion_notify_event", on_hover)

    def _update_calibration_table(self, table_data):
        """Update the calibration table with samples that have quantification data.
        Only calibration standards (samples with a Quantification entry for this compound)
        appear in the table. Unknown samples are tracked separately for plotting."""
        self.calibration_table.blockSignals(True)
        self.calibration_table.setRowCount(0)

        compound_name = self.compound_data.get("Name", "Unknown")
        files_data = self.file_manager.get_files_data()

        calibration_rows = []

        for group, sample_name, peak_area in table_data:
            # Match file by filename (with or without .mzML)
            sample_filename = f"{sample_name}.mzML"
            matching_files = files_data[files_data["filename"] == sample_filename]
            if matching_files.empty:
                matching_files = files_data[files_data["filename"] == sample_name]

            if not matching_files.empty:
                filepath = matching_files.iloc[0]["Filepath"]
                quant_data = self.file_manager.get_quantification_data(filepath, compound_name)

                if quant_data is not None:
                    true_abundance, unit = quant_data
                    dilution = self.file_manager.get_dilution_factor(filepath)
                    inj_vol = self.file_manager.get_injection_volume(filepath)
                    correction_factor = inj_vol * dilution
                    # The stored abundance IS the in-vial concentration.
                    # The actual sample concentration = vial_abundance * dilution.
                    vial_abundance = true_abundance
                    # Optionally normalize peak area by injection volume and dilution
                    corrected_area = peak_area * correction_factor if self.normalize_peak_area_checkbox.isChecked() else peak_area
                    calibration_rows.append(
                        {
                            "sample_name": sample_name,
                            "peak_area": corrected_area,
                            "vial_abundance": vial_abundance,
                            "unit": unit,
                            "dilution": dilution,
                            "inj_vol": inj_vol,
                            "correction_factor": correction_factor,
                        }
                    )

        self.calibration_table.setRowCount(len(calibration_rows))

        for i, row in enumerate(calibration_rows):
            # "Use" checkbox
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            checkbox_item.setCheckState(Qt.CheckState.Checked)
            self.calibration_table.setItem(i, 0, checkbox_item)

            self.calibration_table.setItem(i, 1, QTableWidgetItem(row["sample_name"]))

            area_item = QTableWidgetItem(f"{row['peak_area']:.2e}")
            area_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.calibration_table.setItem(i, 2, area_item)

            # Show vial concentration (true concentration / dilution)
            abundance_item = QTableWidgetItem(f"{row['vial_abundance']:.4g}")
            abundance_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.calibration_table.setItem(i, 3, abundance_item)

            unit_item = QTableWidgetItem(row["unit"])
            unit_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.calibration_table.setItem(i, 4, unit_item)

            dil_item = QTableWidgetItem(f"{row['dilution']:.4g}")
            dil_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.calibration_table.setItem(i, 5, dil_item)

            iv_item = QTableWidgetItem(f"{row['inj_vol']:.4g}")
            iv_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.calibration_table.setItem(i, 6, iv_item)

            cf_item = QTableWidgetItem(f"{row['correction_factor']:.4g}")
            cf_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.calibration_table.setItem(i, 7, cf_item)

        self.calibration_table.blockSignals(False)
        self._update_calibration_plot()

    def _on_calibration_table_changed(self, item):
        """Handle changes in calibration table (checkbox changes)"""
        if item.column() == 0:  # "Use" column
            self._update_calibration_plot()

    def _update_calibration_plot(self):
        """Update the calibration plot.
        Calibration standards (with quantification data) are shown at their actual
        (peak area, vial concentration) coordinates. Unknown samples are shown as
        triangles at their predicted position on the regression curve.
        Calibration standards excluded via checkbox are shown as gray open circles.
        """
        if not hasattr(self, "_all_peak_data"):
            return

        transform = self.axis_transform_combo.currentText()
        model_type = self.regression_model_combo.currentText()

        def apply_transform(v):
            if "Log2" in transform:
                return np.log2(max(float(v), 1e-10))
            elif "Log10" in transform:
                return np.log10(max(float(v), 1e-10))
            return float(v)

        def reverse_transform(v):
            if "Log2" in transform:
                return 2.0**v
            elif "Log10" in transform:
                return 10.0**v
            return v

        # Collect calibration points from table (checked vs unchecked)
        calib_checked_x, calib_checked_y, calib_checked_names = [], [], []
        calib_excl_x, calib_excl_y = [], []
        calib_sample_set = set()

        for i in range(self.calibration_table.rowCount()):
            name_item = self.calibration_table.item(i, 1)
            area_item = self.calibration_table.item(i, 2)
            abund_item = self.calibration_table.item(i, 3)
            checkbox_item = self.calibration_table.item(i, 0)
            if not (name_item and area_item and abund_item):
                continue
            calib_sample_set.add(name_item.text())
            try:
                x_raw = float(area_item.text())
                y_raw = float(abund_item.text())
            except ValueError:
                continue
            if checkbox_item and checkbox_item.checkState() == Qt.CheckState.Checked:
                calib_checked_x.append(x_raw)
                calib_checked_y.append(y_raw)
                calib_checked_names.append(name_item.text())
            else:
                calib_excl_x.append(x_raw)
                calib_excl_y.append(y_raw)

        if len(calib_checked_x) < 2:
            self.calibration_figure.clear()
            ax = self.calibration_figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Not enough calibration points\n(minimum: 2 selected)",
                ha="center",
                va="center",
                fontsize=12,
                color="red",
            )
            ax.axis("off")
            self.calibration_canvas.draw_idle()
            return

        # Collect unknown samples from all peak data
        unknown_x_raw, unknown_names = [], []
        files_data_plot = self.file_manager.get_files_data()
        normalize = self.normalize_peak_area_checkbox.isChecked()
        for _group, sample_name, peak_area in self._all_peak_data:
            if sample_name not in calib_sample_set:
                if normalize:
                    _sfn = f"{sample_name}.mzML"
                    _mf = files_data_plot[files_data_plot["filename"] == _sfn]
                    if _mf.empty:
                        _mf = files_data_plot[files_data_plot["filename"] == sample_name]
                    if not _mf.empty:
                        _fp = _mf.iloc[0]["Filepath"]
                        _iv = self.file_manager.get_injection_volume(_fp)
                        _dil = self.file_manager.get_dilution_factor(_fp)
                        peak_area = peak_area * _iv * _dil
                unknown_x_raw.append(peak_area)
                unknown_names.append(sample_name)

        # Transform calibration data for regression
        x_fit_pts = np.array([apply_transform(v) for v in calib_checked_x])
        y_fit_pts = np.array([apply_transform(v) for v in calib_checked_y])

        poly_degree = 1 if model_type == "Linear" else 2
        coeffs = np.polyfit(x_fit_pts, y_fit_pts, poly_degree)
        poly = np.poly1d(coeffs)

        # R²
        y_pred_pts = poly(x_fit_pts)
        ss_res = np.sum((y_fit_pts - y_pred_pts) ** 2)
        ss_tot = np.sum((y_fit_pts - np.mean(y_fit_pts)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        if poly_degree == 1:
            equation = f"y = {coeffs[0]:.4g}x + {coeffs[1]:.4g}"
        else:
            equation = f"y = {coeffs[0]:.4g}x\u00b2 + {coeffs[1]:.4g}x + {coeffs[2]:.4g}"

        # Store calibration info
        self.calibration_info = {
            "coeffs": coeffs,
            "model_type": model_type,
            "transform": transform,
            "unit": self.calibration_table.item(0, 4).text() if self.calibration_table.rowCount() > 0 else "",
        }

        # Regression line spanning ALL samples (calibration + unknown)
        all_x_raw = calib_checked_x + calib_excl_x + unknown_x_raw
        all_x_t = [apply_transform(v) for v in all_x_raw] if all_x_raw else list(x_fit_pts)
        x_line = np.linspace(min(all_x_t), max(all_x_t), 300)
        y_line = poly(x_line)

        # Axis labels
        if "Log2" in transform:
            x_label, y_label = "Log2(Peak Area)", "Log2(Vial Concentration)"
        elif "Log10" in transform:
            x_label, y_label = "Log10(Peak Area)", "Log10(Vial Concentration)"
        else:
            x_label, y_label = "Peak Area", "Vial Concentration"

        # --- Build plot ---
        self.calibration_figure.clear()
        ax = self.calibration_figure.add_subplot(111)

        # i) Unknown samples first (lowest z-order, alpha=0.5)
        if unknown_x_raw:
            unk_x_t = [apply_transform(v) for v in unknown_x_raw]
            unk_y_pred = [poly(xt) for xt in unk_x_t]
            ax.scatter(
                unk_x_t,
                unk_y_pred,
                s=100,
                color="darkorange",
                alpha=0.5,
                edgecolors="saddlebrown",
                linewidth=1.5,
                marker="^",
                label="Unknown samples (predicted)",
                zorder=1,
            )

        # ii) Regression line on top of unknown samples
        ax.plot(
            x_line,
            y_line,
            "r-",
            linewidth=2,
            label="Regression fit",
            zorder=2,
        )

        # iii) Calibration standards on top of everything
        # — excluded standards (gray open circles)
        if calib_excl_x:
            ax.scatter(
                [apply_transform(v) for v in calib_excl_x],
                [apply_transform(v) for v in calib_excl_y],
                s=80,
                color="gray",
                alpha=0.6,
                edgecolors="gray",
                linewidth=1.5,
                marker="o",
                label="Standards (excluded)",
                zorder=3,
            )

        # — used calibration standards (steel-blue filled circles)
        ax.scatter(
            x_fit_pts,
            y_fit_pts,
            s=100,
            color="steelblue",
            alpha=0.85,
            edgecolors="navy",
            linewidth=1.5,
            marker="o",
            label="Standards (used for calibration)",
            zorder=4,
        )

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(
            f"{equation}    R\u00b2 = {r_squared:.4f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        self.calibration_figure.tight_layout()
        self.calibration_canvas.draw_idle()

        # Refresh the calculated abundances table
        self._update_calculated_abundances_table()

    def _update_calculated_abundances_table(self):
        """Calculate and display abundances for ALL samples using the calibration curve.
        Calibration standards show both the actual (stored) concentration and the
        predicted concentration for comparison.  Unknown samples show only the predicted.
        The predicted concentration accounts for the sample's dilution factor.
        """
        if not hasattr(self, "calibration_info") or self.calibration_info is None:
            self.calculated_abundances_table.setRowCount(0)
            return
        if not hasattr(self, "_all_peak_data") or not self._all_peak_data:
            self.calculated_abundances_table.setRowCount(0)
            return

        compound_name = self.compound_data.get("Name", "Unknown")
        files_data = self.file_manager.get_files_data()
        transform = self.calibration_info["transform"]
        coeffs = self.calibration_info["coeffs"]
        poly = np.poly1d(coeffs)

        # Set of calibration standard names (for type classification)
        calib_sample_set = set()
        for i in range(self.calibration_table.rowCount()):
            name_item = self.calibration_table.item(i, 1)
            if name_item:
                calib_sample_set.add(name_item.text())

        def apply_transform(v):
            if "Log2" in transform:
                return np.log2(max(float(v), 1e-10))
            elif "Log10" in transform:
                return np.log10(max(float(v), 1e-10))
            return float(v)

        def reverse_transform(v):
            if "Log2" in transform:
                return 2.0**v
            elif "Log10" in transform:
                return 10.0**v
            return v

        # Colours for the Type column
        CALIB_BG = QColor(70, 130, 180, 80)  # steelblue, semi-transparent
        UNKNOWN_BG = QColor(255, 165, 0, 80)  # orange, semi-transparent

        rows = []
        for grp, sample_name, peak_area in self._all_peak_data:
            # Find file for dilution and quantification lookup
            sample_filename = f"{sample_name}.mzML"
            matching_files = files_data[files_data["filename"] == sample_filename]
            if matching_files.empty:
                matching_files = files_data[files_data["filename"] == sample_name]

            dilution = 1.0
            inj_vol = 1.0
            quant_data = None
            if not matching_files.empty:
                filepath = matching_files.iloc[0]["Filepath"]
                dilution = self.file_manager.get_dilution_factor(filepath)
                inj_vol = self.file_manager.get_injection_volume(filepath)
                quant_data = self.file_manager.get_quantification_data(filepath, compound_name)

            correction_factor = inj_vol * dilution
            # Optionally correct peak area by injection volume and dilution before prediction
            corrected_pa = peak_area * correction_factor if self.normalize_peak_area_checkbox.isChecked() else peak_area
            # Regression Y-axis = vial concentration (= stored abundance).
            # Actual sample concentration = predicted_vial_conc * dilution.
            pa_t = apply_transform(corrected_pa)
            predicted_vial_conc = reverse_transform(poly(pa_t))
            predicted_actual_conc = predicted_vial_conc * dilution

            if quant_data is not None:
                true_abundance, unit = quant_data
                sample_type = "Calibration standard"
                actual_str = f"{true_abundance:.4g}"
            else:
                unit = self.calibration_info.get("unit", "")
                sample_type = "Unknown"
                actual_str = "N/A"

            rows.append(
                {
                    "type": sample_type,
                    "group": str(grp),
                    "sample_name": sample_name,
                    "peak_area": peak_area,
                    "actual_abundance": actual_str,
                    "predicted_abundance": predicted_actual_conc,
                    "unit": unit,
                    "dilution": dilution,
                    "inj_vol": inj_vol,
                    "correction_factor": correction_factor,
                }
            )

        self.calculated_abundances_table.setRowCount(len(rows))
        for i, data in enumerate(rows):
            # Type column — colour by sample class
            type_item = QTableWidgetItem(data["type"])
            type_item.setBackground(CALIB_BG if data["type"] == "Calibration standard" else UNKNOWN_BG)
            self.calculated_abundances_table.setItem(i, 0, type_item)

            # Group column — colour by group
            group_color = self._get_group_color(data["group"])
            gc = None
            if group_color:
                gc = QColor(group_color)
                gc.setAlphaF(0.5)
            group_item = QTableWidgetItem(data["group"])
            if gc:
                group_item.setBackground(gc)
            self.calculated_abundances_table.setItem(i, 1, group_item)

            # Sample name column — colour by group
            name_item = QTableWidgetItem(data["sample_name"])
            if gc:
                name_item.setBackground(gc)
            self.calculated_abundances_table.setItem(i, 2, name_item)

            self.calculated_abundances_table.setItem(i, 3, QTableWidgetItem(f"{data['peak_area']:.2e}"))
            self.calculated_abundances_table.setItem(i, 4, QTableWidgetItem(data["actual_abundance"]))
            self.calculated_abundances_table.setItem(i, 5, QTableWidgetItem(f"{data['predicted_abundance']:.4g}"))
            self.calculated_abundances_table.setItem(i, 6, QTableWidgetItem(data["unit"]))
            self.calculated_abundances_table.setItem(i, 7, QTableWidgetItem(f"{data['dilution']:.4g}"))
            self.calculated_abundances_table.setItem(i, 8, QTableWidgetItem(f"{data['inj_vol']:.4g}"))
            self.calculated_abundances_table.setItem(i, 9, QTableWidgetItem(f"{data['correction_factor']:.4g}"))

        # Refresh the group summary table
        self._update_quant_summary_table(rows)

    def _copy_abundances_table_excel(self):
        """Copy calculated abundances table to clipboard in Excel format"""
        if self.calculated_abundances_table.rowCount() == 0:
            return

        # Build tab-delimited text
        lines = []

        # Header
        header = []
        for col in range(self.calculated_abundances_table.columnCount()):
            header.append(self.calculated_abundances_table.horizontalHeaderItem(col).text())
        lines.append("\t".join(header))

        # Data rows
        for row in range(self.calculated_abundances_table.rowCount()):
            row_data = []
            for col in range(self.calculated_abundances_table.columnCount()):
                item = self.calculated_abundances_table.item(row, col)
                row_data.append(item.text() if item else "")
            lines.append("\t".join(row_data))

        # Copy to clipboard
        clipboard_text = "\n".join(lines)
        QApplication.clipboard().setText(clipboard_text)

    def _copy_abundances_table_r(self):
        """Copy calculated abundances table to clipboard in R dataframe format"""
        if self.calculated_abundances_table.rowCount() == 0:
            return

        # Build R dataframe format
        lines = []

        # Build data by columns
        # Columns: 0=Type(str), 1=Group(str), 2=SampleName(str), 3=PeakArea(num),
        #          4=ActualAbundance(str/num), 5=PredictedAbundance(num),
        #          6=Unit(str), 7=Dilution(num), 8=InjectionVolume(num), 9=CorrectionFactor(num)
        text_cols = {0, 1, 2, 4, 6}
        for col in range(self.calculated_abundances_table.columnCount()):
            col_name = self.calculated_abundances_table.horizontalHeaderItem(col).text()
            col_name = col_name.replace(" ", "_")

            values = []
            for row in range(self.calculated_abundances_table.rowCount()):
                item = self.calculated_abundances_table.item(row, col)
                if item:
                    text = item.text()
                    if col in text_cols:
                        values.append(f'"{text}"')
                    else:
                        values.append(text)

            lines.append(f"{col_name} = c({', '.join(values)})")

        # Wrap in data.frame
        clipboard_text = f"data.frame(\n  {',\n  '.join(lines)}\n)"
        QApplication.clipboard().setText(clipboard_text)

    def _update_quant_summary_table(self, rows=None):
        """Per-group statistics of predicted concentrations.
        Accepts the rows list from _update_calculated_abundances_table directly
        to avoid fragile column-index reads from the display widget.
        """
        self.quant_summary_table.setRowCount(0)

        if not hasattr(self, "calibration_info") or self.calibration_info is None:
            return
        if not rows:
            return

        # Build group -> [predicted_abundance, ...] from the rows data directly
        group_values: dict = {}
        for data in rows:
            grp = data["group"]
            pred = data["predicted_abundance"]
            try:
                group_values.setdefault(grp, []).append(float(pred))
            except (ValueError, TypeError):
                continue

        if not group_values:
            return

        stats_data = []
        for grp, vals in group_values.items():
            arr = np.array(vals)
            stats_data.append(
                {
                    "group": grp,
                    "min": float(np.min(arr)),
                    "p10": float(np.percentile(arr, 10)),
                    "median": float(np.percentile(arr, 50)),
                    "mean": float(np.mean(arr)),
                    "p90": float(np.percentile(arr, 90)),
                    "max": float(np.max(arr)),
                }
            )

        natsort_key = natsort_keygen()
        stats_data.sort(key=lambda x: natsort_key(x["group"]))

        stat_keys = ["min", "p10", "median", "mean", "p90", "max"]
        col_maxima = {k: max((abs(s[k]) for s in stats_data), default=0) for k in stat_keys}

        self.quant_summary_table.setSortingEnabled(False)
        self.quant_summary_table.setRowCount(len(stats_data))

        for row, stats in enumerate(stats_data):
            group_color = self._get_group_color(stats["group"])
            gc = None
            if group_color:
                gc = QColor(group_color)
                gc.setAlphaF(0.5)

            # Group column
            group_item = QTableWidgetItem(stats["group"])
            if gc:
                group_item.setBackground(gc)
            self.quant_summary_table.setItem(row, 0, group_item)

            # Stat columns
            for col_idx, key in enumerate(stat_keys, 1):
                value = stats[key]
                item = NumericTableWidgetItem(f"{value:.4g}")
                item.setData(Qt.ItemDataRole.UserRole, value)
                col_max = col_maxima[key]
                frac = (abs(value) / col_max) if col_max > 0 else 0.0
                item.setData(BarDelegate.BAR_FRAC_ROLE, frac)
                if group_color:
                    item.setData(BarDelegate.BAR_COLOR_ROLE, QColor(group_color))
                self.quant_summary_table.setItem(row, col_idx, item)

        self.quant_summary_table.setSortingEnabled(True)
        self.quant_summary_table.resizeColumnsToContents()

    def _copy_quant_summary_table_excel(self):
        """Copy quantification group summaries table to clipboard as tab-delimited text."""
        if self.quant_summary_table.rowCount() == 0:
            return
        lines = []
        header = [self.quant_summary_table.horizontalHeaderItem(c).text() for c in range(self.quant_summary_table.columnCount())]
        lines.append("\t".join(header))
        for row in range(self.quant_summary_table.rowCount()):
            row_data = []
            for col in range(self.quant_summary_table.columnCount()):
                item = self.quant_summary_table.item(row, col)
                row_data.append(item.text() if item else "")
            lines.append("\t".join(row_data))
        QApplication.clipboard().setText("\n".join(lines))

    def _copy_quant_summary_table_r(self):
        """Copy quantification group summaries table to clipboard as an R data.frame."""
        if self.quant_summary_table.rowCount() == 0:
            return
        lines = []
        for col in range(self.quant_summary_table.columnCount()):
            col_name = self.quant_summary_table.horizontalHeaderItem(col).text().replace(" ", "_")
            values = []
            for row in range(self.quant_summary_table.rowCount()):
                item = self.quant_summary_table.item(row, col)
                text = item.text() if item else ""
                values.append(f'"{text}"' if col == 0 else text)
            lines.append(f"{col_name} = c({', '.join(values)})")
        QApplication.clipboard().setText(f"data.frame(\n  {',\n  '.join(lines)}\n)")

    def _calculate_peak_area_with_boundaries(self, rt_array, intensity_array, start_rt, end_rt):
        """
        Calculate peak area using proper numerical integration with boundary handling.

        This method:
        1. Includes one data point before and after the boundaries when available
        2. Uses linear interpolation to estimate intensities at exact boundary positions
        3. Applies trapezoidal rule for accurate area calculation
        4. Handles partial areas at boundaries correctly
        """
        if len(rt_array) < 2 or start_rt >= end_rt:
            return 0.0

        # Convert to numpy arrays for easier manipulation
        rt = np.array(rt_array)
        intensity = np.array(intensity_array)

        # Sort by retention time to ensure proper ordering
        sort_indices = np.argsort(rt)
        rt = rt[sort_indices]
        intensity = intensity[sort_indices]

        # Find indices for boundary handling
        # Include one point before start_rt and one point after end_rt if available
        start_idx = np.searchsorted(rt, start_rt, side="left")
        end_idx = np.searchsorted(rt, end_rt, side="right")

        # Extend the range to include boundary points
        extended_start_idx = max(0, start_idx - 1)
        extended_end_idx = min(len(rt), end_idx + 1)

        if extended_start_idx >= extended_end_idx:
            return 0.0

        # Extract the extended range
        rt_extended = rt[extended_start_idx:extended_end_idx]
        intensity_extended = intensity[extended_start_idx:extended_end_idx]

        # Create arrays for integration including interpolated boundary values
        integration_rt = []
        integration_intensity = []

        # Add point before start boundary if it exists
        if extended_start_idx < start_idx and len(rt_extended) > 0:
            integration_rt.append(rt_extended[0])
            integration_intensity.append(intensity_extended[0])

        # Interpolate intensity at start boundary if not exactly on a data point
        if start_rt not in rt_extended and len(rt_extended) >= 2:
            start_intensity = np.interp(start_rt, rt_extended, intensity_extended)
            integration_rt.append(start_rt)
            integration_intensity.append(start_intensity)

        # Add all points within boundaries
        mask_within = (rt_extended >= start_rt) & (rt_extended <= end_rt)
        integration_rt.extend(rt_extended[mask_within].tolist())
        integration_intensity.extend(intensity_extended[mask_within].tolist())

        # Interpolate intensity at end boundary if not exactly on a data point
        if end_rt not in rt_extended and len(rt_extended) >= 2:
            end_intensity = np.interp(end_rt, rt_extended, intensity_extended)
            integration_rt.append(end_rt)
            integration_intensity.append(end_intensity)

        # Add point after end boundary if it exists
        if extended_end_idx > end_idx and len(rt_extended) > 0:
            integration_rt.append(rt_extended[-1])
            integration_intensity.append(intensity_extended[-1])

        # Remove duplicates and sort
        if len(integration_rt) < 2:
            return 0.0

        # Convert to arrays and sort
        integration_rt = np.array(integration_rt)
        integration_intensity = np.array(integration_intensity)

        # Sort by RT
        sort_indices = np.argsort(integration_rt)
        integration_rt = integration_rt[sort_indices]
        integration_intensity = integration_intensity[sort_indices]

        # Remove duplicate RT values (keep first occurrence)
        unique_indices = np.unique(integration_rt, return_index=True)[1]
        integration_rt = integration_rt[unique_indices]
        integration_intensity = integration_intensity[unique_indices]

        if len(integration_rt) < 2:
            return 0.0

        # Now calculate the area only for the portion within boundaries
        # Find the exact indices for the boundary region
        boundary_mask = (integration_rt >= start_rt) & (integration_rt <= end_rt)

        if not np.any(boundary_mask):
            return 0.0

        boundary_rt = integration_rt[boundary_mask]
        boundary_intensity = integration_intensity[boundary_mask]

        # Handle case where boundaries don't align with data points
        if len(boundary_rt) == 0:
            return 0.0
        elif len(boundary_rt) == 1:
            # Single point - estimate area using neighboring points
            single_rt = boundary_rt[0]
            single_intensity = boundary_intensity[0]

            # Find time span by looking at neighboring points
            if len(integration_rt) >= 2:
                # Find the position of this point in the extended array
                pos = np.where(integration_rt == single_rt)[0]
                if len(pos) > 0:
                    pos = pos[0]
                    # Estimate time span based on neighboring points
                    if pos > 0 and pos < len(integration_rt) - 1:
                        # Use average of distances to neighbors
                        dt_before = single_rt - integration_rt[pos - 1]
                        dt_after = integration_rt[pos + 1] - single_rt
                        dt_avg = (dt_before + dt_after) / 2
                    elif pos > 0:
                        dt_avg = single_rt - integration_rt[pos - 1]
                    elif pos < len(integration_rt) - 1:
                        dt_avg = integration_rt[pos + 1] - single_rt
                    else:
                        dt_avg = 0.01  # Fallback

                    return single_intensity * dt_avg

            return 0.0
        else:
            # Multiple points - use trapezoidal integration
            return np.trapz(boundary_intensity, boundary_rt)

    def _restore_peak_boundary_lines(self):
        """Restore peak boundary lines after plot update"""
        # Check if we have stored RT values for boundaries
        if not hasattr(self, "peak_start_rt") or not hasattr(self, "peak_end_rt"):
            return

        if self.peak_start_rt is None and self.peak_end_rt is None:
            return

        # Clear the old boundary line references (they're invalid after chart update)
        self.peak_boundary_lines.clear()

        # Get axes for the new chart
        x_axis = self.chart.axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart.axes(Qt.Orientation.Vertical)[0]
        y_min, y_max = y_axis.min(), y_axis.max()

        # Recreate boundary lines using stored RT values
        stored_rts = []
        if self.peak_start_rt is not None:
            stored_rts.append(self.peak_start_rt)
        if self.peak_end_rt is not None:
            stored_rts.append(self.peak_end_rt)

        for rt_pos in stored_rts:
            # Create new boundary line
            line_series = QLineSeries()
            line_series.setName("")  # No legend entry
            line_series.setProperty("is_decoration", True)
            line_series.append(rt_pos, y_min)
            line_series.append(rt_pos, y_max)

            # Style: solid red line for peak boundaries
            pen = QPen(QColor(255, 0, 0))  # Red
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.SolidLine)
            line_series.setPen(pen)

            # Add to chart
            self.chart.addSeries(line_series)
            line_series.attachAxis(x_axis)
            line_series.attachAxis(y_axis)

            # Store new reference
            self.peak_boundary_lines.append(line_series)

        # Update info and boxplot if we have 2 lines
        if len(self.peak_boundary_lines) == 2:
            self.update_boundary_info()
            self.update_boxplot()

    def update_mz_tolerance_da(self):
        """Update Da value when ppm value changes"""
        ppm = self.mz_tolerance_ppm_spin.value()
        da_value = (self.target_mz * ppm) / 1e6

        # Temporarily disconnect only the reciprocal signal to avoid recursion
        self.mz_tolerance_da_spin.valueChanged.disconnect(self.update_mz_tolerance_ppm)
        self.mz_tolerance_da_spin.setValue(da_value)
        self.mz_tolerance_da_spin.valueChanged.connect(self.update_mz_tolerance_ppm)

    def update_mz_tolerance_ppm(self):
        """Update ppm value when Da value changes"""
        da_value = self.mz_tolerance_da_spin.value()
        ppm = (da_value * 1e6) / self.target_mz if self.target_mz > 0 else 0

        # Temporarily disconnect only the reciprocal signal to avoid recursion
        self.mz_tolerance_ppm_spin.valueChanged.disconnect(self.update_mz_tolerance_da)
        self.mz_tolerance_ppm_spin.setValue(ppm)
        self.mz_tolerance_ppm_spin.valueChanged.connect(self.update_mz_tolerance_da)
        self._notify_setting("mz_tolerance_ppm", ppm)

    def _notify_setting(self, key: str, value) -> None:
        """Persist a single EIC setting via the callback provided by the main window."""
        if self.settings_callback is not None:
            self.settings_callback(key, value)

    def create_chart(self) -> InteractiveChartView:
        """Create the chart widget"""
        self.chart = QChart()
        self.chart.setTitle("")

        # Create axes with better formatting
        self.x_axis = QValueAxis()
        self.x_axis.setTitleText("Retention Time (min)")
        self.x_axis.setLabelFormat("%.2f")  # Two decimal places
        self.x_axis.setTickCount(8)  # Reasonable number of ticks
        self.chart.addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)

        self.y_axis = QValueAxis()
        self.y_axis.setTitleText("Intensity")
        self.y_axis.setLabelFormat("%.2e")  # Scientific notation
        self.y_axis.setTickCount(6)  # Reasonable number of ticks
        self.chart.addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)

        # Use custom interactive chart view
        chart_view = InteractiveChartView(self.chart)

        # Connect context menu signal
        chart_view.contextMenuRequested.connect(self.show_context_menu)

        # Connect main x-axis range changes to sync handler
        self.x_axis.rangeChanged.connect(self._on_main_x_axis_changed)

        # Store original ranges for reset functionality
        self.original_x_range = None
        self.original_y_range = None

        return chart_view

    def reset_x_axis(self):
        """Reset the X-axis to show the full extent of all plotted data."""
        if not self.chart.series():
            return
        min_x = float("inf")
        max_x = float("-inf")
        for series in self.chart.series():
            if series.property("is_decoration"):
                continue
            for i in range(series.count()):
                pt = series.at(i)
                min_x = min(min_x, pt.x())
                max_x = max(max_x, pt.x())
        if min_x == float("inf"):
            return
        x_padding = (max_x - min_x) * 0.02
        self.chart.axes(Qt.Orientation.Horizontal)[0].setRange(min_x - x_padding, max_x + x_padding)

    def reset_y_axis(self):
        """Reset the Y-axis to show the full extent of all plotted data,
        accounting for log transform, ridge offsets, normalization, and
        group separation (all already baked into the plotted series points).
        """
        if not self.chart.series():
            return
        min_y = float("inf")
        max_y = float("-inf")
        for series in self.chart.series():
            if series.property("is_decoration"):
                continue
            for i in range(series.count()):
                pt = series.at(i)
                y = pt.y()
                if y == y:  # NaN check
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        if min_y == float("inf"):
            return

        log_y_active = hasattr(self, "log_y_cb") and self.log_y_cb.isChecked()
        ridge_active = hasattr(self, "ridge_plot_cb") and self.ridge_plot_cb.isChecked()
        normalize = self.normalize_cb.isChecked()

        if normalize and not log_y_active and not ridge_active:
            # Pure normalization: fixed 0-1 range
            self.chart.axes(Qt.Orientation.Vertical)[0].setRange(-0.05, 1.05)
            return

        y_range = max_y - min_y
        padding = y_range * 0.05 if y_range > 0 else abs(max_y) * 0.05
        if log_y_active or ridge_active:
            y_min = min_y - padding
        else:
            y_min = max(0.0, min_y - padding)
        self.chart.axes(Qt.Orientation.Vertical)[0].setRange(y_min, max_y + padding)

    def reset_view(self):
        """Reset the chart view to show all data"""
        if self.chart.series():
            # Calculate the full range of all series
            min_x = float("inf")
            max_x = float("-inf")
            min_y = float("inf")
            max_y = float("-inf")

            for series in self.chart.series():
                if series.property("is_decoration"):
                    continue
                if series.count() > 0:
                    for i in range(series.count()):
                        point = series.at(i)
                        min_x = min(min_x, point.x())
                        max_x = max(max_x, point.x())
                        min_y = min(min_y, point.y())
                        max_y = max(max_y, point.y())

            if min_x != float("inf"):
                # Add small padding to the X-axis range
                x_padding = (max_x - min_x) * 0.02

                # Set X-axis range
                x_axis = self.chart.axes(Qt.Orientation.Horizontal)[0]
                x_axis.setRange(min_x - x_padding, max_x + x_padding)

                # Only adjust Y-axis if normalization is not enabled
                # (normalization sets its own Y-axis range)
                log_y_active = hasattr(self, "log_y_cb") and self.log_y_cb.isChecked()
                ridge_active = hasattr(self, "ridge_plot_cb") and self.ridge_plot_cb.isChecked()
                normalize_only = self.normalize_cb.isChecked() and not log_y_active and not ridge_active
                if not normalize_only:
                    y_padding = (max_y - min_y) * 0.05
                    y_axis = self.chart.axes(Qt.Orientation.Vertical)[0]
                    if log_y_active or ridge_active:
                        y_axis.setRange(min_y - y_padding, max_y + y_padding)
                    else:
                        y_axis.setRange(max(0, min_y - y_padding), max_y + y_padding)

    def extract_eic_data(self):
        """Extract EIC data in a separate thread"""
        if self.target_mz == 0.0:
            QMessageBox.warning(self, "Warning", "Invalid m/z value!")
            return

        self.extract_btn.setEnabled(False)

        # Progress dialog
        self._extraction_progress = QProgressDialog(f"Extracting EIC for m/z {self.target_mz:.4f}", "Cancel", 0, 100, self)
        self._extraction_progress.setWindowTitle("EIC Extraction")
        self._extraction_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._extraction_progress.setMinimumDuration(0)
        self._extraction_progress.setAutoClose(False)
        self._extraction_progress.setAutoReset(False)
        self._extraction_progress.setValue(0)

        # Start extraction in worker thread (always extract full RT range)
        self.extraction_worker = EICExtractionWorker(
            self.file_manager,
            self.target_mz,
            self.mz_tolerance_da_spin.value(),
            None,  # Don't filter RT during extraction
            None,  # Don't filter RT during extraction
            self.eic_method_combo.currentText(),
            self.adduct,  # Pass adduct for polarity determination
            polarity=self.polarity,
        )

        self.extraction_worker.progress.connect(self._extraction_progress.setValue)
        self._extraction_progress.canceled.connect(self._on_extraction_canceled)
        self.extraction_worker.finished.connect(self.on_extraction_finished)
        self.extraction_worker.error.connect(self.on_extraction_error)

        self.extraction_worker.start()

    def _on_extraction_canceled(self):
        """User pressed Cancel in the progress dialog."""
        if hasattr(self, "extraction_worker") and self.extraction_worker.isRunning():
            self.extraction_worker.quit()
            self.extraction_worker.wait()
        self._extraction_progress.close()
        self.extract_btn.setEnabled(True)

    def on_extraction_finished(self, eic_data: dict):
        """Handle completion of EIC extraction"""
        self.eic_data = eic_data
        self._extraction_progress.setValue(100)
        self._extraction_progress.close()
        self.extract_btn.setEnabled(True)

        # Calculate group shifts
        self.calculate_group_shifts()
        self.calculate_file_shifts()

        # Populate group settings table with extracted groups
        self.populate_group_settings_table()

        # Populate sample settings table
        self.populate_sample_settings_table()

        # Update plot
        self.update_plot()

        # Update scatter plot if it exists
        if hasattr(self, "scatter_plot_view") and self.scatter_plot_view is not None:
            self.scatter_plot_view.update_scatter_plot()

    def on_extraction_error(self, error_message: str):
        """Handle EIC extraction error"""
        self._extraction_progress.close()
        self.extract_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"EIC extraction failed: {error_message}")

    def calculate_group_shifts(self):
        """Calculate RT shifts for group separation

        This method gets ALL possible groups from the entire sample matrix,
        not just the ones present in the current EIC data. This ensures
        consistent ordering and spacing even when some groups have no data.
        """
        # Get unique groups from the entire sample matrix (files_data)
        groups = set()

        if hasattr(self.file_manager, "files_data") and self.file_manager.files_data is not None:
            # Get all possible group values from the entire sample matrix
            if self.grouping_column in self.file_manager.files_data.columns:
                group_values = self.file_manager.files_data[self.grouping_column].dropna().unique()
                for value in group_values:
                    groups.add(str(value))

        # Also include groups from current EIC data (in case some aren't in files_data)
        if self.eic_data:
            for data in self.eic_data.values():
                if self.grouping_column in data["metadata"]:
                    group_value = data["metadata"][self.grouping_column]
                    groups.add(str(group_value) if group_value is not None else "Unknown")

        # Sort groups and assign shifts (using natural sort)
        sorted_groups = natsorted(groups)
        shift_amount = self.rt_shift_spin.value()

        self.group_shifts = {}
        for i, group in enumerate(sorted_groups):
            self.group_shifts[group] = i * shift_amount

    def _separation_mode(self) -> str:
        """Return the currently selected separation mode string."""
        return self.separation_mode_combo.currentText()

    def calculate_file_shifts(self):
        """Calculate RT shifts for injection order separation.

        Mode is read from ``self.separation_mode_combo``:

        * ``"By injection order"`` — rank files globally by the numeric value
          in the ``injection_order`` column (or row order as fallback).
        * ``"By group, then injection order"`` — rank files by group first
          (natural sort of the current grouping column), then by
          ``injection_order`` within each group.

        The file with rank 0 receives no shift; each subsequent rank adds one
        ``rt_shift_min`` unit.
        """
        self.file_shifts = {}

        if (
            not hasattr(self.file_manager, "files_data")
            or self.file_manager.files_data is None
            or self.file_manager.files_data.empty
            or "Filepath" not in self.file_manager.files_data.columns
        ):
            return

        df = self.file_manager.files_data
        shift_amount = self.rt_shift_spin.value()
        mode = self._separation_mode()

        # Build numeric injection order series (NaN for missing/non-numeric)
        if "injection_order" in df.columns:
            try:
                inj_order = pd.to_numeric(df["injection_order"], errors="coerce")
            except Exception:
                inj_order = pd.Series([float("nan")] * len(df), index=df.index)
        else:
            inj_order = pd.Series(range(len(df)), index=df.index, dtype=float)

        if mode == "By group, then injection order":
            # Sort by group (natural sort key), then by injection_order within group
            group_col = self.grouping_column if self.grouping_column in df.columns else "group"
            if group_col not in df.columns:
                group_col = None

            work_df = df.assign(_inj_order=inj_order)
            if group_col is not None:
                group_vals = work_df[group_col].astype(str)
                all_groups = natsorted(group_vals.unique())
                group_rank = group_vals.map({g: i for i, g in enumerate(all_groups)})
                work_df = work_df.assign(_group_rank=group_rank)
                sorted_df = work_df.sort_values(["_group_rank", "_inj_order"], na_position="last")
            else:
                sorted_df = work_df.sort_values("_inj_order", na_position="last")

            for rank, (_, row) in enumerate(sorted_df.iterrows()):
                self.file_shifts[row["Filepath"]] = rank * shift_amount
        else:
            # "By injection order": sort globally by injection_order
            sorted_df = df.assign(_inj_order=inj_order).sort_values("_inj_order", na_position="last")
            for rank, (_, row) in enumerate(sorted_df.iterrows()):
                self.file_shifts[row["Filepath"]] = rank * shift_amount

    def _get_group_color(self, group_name):
        """Get the color for a group based on the selected grouping column

        Args:
            group_name: The name/value of the group

        Returns:
            QColor or color string if found, None otherwise
        """
        # If grouping by 'color' column, use the color value directly
        if self.grouping_column == "color":
            # Find any file with this color value and return it
            for data in self.eic_data.values():
                metadata = data.get("metadata", {})
                if metadata.get("color") == group_name:
                    # The group_name IS the color in this case
                    try:
                        return QColor(group_name)
                    except:
                        pass
            return None

        # If grouping by 'group' column, use the file_manager's group colors
        if self.grouping_column == "group":
            return self.file_manager.get_group_color(group_name)

        # For other columns, try to find the corresponding group and use its color
        # Find any file that has this grouping value and get its group
        for data in self.eic_data.values():
            metadata = data.get("metadata", {})
            if str(metadata.get(self.grouping_column)) == group_name:
                # Get the actual group for this sample
                actual_group = metadata.get("group", None)
                if actual_group:
                    return self.file_manager.get_group_color(actual_group)
                break

        return None

    # ------------------------------------------------------------------
    # Ridge plot & log Y-axis helpers
    # ------------------------------------------------------------------

    def _on_ridge_plot_toggled(self, state):
        """Show/hide ridge increment controls and refresh the plot."""
        enabled = bool(state == Qt.CheckState.Checked.value or state is True or state == 2)
        self.ridge_increment_widget.setVisible(enabled)
        self.update_plot()

    def _on_ridge_slider_changed(self, value):
        """Update the increment label and refresh the plot when slider moves."""
        float_val = self._get_ridge_increment()
        self.ridge_increment_label.setText(f"{float_val:.2e}")
        self.update_plot(preserve_view=True)

    def _get_ridge_increment(self) -> float:
        """Return the current ridge increment in plot-space units."""
        return self.ridge_increment_slider.value() / 10000.0 * self._ridge_increment_max

    def _set_ridge_slider_max(self, max_val: float):
        """Update the ridge slider's scale without resetting its fractional position."""
        if max_val <= 0:
            max_val = 1.0
        old_max = self._ridge_increment_max
        old_frac = self.ridge_increment_slider.value() / 10000.0
        self._ridge_increment_max = max_val
        # If max changed significantly, keep the same fraction (proportional scaling)
        new_val = int(old_frac * 10000.0)
        # Temporarily block signals to avoid a recursive update_plot call
        self.ridge_increment_slider.blockSignals(True)
        self.ridge_increment_slider.setValue(new_val)
        self.ridge_increment_slider.blockSignals(False)
        self.ridge_increment_label.setText(f"{self._get_ridge_increment():.2e}")

    def _apply_log_transform(self, intensity: np.ndarray) -> np.ndarray:
        """Apply a signed log10 transform to an intensity array.

        Values > 0 → log10(max(v, 1.0))
        Values ≤ 0 → -log10(max(|v|, 1.0))  (for the "negative" group setting)
        """
        sign = np.sign(intensity)
        sign[sign == 0] = 1  # Treat 0 as positive
        return sign * np.log10(np.maximum(np.abs(intensity), 1.0))

    def update_plot(self, preserve_view=False):
        """Update the EIC plot

        Args:
            preserve_view: If True, maintain current zoom/pan state instead of resetting view
        """
        if not self.eic_data:
            return

        # Save current axis ranges if we need to preserve the view
        saved_x_range = None
        saved_y_range = None
        if preserve_view and self.chart.axes(Qt.Orientation.Horizontal):
            x_axis = self.chart.axes(Qt.Orientation.Horizontal)[0]
            y_axis = self.chart.axes(Qt.Orientation.Vertical)[0]
            saved_x_range = (x_axis.min(), x_axis.max())
            saved_y_range = (y_axis.min(), y_axis.max())

        # Recalculate group shifts with current RT shift value
        self.calculate_group_shifts()
        self.calculate_file_shifts()

        # Clear existing series
        self.chart.removeAllSeries()

        sep_mode = self._separation_mode()
        separate_groups = sep_mode == "By group"
        separate_injection = sep_mode in (
            "By injection order",
            "By group, then injection order",
        )
        crop_rt = self.crop_rt_cb.isChecked()
        normalize = self.normalize_cb.isChecked()

        # Get RT window if cropping is enabled
        rt_start = self.compound_data.get("RT_start_min") if crop_rt else None
        rt_end = self.compound_data.get("RT_end_min") if crop_rt else None

        # Organize data by groups first
        groups_data = {}
        for filepath, data in self.eic_data.items():
            rt = data["rt"]
            intensity = data["intensity"]
            metadata = data["metadata"]

            if len(rt) == 0 or len(intensity) == 0:
                continue

            # Apply RT cropping if enabled
            if crop_rt and rt_start is not None and rt_end is not None:
                mask = (rt >= rt_start) & (rt <= rt_end)
                rt = rt[mask]
                intensity = intensity[mask]

                if len(rt) == 0:
                    continue

            # Apply normalization if enabled
            if normalize and len(intensity) > 0:
                max_intensity = np.max(intensity)
                if max_intensity > 0:
                    intensity = intensity / max_intensity  # Normalize to 0-1 range

            # Get group value from the selected grouping column
            group_value = metadata.get(self.grouping_column, "Unknown")
            group = str(group_value) if group_value is not None else "Unknown"
            if group not in groups_data:
                groups_data[group] = []

            # Apply group shift only if separation is enabled
            rt_plot = rt.copy()
            if separate_groups:
                shift = self.group_shifts.get(group, 0.0)
                rt_plot = rt_plot + shift
            if separate_injection:
                rt_plot = rt_plot + self.file_shifts.get(filepath, 0.0)

            groups_data[group].append(
                {
                    "rt": rt_plot,
                    "intensity": intensity,
                    "metadata": metadata,
                    "filepath": filepath,
                }
            )

        # Read display-transform flags
        log_y = self.log_y_cb.isChecked()
        ridge = self.ridge_plot_cb.isChecked()

        # When ridge plot is active, recompute the slider scale from the data that
        # will be plotted (after normalization & group scaling, before ridge shift).
        if ridge:
            max_plot_intensity = 0.0
            for _grp in natsorted(groups_data.keys()):
                for _fd in groups_data.get(_grp, []):
                    _fn = _fd["metadata"].get("filename", os.path.basename(_fd["filepath"]))
                    _ss = self.sample_settings.get(_fn, {"scaling": 1.0, "plot": True, "negative": False})
                    if not _ss.get("plot", True):
                        continue
                    _ints = _fd["intensity"] * _ss.get("scaling", 1.0)
                    if _ss.get("negative", False):
                        _ints = -_ints
                    if log_y:
                        _ints = self._apply_log_transform(_ints)
                    if len(_ints) > 0:
                        max_plot_intensity = max(max_plot_intensity, float(np.max(np.abs(_ints))))
            if max_plot_intensity > 0:
                self._set_ridge_slider_max(max_plot_intensity)
            ridge_increment = self._get_ridge_increment()
        else:
            ridge_increment = 0.0

        # Ridge file counter – incremented for every file actually added to the chart
        ridge_file_index = 0

        # Create separate series for each file, but group them for legend display
        # Iterate through groups in the same sorted order as group_shifts
        sorted_groups = natsorted(groups_data.keys())

        for group_name in sorted_groups:
            # Get group color
            group_color = self._get_group_color(group_name)

            group_files = groups_data.get(group_name, [])

            # Create a placeholder legend entry when all samples in the group are hidden
            # or there is no data (keeps legend order consistent).
            any_visible = any(self.sample_settings.get(fd["metadata"].get("filename", os.path.basename(fd["filepath"])), {}).get("plot", True) for fd in group_files)
            if not any_visible or len(group_files) == 0:
                series = QLineSeries()
                if separate_groups:
                    shift = self.group_shifts.get(group_name, 0.0)
                    legend_name = f"{group_name} (+ {shift:.1f} min)"
                else:
                    legend_name = group_name
                series.setName(legend_name)

                if group_color:
                    color = QColor(group_color)
                    color.setAlpha(180)
                    pen = QPen(color)
                    pen.setWidthF(1.0)
                    series.setPen(pen)

                self.chart.addSeries(series)
                series.attachAxis(self.x_axis)
                series.attachAxis(self.y_axis)
                continue  # Skip to next group

            first_file_in_group = True

            for file_data in group_files:
                _fn = file_data["metadata"].get(
                    "filename",
                    os.path.basename(file_data["filepath"]),
                )
                # Per-sample rendering settings (authoritative source for all render decisions)
                _ss = self.sample_settings.get(_fn, {"scaling": 1.0, "plot": True, "negative": False, "line_width": 1.0})
                if not _ss.get("plot", True):
                    continue

                rt = file_data["rt"]
                intensity = file_data["intensity"]

                # Apply per-sample scaling and negation
                intensity = intensity * _ss.get("scaling", 1.0)
                if _ss.get("negative", False):
                    intensity = -intensity

                # Apply log10 transform if enabled
                if log_y:
                    intensity = self._apply_log_transform(intensity)

                # Apply ridge shift (shift this file's EIC upward by its index)
                if ridge:
                    intensity = intensity + ridge_file_index * ridge_increment
                    ridge_file_index += 1

                # Create individual series for each file
                series = QLineSeries()

                # Store the sample filename for hover tooltips
                filepath = file_data["filepath"]
                filename = file_data["metadata"].get(
                    "filename",
                    filepath.split("\\")[-1] if "\\" in filepath else filepath.split("/")[-1] if "/" in filepath else filepath,
                )

                # Store custom property for hover detection
                series.setProperty("sample_filename", filename)

                # Only the first file in each group gets the group name for legend
                if separate_injection:
                    # Per-file legend entry showing its individual shift
                    file_shift = self.file_shifts.get(filepath, 0.0)
                    name_without_ext = filename.rsplit(".", 1)[0] if "." in filename else filename
                    series.setName(f"{name_without_ext} (+ {file_shift:.1f} min)")
                elif first_file_in_group:
                    # Add shift information to legend if groups are separated
                    if separate_groups:
                        shift = self.group_shifts.get(group_name, 0.0)
                        legend_name = f"{group_name} (+ {shift:.1f} min)"
                    else:
                        legend_name = group_name
                    series.setName(legend_name)
                    first_file_in_group = False
                else:
                    series.setName("")  # Empty name won't appear in legend

                # Add data points for this file
                for x, y in zip(rt, intensity):
                    series.append(float(x) * self._rt_factor, float(y))

                # Apply group color with transparency and per-sample line width
                _effective_lw = _ss.get("line_width", 1.0)
                if group_color:
                    color = QColor(group_color)
                    color.setAlpha(180)
                    pen = QPen(color)
                    pen.setWidthF(_effective_lw)
                    series.setPen(pen)

                # Add series to chart
                self.chart.addSeries(series)
                series.attachAxis(self.x_axis)
                series.attachAxis(self.y_axis)

        # Add reference lines
        self._add_reference_lines(groups_data, separate_groups, separate_injection, sep_mode)

        # Show legend with better formatting
        legend = self.chart.legend()
        legend_pos = self.legend_position_combo.currentText() if hasattr(self, "legend_position_combo") else "Right"
        if legend_pos == "Off":
            legend.setVisible(False)
        else:
            legend.setVisible(True)
            if legend_pos == "Top":
                legend.setAlignment(Qt.AlignmentFlag.AlignTop)
            else:
                legend.setAlignment(Qt.AlignmentFlag.AlignRight)
            legend.setMarkerShape(legend.MarkerShape.MarkerShapeRectangle)

        # Hide legend markers for series with empty names
        for marker in legend.markers():
            series = marker.series()
            if series.name() == "":
                marker.setVisible(False)

        # Update Y-axis title and range based on normalization / log / ridge status
        if normalize and log_y:
            self.y_axis.setTitleText("Log\u2081\u2080(Normalized Intensity)")
        elif normalize:
            self.y_axis.setTitleText("Normalized Intensity")
        elif log_y:
            self.y_axis.setTitleText("Log\u2081\u2080(Intensity)")
        else:
            self.y_axis.setTitleText("Intensity")

        # Update x-axis title and tick-label visibility based on separation mode
        is_separated = sep_mode != "None"
        if is_separated:
            rt_shift = self.rt_shift_spin.value()
            self.x_axis.setTitleText(f"Retention Time ({self._rt_label}) + i \u00d7 {rt_shift:.1f} {self._rt_label}")
            self.x_axis.setTickCount(2)  # Only two silent anchors; labels are hidden
            self.x_axis.setLabelsVisible(False)
        else:
            self.x_axis.setTitleText(f"Retention Time ({self._rt_label})")
            self.x_axis.setTickCount(8)
            self.x_axis.setLabelsVisible(True)

        # Clear any previous group-name annotations from the chart scene
        self._clear_group_annotations()

        if normalize and not log_y and not ridge:
            # Pure normalization: fixed 0–1 range
            # Use a timer to ensure the range is applied after series are fully added
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(50, lambda: self.y_axis.setRange(-0.05, 1.05))
        else:
            # For non-normalized data (or when log/ridge is active), calculate range from actual data
            self._set_y_axis_from_data()

        # Restore saved view or reset to show all data
        from PyQt6.QtCore import QTimer

        if preserve_view and saved_x_range and saved_y_range:
            # Restore the saved axis ranges
            def restore_ranges():
                x_axis = self.chart.axes(Qt.Orientation.Horizontal)[0]
                y_axis = self.chart.axes(Qt.Orientation.Vertical)[0]
                x_axis.setRange(saved_x_range[0], saved_x_range[1])
                # For pure normalization (no log/ridge), enforce the normalized range
                if normalize and not log_y and not ridge:
                    y_axis.setRange(-0.05, 1.05)
                else:
                    y_axis.setRange(saved_y_range[0], saved_y_range[1])

            QTimer.singleShot(60, restore_ranges)
        else:
            # Automatically reset view to show all data after any changes
            QTimer.singleShot(50, self.reset_view)  # Small delay to ensure chart is fully updated

        # Schedule group-name annotations on the vertical reference lines when "By group"
        # separation is active.  The delay (120 ms) is longer than reset_view (50 ms) so
        # that axis ranges are fully settled before we map data → pixel coordinates.
        if sep_mode == "By group":
            _groups_snap = dict(groups_data)
            QTimer.singleShot(120, lambda: self._place_group_annotations(_groups_snap))

        # Update series cache for hover detection
        if hasattr(self.chart_view, "update_series_cache"):
            self.chart_view.update_series_cache()

        # Re-add peak boundary lines if they exist
        self._restore_peak_boundary_lines()

        # Refresh all extra EIC trace charts with the same view settings
        if self._extra_eic_traces:
            for trace in self._extra_eic_traces:
                self._update_extra_trace_chart(trace)
            # Re-sync x-axis range of extra traces to the main chart after reset
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(80, self._sync_all_x_axes_to_main)
            # Re-equalize y-axis label widths
            QTimer.singleShot(150, self._equalize_y_axis_widths)

    def _clear_group_annotations(self):
        """Remove all group-name text annotations and disconnect live-repositioning signals."""
        # Disconnect before removing items so signals don't fire on stale references
        try:
            self.x_axis.rangeChanged.disconnect(self._reposition_group_annotations)
        except (TypeError, RuntimeError):
            pass
        try:
            self.chart.plotAreaChanged.disconnect(self._reposition_group_annotations)
        except (TypeError, RuntimeError):
            pass
        for item in self._group_annotations:
            if item.scene() is not None:
                item.scene().removeItem(item)
        self._group_annotations.clear()
        self._group_annotation_data.clear()

    def _reposition_group_annotations(self):
        """Recompute pixel positions of all group-name labels from current axis state.

        Connected to x_axis.rangeChanged and chart.plotAreaChanged so that
        labels always stay fixed to their data-space x-coordinate when the
        user pans, zooms, or resizes the chart.
        """
        if not self._group_annotation_data:
            return
        plot_area = self.chart.plotArea()
        for text, x_rt in self._group_annotation_data:
            pos = self.chart.mapToPosition(QPointF(x_rt, 0))
            outside = pos.x() < plot_area.left() or pos.x() > plot_area.right()
            text.setVisible(not outside)
            if not outside:
                text_rect = text.boundingRect()
                text_w = text_rect.width()
                text_h = text_rect.height()
                # Centre label on the line; pin the top char near plot_area.top()
                text.setPos(pos.x() - text_h / 2, plot_area.top() + text_w + 6)

    def _place_group_annotations(self, groups_data):
        """Create rotated group-name labels along the vertical reference lines.

        Items are QGraphicsSimpleTextItem children of the chart, rotated -90°
        so they read bottom-to-top.  Actual pixel positions are computed by
        _reposition_group_annotations, which is also connected to
        rangeChanged / plotAreaChanged so they follow every zoom or pan.
        """
        from PyQt6.QtWidgets import QGraphicsSimpleTextItem

        self._clear_group_annotations()

        compound_rt = self.compound_data.get("RT_min", 0.0)
        if not (compound_rt > 0):
            return

        sorted_groups = natsorted(groups_data.keys())
        if not sorted_groups or self.chart.scene() is None:
            return

        for group_name in sorted_groups:
            x_rt = compound_rt + self.group_shifts.get(group_name, 0.0)

            text = QGraphicsSimpleTextItem(group_name, self.chart)
            font = text.font()
            font.setPointSize(8)
            text.setFont(font)

            group_color = self._get_group_color(group_name)
            if group_color:
                text.setBrush(QBrush(QColor(group_color)))

            # Rotate -90° so the label reads bottom-to-top along the reference line
            text.setRotation(-90)
            # Render on top of all chart content
            text.setZValue(10)

            self._group_annotations.append(text)
            self._group_annotation_data.append((text, x_rt))

        # Connect signals so positions update on every zoom / pan / resize
        self.x_axis.rangeChanged.connect(self._reposition_group_annotations)
        self.chart.plotAreaChanged.connect(self._reposition_group_annotations)

        # Initial positioning
        self._reposition_group_annotations()

    def _add_reference_lines(self, groups_data, separate_groups, separate_injection=False, sep_mode="None"):
        """Add reference lines to the chart"""
        # Get the current axis ranges to draw lines across the full chart
        x_axis = self.chart.axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart.axes(Qt.Orientation.Vertical)[0]

        # Calculate overall RT range for reference lines (using the final shifted values)
        min_rt, max_rt = float("inf"), float("-inf")
        for group_files in groups_data.values():
            for file_data in group_files:
                rt_values = file_data["rt"]
                if len(rt_values) > 0:
                    min_rt = min(min_rt, np.min(rt_values))
                    max_rt = max(max_rt, np.max(rt_values))

        if min_rt == float("inf"):
            return  # No data to work with

        # Extend range slightly beyond data for full coverage
        rt_range = max_rt - min_rt
        line_start = min_rt - rt_range * 0.1
        line_end = max_rt + rt_range * 0.1

        # Add horizontal baseline at intensity 0
        baseline_series = QLineSeries()
        baseline_series.setName("")  # No legend entry
        baseline_series.setProperty("is_decoration", True)
        baseline_series.append(line_start, 0)
        baseline_series.append(line_end, 0)

        # Style: solid black line
        baseline_pen = QPen(QColor(0, 0, 0))  # Black
        baseline_pen.setWidth(1)
        baseline_pen.setStyle(Qt.PenStyle.SolidLine)
        baseline_series.setPen(baseline_pen)

        self.chart.addSeries(baseline_series)
        baseline_series.attachAxis(x_axis)
        baseline_series.attachAxis(y_axis)

        # Add vertical lines using compound RT_min value
        # Use the expected RT from compound data instead of calculating from actual data
        compound_rt = self.compound_data.get("RT_min", 0.0)

        def _add_vline(reference_rt):
            vertical_line = QLineSeries()
            vertical_line.setName("")  # No legend entry
            vertical_line.setProperty("is_decoration", True)
            y_min = y_axis.min()
            y_max = y_axis.max()
            y_range = y_max - y_min
            vertical_line.append(reference_rt, y_min - y_range * 0.1)
            vertical_line.append(reference_rt, y_max + y_range * 0.1)
            vertical_pen = QPen(QColor(128, 128, 128))  # Grey
            vertical_pen.setWidth(1)
            vertical_pen.setStyle(Qt.PenStyle.DashLine)
            vertical_line.setPen(vertical_pen)
            self.chart.addSeries(vertical_line)
            vertical_line.attachAxis(x_axis)
            vertical_line.attachAxis(y_axis)

        if compound_rt > 0:
            if separate_injection:
                # One dashed line per unique total shift across all files
                seen_shifts = set()
                for group_name, group_files in groups_data.items():
                    group_shift = self.group_shifts.get(group_name, 0.0) if separate_groups else 0.0
                    for file_data in group_files:
                        total_shift = group_shift + self.file_shifts.get(file_data["filepath"], 0.0)
                        if total_shift not in seen_shifts:
                            seen_shifts.add(total_shift)
                            _add_vline(compound_rt + total_shift)
            elif separate_groups:
                for group_name in groups_data.keys():
                    _add_vline(compound_rt + self.group_shifts.get(group_name, 0.0))
            else:
                _add_vline(compound_rt)

    def _set_y_axis_from_data(self):
        """Set Y-axis range based on actual data in the chart"""
        if not self.chart.series():
            return

        # Find Y data range from all series (skip decoration/reference lines)
        min_y, max_y = float("inf"), float("-inf")

        for series in self.chart.series():
            if series.property("is_decoration"):
                continue
            if hasattr(series, "pointsVector"):
                points = series.pointsVector()
                if points:
                    y_values = [p.y() for p in points if not np.isnan(p.y())]
                    if y_values:
                        min_y = min(min_y, min(y_values))
                        max_y = max(max_y, max(y_values))

        # Set Y range with padding if we found valid data
        if min_y != float("inf") and max_y != float("-inf"):
            y_range = max_y - min_y
            padding = y_range * 0.05 if y_range > 0 else abs(max_y) * 0.05
            log_y_active = hasattr(self, "log_y_cb") and self.log_y_cb.isChecked()
            ridge_active = hasattr(self, "ridge_plot_cb") and self.ridge_plot_cb.isChecked()
            # Don't clamp to 0 when log scale or ridge plot is active (Y values may be negative)
            if log_y_active or ridge_active:
                y_min = min_y - padding
            else:
                y_min = max(0, min_y - padding)
            y_max = max_y + padding
            self.y_axis.setRange(y_min, y_max)

    def update_axes_ranges(self, force_y_auto_zoom=False, force_x_auto_zoom=False):
        """Update the axes ranges based on current data"""
        if not self.chart.series():
            return

        # Find data ranges
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for series in self.chart.series():
            if hasattr(series, "pointsVector"):
                points = series.pointsVector()
                if points:
                    x_values = [p.x() for p in points]
                    y_values = [p.y() for p in points]

                    if x_values:
                        min_x = min(min_x, min(x_values))
                        max_x = max(max_x, max(x_values))

                    if y_values:
                        min_y = min(min_y, min(y_values))
                        max_y = max(max_y, max(y_values))

        # Set X ranges with some padding
        if min_x != float("inf") and max_x != float("-inf"):
            x_range = max_x - min_x
            padding = x_range * 0.05
            x_min, x_max = min_x - padding, max_x + padding
            self.x_axis.setRange(x_min, x_max)

            # Store original range for reset functionality (only if not set or forced)
            if self.original_x_range is None or force_x_auto_zoom:
                self.original_x_range = (x_min, x_max)

        # Set Y ranges with some padding (always update if forced or no original range)
        if min_y != float("inf") and max_y != float("-inf"):
            y_range = max_y - min_y
            padding = y_range * 0.05
            y_min, y_max = max(0, min_y - padding), max_y + padding
            self.y_axis.setRange(y_min, y_max)

            # Store original range for reset functionality (always update if forced)
            if self.original_y_range is None or force_y_auto_zoom:
                self.original_y_range = (y_min, y_max)

    def reset_zoom(self):
        """Reset zoom to original view"""
        if self.original_x_range and self.original_y_range:
            self.x_axis.setRange(self.original_x_range[0], self.original_x_range[1])
            self.y_axis.setRange(self.original_y_range[0], self.original_y_range[1])
        else:
            # Fallback to recalculating ranges
            self.update_axes_ranges()

    def _parse_filter_string_type(self, filter_string: str) -> Optional[str]:
        """Apply the user-configured regex to a filter string and return the type label.

        Returns None if no pattern is configured or no match is found.
        """
        pattern = self.defaults.get("msms_filter_regex", "")
        replacement = self.defaults.get("msms_filter_replacement", "")
        if not pattern or not filter_string:
            return None
        try:
            m = re.search(pattern, filter_string)
            if m:
                return m.expand(replacement)
        except re.error:
            pass
        return None

    def _get_msms_filter_types_at_rt(self, rt_center: float, rt_window: float) -> list:
        """Return a sorted list of unique filter-string type labels for MS2 spectra near *rt_center*.

        Only works when cached data is available (memory mode). Returns an empty list otherwise.
        """
        rt_start = rt_center - rt_window
        rt_end = rt_center + rt_window
        precursor_tolerance = 0.01  # 10 mDa

        pattern = self.defaults.get("msms_filter_regex", "")
        if not pattern:
            return []
        try:
            compiled = re.compile(pattern)
        except re.error:
            return []
        replacement = self.defaults.get("msms_filter_replacement", "")

        types_seen: set = set()
        files_data = self.file_manager.get_files_data()

        for _, row in files_data.iterrows():
            filepath = row["Filepath"]
            if not (self.file_manager.keep_in_memory and filepath in self.file_manager.cached_data):
                continue
            cached = self.file_manager.cached_data[filepath]
            if not (isinstance(cached, dict) and "ms2" in cached):
                continue
            for sd in cached["ms2"]:
                if not (rt_start <= sd["scan_time"] <= rt_end):
                    continue
                pmz = sd.get("precursor_mz")
                if pmz is None or abs(pmz - self.target_mz) > precursor_tolerance:
                    continue
                fs = sd.get("filter_string") or ""
                m = compiled.search(fs)
                if m:
                    types_seen.add(m.expand(replacement))

        return sorted(types_seen)

    # ------------------------------------------------------------------
    # Extra EIC trace support
    # ------------------------------------------------------------------

    def _on_main_x_axis_changed(self, min_val: float, max_val: float):
        """Propagate main chart x-axis range to all extra trace charts."""
        if self._syncing_x_axes:
            return
        self._syncing_x_axes = True
        try:
            for trace in self._extra_eic_traces:
                x_ax = trace.get("x_axis")
                if x_ax is not None:
                    x_ax.setRange(min_val, max_val)
        finally:
            self._syncing_x_axes = False

    def _on_trace_x_axis_changed(self, min_val: float, max_val: float):
        """Propagate an extra trace x-axis range change to all other charts."""
        if self._syncing_x_axes:
            return
        self._syncing_x_axes = True
        try:
            # Sync main chart
            self.x_axis.setRange(min_val, max_val)
            # Sync all other extra traces
            for trace in self._extra_eic_traces:
                x_ax = trace.get("x_axis")
                if x_ax is not None:
                    x_ax.setRange(min_val, max_val)
        finally:
            self._syncing_x_axes = False

    def _sync_all_x_axes_to_main(self):
        """Copy the current main x-axis range to every extra trace chart."""
        if not self.chart.axes(Qt.Orientation.Horizontal):
            return
        main_x = self.chart.axes(Qt.Orientation.Horizontal)[0]
        min_val, max_val = main_x.min(), main_x.max()
        self._syncing_x_axes = True
        try:
            for trace in self._extra_eic_traces:
                x_ax = trace.get("x_axis")
                if x_ax is not None:
                    x_ax.setRange(min_val, max_val)
        finally:
            self._syncing_x_axes = False

    def _equalize_y_axis_widths(self):
        """Align the left edge of all EIC chart plot-areas so that y-axis tick
        labels are all the same width and the x-axes are visually aligned.

        Uses a two-phase approach to avoid infinite feedback loops:
          1. Reset all extra left margins to 0.
          2. After a short QTimer delay (so Qt can re-layout), measure the
             natural plotArea().left() of each chart (= y-axis label width)
             and add compensating left margins to narrower charts.
        A reentrance guard prevents recursive calls from plotAreaChanged.
        """
        if self._equalizing_y_widths:
            return
        self._equalizing_y_widths = True

        all_charts = [self.chart]
        for trace in self._extra_eic_traces:
            c = trace.get("chart")
            if c is not None:
                all_charts.append(c)

        if len(all_charts) < 2:
            # No extra traces — reset any stale padding
            self.chart.setMargins(QMargins(0, 0, 0, 0))
            self._equalizing_y_widths = False
            return

        # Phase 1 — reset all left padding so natural widths can be measured
        for c in all_charts:
            c.setMargins(QMargins(0, 0, 0, 0))

        # Phase 2 — after Qt has laid out the charts, measure and compensate
        from PyQt6.QtCore import QTimer

        def _apply_equalization():
            try:
                lefts = [c.plotArea().left() for c in all_charts]
                max_left = max(lefts)
                for c, left in zip(all_charts, lefts):
                    extra = max_left - left
                    if extra > 0.5:
                        c.setMargins(QMargins(int(extra), 0, 0, 0))
            finally:
                self._equalizing_y_widths = False

        QTimer.singleShot(60, _apply_equalization)

    def _show_add_eic_trace_dialog(self):
        """Open the Add EIC Trace dialog and, if accepted, extract and display the new trace."""
        dialog = _AddEICTraceDialog(
            adducts_data=self._adducts_data,
            ppm_default=self.defaults.get("mz_tolerance_ppm", 5.0),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        label, mz, polarity, ppm = dialog.get_result()
        self._add_extra_eic_trace(label, mz, ppm, polarity)

    def _add_extra_eic_trace(self, label: str, target_mz: float, ppm: float, polarity: str):
        """Extract EIC data for *target_mz* and add a new subplot below the main chart."""
        # Pick a color for this trace (cycle through the palette)
        color = _EXTRA_TRACE_COLORS[len(self._extra_eic_traces) % len(_EXTRA_TRACE_COLORS)]

        # Determine mz tolerance in Da
        mz_tolerance_da = target_mz * ppm * 1e-6

        # Extract EIC data for every loaded file synchronously
        eic_data = {}
        files_data = self.file_manager.get_files_data()
        total_files = len(files_data)
        progress = QProgressDialog(f"Extracting EIC for {label} (m/z {target_mz:.4f})...", "Cancel", 0, total_files, self)
        progress.setWindowTitle("Adding EIC Trace")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        failed_files = []
        for file_idx, (_, row) in enumerate(files_data.iterrows()):
            if progress.wasCanceled():
                progress.close()
                return
            progress.setValue(file_idx)
            filepath = row["Filepath"]
            try:
                rt, intensity = self.file_manager.extract_eic(
                    filepath,
                    target_mz,
                    mz_tolerance_da,
                    None,
                    None,
                    self.eic_method_combo.currentText(),
                    polarity,
                )
                eic_data[filepath] = {
                    "rt": rt,
                    "intensity": intensity,
                    "metadata": row.to_dict(),
                }
            except Exception as e:
                failed_files.append(str(row.get("filename", os.path.basename(filepath))))
                print(f"[Extra EIC trace] failed for {filepath}: {e}")
        progress.setValue(total_files)
        progress.close()

        if not eic_data:
            QMessageBox.warning(self, "No Data", f"No EIC data found for m/z {target_mz:.4f}.")
            return

        # Create chart and chart view for this trace
        trace_chart = QChart()
        trace_chart.setTitle("")
        trace_chart.legend().setVisible(False)
        trace_chart.setMargins(QMargins(0, 0, 0, 0))

        trace_x_axis = QValueAxis()
        trace_x_axis.setTitleText("Retention Time (min)")
        trace_x_axis.setLabelFormat("%.2f")
        trace_x_axis.setTickCount(8)
        trace_chart.addAxis(trace_x_axis, Qt.AlignmentFlag.AlignBottom)

        trace_y_axis = QValueAxis()
        pol_str = f" [{polarity[0].upper() if polarity else '?'}]"
        trace_y_axis.setTitleText(f"Intensity  {label}{pol_str}")
        trace_y_axis.setLabelFormat("%.2e")
        trace_y_axis.setTickCount(4)
        trace_chart.addAxis(trace_y_axis, Qt.AlignmentFlag.AlignLeft)

        trace_chart_view = InteractiveChartView(trace_chart)

        # Synchronize scroll/zoom with the main chart
        trace_x_axis.rangeChanged.connect(self._on_trace_x_axis_changed)

        # Connect context menu so the extra trace chart shows the same menu
        trace_chart_view.contextMenuRequested.connect(lambda rt_val, pos, cv=trace_chart_view: self.show_context_menu(rt_val, pos, source_chart_view=cv))

        # Assemble trace info dict
        trace_info = {
            "label": label,
            "mz": target_mz,
            "ppm": ppm,
            "polarity": polarity,
            "eic_data": eic_data,
            "chart": trace_chart,
            "chart_view": trace_chart_view,
            "x_axis": trace_x_axis,
            "y_axis": trace_y_axis,
            "color": color,
        }
        self._extra_eic_traces.append(trace_info)

        # Populate chart with data
        self._update_extra_trace_chart(trace_info)

        # Add the chart view to the charts splitter
        self._eic_charts_splitter.addWidget(trace_chart_view)

        # Move legend to Top automatically when the user adds the first extra trace
        # and the legend is currently on the Right (it would overlap extra charts).
        if len(self._extra_eic_traces) == 1 and hasattr(self, "legend_position_combo"):
            if self.legend_position_combo.currentText() == "Right":
                self.legend_position_combo.setCurrentText("Top")

        # Equalise heights: give each pane roughly equal space
        n = self._eic_charts_splitter.count()
        if n > 1:
            total = 800  # nominal pixels
            self._eic_charts_splitter.setSizes([total // n] * n)

        # Sync x-axis of the new trace to the current main chart range
        if self.chart.axes(Qt.Orientation.Horizontal):
            main_x = self.chart.axes(Qt.Orientation.Horizontal)[0]
            trace_x_axis.setRange(main_x.min(), main_x.max())

        # Align y-axis label widths after the chart has been laid out
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(200, self._equalize_y_axis_widths)

        # Warn only if some files failed
        if failed_files:
            n_loaded = len(eic_data)
            n_total = len(files_data)
            QMessageBox.warning(
                self,
                "EIC Trace - Partial Failure",
                f"Extracted {n_loaded}/{n_total} files for '{label}' (m/z {target_mz:.4f}).\n"
                f"{len(failed_files)} file(s) failed: {', '.join(failed_files[:3])}{'…' if len(failed_files) > 3 else ''}",
            )

    def _update_extra_trace_chart(self, trace_info: dict):
        """Populate a trace chart with EIC series for all files,
        applying the same view settings as the main EIC chart."""
        trace_chart: QChart = trace_info["chart"]
        trace_chart.removeAllSeries()

        trace_x_axis: QValueAxis = trace_info["x_axis"]
        trace_y_axis: QValueAxis = trace_info["y_axis"]
        eic_data: dict = trace_info["eic_data"]
        label: str = trace_info["label"]
        polarity: str = trace_info.get("polarity") or ""

        # Read the same view settings as the main plot
        sep_mode = self._separation_mode()
        separate_groups = sep_mode == "By group"
        separate_injection = sep_mode in ("By injection order", "By group, then injection order")
        crop_rt = self.crop_rt_cb.isChecked()
        normalize = self.normalize_cb.isChecked()
        log_y = self.log_y_cb.isChecked()
        ridge = self.ridge_plot_cb.isChecked()
        ridge_increment = self._get_ridge_increment() if ridge else 0.0

        rt_start_crop = self.compound_data.get("RT_start_min") if crop_rt else None
        rt_end_crop = self.compound_data.get("RT_end_min") if crop_rt else None

        # Update y-axis title to reflect transforms
        pol_str = f" [{polarity[0].upper()}]" if polarity else ""
        if normalize and log_y:
            y_title = f"Log\u2081\u2080(Norm.)  {label}{pol_str}"
        elif normalize:
            y_title = f"Norm. Intensity  {label}{pol_str}"
        elif log_y:
            y_title = f"Log\u2081\u2080(Intensity)  {label}{pol_str}"
        elif ridge:
            y_title = f"Offset Intensity  {label}{pol_str}"
        else:
            y_title = f"Intensity  {label}{pol_str}"
        trace_y_axis.setTitleText(y_title)

        # Update x-axis title exactly as the main chart does
        if sep_mode != "None":
            rt_shift = self.rt_shift_spin.value()
            trace_x_axis.setTitleText(f"Retention Time ({self._rt_label}) + i \u00d7 {rt_shift:.1f} {self._rt_label}")
            trace_x_axis.setTickCount(2)
            trace_x_axis.setLabelsVisible(False)
        else:
            trace_x_axis.setTitleText(f"Retention Time ({self._rt_label})")
            trace_x_axis.setTickCount(8)
            trace_x_axis.setLabelsVisible(True)

        all_y = []
        all_x_min = float("inf")
        all_x_max = float("-inf")
        ridge_file_index = 0

        for filepath, data in eic_data.items():
            rt = data["rt"]
            intensity = data["intensity"]
            if len(rt) == 0:
                continue

            # Apply RT cropping
            if crop_rt and rt_start_crop is not None and rt_end_crop is not None:
                mask = (rt >= rt_start_crop) & (rt <= rt_end_crop)
                rt = rt[mask]
                intensity = intensity[mask]
                if len(rt) == 0:
                    continue

            metadata = data["metadata"]
            group_value = metadata.get(self.grouping_column, "Unknown")
            group = str(group_value) if group_value is not None else "Unknown"

            # Per-sample rendering settings (authoritative source for all render decisions)
            fn = metadata.get("filename", os.path.basename(filepath))
            _ss = self.sample_settings.get(fn, {"scaling": 1.0, "plot": True, "negative": False, "line_width": 1.0})
            if not _ss.get("plot", True):
                continue

            # Apply RT shifts (same offsets as main chart)
            rt_plot = rt.copy()
            if separate_groups:
                rt_plot = rt_plot + self.group_shifts.get(group, 0.0)
            if separate_injection:
                rt_plot = rt_plot + self.file_shifts.get(filepath, 0.0)

            # Apply per-sample scaling and negation
            intensity_plot = intensity * _ss.get("scaling", 1.0)
            if _ss.get("negative", False):
                intensity_plot = -intensity_plot

            # Apply normalization
            if normalize and len(intensity_plot) > 0:
                max_i = float(np.max(intensity_plot))
                if max_i > 0:
                    intensity_plot = intensity_plot / max_i

            # Apply log transform
            if log_y:
                intensity_plot = self._apply_log_transform(intensity_plot)

            # Apply ridge shift (same per-file counter as main chart)
            if ridge:
                intensity_plot = intensity_plot + ridge_file_index * ridge_increment
                ridge_file_index += 1

            # Build QLineSeries
            series = QLineSeries()
            series.setProperty("sample_filename", fn)
            for x, y in zip(rt_plot, intensity_plot):
                series.append(float(x) * self._rt_factor, float(y))

            # Use the group color (same as main chart) with partial opacity
            group_color_raw = self._get_group_color(group)
            if group_color_raw:
                pen_color = QColor(group_color_raw) if isinstance(group_color_raw, str) else QColor(group_color_raw)
            else:
                pen_color = QColor(trace_info["color"])
            pen_color.setAlpha(180)
            lw = _ss.get("line_width", 1.0)
            pen = QPen(pen_color)
            pen.setWidthF(lw)
            series.setPen(pen)

            trace_chart.addSeries(series)
            series.attachAxis(trace_x_axis)
            series.attachAxis(trace_y_axis)

            for val in intensity_plot:
                all_y.append(float(val))
            if len(rt_plot):
                all_x_min = min(all_x_min, float(np.min(rt_plot)))
                all_x_max = max(all_x_max, float(np.max(rt_plot)))

        # Set y-axis range
        if all_y:
            y_min_data = min(all_y)
            y_max_data = max(all_y)
            y_pad = (y_max_data - y_min_data) * 0.05 if y_max_data != y_min_data else abs(y_max_data) * 0.05
            if normalize and not log_y and not ridge:
                trace_y_axis.setRange(-0.05, 1.05)
            elif log_y or ridge:
                trace_y_axis.setRange(y_min_data - y_pad, y_max_data + y_pad)
            else:
                trace_y_axis.setRange(max(0.0, y_min_data - y_pad), y_max_data + y_pad)

        if all_x_min < float("inf"):
            x_pad = (all_x_max - all_x_min) * 0.02
            trace_x_axis.setRange(all_x_min - x_pad, all_x_max + x_pad)

    def _remove_all_extra_traces(self):
        """Remove all extra EIC trace subplots."""
        for trace in self._extra_eic_traces:
            cv = trace.get("chart_view")
            if cv is not None:
                cv.setParent(None)
                cv.deleteLater()
        self._extra_eic_traces.clear()
        # Reset main chart margins now that there are no extra traces
        self.chart.setMargins(QMargins(0, 0, 0, 0))

    def _remove_extra_trace_at(self, index: int):
        """Remove a single extra EIC trace by its index (0 = first extra trace)."""
        if index < 0 or index >= len(self._extra_eic_traces):
            return
        trace = self._extra_eic_traces.pop(index)
        cv = trace.get("chart_view")
        if cv is not None:
            cv.setParent(None)
            cv.deleteLater()
        # Re-equalize (or reset if no more traces remain)
        if not self._extra_eic_traces:
            self.chart.setMargins(QMargins(0, 0, 0, 0))
        else:
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(100, self._equalize_y_axis_widths)

    def show_context_menu(self, rt_value: float, position: QPointF, source_chart_view=None):
        """Show context menu at the specified position.

        *source_chart_view* is the chart view that emitted the signal; when
        None the main self.chart_view is used for global position mapping.
        """
        context_menu = QMenu(self)
        has_extra = bool(self._extra_eic_traces)

        # Determine the m/z and polarity that MS1/MSMS lookups should use.
        # When the menu was triggered from an extra-trace chart view we use
        # that trace's m/z, not the main window's target m/z.
        _ctx_mz = self.target_mz
        _ctx_polarity = self.polarity
        if source_chart_view is not None and source_chart_view is not self.chart_view:
            for _trace in self._extra_eic_traces:
                if _trace.get("chart_view") is source_chart_view:
                    _ctx_mz = _trace.get("mz", self.target_mz)
                    _ctx_polarity = _trace.get("polarity") or self.polarity
                    break

        # Helper: temporarily swap self.target_mz/polarity, call fn, then restore.
        def _with_ctx_mz(fn, *args, **kwargs):
            _saved_mz, _saved_pol = self.target_mz, self.polarity
            self.target_mz, self.polarity = _ctx_mz, _ctx_polarity
            try:
                return fn(*args, **kwargs)
            finally:
                self.target_mz, self.polarity = _saved_mz, _saved_pol

        # Determine the m/z and polarity that MS1/MSMS lookups should use.
        # When the menu was triggered from an extra-trace chart view we want to
        # search for *that* trace's precursor m/z, not the main window's m/z.
        _ctx_mz = self.target_mz
        _ctx_polarity = self.polarity
        if source_chart_view is not None and source_chart_view is not self.chart_view:
            for _trace in self._extra_eic_traces:
                if _trace.get("chart_view") is source_chart_view:
                    _ctx_mz = _trace.get("mz", self.target_mz)
                    _ctx_polarity = _trace.get("polarity") or self.polarity
                    break

        # Helper: temporarily swap self.target_mz/polarity, call fn, then restore.
        # This lets all existing finder/viewer methods work without modification.
        def _with_ctx_mz(fn, *args, **kwargs):
            _saved_mz, _saved_pol = self.target_mz, self.polarity
            self.target_mz, self.polarity = _ctx_mz, _ctx_polarity
            try:
                return fn(*args, **kwargs)
            finally:
                self.target_mz, self.polarity = _saved_mz, _saved_pol

        # --- View submenu at the top ---
        view_menu = context_menu.addMenu("View")

        reset_view_action = QAction("Auto fit", self)
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)

        reset_x_action = QAction("Reset X-axis", self)
        reset_x_action.triggered.connect(self.reset_x_axis)
        view_menu.addAction(reset_x_action)

        reset_y_action = QAction("Reset Y-axis", self)
        reset_y_action.triggered.connect(self.reset_y_axis)
        view_menu.addAction(reset_y_action)

        # Scatter plot hidden when extra traces are present
        if not has_extra:
            view_menu.addSeparator()
            scatter_action = QAction(self.scatter_plot_menu_text, self)
            scatter_action.triggered.connect(self.toggle_scatter_plot)
            view_menu.addAction(scatter_action)

        context_menu.addSeparator()

        # Add EIC trace - show submenu if formula/adducts are available
        if has_extra or (self._adducts_data is not None and not self._adducts_data.empty and self.compound_data.get("ChemicalFormula")):
            add_trace_menu = context_menu.addMenu("Add EIC trace")
            # Top option: open dialog (always available)
            custom_action = QAction("Custom (dialog)…", self)
            custom_action.triggered.connect(self._show_add_eic_trace_dialog)
            add_trace_menu.addAction(custom_action)

            formula = self.compound_data.get("ChemicalFormula", "")
            if formula and self._adducts_data is not None and not self._adducts_data.empty:
                add_trace_menu.addSeparator()
                # Other adducts of this compound
                adducts_header = QAction("— Other adducts —", self)
                adducts_header.setEnabled(False)
                add_trace_menu.addAction(adducts_header)
                for _, adduct_row in self._adducts_data.iterrows():
                    adduct_name = str(adduct_row.get("Adduct", ""))
                    if adduct_name == self.adduct:
                        continue  # skip current adduct
                    try:
                        mz_val = calculate_mz_from_formula(formula, adduct_name, self._adducts_data)
                        if mz_val > 0:
                            if adduct_name.endswith("-"):
                                polarity = "negative"
                            else:
                                polarity = "positive"
                            act = QAction(f"{adduct_name}  (m/z {mz_val:.4f})", self)
                            ppm = self.defaults.get("mz_tolerance_ppm", 5.0)
                            act.triggered.connect(lambda checked=False, _lbl=adduct_name, _mz=mz_val, _ppm=ppm, _pol=polarity: self._add_extra_eic_trace(_lbl, _mz, _ppm, _pol))
                            add_trace_menu.addAction(act)
                    except Exception:
                        pass

                # Isotopologs: per-element submenu with values -2,-1,0,1,...n,n+1,n+2
                try:
                    parsed = parse_molecular_formula(formula)
                    # Isotopic mass shifts for common elements (monoisotopic spacing in Da)
                    _ISO_MASSES = {
                        "C": 1.003355,   # 13C - 12C
                        "N": 0.997035,   # 15N - 14N
                        "O": 2.004244,   # 18O - 16O (most common heavy isotope)
                        "S": 1.995796,   # 34S - 32S
                        "H": 1.006277,   # 2H - 1H
                    }
                    adduct_row_cur = self._adducts_data[self._adducts_data["Adduct"] == self.adduct]
                    if not adduct_row_cur.empty and any(el in parsed for el in _ISO_MASSES):
                        _, _charge, _ = adduct_mass_change(adduct_row_cur.iloc[0])
                        ppm = self.defaults.get("mz_tolerance_ppm", 5.0)

                        add_trace_menu.addSeparator()
                        iso_menu = add_trace_menu.addMenu("— Isotopologs —")

                        for elem, elem_mass_step in _ISO_MASSES.items():
                            n_elem = parsed.get(elem, 0)
                            if n_elem == 0:
                                continue
                            elem_menu = iso_menu.addMenu(f"{elem} (n={n_elem})")
                            # Include values: -2,-1, 0,1,...,n, n+1,n+2
                            iso_values = list(range(-2, n_elem + 3))
                            for iso_n in iso_values:
                                iso_mz = self.target_mz + iso_n * elem_mass_step / abs(_charge)
                                if iso_mz <= 0:
                                    continue
                                sign = "+" if iso_n >= 0 else ""
                                iso_label = f"M{sign}{iso_n}{elem} ({self.adduct})"
                                entry_text = f"M{sign}{iso_n}{elem}  (m/z {iso_mz:.4f})"
                                act = QAction(entry_text, self)
                                act.triggered.connect(
                                    lambda checked=False, _lbl=iso_label, _mz=iso_mz, _ppm=ppm, _pol=self.polarity: self._add_extra_eic_trace(_lbl, _mz, _ppm, _pol)
                                )
                                elem_menu.addAction(act)
                except Exception:
                    pass
        else:
            add_trace_action = QAction("Add EIC trace", self)
            add_trace_action.triggered.connect(self._show_add_eic_trace_dialog)
            context_menu.addAction(add_trace_action)

        if has_extra:
            # Submenu for removing individual traces
            remove_menu = context_menu.addMenu(f"Remove EIC trace ({len(self._extra_eic_traces)})")
            remove_all_action = QAction("Remove all extra traces", self)
            remove_all_action.triggered.connect(self._remove_all_extra_traces)
            remove_menu.addAction(remove_all_action)
            remove_menu.addSeparator()
            for i, trace in enumerate(self._extra_eic_traces):
                lbl = trace.get("label", f"Trace {i + 1}")
                act = remove_menu.addAction(f"Remove: {lbl}")
                act.triggered.connect(lambda checked=False, idx=i: self._remove_extra_trace_at(idx))

        # Settings template submenu
        templates = self.defaults.get("settings_templates", [])
        if templates:
            template_menu = context_menu.addMenu("Settings template")
            for tmpl in templates:
                tname = tmpl.get("name", "Unnamed")
                act = template_menu.addAction(tname)
                act.triggered.connect(lambda checked=False, t=tmpl: self._apply_settings_template(t))

        context_menu.addSeparator()

        # Add RT info
        rt_action = QAction(f"Clicked at {rt_value:.2f} min", self)
        rt_action.setEnabled(False)  # Make it non-clickable header
        context_menu.addAction(rt_action)
        context_menu.addSeparator()

        # Peak boundary options hidden when extra traces are present
        if not has_extra:
            if len(self.peak_boundary_lines) == 0:
                add_boundary_action = QAction("Add peak boundary", self)
                add_boundary_action.triggered.connect(lambda: self.add_peak_boundary(rt_value))
                context_menu.addAction(add_boundary_action)
            elif len(self.peak_boundary_lines) == 1:
                add_second_boundary_action = QAction("Add second peak boundary", self)
                add_second_boundary_action.triggered.connect(lambda: self.add_peak_boundary(rt_value))
                context_menu.addAction(add_second_boundary_action)

                remove_boundary_action = QAction("Remove peak boundaries", self)
                remove_boundary_action.triggered.connect(self.remove_peak_boundaries)
                context_menu.addAction(remove_boundary_action)
            else:  # len == 2
                remove_boundary_action = QAction("Remove peak boundaries", self)
                remove_boundary_action.triggered.connect(self.remove_peak_boundaries)
                context_menu.addAction(remove_boundary_action)

            context_menu.addSeparator()

        # Add MS1 viewing option
        ms1_action = QAction("Show MS1 spectra", self)
        ms1_action.triggered.connect(lambda: _with_ctx_mz(self.view_ms1_spectra, rt_value))
        context_menu.addAction(ms1_action)

        # Determine which MSMS actions are enabled via options
        show_msms_closest = self.defaults.get("show_msms_closest", True)
        show_msms_3s = self.defaults.get("show_msms_3s", False)
        show_msms_6s = self.defaults.get("show_msms_6s", False)
        show_msms_9s = self.defaults.get("show_msms_9s", False)
        show_msms_most_abundant_3s = self.defaults.get("show_msms_most_abundant_3s", False)
        show_msms_most_abundant_6s = self.defaults.get("show_msms_most_abundant_6s", False)
        show_msms_most_abundant_9s = self.defaults.get("show_msms_most_abundant_9s", False)
        any_msms_enabled = (
            show_msms_closest or show_msms_3s or show_msms_6s or show_msms_9s or show_msms_most_abundant_3s or show_msms_most_abundant_6s or show_msms_most_abundant_9s
        )

        if any_msms_enabled:
            # Add MSMS viewing options (unfiltered)
            if show_msms_closest:
                msms_closest_action = QAction("Show MSMS spectra", self)
                msms_closest_action.triggered.connect(lambda: _with_ctx_mz(self.view_closest_msms_spectrum, rt_value))
                context_menu.addAction(msms_closest_action)

            if show_msms_3s:
                msms_3s_action = QAction("Show MSMS spectra (±3 sec)", self)
                msms_3s_action.triggered.connect(lambda: _with_ctx_mz(self.view_msms_spectra, rt_value, 3.0 / 60.0))
                context_menu.addAction(msms_3s_action)

            if show_msms_6s:
                msms_6s_action = QAction("Show MSMS spectra (±6 sec)", self)
                msms_6s_action.triggered.connect(lambda: _with_ctx_mz(self.view_msms_spectra, rt_value, 6.0 / 60.0))
                context_menu.addAction(msms_6s_action)

            if show_msms_9s:
                msms_9s_action = QAction("Show MSMS spectra (±9 sec)", self)
                msms_9s_action.triggered.connect(lambda: _with_ctx_mz(self.view_msms_spectra, rt_value, 9.0 / 60.0))
                context_menu.addAction(msms_9s_action)

            if show_msms_most_abundant_3s:
                act = QAction("Most abundant MSMS per file (±3 sec)", self)
                act.triggered.connect(lambda: _with_ctx_mz(self.view_most_abundant_msms, rt_value, 3.0 / 60.0))
                context_menu.addAction(act)

            if show_msms_most_abundant_6s:
                act = QAction("Most abundant MSMS per file (±6 sec)", self)
                act.triggered.connect(lambda: _with_ctx_mz(self.view_most_abundant_msms, rt_value, 6.0 / 60.0))
                context_menu.addAction(act)

            if show_msms_most_abundant_9s:
                act = QAction("Most abundant MSMS per file (±9 sec)", self)
                act.triggered.connect(lambda: _with_ctx_mz(self.view_most_abundant_msms, rt_value, 9.0 / 60.0))
                context_menu.addAction(act)

            # Add per-type MSMS submenus when a filter regex is configured
            filter_types = _with_ctx_mz(self._get_msms_filter_types_at_rt, rt_value, 9.0 / 60.0)
            if filter_types:
                type_header = QAction("by filter-string", self)
                type_header.setEnabled(False)
                context_menu.addAction(type_header)
                for ftype in filter_types:
                    sub = context_menu.addMenu(f"MSMS: {ftype}")
                    if show_msms_closest:
                        closest_action = sub.addAction("Closest spectrum")
                        closest_action.triggered.connect(lambda checked=False, ft=ftype: _with_ctx_mz(self.view_closest_msms_spectrum, rt_value, filter_type=ft))
                    for secs, secs_min, enabled in (
                        (3, 3.0 / 60.0, show_msms_3s),
                        (6, 6.0 / 60.0, show_msms_6s),
                        (9, 9.0 / 60.0, show_msms_9s),
                    ):
                        if enabled:
                            action = sub.addAction(f"±{secs} seconds")
                            action.triggered.connect(lambda checked=False, ft=ftype, sw=secs_min: _with_ctx_mz(self.view_msms_spectra, rt_value, sw, filter_type=ft))

        # Show the menu — use the source chart view for correct screen coordinates
        cv = source_chart_view if source_chart_view is not None else self.chart_view
        global_pos = cv.mapToGlobal(position.toPoint())
        context_menu.exec(global_pos)

    def view_msms_spectra(self, rt_center: float, rt_window: float, filter_type: Optional[str] = None):
        """View MSMS spectra within the specified RT window.

        If *filter_type* is given, only spectra whose filter-string matches the
        configured regex and produces that type label are shown.
        """
        try:
            # Calculate RT window
            rt_start = rt_center - rt_window
            rt_end = rt_center + rt_window

            # Find MSMS spectra (optionally restricted to one filter-string type)
            msms_spectra = self.find_msms_spectra(rt_start, rt_end, filter_type=filter_type)

            if not msms_spectra:
                type_hint = f" [{filter_type}]" if filter_type else ""
                QMessageBox.information(
                    self,
                    "No MSMS Found",
                    f"No MSMS spectra{type_hint} found for m/z {self.target_mz:.4f} in RT window {rt_center:.2f} ± {rt_window * 60:.0f} s",
                )
                return

            # Open MSMS viewer window (parent=None → independent top-level window)
            msms_viewer = MSMSViewerWindow(
                msms_spectra,
                self.target_mz,
                rt_center,
                rt_window,
                self.compound_data["Name"],
                self.adduct,
                None,
                file_manager=self.file_manager,
                filter_type=filter_type,
                defaults=self.defaults,
                compound_formula=self.compound_data.get("ChemicalFormula"),
                adduct_info=self._lookup_adduct_info(),
                compound_smiles=self.compound_data.get("SMILES"),
            )
            self._msms_windows.append(msms_viewer)
            msms_viewer.destroyed.connect(lambda _, w=msms_viewer: self._msms_windows.remove(w) if w in self._msms_windows else None)
            msms_viewer.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to view MSMS spectra: {str(e)}")

    def view_closest_msms_spectrum(self, rt_center: float, filter_type: Optional[str] = None):
        """View the single closest MSMS spectrum (per file) to rt_center.

        If *filter_type* is given, only spectra whose filter-string matches the
        configured regex and produces that type label are considered.
        """
        try:
            msms_spectra = self.find_closest_msms_spectra(rt_center, filter_type=filter_type)

            if not msms_spectra:
                type_hint = f" [{filter_type}]" if filter_type else ""
                QMessageBox.information(
                    self,
                    "No MSMS Found",
                    f"No MSMS spectra{type_hint} found for m/z {self.target_mz:.4f} near RT {rt_center:.2f} min",
                )
                return

            # Derive rt_window from the actual maximum offset so the viewer
            # title / display reflects the real distance.
            max_offset = 0.0
            for entry in msms_spectra.values():
                for s in entry["spectra"]:
                    max_offset = max(max_offset, abs(s["rt"] - rt_center))

            # Warn if the nearest spectrum is more than the warning threshold from the clicked RT
            if max_offset * 60 > _MSMS_RT_WARNING_THRESHOLD_S:
                reply = QMessageBox.question(
                    self,
                    "Distant MSMS Spectrum",
                    f"The nearest MSMS spectrum found is {max_offset * 60:.1f} seconds away from "
                    f"the clicked retention time ({rt_center:.2f} min).\n\n"
                    "This spectrum may not be representative. Do you still want to view it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            # Independent top-level window (parent=None)
            msms_viewer = MSMSViewerWindow(
                msms_spectra,
                self.target_mz,
                rt_center,
                max_offset,
                self.compound_data["Name"],
                self.adduct,
                None,
                file_manager=self.file_manager,
                filter_type=filter_type,
                defaults=self.defaults,
                compound_formula=self.compound_data.get("ChemicalFormula"),
                adduct_info=self._lookup_adduct_info(),
                compound_smiles=self.compound_data.get("SMILES"),
            )
            self._msms_windows.append(msms_viewer)
            msms_viewer.destroyed.connect(lambda _, w=msms_viewer: self._msms_windows.remove(w) if w in self._msms_windows else None)
            msms_viewer.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to view closest MSMS spectrum: {str(e)}")

    def view_most_abundant_msms(self, rt_center: float, rt_window: float):
        """View the single most-abundant MSMS spectrum per file within an RT window.

        Only the spectrum with the highest precursor intensity (or highest
        total fragment intensity) in the RT window is retained per file.
        The result is passed to MSMSViewerWindow.
        """
        try:
            rt_start = rt_center - rt_window
            rt_end = rt_center + rt_window

            # Get all spectra in the window from all files
            all_spectra = self.find_msms_spectra(rt_start, rt_end)

            if not all_spectra:
                QMessageBox.information(
                    self,
                    "No MSMS Found",
                    f"No MSMS spectra found for m/z {self.target_mz:.4f} in RT window {rt_center:.2f} ± {rt_window * 60:.0f} s",
                )
                return

            # Keep only the most abundant spectrum per file
            most_abundant = {}
            for filepath, data in all_spectra.items():
                best_spec = None
                best_intensity = -1.0
                for spec in data["spectra"]:
                    # Use precursor_intensity first, fall back to max fragment intensity
                    pi = float(spec.get("precursor_intensity") or 0)
                    if pi <= 0:
                        ints = spec.get("intensity")
                        pi = float(max(ints)) if ints is not None and len(ints) > 0 else 0.0
                    if pi > best_intensity:
                        best_intensity = pi
                        best_spec = spec
                if best_spec is not None:
                    most_abundant[filepath] = {
                        "filename": data["filename"],
                        "group": data.get("group", "Unknown"),
                        "spectra": [best_spec],
                        "metadata": data.get("metadata", {}),
                    }

            if not most_abundant:
                QMessageBox.information(self, "No MSMS Found", "No valid spectra found after filtering.")
                return

            # Warn if any selected spectrum is more than the warning threshold from the clicked RT
            max_rt_offset = 0.0
            for data in most_abundant.values():
                for spec in data["spectra"]:
                    max_rt_offset = max(max_rt_offset, abs(spec["rt"] - rt_center))
            if max_rt_offset * 60 > _MSMS_RT_WARNING_THRESHOLD_S:
                reply = QMessageBox.question(
                    self,
                    "Distant MSMS Spectrum",
                    f"The most abundant MSMS spectrum found is {max_rt_offset * 60:.1f} seconds away from "
                    f"the clicked retention time ({rt_center:.2f} min).\n\n"
                    "This spectrum may not be representative. Do you still want to view it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            msms_viewer = MSMSViewerWindow(
                most_abundant,
                self.target_mz,
                rt_center,
                rt_window,
                self.compound_data["Name"],
                self.adduct,
                None,
                file_manager=self.file_manager,
                defaults=self.defaults,
                compound_formula=self.compound_data.get("ChemicalFormula"),
                adduct_info=self._lookup_adduct_info(),
                compound_smiles=self.compound_data.get("SMILES"),
            )
            self._msms_windows.append(msms_viewer)
            msms_viewer.destroyed.connect(lambda _, w=msms_viewer: self._msms_windows.remove(w) if w in self._msms_windows else None)
            msms_viewer.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to view most-abundant MSMS spectrum: {str(e)}")

    def view_ms1_spectra(self, rt_center: float):
        """View MS1 spectra at the specified RT for all files"""
        try:
            # Find MS1 spectra closest to the specified RT
            ms1_spectra = self.find_ms1_spectra(rt_center)

            if not ms1_spectra:
                QMessageBox.information(
                    self,
                    "No MS1 Found",
                    f"No MS1 spectra found at RT {rt_center:.2f} min",
                )
                return

            # Open MS1 viewer window (parent=None → independent top-level window)
            ms1_viewer = MS1ViewerWindow(
                ms1_spectra,
                self.target_mz,
                rt_center,
                self.compound_data["Name"],
                self.adduct,
                self.mz_tolerance_da_spin.value(),
                self.compound_data.get("ChemicalFormula", ""),
                None,
            )
            self._msms_windows.append(ms1_viewer)
            ms1_viewer.destroyed.connect(lambda _, w=ms1_viewer: self._msms_windows.remove(w) if w in self._msms_windows else None)
            ms1_viewer.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to view MS1 spectra: {str(e)}")

    def _is_file_visible(self, row) -> bool:
        """Return True if the file described by *row* is visible per current group/sample settings."""
        filename = str(row.get("filename", ""))
        group_value = row.get(self.grouping_column, row.get("group", "Unknown"))
        group = str(group_value) if group_value is not None else "Unknown"
        # Check group visibility
        if not self.group_settings.get(group, {"plot": True}).get("plot", True):
            return False
        # Check sample visibility
        if not self.sample_settings.get(filename, {"plot": True}).get("plot", True):
            return False
        return True

    def find_ms1_spectra(self, rt_center: float):
        """Find MS1 spectra closest to the specified RT for each file"""
        ms1_spectra = {}  # filepath -> spectrum data

        files_data = self.file_manager.get_files_data()

        for _, row in files_data.iterrows():
            filepath = row["Filepath"]
            filename = row["filename"]
            group = row.get("group", "Unknown")

            # Skip hidden groups/samples
            if not self._is_file_visible(row):
                continue

            try:
                closest_spectrum = None
                min_rt_diff = float("inf")
                all_same_polarity = []
                closest_idx = 0

                # Check if we have cached data (memory mode)
                if self.file_manager.keep_in_memory and filepath in self.file_manager.cached_data:
                    cached_file_data = self.file_manager.cached_data[filepath]

                    # Handle both old format (list) and new format (dict with ms1/ms2)
                    if isinstance(cached_file_data, dict) and "ms1" in cached_file_data:
                        ms1_spectra_data = cached_file_data["ms1"]

                        for spectrum_data in ms1_spectra_data:
                            spectrum_rt = spectrum_data["scan_time"]
                            spectrum_polarity = spectrum_data.get("polarity")

                            # Check polarity if available
                            if (
                                self.polarity
                                and spectrum_polarity
                                and not (
                                    (self.polarity.lower() in ["+", "positive", "pos"] and spectrum_polarity.lower() in ["+", "positive", "pos"])
                                    or (self.polarity.lower() in ["-", "negative", "neg"] and spectrum_polarity.lower() in ["-", "negative", "neg"])
                                )
                            ):
                                continue

                            spec_dict = {
                                "rt": spectrum_rt,
                                "mz": spectrum_data["mz"],
                                "intensity": spectrum_data["intensity"],
                                "polarity": spectrum_polarity,
                                "filename": filename,
                                "group": group,
                                "scan_id": spectrum_data.get("scan_id"),
                                "filter_string": spectrum_data.get("filter_string", "NA"),
                            }
                            all_same_polarity.append(spec_dict)

                            # Find closest spectrum to RT
                            rt_diff = abs(spectrum_rt - rt_center)
                            if rt_diff < min_rt_diff:
                                min_rt_diff = rt_diff
                                closest_spectrum = spec_dict
                                closest_idx = len(all_same_polarity) - 1

                else:
                    # Read from file to get MS1 spectra
                    reader = self.file_manager.get_mzml_reader(filepath)

                    for spectrum in reader:
                        if spectrum.ms_level == 1:  # MS1 spectra
                            spectrum_rt = spectrum.scan_time_in_minutes()

                            # Check polarity if available
                            spectrum_polarity = self.file_manager._get_spectrum_polarity(spectrum)
                            if (
                                self.polarity
                                and spectrum_polarity
                                and not (
                                    (self.polarity.lower() in ["+", "positive", "pos"] and spectrum_polarity.lower() in ["+", "positive", "pos"])
                                    or (self.polarity.lower() in ["-", "negative", "neg"] and spectrum_polarity.lower() in ["-", "negative", "neg"])
                                )
                            ):
                                continue

                            # Extract spectrum data
                            mz_array = spectrum.mz
                            intensity_array = spectrum.i

                            if len(mz_array) > 0:
                                spec_dict = {
                                    "rt": spectrum_rt,
                                    "mz": mz_array,
                                    "intensity": intensity_array,
                                    "polarity": spectrum_polarity,
                                    "filename": filename,
                                    "group": group,
                                    "scan_id": spectrum.ID,
                                    "filter_string": spectrum.get("MS:1000512", "NA"),
                                }
                                all_same_polarity.append(spec_dict)

                                # Find closest spectrum to RT
                                rt_diff = abs(spectrum_rt - rt_center)
                                if rt_diff < min_rt_diff:
                                    min_rt_diff = rt_diff
                                    closest_spectrum = spec_dict
                                    closest_idx = len(all_same_polarity) - 1

                # Add the closest spectrum if found
                if closest_spectrum is not None:
                    ms1_spectra[filepath] = {
                        "filename": filename,
                        "group": group,
                        "spectrum": closest_spectrum,
                        "all_spectra": all_same_polarity,
                        "current_index": closest_idx,
                    }

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue

        return ms1_spectra

    def view_2d_scatter_plot(self, rt_center: float):
        """Add/remove 2D scatter plot (RT vs m/z) underneath the EIC plot"""
        try:
            if hasattr(self, "scatter_plot_view") and self.scatter_plot_view is not None:
                # Remove existing scatter plot
                self.remove_scatter_plot()
            else:
                # Add scatter plot
                self.add_scatter_plot(rt_center)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to toggle 2D scatter plot: {str(e)}")

    def get_rt_range(self):
        """Get the RT range from the current chart view"""
        try:
            # Get RT range from the chart axes
            if hasattr(self, "chart_view") and self.chart_view:
                chart = self.chart_view.chart()
                axes = chart.axes(Qt.Orientation.Horizontal)
                if axes:
                    x_axis = axes[0]
                    return x_axis.min(), x_axis.max()

            # Fallback: use a reasonable default range
            return 0.0, 30.0  # Default 30-minute range

        except Exception:
            # If there's any error, return a default range
            return 0.0, 30.0

    def add_scatter_plot(self, rt_center: float):
        """Add 2D scatter plot underneath the EIC and boxplot with resizable splitter"""
        # Get RT range from the current chart view
        rt_min, rt_max = self.get_rt_range()

        # Get the right panel
        right_panel = self.eic_boxplot_splitter.parent()
        current_layout = right_panel.layout()

        # Remove the current eic_boxplot_splitter from layout temporarily
        current_layout.removeWidget(self.eic_boxplot_splitter)

        # Create a vertical splitter to hold EIC chart, boxplot, and scatter plot
        three_way_splitter = QSplitter(Qt.Orientation.Vertical)
        three_way_splitter.setChildrenCollapsible(False)  # Prevent completely collapsing widgets

        # Add the EIC chart view to the new splitter
        three_way_splitter.addWidget(self.chart_view)

        # Add the boxplot widget to the new splitter
        three_way_splitter.addWidget(self.boxplot_widget)

        # Create a widget to contain the scatter plot
        scatter_container = QWidget()
        scatter_layout = QVBoxLayout(scatter_container)
        scatter_layout.setContentsMargins(0, 5, 0, 0)
        scatter_layout.setSpacing(2)

        # Create embedded scatter plot view
        self.scatter_plot_view = EmbeddedScatterPlotView(
            self.file_manager,
            self.target_mz,
            self.mz_tolerance_da_spin.value() * 3,  # 3x the extraction window
            rt_min,
            rt_max,
            self.polarity,
            rt_center,
            self,
        )

        # Remove the fixed height from scatter plot to make it resizable
        self.scatter_plot_view.setMinimumHeight(200)
        self.scatter_plot_view.setMaximumHeight(16777215)  # Remove the fixed height constraint

        scatter_layout.addWidget(self.scatter_plot_view)

        # Add the scatter container to the new splitter
        three_way_splitter.addWidget(scatter_container)

        # Set initial splitter sizes (50% EIC, 25% boxplot, 25% scatter plot)
        three_way_splitter.setSizes([500, 250, 250])
        three_way_splitter.setStretchFactor(0, 1)  # EIC chart is stretchable
        three_way_splitter.setStretchFactor(1, 0)  # Boxplot maintains its proportion
        three_way_splitter.setStretchFactor(2, 0)  # Scatter plot maintains its proportion

        # Add the new splitter to the right panel layout
        current_layout.addWidget(three_way_splitter)

        # Store reference to new splitter for cleanup
        self.chart_scatter_splitter = three_way_splitter

        # Keep both charts in lockstep horizontally
        self._connect_scatter_x_axis_sync()

        # Update context menu text
        self.update_context_menu_text("Hide 2D scatter plot")

    def remove_scatter_plot(self):
        """Remove the 2D scatter plot from the EIC window and restore two-way splitter layout"""
        if hasattr(self, "chart_scatter_splitter") and self.chart_scatter_splitter is not None:
            # Get the right panel
            right_panel = self.chart_scatter_splitter.parent()
            layout = right_panel.layout()

            # Remove the three-way splitter from layout
            layout.removeWidget(self.chart_scatter_splitter)

            # Create a new two-way splitter for EIC and boxplot
            new_eic_boxplot_splitter = QSplitter(Qt.Orientation.Vertical)
            new_eic_boxplot_splitter.setChildrenCollapsible(False)

            # Re-parent the chart view and boxplot to the new two-way splitter
            new_eic_boxplot_splitter.addWidget(self.chart_view)
            new_eic_boxplot_splitter.addWidget(self.boxplot_widget)

            # Set initial sizes (75% EIC, 25% boxplot)
            new_eic_boxplot_splitter.setSizes([750, 250])
            new_eic_boxplot_splitter.setStretchFactor(0, 1)  # EIC chart is stretchable
            new_eic_boxplot_splitter.setStretchFactor(1, 0)  # Boxplot maintains proportion

            # Add the new two-way splitter to the layout
            layout.addWidget(new_eic_boxplot_splitter)

            # Update the reference
            self.eic_boxplot_splitter = new_eic_boxplot_splitter

            # Clean up the old three-way splitter
            self.chart_scatter_splitter.deleteLater()
            self.chart_scatter_splitter = None

        if hasattr(self, "scatter_plot_view") and self.scatter_plot_view is not None:
            self.scatter_plot_view.deleteLater()
            self.scatter_plot_view = None

        # Update context menu text
        self.update_context_menu_text("View 2D scatter plot (RT vs m/z)")

    def toggle_scatter_plot(self):
        """Show/hide the optional RT vs m/z scatter plot."""
        if self.scatter_plot_view is not None:
            self.remove_scatter_plot()
            return

        rt_min, rt_max = self.get_rt_range()
        self.add_scatter_plot((rt_min + rt_max) / 2.0)

    def _get_scatter_x_axis(self):
        """Return the scatter chart x-axis if available."""
        if self.scatter_plot_view is None or not hasattr(self.scatter_plot_view, "chart"):
            return None

        axes = self.scatter_plot_view.chart.axes(Qt.Orientation.Horizontal)
        return axes[0] if axes else None

    def _connect_scatter_x_axis_sync(self):
        """Synchronize EIC and scatter x-axis ranges in both directions."""
        scatter_x_axis = self._get_scatter_x_axis()
        if scatter_x_axis is None:
            return

        # Qt auto-disconnects deleted QObject signal connections; reconnect each time scatter is created.
        try:
            self.x_axis.rangeChanged.disconnect(self._on_eic_x_range_changed)
        except Exception:
            pass
        self.x_axis.rangeChanged.connect(self._on_eic_x_range_changed)

        try:
            scatter_x_axis.rangeChanged.disconnect(self._on_scatter_x_range_changed)
        except Exception:
            pass
        scatter_x_axis.rangeChanged.connect(self._on_scatter_x_range_changed)

        # Start with the current EIC x range.
        self._on_eic_x_range_changed(self.x_axis.min(), self.x_axis.max())

    def _on_eic_x_range_changed(self, minimum: float, maximum: float):
        """Push EIC x-axis changes to the scatter plot."""
        if self._syncing_scatter_x_axis:
            return

        scatter_x_axis = self._get_scatter_x_axis()
        if scatter_x_axis is None:
            return

        if abs(scatter_x_axis.min() - minimum) < 1e-12 and abs(scatter_x_axis.max() - maximum) < 1e-12:
            return

        self._syncing_scatter_x_axis = True
        try:
            scatter_x_axis.setRange(minimum, maximum)
        finally:
            self._syncing_scatter_x_axis = False

    def _on_scatter_x_range_changed(self, minimum: float, maximum: float):
        """Push scatter x-axis changes to the EIC plot."""
        if self._syncing_scatter_x_axis:
            return

        if abs(self.x_axis.min() - minimum) < 1e-12 and abs(self.x_axis.max() - maximum) < 1e-12:
            return

        self._syncing_scatter_x_axis = True
        try:
            self.x_axis.setRange(minimum, maximum)
        finally:
            self._syncing_scatter_x_axis = False

    def update_context_menu_text(self, new_text):
        """Update the context menu text for the scatter plot option"""
        # This will be handled in the context menu creation
        # Store the current state for context menu
        self.scatter_plot_menu_text = new_text

    def auto_add_scatter_plot(self):
        """Automatically add scatter plot after EIC data is loaded"""
        try:
            # Use the center of the visible RT range
            rt_min, rt_max = self.get_rt_range()
            rt_center = (rt_min + rt_max) / 2
            self.add_scatter_plot(rt_center)
        except Exception as e:
            print(f"Could not auto-add scatter plot: {str(e)}")

    def find_msms_spectra(self, rt_start: float, rt_end: float, filter_type: Optional[str] = None):
        """Find MSMS spectra within RT window for the target m/z and polarity.

        If *filter_type* is given only spectra whose filter string, parsed by
        the configured regex, produces that type label are returned.
        """
        msms_spectra = {}  # filepath -> list of spectra

        # Define precursor tolerance (in Da)
        precursor_tolerance = 0.01  # 10 mDa tolerance for precursor matching

        files_data = self.file_manager.get_files_data()

        for _, row in files_data.iterrows():
            filepath = row["Filepath"]
            filename = row["filename"]

            # Skip hidden groups/samples
            if not self._is_file_visible(row):
                continue

            try:
                file_msms = []

                # Check if we have cached data (memory mode)
                if self.file_manager.keep_in_memory and filepath in self.file_manager.cached_data:
                    cached_file_data = self.file_manager.cached_data[filepath]

                    # Handle both old format (list) and new format (dict with ms1/ms2)
                    if isinstance(cached_file_data, dict) and "ms2" in cached_file_data:
                        ms2_spectra = cached_file_data["ms2"]

                        for spectrum_data in ms2_spectra:
                            spectrum_rt = spectrum_data["scan_time"]
                            precursor_mz = spectrum_data.get("precursor_mz")
                            spectrum_polarity = spectrum_data.get("polarity")

                            # Check RT window
                            if not (rt_start <= spectrum_rt <= rt_end):
                                continue

                            # Check precursor m/z
                            if precursor_mz is None:
                                continue

                            # Check if precursor matches target m/z
                            if abs(precursor_mz - self.target_mz) > precursor_tolerance:
                                continue

                            # Check polarity if available
                            if not (
                                (self.polarity.lower() in ["+", "positive", "pos"] and spectrum_polarity.lower() in ["+", "positive", "pos"])
                                or (self.polarity.lower() in ["-", "negative", "neg"] and spectrum_polarity.lower() in ["-", "negative", "neg"])
                            ):
                                continue

                            # Extract spectrum data
                            mz_array = spectrum_data["mz"]
                            intensity_array = spectrum_data["intensity"]

                            if len(mz_array) > 0:
                                msms_spectrum = {
                                    "rt": spectrum_rt,
                                    "precursor_mz": precursor_mz,
                                    "precursor_intensity": spectrum_data.get("precursor_intensity", 0),
                                    "mz": mz_array,
                                    "intensity": intensity_array,
                                    "scan_id": spectrum_data.get("scan_id", f"RT_{spectrum_rt:.2f}"),
                                    "polarity": spectrum_polarity,
                                    "filter_string": spectrum_data.get("filter_string"),
                                    "collision_energy": spectrum_data.get("collision_energy"),
                                }
                                # Apply filter-type restriction
                                if filter_type is not None:
                                    parsed = self._parse_filter_string_type(msms_spectrum["filter_string"] or "")
                                    if parsed != filter_type:
                                        continue
                                file_msms.append(msms_spectrum)

                else:
                    # Read from file to get MSMS spectra
                    reader = self.file_manager.get_mzml_reader(filepath)

                    for spectrum in reader:
                        if spectrum.ms_level == 2:  # MSMS spectra
                            spectrum_rt = spectrum.scan_time_in_minutes()

                            # Check RT window
                            if not (rt_start <= spectrum_rt <= rt_end):
                                continue

                            # Check precursor m/z
                            try:
                                precursor_mz = spectrum.selected_precursors[0]["mz"] if spectrum.selected_precursors else None
                                if precursor_mz is None:
                                    continue

                                # Check if precursor matches target m/z
                                if abs(precursor_mz - self.target_mz) > precursor_tolerance:
                                    continue

                                # Check polarity if available
                                spectrum_polarity = self.file_manager._get_spectrum_polarity(spectrum)
                                if self.polarity and spectrum_polarity and self.polarity != spectrum_polarity:
                                    continue

                                # Extract spectrum data
                                mz_array = spectrum.mz
                                intensity_array = spectrum.i

                                # Try to get precursor intensity
                                precursor_intensity = 0
                                try:
                                    if spectrum.selected_precursors and len(spectrum.selected_precursors) > 0:
                                        precursor_info = spectrum.selected_precursors[0]
                                        precursor_intensity = precursor_info.get("intensity", 0)
                                        if precursor_intensity is None:
                                            precursor_intensity = 0
                                except:
                                    precursor_intensity = 0

                                if len(mz_array) > 0:
                                    spectrum_data = {
                                        "rt": spectrum_rt,
                                        "precursor_mz": precursor_mz,
                                        "precursor_intensity": precursor_intensity,
                                        "mz": mz_array,
                                        "intensity": intensity_array,
                                        "scan_id": spectrum.ID,
                                        "polarity": spectrum_polarity,
                                        "filter_string": self.file_manager._get_filter_string(spectrum),
                                        "collision_energy": self.file_manager._get_collision_energy(spectrum),
                                    }
                                    # Apply filter-type restriction
                                    if filter_type is not None:
                                        parsed = self._parse_filter_string_type(spectrum_data["filter_string"] or "")
                                        if parsed != filter_type:
                                            continue
                                    file_msms.append(spectrum_data)

                            except Exception as e:
                                print(f"Error processing spectrum in {filename}: {e}")
                                continue

                if file_msms:
                    # Sort by RT
                    file_msms.sort(key=lambda x: x["rt"])
                    msms_spectra[filepath] = {
                        "filename": filename,
                        "group": row.get("group", "Unknown"),
                        "spectra": file_msms,
                        "metadata": row.to_dict(),
                    }

            except Exception as e:
                print(f"Error reading MSMS from {filepath}: {e}")
                continue

        return msms_spectra

    def find_closest_msms_spectra(self, rt_center: float, filter_type: Optional[str] = None):
        """For each file, find the single MSMS spectrum whose RT is closest to
        rt_center and whose precursor m/z matches target_mz.

        Returns the same dict structure as find_msms_spectra so the result can
        be passed directly to MSMSViewerWindow.
        """
        msms_spectra = {}
        precursor_tolerance = 0.01

        files_data = self.file_manager.get_files_data()

        for _, row in files_data.iterrows():
            filepath = row["Filepath"]
            filename = row["filename"]

            # Skip hidden groups/samples
            if not self._is_file_visible(row):
                continue

            try:
                closest_spectrum = None
                min_rt_diff = float("inf")

                if self.file_manager.keep_in_memory and filepath in self.file_manager.cached_data:
                    cached_file_data = self.file_manager.cached_data[filepath]

                    if isinstance(cached_file_data, dict) and "ms2" in cached_file_data:
                        for spectrum_data in cached_file_data["ms2"]:
                            spectrum_rt = spectrum_data["scan_time"]
                            precursor_mz = spectrum_data.get("precursor_mz")
                            spectrum_polarity = spectrum_data.get("polarity")

                            if precursor_mz is None:
                                continue
                            if abs(precursor_mz - self.target_mz) > precursor_tolerance:
                                continue
                            if not (
                                (self.polarity.lower() in ["+", "positive", "pos"] and spectrum_polarity.lower() in ["+", "positive", "pos"])
                                or (self.polarity.lower() in ["-", "negative", "neg"] and spectrum_polarity.lower() in ["-", "negative", "neg"])
                            ):
                                continue

                            mz_array = spectrum_data["mz"]
                            intensity_array = spectrum_data["intensity"]
                            if len(mz_array) == 0:
                                continue

                            candidate = {
                                "rt": spectrum_rt,
                                "precursor_mz": precursor_mz,
                                "precursor_intensity": spectrum_data.get("precursor_intensity", 0),
                                "mz": mz_array,
                                "intensity": intensity_array,
                                "scan_id": spectrum_data.get("scan_id", f"RT_{spectrum_rt:.2f}"),
                                "polarity": spectrum_polarity,
                                "filter_string": spectrum_data.get("filter_string"),
                                "collision_energy": spectrum_data.get("collision_energy"),
                            }
                            if filter_type is not None:
                                parsed = self._parse_filter_string_type(candidate["filter_string"] or "")
                                if parsed != filter_type:
                                    continue

                            rt_diff = abs(spectrum_rt - rt_center)
                            if rt_diff < min_rt_diff:
                                min_rt_diff = rt_diff
                                closest_spectrum = candidate

                else:
                    reader = self.file_manager.get_mzml_reader(filepath)

                    for spectrum in reader:
                        if spectrum.ms_level != 2:
                            continue

                        spectrum_rt = spectrum.scan_time_in_minutes()

                        try:
                            precursor_mz = spectrum.selected_precursors[0]["mz"] if spectrum.selected_precursors else None
                            if precursor_mz is None:
                                continue
                            if abs(precursor_mz - self.target_mz) > precursor_tolerance:
                                continue

                            spectrum_polarity = self.file_manager._get_spectrum_polarity(spectrum)
                            if self.polarity and spectrum_polarity and self.polarity != spectrum_polarity:
                                continue

                            mz_array = spectrum.mz
                            intensity_array = spectrum.i
                            if len(mz_array) == 0:
                                continue

                            precursor_intensity = 0
                            try:
                                if spectrum.selected_precursors:
                                    precursor_intensity = spectrum.selected_precursors[0].get("intensity", 0) or 0
                            except Exception:
                                precursor_intensity = 0

                            candidate = {
                                "rt": spectrum_rt,
                                "precursor_mz": precursor_mz,
                                "precursor_intensity": precursor_intensity,
                                "mz": mz_array,
                                "intensity": intensity_array,
                                "scan_id": spectrum.ID,
                                "polarity": spectrum_polarity,
                                "filter_string": self.file_manager._get_filter_string(spectrum),
                                "collision_energy": self.file_manager._get_collision_energy(spectrum),
                            }
                            if filter_type is not None:
                                parsed = self._parse_filter_string_type(candidate["filter_string"] or "")
                                if parsed != filter_type:
                                    continue

                            rt_diff = abs(spectrum_rt - rt_center)
                            if rt_diff < min_rt_diff:
                                min_rt_diff = rt_diff
                                closest_spectrum = candidate

                        except Exception as e:
                            print(f"Error processing spectrum in {filename}: {e}")
                            continue

                if closest_spectrum is not None:
                    msms_spectra[filepath] = {
                        "filename": filename,
                        "group": row.get("group", "Unknown"),
                        "spectra": [closest_spectrum],
                        "metadata": row.to_dict(),
                    }

            except Exception as e:
                print(f"Error reading closest MSMS from {filepath}: {e}")
                continue

        return msms_spectra


# Number of seconds threshold for warning about distant MSMS spectra
_MSMS_RT_WARNING_THRESHOLD_S = 5.0

# Distinct colors cycled for extra EIC traces (colorblind-friendly)
_EXTRA_TRACE_COLORS = [
    "#4477AA",  # blue
    "#EE6677",  # red-pink
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # gray
]


class _AddEICTraceDialog(QDialog):
    """Dialog for adding an extra EIC trace to an existing EIC window.

    Supports two modes:
      * Tab 0 (default): enter an m/z value + polarity directly
      * Tab 1: enter a chemical formula / neutral mass + adduct
    """

    def __init__(self, adducts_data, ppm_default=5.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add EIC Trace")
        self.setMinimumWidth(480)
        self._adducts_data = adducts_data
        self._ppm_default = ppm_default
        self._result_label = ""
        self._result_mz = None
        self._result_polarity = None
        self._result_ppm = ppm_default
        self._setup_ui()
        self._connect_signals()
        self._validate_tab1()
        self._validate_tab2()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()

        # ---- Tab 0: m/z value ----
        tab0 = QWidget()
        form0 = QFormLayout(tab0)
        form0.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form0.setContentsMargins(12, 12, 12, 12)
        form0.setVerticalSpacing(8)

        self.mz_spin = NoScrollDoubleSpinBox()
        self.mz_spin.setRange(0.001, 100000.0)
        self.mz_spin.setDecimals(6)
        self.mz_spin.setSuffix(" m/z")
        self.mz_spin.setValue(200.0)
        form0.addRow("m/z value:", self.mz_spin)

        self.polarity_combo = QComboBox()
        self.polarity_combo.addItem("Positive", "positive")
        self.polarity_combo.addItem("Negative", "negative")
        form0.addRow("Polarity:", self.polarity_combo)

        self.tabs.addTab(tab0, "m/z value")

        # ---- Tab 1: Formula / Mass ----
        tab1 = QWidget()
        form1 = QFormLayout(tab1)
        form1.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form1.setContentsMargins(12, 12, 12, 12)
        form1.setVerticalSpacing(8)

        self.formula_edit = QLineEdit()
        self.formula_edit.setPlaceholderText("e.g. C6H12O6")
        form1.addRow("Chemical Formula:", self.formula_edit)

        self.mass_spin = NoScrollDoubleSpinBox()
        self.mass_spin.setRange(0.0, 100000.0)
        self.mass_spin.setDecimals(6)
        self.mass_spin.setSuffix(" Da")
        self.mass_spin.setValue(0.0)
        self.mass_spin.setSpecialValueText("(derive from formula)")
        form1.addRow("Neutral Mass:", self.mass_spin)

        self.adduct_combo = QComboBox()
        self._fill_adducts_combo(self.adduct_combo)
        form1.addRow("Adduct:", self.adduct_combo)

        self.mz_preview_label = QLabel("(enter formula or mass + adduct to preview m/z)")
        self.mz_preview_label.setStyleSheet("color: gray; font-style: italic;")
        form1.addRow("Calculated m/z:", self.mz_preview_label)

        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        form1.addRow("", self.validation_label)

        self.tabs.addTab(tab1, "Formula / Mass")

        layout.addWidget(self.tabs)

        # ---- Shared ppm tolerance spinner ----
        ppm_layout = QHBoxLayout()
        ppm_label = QLabel("m/z Tolerance:")
        self.ppm_spin = NoScrollDoubleSpinBox()
        self.ppm_spin.setRange(0.1, 500.0)
        self.ppm_spin.setDecimals(1)
        self.ppm_spin.setSingleStep(1.0)
        self.ppm_spin.setSuffix(" ppm")
        self.ppm_spin.setValue(self._ppm_default)
        ppm_layout.addWidget(ppm_label)
        ppm_layout.addWidget(self.ppm_spin)
        ppm_layout.addStretch()
        layout.addLayout(ppm_layout)

        # ---- Buttons ----
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.ok_btn = QPushButton("Add Trace")
        self.ok_btn.setEnabled(False)
        self.ok_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.ok_btn.clicked.connect(self._on_accept)
        cancel_btn.clicked.connect(self.reject)

    def _fill_adducts_combo(self, combo):
        """Populate adduct combo from the adducts table or a built-in fallback list."""
        _BUILTIN_ADDUCTS = [
            "[M+H]+",
            "[M+NH4]+",
            "[M+Na]+",
            "[M+K]+",
            "[M+2H]++",
            "[2M+H]+",
            "[M-H]-",
            "[M+Cl]-",
            "[M+FA-H]-",
            "[M-H2O-H]-",
            "[2M-H]-",
        ]
        if self._adducts_data is not None and not self._adducts_data.empty:
            adducts = self._adducts_data["Adduct"].tolist()
        else:
            adducts = _BUILTIN_ADDUCTS
        for a in adducts:
            combo.addItem(a)

    # ------------------------------------------------------------------
    # Signal connections / validation
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.formula_edit.textChanged.connect(self._validate_tab1)
        self.mass_spin.valueChanged.connect(self._validate_tab1)
        self.adduct_combo.currentIndexChanged.connect(self._validate_tab1)
        self.mz_spin.valueChanged.connect(self._validate_tab2)
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _get_adduct_polarity(self, adduct: str):
        stripped = adduct.rstrip()
        if stripped.endswith("+"):
            return "positive"
        if stripped.endswith("-"):
            return "negative"
        if self._adducts_data is not None and not self._adducts_data.empty:
            row = self._adducts_data[self._adducts_data["Adduct"] == adduct]
            if not row.empty:
                try:
                    charge = int(row.iloc[0]["Charge"])
                    return "positive" if charge > 0 else "negative"
                except Exception:
                    pass
        return None

    def _validate_tab1(self):
        formula = self.formula_edit.text().strip()
        mass_override = self.mass_spin.value()
        adduct = self.adduct_combo.currentText()
        errors = []
        neutral_mass = None

        if formula:
            try:
                parse_molecular_formula(formula)
                neutral_mass = calculate_molecular_mass(formula)
                self.formula_edit.setStyleSheet("QLineEdit { border: 1.5px solid #3a3; }")
            except Exception as exc:
                errors.append(f"Invalid formula: {exc}")
                self.formula_edit.setStyleSheet("QLineEdit { border: 1.5px solid red; }")
        else:
            self.formula_edit.setStyleSheet("")

        if neutral_mass is None and mass_override > 0.0:
            neutral_mass = mass_override

        mz_value = None
        if neutral_mass is not None and adduct:
            if self._adducts_data is not None and not self._adducts_data.empty:
                adduct_row = self._adducts_data[self._adducts_data["Adduct"] == adduct]
                if not adduct_row.empty:
                    try:
                        mass_change, charge, multiplier = adduct_mass_change(adduct_row.iloc[0])
                        mz_value = (multiplier * neutral_mass + mass_change) / abs(charge)
                    except Exception as exc:
                        errors.append(f"m/z calculation error: {exc}")
            else:
                errors.append("No adducts table – cannot calculate m/z")

        if mz_value is not None:
            self.mz_preview_label.setText(f"{mz_value:.6f}")
            self.mz_preview_label.setStyleSheet("color: #1a7a1a; font-weight: bold;")
        else:
            self.mz_preview_label.setText("(enter formula or mass + adduct to preview m/z)")
            self.mz_preview_label.setStyleSheet("color: gray; font-style: italic;")

        if errors:
            self.validation_label.setText("❌ " + "; ".join(errors))
            self.validation_label.setStyleSheet("color: red;")
        else:
            self.validation_label.setText("")

        if self.tabs.currentIndex() == 1:
            self.ok_btn.setEnabled(bool(not errors and mz_value is not None))

    def _validate_tab2(self):
        valid = self.mz_spin.value() > 0.0
        if self.tabs.currentIndex() == 0:
            self.ok_btn.setEnabled(valid)

    def _on_tab_changed(self, index):
        if index == 0:
            self._validate_tab2()
        else:
            self._validate_tab1()

    # ------------------------------------------------------------------
    # Accept handler
    # ------------------------------------------------------------------

    def _on_accept(self):
        if self.tabs.currentIndex() == 1:
            formula = self.formula_edit.text().strip()
            mass_override = self.mass_spin.value()
            adduct = self.adduct_combo.currentText()

            neutral_mass = None
            if formula:
                try:
                    neutral_mass = calculate_molecular_mass(formula)
                except Exception:
                    pass
            if neutral_mass is None and mass_override > 0.0:
                neutral_mass = mass_override

            adduct_row = self._adducts_data[self._adducts_data["Adduct"] == adduct]
            mass_change, charge, multiplier = adduct_mass_change(adduct_row.iloc[0])
            self._result_mz = (multiplier * neutral_mass + mass_change) / abs(charge)
            self._result_polarity = self._get_adduct_polarity(adduct)
            name = formula if formula else f"Mass {neutral_mass:.4f} Da"
            self._result_label = f"{name} {adduct}"
        else:
            self._result_mz = self.mz_spin.value()
            self._result_polarity = self.polarity_combo.currentData()
            pol_sign = "+" if self._result_polarity == "positive" else "-"
            self._result_label = f"m/z {self._result_mz:.6f} [{pol_sign}]"

        self._result_ppm = self.ppm_spin.value()
        self.accept()

    def get_result(self):
        """Return (label, mz, polarity, ppm) after the dialog was accepted."""
        return (self._result_label, self._result_mz, self._result_polarity, self._result_ppm)


class EmbeddedScatterPlotView(QWidget):
    """Embedded scatter plot view for integration into EIC window"""

    def __init__(
        self,
        file_manager,
        target_mz,
        mz_tolerance,
        rt_min,
        rt_max,
        polarity,
        rt_center,
        parent=None,
    ):
        super().__init__(parent)

        self.file_manager = file_manager
        self.target_mz = target_mz
        self.mz_tolerance = mz_tolerance
        self.rt_min = rt_min
        self.rt_max = rt_max
        self.polarity = polarity
        self.rt_center = rt_center
        self.eic_window = parent  # Direct reference to EIC window

        # Store all signal data for hover functionality
        self.signal_data = []  # List of dicts with rt, mz, intensity, group, filename
        self.chart_view = None
        self.current_hover_group = None

        self.init_ui()
        self.load_data()

    def init_ui(self):
        """Initialize the embedded scatter plot UI"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create chart
        self.create_scatter_chart()
        layout.addWidget(self.chart_view)

        # Set minimum height for the scatter plot (will be resizable via splitter)
        self.setMinimumHeight(200)

    def normalize_group_column(self, files_data):
        """Normalize group column name to handle case differences"""
        if "Group" in files_data.columns and "group" not in files_data.columns:
            files_data = files_data.copy()
            files_data["group"] = files_data["Group"]
        elif "group" not in files_data.columns and "Group" not in files_data.columns:
            files_data = files_data.copy()
            files_data["group"] = "Unknown"
        return files_data

    def create_scatter_chart(self):
        """Create the 2D scatter chart"""
        from PyQt6.QtCharts import QChart, QScatterSeries

        self.chart = QChart()
        self.chart.setTitle("")
        self.chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        self.chart.setMargins(QMargins(0, 0, 0, 0))

        # Create chart view
        self.chart_view = Interactive2DScatterChartView(self.chart, self)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Set background color to white
        self.chart.setBackgroundBrush(QBrush(QColor(255, 255, 255)))  # White background

    def load_data(self):
        """Load and process data for the scatter plot"""
        MAX_TOTAL_POINTS = 10000

        try:
            # Calculate extended m/z range (3x the extraction window)
            mz_min = self.target_mz - self.mz_tolerance
            mz_max = self.target_mz + self.mz_tolerance

            # Get EIC window boundaries for background highlighting
            eic_mz_min = self.target_mz - (self.mz_tolerance / 3)  # Original EIC window
            eic_mz_max = self.target_mz + (self.mz_tolerance / 3)

            files_data = self.file_manager.get_files_data()
            files_data = self.normalize_group_column(files_data)

            # Group data by group for consistent coloring
            group_signals = {}

            for _, row in files_data.iterrows():
                filepath = row["Filepath"]
                filename = row["filename"]
                group = row["group"]  # Use lowercase after normalization

                try:
                    # Use the same data access pattern as extract_eic function
                    if self.file_manager.keep_in_memory and filepath in self.file_manager.cached_data:
                        # Use cached memory data
                        cached_file_data = self.file_manager.cached_data[filepath]

                        # Handle both old format (list) and new format (dict with ms1/ms2)
                        if isinstance(cached_file_data, dict) and "ms1" in cached_file_data:
                            spectra_data = cached_file_data["ms1"]
                        else:
                            # Old format - assume it's MS1 data
                            spectra_data = cached_file_data

                        for spectrum_data in spectra_data:
                            # Check polarity if specified

                            if (
                                self.polarity
                                and spectrum_data.get("polarity")
                                and not (
                                    (self.polarity.lower() in ["+", "positive", "pos"] and spectrum_data.get("polarity").lower() in ["+", "positive", "pos"])
                                    or (self.polarity.lower() in ["-", "negative", "neg"] and spectrum_data.get("polarity").lower() in ["-", "negative", "neg"])
                                )
                            ):
                                continue

                            spectrum_rt = spectrum_data["scan_time"]

                            # Filter by RT range
                            if self.rt_min <= spectrum_rt <= self.rt_max:
                                mz_array = spectrum_data["mz"]
                                intensity_array = spectrum_data["intensity"]

                                if len(mz_array) > 0:
                                    # Filter by extended m/z range
                                    mz_mask = (mz_array >= mz_min) & (mz_array <= mz_max)
                                    if np.any(mz_mask):
                                        filtered_mz = mz_array[mz_mask]
                                        filtered_intensity = intensity_array[mz_mask]

                                        # Add signals to group
                                        if group not in group_signals:
                                            group_signals[group] = []

                                        for mz, intensity in zip(filtered_mz, filtered_intensity):
                                            signal_info = {
                                                "rt": spectrum_rt,
                                                "mz": mz,
                                                "intensity": intensity,
                                                "group": group,
                                                "filename": filename,
                                                "in_eic_window": eic_mz_min <= mz <= eic_mz_max,
                                            }
                                            group_signals[group].append(signal_info)
                                            self.signal_data.append(signal_info)

                    else:
                        # Read from file directly (similar to extract_eic function)
                        reader = self.file_manager.get_mzml_reader(filepath)

                        for spectrum in reader:
                            if spectrum.ms_level == 1:  # Only MS1 spectra
                                # Check polarity if specified
                                if self.polarity is not None:
                                    spectrum_polarity = self.file_manager._get_spectrum_polarity(spectrum)
                                    if (
                                        self.polarity
                                        and spectrum_polarity
                                        and not (
                                            (self.polarity.lower() in ["+", "positive", "pos"] and spectrum_polarity.lower() in ["+", "positive", "pos"])
                                            or (self.polarity.lower() in ["-", "negative", "neg"] and spectrum_polarity.lower() in ["-", "negative", "neg"])
                                        )
                                    ):
                                        continue

                                spectrum_rt = spectrum.scan_time_in_minutes()

                                # Filter by RT range
                                if self.rt_min <= spectrum_rt <= self.rt_max:
                                    mz_array = spectrum.mz
                                    intensity_array = spectrum.i

                                    if len(mz_array) > 0:
                                        # Filter by extended m/z range
                                        mz_mask = (mz_array >= mz_min) & (mz_array <= mz_max)
                                        if np.any(mz_mask):
                                            filtered_mz = mz_array[mz_mask]
                                            filtered_intensity = intensity_array[mz_mask]

                                            # Add signals to group
                                            if group not in group_signals:
                                                group_signals[group] = []

                                            for mz, intensity in zip(filtered_mz, filtered_intensity):
                                                signal_info = {
                                                    "rt": spectrum_rt,
                                                    "mz": mz,
                                                    "intensity": intensity,
                                                    "group": group,
                                                    "filename": filename,
                                                    "in_eic_window": eic_mz_min <= mz <= eic_mz_max,
                                                }
                                                group_signals[group].append(signal_info)
                                                self.signal_data.append(signal_info)

                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")
                    continue

            # Subsample globally to the top MAX_TOTAL_POINTS most abundant points
            total_points = len(self.signal_data)
            chart_title = ""
            if total_points > MAX_TOTAL_POINTS:
                self.signal_data.sort(key=lambda s: s["intensity"], reverse=True)
                self.signal_data = self.signal_data[:MAX_TOTAL_POINTS]
                # Rebuild group_signals from the subsampled data
                group_signals = {}
                for signal_info in self.signal_data:
                    g = signal_info["group"]
                    if g not in group_signals:
                        group_signals[g] = []
                    group_signals[g].append(signal_info)
                chart_title = f"{total_points:,} points total — showing {MAX_TOTAL_POINTS:,} most abundant"

            # Create scatter series for each group
            self.create_scatter_series(group_signals)
            self.chart.setTitle(chart_title)

        except Exception as e:
            print(f"Failed to load scatter plot data: {str(e)}")
            import traceback

            traceback.print_exc()

    def create_scatter_series(self, group_signals):
        """Create scatter series for each group with intensity-based coloring"""
        from PyQt6.QtCharts import QScatterSeries, QValueAxis

        if not group_signals:
            return

        # Find global intensity range for normalization
        all_intensities = [signal["intensity"] for signal in self.signal_data]
        if not all_intensities:
            return

        min_intensity = min(all_intensities)
        max_intensity = max(all_intensities)

        # Get group colors from file manager
        group_colors = {}
        files_data = self.file_manager.get_files_data()
        files_data = self.normalize_group_column(files_data)
        for _, row in files_data.iterrows():
            group = row.get("group", "Unknown")
            if group not in group_colors:
                # Get group color from file manager or use a default
                if hasattr(self.file_manager, "get_group_color"):
                    color = self.file_manager.get_group_color(group)
                else:
                    # Default colors if group color method doesn't exist
                    colors = [
                        QColor(31, 119, 180),
                        QColor(255, 127, 14),
                        QColor(44, 160, 44),
                        QColor(214, 39, 40),
                        QColor(148, 103, 189),
                        QColor(140, 86, 75),
                    ]
                    color_idx = hash(group) % len(colors)
                    color = colors[color_idx]
                group_colors[group] = color

        # Create series for each group
        self.series_by_group = {}

        for group, signals in group_signals.items():
            if not signals:
                continue

            base_color = group_colors.get(group, QColor(100, 100, 100))

            # Create separate series for different intensity levels to achieve gradient effect
            # We'll create multiple series with different transparency levels
            intensity_levels = 5  # Number of intensity levels

            for level in range(intensity_levels):
                series = QScatterSeries()
                series.setName(f"{group}_level_{level}")
                series.setMarkerSize(6)  # Smaller markers for embedded view

                # Calculate intensity range for this level
                intensity_range = max_intensity - min_intensity
                level_min = min_intensity + (intensity_range * level / intensity_levels)
                level_max = min_intensity + (intensity_range * (level + 1) / intensity_levels)

                # Calculate transparency (10% for lowest, 100% for highest)
                transparency = int(25 + (230 * level / (intensity_levels - 1)))  # 25-255 range

                # Set color with transparency
                color = QColor(base_color)
                color.setAlpha(transparency)
                series.setColor(color)
                series.setBorderColor(color.darker(110))

                # Add points for this intensity level
                for signal in signals:
                    if level_min <= signal["intensity"] <= level_max or (level == intensity_levels - 1 and signal["intensity"] >= level_min):
                        series.append(signal["rt"], signal["mz"])

                if series.count() > 0:
                    self.chart.addSeries(series)
                    if group not in self.series_by_group:
                        self.series_by_group[group] = []
                    self.series_by_group[group].append(series)

        # Create and configure axes
        self.setup_axes()

        # Store reference for hover functionality
        self.chart_view.signal_data = self.signal_data
        self.chart_view.series_by_group = self.series_by_group

    def setup_axes(self):
        """Setup chart axes"""
        from PyQt6.QtCharts import QValueAxis

        # Create axes
        x_axis = QValueAxis()
        y_axis = QValueAxis()

        # Configure X-axis (RT)
        x_axis.setTitleText("Retention Time (min)")
        x_axis.setRange(self.rt_min, self.rt_max)
        x_axis.setTickCount(6)  # Fewer ticks for embedded view
        x_axis.setLabelFormat("%.2f")

        # Configure Y-axis (m/z)
        mz_padding = self.mz_tolerance * 0.1  # 10% padding
        y_axis.setTitleText("m/z")
        y_axis.setRange(
            self.target_mz - self.mz_tolerance - mz_padding,
            self.target_mz + self.mz_tolerance + mz_padding,
        )
        y_axis.setTickCount(6)  # Fewer ticks for embedded view
        y_axis.setLabelFormat("%.4f")

        # Add axes to chart
        self.chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

        # Attach series to axes
        for group_series_list in self.series_by_group.values():
            for series in group_series_list:
                series.attachAxis(x_axis)
                series.attachAxis(y_axis)

        # Hide legend since we have many series
        self.chart.legend().setVisible(False)

        # Add horizontal line to indicate target m/z
        self.add_target_mz_line(x_axis, y_axis)

    def add_target_mz_line(self, x_axis, y_axis):
        """Add horizontal lines to indicate the target m/z and extraction window"""
        # Create a horizontal line series at the target m/z
        target_line = QLineSeries()
        target_line.setName("Target m/z")

        # Add points across the full RT range
        target_line.append(self.rt_min, self.target_mz)
        target_line.append(self.rt_max, self.target_mz)

        # Style the line (red solid line)
        pen = QPen(QColor(255, 0, 0))  # Red color
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.SolidLine)
        target_line.setPen(pen)

        # Add to chart
        self.chart.addSeries(target_line)
        target_line.attachAxis(x_axis)
        target_line.attachAxis(y_axis)

        # Add extraction window indicator lines (original EIC extraction window, not the 3x extended window)
        eic_tolerance = self.mz_tolerance / 3  # Original EIC extraction window

        # Upper extraction window line
        upper_line = QLineSeries()
        upper_line.setName("EIC Upper Window")
        upper_line.append(self.rt_min, self.target_mz + eic_tolerance)
        upper_line.append(self.rt_max, self.target_mz + eic_tolerance)

        # Style the upper line (orange dashed line)
        upper_pen = QPen(QColor(255, 165, 0))  # Orange color
        upper_pen.setWidth(1)
        upper_pen.setStyle(Qt.PenStyle.DashLine)
        upper_line.setPen(upper_pen)

        # Add to chart
        self.chart.addSeries(upper_line)
        upper_line.attachAxis(x_axis)
        upper_line.attachAxis(y_axis)

        # Lower extraction window line
        lower_line = QLineSeries()
        lower_line.setName("EIC Lower Window")
        lower_line.append(self.rt_min, self.target_mz - eic_tolerance)
        lower_line.append(self.rt_max, self.target_mz - eic_tolerance)

        # Style the lower line (orange dashed line)
        lower_pen = QPen(QColor(255, 165, 0))  # Orange color
        lower_pen.setWidth(1)
        lower_pen.setStyle(Qt.PenStyle.DashLine)
        lower_line.setPen(lower_pen)

        # Add to chart
        self.chart.addSeries(lower_line)
        lower_line.attachAxis(x_axis)
        lower_line.attachAxis(y_axis)

    def update_scatter_plot(self):
        """Update the scatter plot data after EIC extraction - completely rebuild everything"""
        try:
            print("Updating scatter plot with fresh parameters...")

            # Get current m/z tolerance from the EIC window
            if self.eic_window and hasattr(self.eic_window, "mz_tolerance_da_spin"):
                current_mz_tolerance = self.eic_window.mz_tolerance_da_spin.value() * 3  # 3x for scatter plot
                print(f"Using updated m/z tolerance: {current_mz_tolerance:.6f} Da")
                self.mz_tolerance = current_mz_tolerance

            # Get current RT range from EIC window
            if self.eic_window and hasattr(self.eic_window, "get_rt_range"):
                self.rt_min, self.rt_max = self.eic_window.get_rt_range()
                print(f"Using updated RT range: {self.rt_min:.2f} - {self.rt_max:.2f} min")

            # Complete cleanup - remove all series and axes
            self.chart.removeAllSeries()

            # Remove all axes
            for axis in self.chart.axes():
                self.chart.removeAxis(axis)

            # Clear stored data
            self.signal_data = []
            if hasattr(self, "series_by_group"):
                self.series_by_group = {}

            # Reload data with current parameters
            self.load_data()

            # Force chart view refresh
            if hasattr(self, "chart_view"):
                self.chart_view.update()
                self.chart_view.repaint()

            print("Scatter plot updated successfully with fresh parameters")

        except Exception as e:
            print(f"Error updating scatter plot: {str(e)}")

            traceback.print_exc()


class Interactive2DScatterChartView(QChartView):
    """Interactive chart view for 2D scatter plot with hover functionality"""

    def __init__(self, chart, parent_window):
        super().__init__(chart)
        self.parent_window = parent_window
        self.signal_data = []
        self.series_by_group = {}
        self.setMouseTracking(True)
        self.hover_tolerance_rt = 0.1  # 0.1 minutes tolerance
        self.hover_tolerance_mz = 0.01  # 0.01 Da tolerance

        # Hover state
        self.current_hover_group = None
        self.original_colors = {}  # Store original colors for restoration

        # Pan and zoom state
        self.is_panning = False
        self.is_zooming = False
        self.last_mouse_pos = None
        self.pan_start_pos = None
        self.zoom_start_pos = None
        self.interaction_start_x_range = None
        self.interaction_start_y_range = None
        self.zoom_anchor_x = 0
        self.zoom_anchor_y = 0

    def mouseMoveEvent(self, event):
        """Handle mouse move events for pan, zoom, and hover functionality"""
        if self.is_panning:
            self._handle_panning(event)
        elif self.is_zooming:
            self._handle_zooming(event)
        else:
            # Handle hover functionality when not panning or zooming
            plot_area = self.chart().plotArea()
            if plot_area.contains(event.position()):
                # Get chart axes
                axes = self.chart().axes()
                if len(axes) >= 2:
                    x_axis = None
                    y_axis = None
                    for axis in axes:
                        if axis.orientation() == Qt.Orientation.Horizontal:
                            x_axis = axis
                        else:
                            y_axis = axis

                    if x_axis and y_axis:
                        # Convert pixel position to data coordinates
                        rel_x = (event.position().x() - plot_area.left()) / plot_area.width()
                        rel_y = (event.position().y() - plot_area.top()) / plot_area.height()

                        data_rt = x_axis.min() + rel_x * (x_axis.max() - x_axis.min())
                        data_mz = y_axis.max() - rel_y * (y_axis.max() - y_axis.min())  # Y is inverted

                        # Find nearby signals and determine which group to highlight
                        nearby_group = self.find_nearby_group(data_rt, data_mz)

                        if nearby_group != self.current_hover_group:
                            self.highlight_group(nearby_group)
                            self.current_hover_group = nearby_group

        self.last_mouse_pos = event.position()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press events for pan and zoom"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Start panning
            self.is_panning = True
            self.pan_start_pos = event.position()

            # Store current axis ranges
            axes = self.chart().axes()
            x_axis = None
            y_axis = None
            for axis in axes:
                if axis.orientation() == Qt.Orientation.Horizontal:
                    x_axis = axis
                else:
                    y_axis = axis

            if x_axis and y_axis:
                self.interaction_start_x_range = (x_axis.min(), x_axis.max())
                self.interaction_start_y_range = (y_axis.min(), y_axis.max())

            self.setCursor(Qt.CursorShape.ClosedHandCursor)

        elif event.button() == Qt.MouseButton.RightButton:
            # Start zooming
            self.is_zooming = True
            self.zoom_start_pos = event.position()

            # Store current axis ranges
            axes = self.chart().axes()
            x_axis = None
            y_axis = None
            for axis in axes:
                if axis.orientation() == Qt.Orientation.Horizontal:
                    x_axis = axis
                else:
                    y_axis = axis

            if x_axis and y_axis:
                self.interaction_start_x_range = (x_axis.min(), x_axis.max())
                self.interaction_start_y_range = (y_axis.min(), y_axis.max())

                # Set zoom anchor point
                plot_area = self.chart().plotArea()
                rel_x = (event.position().x() - plot_area.left()) / plot_area.width()
                rel_y = (event.position().y() - plot_area.top()) / plot_area.height()
                rel_x = max(0.0, min(1.0, rel_x))
                rel_y = max(0.0, min(1.0, rel_y))

                x_range = self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
                y_range = self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
                self.zoom_anchor_x = self.interaction_start_x_range[0] + rel_x * x_range
                self.zoom_anchor_y = self.interaction_start_y_range[1] - rel_y * y_range

            self.setCursor(Qt.CursorShape.SizeAllCursor)

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_zooming = False

        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15

        # Get mouse position in plot coordinates
        plot_area = self.chart().plotArea()
        if plot_area.contains(event.position()):
            rel_x = (event.position().x() - plot_area.left()) / plot_area.width()
            rel_y = (event.position().y() - plot_area.top()) / plot_area.height()

            # Get current axis ranges
            axes = self.chart().axes()
            x_axis = None
            y_axis = None
            for axis in axes:
                if axis.orientation() == Qt.Orientation.Horizontal:
                    x_axis = axis
                else:
                    y_axis = axis

            if x_axis and y_axis:
                current_x_range = x_axis.max() - x_axis.min()
                current_y_range = y_axis.max() - y_axis.min()

                # Calculate new ranges
                new_x_range = current_x_range / zoom_factor
                new_y_range = current_y_range / zoom_factor

                # Calculate anchor point
                anchor_x = x_axis.min() + rel_x * current_x_range
                anchor_y = y_axis.max() - rel_y * current_y_range

                # Calculate new axis bounds
                new_x_min = anchor_x - rel_x * new_x_range
                new_x_max = anchor_x + (1 - rel_x) * new_x_range
                new_y_min = anchor_y - (1 - rel_y) * new_y_range
                new_y_max = anchor_y + rel_y * new_y_range

                x_axis.setRange(new_x_min, new_x_max)
                y_axis.setRange(new_y_min, new_y_max)

    def _handle_panning(self, event):
        """Handle panning interaction"""
        if not self.interaction_start_x_range or not self.interaction_start_y_range:
            return

        delta_x = event.position().x() - self.pan_start_pos.x()
        delta_y = event.position().y() - self.pan_start_pos.y()

        plot_area = self.chart().plotArea()
        axes = self.chart().axes()
        x_axis = None
        y_axis = None
        for axis in axes:
            if axis.orientation() == Qt.Orientation.Horizontal:
                x_axis = axis
            else:
                y_axis = axis

        if x_axis and y_axis:
            x_range = self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
            y_range = self.interaction_start_y_range[1] - self.interaction_start_y_range[0]

            x_per_pixel = x_range / plot_area.width()
            y_per_pixel = y_range / plot_area.height()

            x_offset = -delta_x * x_per_pixel
            y_offset = delta_y * y_per_pixel  # Y is inverted

            new_x_min = self.interaction_start_x_range[0] + x_offset
            new_x_max = self.interaction_start_x_range[1] + x_offset
            new_y_min = self.interaction_start_y_range[0] + y_offset
            new_y_max = self.interaction_start_y_range[1] + y_offset

            x_axis.setRange(new_x_min, new_x_max)
            y_axis.setRange(new_y_min, new_y_max)

    def _handle_zooming(self, event):
        """Handle zooming interaction"""
        if not self.interaction_start_x_range or not self.interaction_start_y_range:
            return

        delta_x = event.position().x() - self.zoom_start_pos.x()
        delta_y = event.position().y() - self.zoom_start_pos.y()

        zoom_sensitivity = 0.005
        x_zoom_factor = 1.0 - (delta_x * zoom_sensitivity)
        y_zoom_factor = 1.0 + (delta_y * zoom_sensitivity)

        x_zoom_factor = max(0.1, min(10.0, x_zoom_factor))
        y_zoom_factor = max(0.1, min(10.0, y_zoom_factor))

        original_x_range = self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
        original_y_range = self.interaction_start_y_range[1] - self.interaction_start_y_range[0]

        new_x_range = original_x_range * x_zoom_factor
        new_y_range = original_y_range * y_zoom_factor

        anchor_to_left = self.zoom_anchor_x - self.interaction_start_x_range[0]
        anchor_to_right = self.interaction_start_x_range[1] - self.zoom_anchor_x
        anchor_to_bottom = self.zoom_anchor_y - self.interaction_start_y_range[0]
        anchor_to_top = self.interaction_start_y_range[1] - self.zoom_anchor_y

        new_x_min = self.zoom_anchor_x - (anchor_to_left / original_x_range) * new_x_range
        new_x_max = self.zoom_anchor_x + (anchor_to_right / original_x_range) * new_x_range
        new_y_min = self.zoom_anchor_y - (anchor_to_bottom / original_y_range) * new_y_range
        new_y_max = self.zoom_anchor_y + (anchor_to_top / original_y_range) * new_y_range

        axes = self.chart().axes()
        x_axis = None
        y_axis = None
        for axis in axes:
            if axis.orientation() == Qt.Orientation.Horizontal:
                x_axis = axis
            else:
                y_axis = axis

        if x_axis and y_axis:
            x_axis.setRange(new_x_min, new_x_max)
            y_axis.setRange(new_y_min, new_y_max)

    def find_nearby_group(self, rt, mz):
        """Find the group of signals near the mouse position"""
        for signal in self.signal_data:
            if abs(signal["rt"] - rt) <= self.hover_tolerance_rt and abs(signal["mz"] - mz) <= self.hover_tolerance_mz:
                return signal["group"]
        return None

    def highlight_group(self, group_to_highlight):
        """Highlight all signals from a specific group"""
        if not hasattr(self, "series_by_group"):
            return

        # Restore original colors if we were highlighting another group
        if self.current_hover_group and self.current_hover_group in self.original_colors:
            self.restore_group_colors(self.current_hover_group)

        if group_to_highlight:
            # Store original colors and apply highlight
            if group_to_highlight not in self.original_colors:
                self.store_original_colors(group_to_highlight)

            # Apply light blue highlight while maintaining intensity gradient
            self.apply_group_highlight(group_to_highlight)
        else:
            # No group to highlight, restore all to original colors
            for group in self.original_colors:
                self.restore_group_colors(group)

    def store_original_colors(self, group):
        """Store the original colors of a group's series"""
        if group in self.series_by_group:
            self.original_colors[group] = []
            for series in self.series_by_group[group]:
                self.original_colors[group].append(series.color())

    def apply_group_highlight(self, group):
        """Apply light blue highlight to a group while maintaining intensity gradient"""
        if group in self.series_by_group:
            base_highlight_color = QColor(173, 216, 230)  # Light blue

            for i, series in enumerate(self.series_by_group[group]):
                # Maintain the transparency level but change to light blue
                original_alpha = series.color().alpha()
                highlight_color = QColor(base_highlight_color)
                highlight_color.setAlpha(original_alpha)

                series.setColor(highlight_color)
                series.setBorderColor(highlight_color.darker(110))

    def restore_group_colors(self, group):
        """Restore the original colors of a group's series"""
        if group in self.original_colors and group in self.series_by_group:
            for i, series in enumerate(self.series_by_group[group]):
                if i < len(self.original_colors[group]):
                    original_color = self.original_colors[group][i]
                    series.setColor(original_color)
                    series.setBorderColor(original_color.darker(110))

    def leaveEvent(self, event):
        """Restore all colors when mouse leaves the chart"""
        if self.current_hover_group:
            self.highlight_group(None)
            self.current_hover_group = None
        super().leaveEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# MS1 single-spectrum detail window
# ─────────────────────────────────────────────────────────────────────────────
