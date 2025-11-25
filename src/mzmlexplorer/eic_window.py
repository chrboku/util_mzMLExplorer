"""
EIC (Extracted Ion Chromatogram) window for displaying chromatographic data
"""

import sys
import pandas as pd
import time
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QComboBox,
    QMenu,
    QScrollArea,
    QGridLayout,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QTabWidget,
    QApplication,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QMargins
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPen, QColor, QPainter, QMouseEvent, QAction, QBrush
from PyQt6.QtWidgets import QSizePolicy
from .utils import calculate_cosine_similarity, calculate_similarity_statistics
import numpy as np
from typing import Dict, Tuple, Optional, List
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from .utils import (
    calculate_mz_from_formula,
    format_mz,
    format_retention_time,
    parse_molecular_formula,
)
from natsort import natsorted, natsort_keygen


class CollapsibleBox(QWidget):
    """A collapsible widget with a clickable header"""

    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 5px;
                border: 1px solid #ccc;
                background-color: #f0f0f0;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_content)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_area.setVisible(False)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

    def toggle_content(self):
        """Toggle visibility of content area"""
        is_visible = self.content_area.isVisible()
        self.content_area.setVisible(not is_visible)
        # Update button text to show collapse state
        current_text = self.toggle_button.text()
        if "▼" in current_text:
            self.toggle_button.setText(current_text.replace("▼", "▶"))
        elif "▶" in current_text:
            self.toggle_button.setText(current_text.replace("▶", "▼"))
        else:
            if not is_visible:
                self.toggle_button.setText("▼ " + current_text)
            else:
                self.toggle_button.setText("▶ " + current_text)

    def add_widget(self, widget):
        """Add a widget to the content area"""
        self.content_layout.addWidget(widget)

    def set_expanded(self, expanded):
        """Set the expanded state"""
        if expanded != self.content_area.isVisible():
            self.toggle_content()


class MSMSPopupWindow(QWidget):
    """Popup window for displaying a single MSMS spectrum in a larger view"""

    def __init__(self, spectrum_data, filename, group, parent=None):
        super().__init__(parent)
        self.spectrum_data = spectrum_data
        self.filename = filename
        self.group = group
        self.selected_mz = None  # Track selected m/z for highlighting

        self.setWindowTitle(f"MSMS Spectrum - {filename}")
        self.setWindowFlags(Qt.WindowType.Window)  # Make it a separate window
        self.resize(1000, 800)

        self.setup_ui()

    def setup_ui(self):
        """Setup the popup window UI"""
        layout = QVBoxLayout(self)

        # Header with spectrum information
        precursor_intensity = self.spectrum_data.get("precursor_intensity", 0)
        intensity_text = (
            f"{precursor_intensity:.1e}" if precursor_intensity > 0 else "N/A"
        )

        header_text = (
            f"<b>File:</b> {self.filename}<br>"
            f"<b>Group:</b> {self.group}<br>"
            f"<b>RT:</b> {self.spectrum_data['rt']:.2f} min<br>"
            f"<b>Precursor m/z:</b> {self.spectrum_data['precursor_mz']:.4f}<br>"
            f"<b>Precursor Intensity:</b> {intensity_text}"
        )

        header_label = QLabel(header_text)
        header_label.setStyleSheet("""
            QLabel { 
                background-color: #f0f0f0; 
                padding: 5px; 
                margin: 2px;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-size: 11px;
            }
        """)
        header_label.setMaximumHeight(50)  # Set maximum height for minimal appearance
        layout.addWidget(header_label)

        # Create splitter for table and chart (horizontal layout)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create table for m/z and intensity values (left side)
        self.create_data_table()
        splitter.addWidget(self.table_widget)

        # Create large MSMS chart with interactive capabilities (right side)
        chart = self.create_large_msms_chart()
        self.chart_view = InteractiveMSMSChartView(chart)
        self.chart_view.setMinimumSize(600, 400)
        splitter.addWidget(self.chart_view)

        # Set splitter proportions (30% table, 70% chart)
        splitter.setSizes([300, 700])

        layout.addWidget(splitter)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def create_data_table(self):
        """Create a sortable table with m/z and intensity data"""
        self.table_widget = QTableWidget()

        # Get data and ensure they're numpy arrays
        mz_array = np.array(self.spectrum_data["mz"])
        intensity_array = np.array(self.spectrum_data["intensity"])

        # Calculate relative abundance (% of base peak)
        max_intensity = np.max(intensity_array) if len(intensity_array) > 0 else 1
        relative_abundance = (
            (intensity_array / max_intensity * 100)
            if max_intensity > 0
            else np.zeros_like(intensity_array)
        )

        # Setup table
        num_rows = len(mz_array)
        self.table_widget.setRowCount(num_rows)
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(
            ["m/z", "Intensity", "Rel. Abundance (%)"]
        )

        # Disable sorting during population to prevent data loss
        self.table_widget.setSortingEnabled(False)

        # Set selection behavior
        self.table_widget.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

        # Create custom table items that sort numerically
        for i, (mz, intensity, rel_abund) in enumerate(
            zip(mz_array, intensity_array, relative_abundance)
        ):
            # Convert numpy types to Python native types explicitly
            mz_val = float(mz)
            intensity_val = float(intensity)
            rel_abund_val = float(rel_abund)

            # Create m/z item
            mz_item = QTableWidgetItem()
            mz_item.setData(Qt.ItemDataRole.DisplayRole, f"{mz_val:.4f}")
            mz_item.setData(Qt.ItemDataRole.UserRole, mz_val)  # Store for selection
            # Set the data type for proper sorting
            mz_item.setData(Qt.ItemDataRole.EditRole, mz_val)

            # Create intensity item
            intensity_item = QTableWidgetItem()
            if intensity_val >= 1000:
                intensity_item.setData(
                    Qt.ItemDataRole.DisplayRole, f"{intensity_val:.2e}"
                )
            else:
                intensity_item.setData(
                    Qt.ItemDataRole.DisplayRole, f"{intensity_val:.2f}"
                )
            intensity_item.setData(
                Qt.ItemDataRole.UserRole, intensity_val
            )  # Store for selection
            # Set the data type for proper sorting
            intensity_item.setData(Qt.ItemDataRole.EditRole, intensity_val)

            # Create relative abundance item
            rel_abund_item = QTableWidgetItem()
            rel_abund_item.setData(Qt.ItemDataRole.DisplayRole, f"{rel_abund_val:.1f}")
            rel_abund_item.setData(Qt.ItemDataRole.UserRole, rel_abund_val)
            # Set the data type for proper sorting
            rel_abund_item.setData(Qt.ItemDataRole.EditRole, rel_abund_val)

            self.table_widget.setItem(i, 0, mz_item)
            self.table_widget.setItem(i, 1, intensity_item)
            self.table_widget.setItem(i, 2, rel_abund_item)

        # Enable sorting after all data is populated
        self.table_widget.setSortingEnabled(True)

        # Connect selection change
        self.table_widget.itemSelectionChanged.connect(self.on_table_selection_changed)

        # Resize columns to content
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        # Set minimum height
        self.table_widget.setMinimumHeight(200)

    def on_table_selection_changed(self):
        """Handle table selection changes to highlight peaks in the chart"""
        selected_items = self.table_widget.selectedItems()
        if selected_items:
            # Get the m/z value from the first column of the selected row
            mz_item = (
                selected_items[0]
                if selected_items[0].column() == 0
                else selected_items[1]
            )
            if mz_item.column() != 0:
                # Find the m/z item in the same row
                row = mz_item.row()
                mz_item = self.table_widget.item(row, 0)

            self.selected_mz = mz_item.data(Qt.ItemDataRole.UserRole)
            self.update_chart_highlighting()
        else:
            self.selected_mz = None
            self.update_chart_highlighting()

    def update_chart_highlighting(self):
        """Update the chart to highlight the selected m/z peak"""
        chart = self.chart_view.chart()

        # Clear existing series
        chart.removeAllSeries()

        # Recreate series with highlighting
        mz_array = self.spectrum_data["mz"]
        intensity_array = self.spectrum_data["intensity"]

        # Normalize intensities
        if len(intensity_array) > 0:
            max_intensity = np.max(intensity_array)
            if max_intensity > 0:
                normalized_intensity = (intensity_array / max_intensity) * 100
            else:
                normalized_intensity = intensity_array
        else:
            normalized_intensity = intensity_array

        # Create main series (blue)
        main_series = QLineSeries()
        main_pen = QPen(QColor(0, 100, 200))
        main_pen.setWidth(2)
        main_series.setPen(main_pen)

        # Create highlight series (firebrick)
        highlight_series = QLineSeries()
        highlight_pen = QPen(QColor(178, 34, 34))  # Firebrick color
        highlight_pen.setWidth(3)
        highlight_series.setPen(highlight_pen)

        # Add data points
        tolerance = 0.01  # m/z tolerance for highlighting

        for mz, intensity in zip(mz_array, normalized_intensity):
            # Check if this peak should be highlighted
            is_highlighted = (
                self.selected_mz is not None and abs(mz - self.selected_mz) < tolerance
            )

            if is_highlighted:
                # Add to highlight series
                highlight_series.append(float(mz), 0.0)
                highlight_series.append(float(mz), float(intensity))
                highlight_series.append(float(mz), 0.0)
            else:
                # Add to main series
                main_series.append(float(mz), 0.0)
                main_series.append(float(mz), float(intensity))
                main_series.append(float(mz), 0.0)

        # Add series to chart
        chart.addSeries(main_series)
        if highlight_series.count() > 0:
            chart.addSeries(highlight_series)

        # Reattach axes
        x_axis = (
            chart.axes(Qt.Orientation.Horizontal)[0]
            if chart.axes(Qt.Orientation.Horizontal)
            else None
        )
        y_axis = (
            chart.axes(Qt.Orientation.Vertical)[0]
            if chart.axes(Qt.Orientation.Vertical)
            else None
        )

        if x_axis and y_axis:
            main_series.attachAxis(x_axis)
            main_series.attachAxis(y_axis)
            if highlight_series.count() > 0:
                highlight_series.attachAxis(x_axis)
                highlight_series.attachAxis(y_axis)

    def create_large_msms_chart(self):
        """Create a large MSMS spectrum chart"""
        chart = QChart()

        # Get precursor intensity for display
        precursor_intensity = self.spectrum_data.get("precursor_intensity", 0)
        intensity_text = (
            f"{precursor_intensity:.1e}" if precursor_intensity > 0 else "N/A"
        )

        chart.setTitle(
            f"MSMS Spectrum\n"
            f"RT: {self.spectrum_data['rt']:.2f} min, "
            f"Precursor: {self.spectrum_data['precursor_mz']:.4f}, "
            f"Intensity: {intensity_text}"
        )

        # Create series for the spectrum
        series = QLineSeries()

        # Add spectrum data as vertical lines (stick spectrum)
        mz_array = self.spectrum_data["mz"]
        intensity_array = self.spectrum_data["intensity"]

        # Normalize intensities to 0-100 for better visualization
        if len(intensity_array) > 0:
            max_intensity = np.max(intensity_array)
            if max_intensity > 0:
                normalized_intensity = (intensity_array / max_intensity) * 100
            else:
                normalized_intensity = intensity_array
        else:
            normalized_intensity = intensity_array

        # Create stick spectrum by adding points at baseline and peak height
        for mz, intensity in zip(mz_array, normalized_intensity):
            # Add baseline point
            series.append(float(mz), 0.0)
            # Add peak point
            series.append(float(mz), float(intensity))
            # Add baseline point after peak
            series.append(float(mz), 0.0)

        # Style the series
        pen = QPen(QColor(0, 100, 200))  # Blue color
        pen.setWidth(2)  # Thicker lines for larger view
        series.setPen(pen)

        chart.addSeries(series)

        # Create and configure axes
        x_axis = QValueAxis()
        x_axis.setTitleText("m/z")
        x_axis.setLabelFormat("%.2f")
        x_axis.setTickCount(10)

        y_axis = QValueAxis()
        y_axis.setTitleText("Intensity (%)")
        y_axis.setLabelFormat("%.0f")
        y_axis.setRange(0, 105)  # 0-100% plus some padding
        y_axis.setTickCount(6)

        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

        series.attachAxis(x_axis)
        series.attachAxis(y_axis)

        # Set x-axis range to show all data with some padding
        if len(mz_array) > 0:
            mz_min = np.min(mz_array)
            mz_max = np.max(mz_array)
            mz_range = mz_max - mz_min
            padding = mz_range * 0.05 if mz_range > 0 else 1.0
            x_axis.setRange(mz_min - padding, mz_max + padding)
        else:
            x_axis.setRange(0, 1000)

        # Hide legend since we only have one series
        chart.legend().setVisible(False)

        return chart


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
                font-size: 10px;
                font-weight: bold;
            }
        """)
        self.hover_label.hide()
        self.hover_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )  # Don't interfere with mouse events

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
            x_range = (
                self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
            )
            y_range = (
                self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
            )

            self.zoom_anchor_x = self.interaction_start_x_range[0] + rel_x * x_range
            self.zoom_anchor_y = (
                self.interaction_start_y_range[1] - rel_y * y_range
            )  # Y is inverted

            self.setCursor(Qt.CursorShape.SizeAllCursor)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events"""
        # Check if we're tracking a potential right-click for context menu
        if (
            self.right_click_pending
            and self.mouse_press_pos is not None
            and event.buttons() & Qt.MouseButton.RightButton
        ):
            current_pos = event.position().toPoint()
            distance = (
                (current_pos.x() - self.mouse_press_pos.x()) ** 2
                + (current_pos.y() - self.mouse_press_pos.y()) ** 2
            ) ** 0.5

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

                x_range = (
                    self.interaction_start_x_range[1]
                    - self.interaction_start_x_range[0]
                )
                y_range = (
                    self.interaction_start_y_range[1]
                    - self.interaction_start_y_range[0]
                )
                self.zoom_anchor_x = self.interaction_start_x_range[0] + rel_x * x_range
                self.zoom_anchor_y = self.interaction_start_y_range[0] + rel_y * y_range

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
                min_distance = self._find_closest_distance_to_series(
                    mouse_x, mouse_y, points
                )

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
                        font-size: 10px;
                        font-weight: bold;
                        color: {color_hex};
                    }}
                """)

                # Position the label near the mouse cursor
                label_pos = event.position().toPoint()
                label_pos.setX(
                    label_pos.x() + 10
                )  # Offset to avoid covering the cursor
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
        x_axis = (
            self.chart().axes(Qt.Orientation.Horizontal)[0]
            if self.chart().axes(Qt.Orientation.Horizontal)
            else None
        )
        y_axis = (
            self.chart().axes(Qt.Orientation.Vertical)[0]
            if self.chart().axes(Qt.Orientation.Vertical)
            else None
        )

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
            distance = self._point_to_line_segment_distance(
                norm_mouse_x, norm_mouse_y, norm_x1, norm_y1, norm_x2, norm_y2
            )
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
                distance = (
                    (release_pos.x() - self.mouse_press_pos.x()) ** 2
                    + (release_pos.y() - self.mouse_press_pos.y()) ** 2
                ) ** 0.5

                # Show context menu only if:
                # 1. Time between press and release is less than timeout
                # 2. Mouse didn't move more than threshold (not a drag)
                if time_diff <= self.click_timeout and distance <= self.drag_threshold:
                    plot_area = self.chart().plotArea()
                    if plot_area.contains(event.position()):
                        # Convert mouse position to data coordinates
                        rel_x = (
                            event.position().x() - plot_area.left()
                        ) / plot_area.width()

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
        y_offset = (
            delta_y * y_per_pixel
        )  # Positive because Y axis is inverted in screen coordinates

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
        y_zoom_factor = 1.0 + (
            delta_y * zoom_sensitivity
        )  # Inverted for intuitive behavior

        # Clamp zoom factors to reasonable limits
        x_zoom_factor = max(0.1, min(10.0, x_zoom_factor))
        y_zoom_factor = max(0.1, min(10.0, y_zoom_factor))

        # Calculate new ranges anchored at the mouse click position
        original_x_range = (
            self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
        )
        original_y_range = (
            self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
        )

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
    ):
        super().__init__(parent)

        # Configure as independent window
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.compound_data = compound_data
        self.adduct = adduct
        self.file_manager = file_manager
        self.eic_data = {}
        self.group_shifts = {}
        self.integration_callback = integration_callback
        self.grouping_column = "group"  # Default grouping column

        # Store defaults (use application defaults if none provided)
        self.defaults = (
            defaults
            if defaults is not None
            else {
                "mz_tolerance_ppm": 5.0,
                "separate_groups": True,
                "rt_shift_min": 1.0,
                "crop_rt_window": False,
                "normalize_samples": False,
            }
        )

        # Use pre-calculated m/z value if provided, otherwise calculate
        if mz_value is not None:
            self.target_mz = mz_value
            self.polarity = polarity
        else:
            # Fallback: Calculate target m/z using compound manager
            try:
                # Use the compound manager to calculate m/z properly
                from .compound_manager import CompoundManager

                temp_manager = CompoundManager()

                # Create a temporary DataFrame with just this compound
                temp_compound_data = pd.DataFrame([compound_data])
                temp_manager.compounds_data = temp_compound_data

                self.target_mz = temp_manager.calculate_compound_mz(
                    compound_data["Name"], adduct
                )

                if self.target_mz is None:
                    raise ValueError("Could not calculate m/z value")

                self.polarity = None  # Polarity not available in fallback mode

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to calculate m/z: {str(e)}"
                )
                self.target_mz = 0.0
                self.polarity = None

        # Initialize scatter plot attributes
        self.scatter_plot_view = None
        self.scatter_separator_label = None
        self.scatter_plot_menu_text = "View 2D scatter plot (RT vs m/z)"

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

        self.init_ui()
        self.extract_eic_data()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"EIC: {self.compound_data['Name']} - {self.adduct}")
        self.setGeometry(200, 200, 1400, 800)

        layout = QHBoxLayout(self)

        # Create splitter for left panel and right panel
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel for compound info and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel for chart
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions (left panel much narrower)
        splitter.setSizes([250, 1150])

        layout.addWidget(splitter)

    def create_left_panel(self) -> QWidget:
        """Create the left panel with compound info and controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Compound information
        info_group = self.create_compound_info_group()
        layout.addWidget(info_group)

        # Extraction parameters
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)

        # Extract button
        self.extract_btn = QPushButton("Extract EIC")
        self.extract_btn.clicked.connect(self.extract_eic_data)
        layout.addWidget(self.extract_btn)

        # Reset View button
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.reset_view_btn.setEnabled(False)  # Disabled until data is loaded
        layout.addWidget(self.reset_view_btn)

        # Progress bar (underneath the buttons)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Group settings table
        self.create_group_settings_table()
        layout.addWidget(self.group_settings_box)

        layout.addStretch()
        return panel

    def create_right_panel(self) -> QWidget:
        """Create the right panel with the chart and boxplot"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create a vertical splitter for EIC chart and boxplot
        self.eic_boxplot_splitter = QSplitter(Qt.Orientation.Vertical)
        self.eic_boxplot_splitter.setChildrenCollapsible(
            False
        )  # Prevent complete collapse

        # Main chart (EIC)
        self.chart_view = self.create_chart()
        self.eic_boxplot_splitter.addWidget(self.chart_view)

        # Boxplot widget
        self.create_boxplot_widget()
        self.eic_boxplot_splitter.addWidget(self.boxplot_widget)

        # Set initial sizes (75% EIC, 25% boxplot)
        self.eic_boxplot_splitter.setSizes([750, 250])
        self.eic_boxplot_splitter.setStretchFactor(0, 1)  # EIC chart is stretchable
        self.eic_boxplot_splitter.setStretchFactor(
            1, 1
        )  # Boxplot adapts with available space

        # Add the splitter to the main layout
        layout.addWidget(self.eic_boxplot_splitter)

        return panel

    def create_compound_info_group(self) -> QGroupBox:
        """Create the compound information group"""
        group = QGroupBox("Compound Information")
        layout = QVBoxLayout(group)

        # Determine compound info display
        formula_info = ""
        if (
            "ChemicalFormula" in self.compound_data
            and self.compound_data["ChemicalFormula"]
        ):
            formula_info = (
                f"<b>Formula:</b> {self.compound_data['ChemicalFormula']}<br>"
            )
        elif "Mass" in self.compound_data and self.compound_data["Mass"]:
            formula_info = f"<b>Mass:</b> {self.compound_data['Mass']} Da<br>"

        # Compound info
        compound_info = QLabel(
            f"<b>Compound:</b> {self.compound_data['Name']}<br>"
            f"{formula_info}"
            f"<b>Adduct:</b> {self.adduct}<br>"
            f"<b>m/z:</b> {format_mz(self.target_mz)}"
        )
        layout.addWidget(compound_info)

        # RT info
        rt_info = QLabel(
            f"<b>RT:</b> {format_retention_time(self.compound_data['RT_min'])}<br>"
            f"<b>RT Window:</b> {format_retention_time(self.compound_data['RT_start_min'])} - "
            f"{format_retention_time(self.compound_data['RT_end_min'])}"
        )
        layout.addWidget(rt_info)

        return group

    def create_control_panel(self) -> QGroupBox:
        """Create the control panel"""
        group = QGroupBox("Extraction Parameters")
        layout = QFormLayout(group)

        # EIC calculation method
        self.eic_method_combo = QComboBox()
        self.eic_method_combo.addItems(["Sum of all signals", "Most intensive signal"])
        self.eic_method_combo.setCurrentIndex(0)  # Default to sum
        layout.addRow("EIC Method:", self.eic_method_combo)

        # m/z tolerance in ppm (primary)
        self.mz_tolerance_ppm_spin = QDoubleSpinBox()
        self.mz_tolerance_ppm_spin.setRange(0.1, 10000.0)
        self.mz_tolerance_ppm_spin.setValue(
            self.defaults["mz_tolerance_ppm"]
        )  # Use default
        self.mz_tolerance_ppm_spin.setSuffix(" ppm")
        self.mz_tolerance_ppm_spin.setDecimals(1)
        self.mz_tolerance_ppm_spin.setSingleStep(1.0)
        self.mz_tolerance_ppm_spin.valueChanged.connect(self.update_mz_tolerance_da)
        layout.addRow("m/z Tolerance (ppm):", self.mz_tolerance_ppm_spin)

        # m/z tolerance in Da (linked to ppm)
        self.mz_tolerance_da_spin = QDoubleSpinBox()
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
        self.grouping_column_combo.currentTextChanged.connect(
            self.on_grouping_column_changed
        )
        layout.addRow("Group by Column:", self.grouping_column_combo)

        # Group separation
        self.separate_groups_cb = QCheckBox("Separate by groups")
        self.separate_groups_cb.setChecked(
            self.defaults["separate_groups"]
        )  # Use default
        self.separate_groups_cb.stateChanged.connect(self.update_plot)
        layout.addRow(self.separate_groups_cb)

        # RT shift for group separation (more flexible range)
        self.rt_shift_spin = QDoubleSpinBox()
        self.rt_shift_spin.setRange(0.0, 60.0)  # Allow up to 60 minutes
        self.rt_shift_spin.setValue(self.defaults["rt_shift_min"])  # Use default
        self.rt_shift_spin.setSuffix(" min")
        self.rt_shift_spin.setDecimals(1)
        self.rt_shift_spin.setEnabled(True)  # Always enabled
        self.rt_shift_spin.valueChanged.connect(self.update_plot)
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

        return group

    def create_group_settings_table(self):
        """Create the group settings table for EIC display controls"""
        # Create a collapsible box for the table
        self.group_settings_box = CollapsibleBox("Group Display Settings")

        # Create the table
        self.group_settings_table = QTableWidget()
        self.group_settings_table.setColumnCount(4)
        self.group_settings_table.setHorizontalHeaderLabels(
            ["Scaling", "Plot", "Neg.", "Line Width"]
        )

        # Configure table appearance
        self.group_settings_table.setAlternatingRowColors(True)
        self.group_settings_table.horizontalHeader().setStretchLastSection(False)
        self.group_settings_table.verticalHeader().setVisible(True)
        self.group_settings_table.setMaximumHeight(800)
        self.group_settings_table.setMinimumHeight(340)

        # Set column widths
        self.group_settings_table.setColumnWidth(0, 95)  # Scaling column
        self.group_settings_table.setColumnWidth(1, 30)  # Plot checkbox column
        self.group_settings_table.setColumnWidth(2, 30)  # Negative checkbox column
        self.group_settings_table.setColumnWidth(3, 60)  # Line Width column

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

        if (
            hasattr(self.file_manager, "files_data")
            and self.file_manager.files_data is not None
        ):
            # Get all possible group values from the entire sample matrix
            if self.grouping_column in self.file_manager.files_data.columns:
                group_values = (
                    self.file_manager.files_data[self.grouping_column].dropna().unique()
                )
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
            # Set row header (group name) with group color
            header_item = QTableWidgetItem(group)

            # Get and apply the group color
            group_color = self._get_group_color(group)
            if group_color:
                # group_color is already a QColor, create QBrush from it
                header_item.setForeground(QColor(group_color))

            table.setVerticalHeaderItem(row, header_item)

            # Column 0: Scaling factor (QDoubleSpinBox)
            scaling_spin = QDoubleSpinBox()
            scaling_spin.setRange(0.00001, 100000.0)
            scaling_spin.setValue(self.group_settings[group]["scaling"])
            scaling_spin.setDecimals(5)
            scaling_spin.setSingleStep(0.1)
            scaling_spin.valueChanged.connect(
                lambda value, g=group: self.on_group_setting_changed(
                    g, "scaling", value
                )
            )
            table.setCellWidget(row, 0, scaling_spin)

            # Column 1: Plot checkbox
            plot_checkbox = QCheckBox()
            plot_checkbox.setChecked(self.group_settings[group]["plot"])
            plot_checkbox.stateChanged.connect(
                lambda state, g=group: self.on_group_setting_changed(
                    g, "plot", state == Qt.CheckState.Checked.value
                )
            )
            # Center the checkbox
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(plot_checkbox)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            table.setCellWidget(row, 1, checkbox_widget)

            # Column 2: Negative intensities checkbox
            negative_checkbox = QCheckBox()
            negative_checkbox.setChecked(self.group_settings[group]["negative"])
            negative_checkbox.stateChanged.connect(
                lambda state, g=group: self.on_group_setting_changed(
                    g, "negative", state == Qt.CheckState.Checked.value
                )
            )
            # Center the checkbox
            neg_checkbox_widget = QWidget()
            neg_checkbox_layout = QHBoxLayout(neg_checkbox_widget)
            neg_checkbox_layout.addWidget(negative_checkbox)
            neg_checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            neg_checkbox_layout.setContentsMargins(0, 0, 0, 0)
            table.setCellWidget(row, 2, neg_checkbox_widget)

            # Column 3: Line width (QDoubleSpinBox)
            width_spin = QDoubleSpinBox()
            width_spin.setRange(0.5, 10.0)
            width_spin.setValue(self.group_settings[group]["line_width"])
            width_spin.setDecimals(1)
            width_spin.setSingleStep(0.5)
            width_spin.valueChanged.connect(
                lambda value, g=group: self.on_group_setting_changed(
                    g, "line_width", value
                )
            )
            table.setCellWidget(row, 3, width_spin)

    def _populate_grouping_columns(self):
        """Populate the grouping column selector with available columns"""
        self.grouping_column_combo.clear()

        # Get available columns from files_data
        files_data = self.file_manager.get_files_data()
        if not files_data.empty:
            # Exclude system columns that shouldn't be used for grouping
            exclude_columns = {"Filepath", "filename"}
            available_columns = [
                col for col in files_data.columns if col not in exclude_columns
            ]

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
                self.populate_group_settings_table()
                self.update_plot()

    def on_group_setting_changed(self, group, setting, value):
        """Handle changes to group settings"""
        if group in self.group_settings:
            self.group_settings[group][setting] = value
            # Update the plot when settings change without resetting the view
            self.update_plot(preserve_view=True)

    def create_boxplot_widget(self):
        """Create the tabbed widget for boxplots and peak area table"""
        # Create the main tabbed widget
        self.boxplot_widget = QTabWidget()

        # Tab 1: Boxplot
        self.boxplot_figure = Figure(figsize=(10, 3))
        self.boxplot_canvas = FigureCanvas(self.boxplot_figure)
        self.boxplot_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.boxplot_canvas.setMinimumSize(0, 0)
        self.boxplot_widget.addTab(self.boxplot_canvas, "Boxplot")

        self.boxplot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

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
        self.peak_area_table.setHorizontalHeaderLabels(
            ["Group", "Sample Name", "Peak Area"]
        )

        # Configure table appearance
        self.peak_area_table.setAlternatingRowColors(True)
        self.peak_area_table.setSortingEnabled(True)
        self.peak_area_table.horizontalHeader().setStretchLastSection(True)
        self.peak_area_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.peak_area_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        peak_area_layout.addWidget(self.peak_area_table)
        self.boxplot_widget.addTab(peak_area_tab, "Peak Area Table")

        # Tab 3: Summary Statistics Table
        summary_stats_tab = QWidget()
        summary_stats_layout = QVBoxLayout(summary_stats_tab)

        # Buttons for summary statistics table
        summary_buttons_layout = QHBoxLayout()

        self.copy_summary_excel_btn = QPushButton("Copy as Excel Tab-delimited")
        self.copy_summary_excel_btn.clicked.connect(
            self._copy_summary_stats_table_excel
        )
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
        self.summary_stats_table.horizontalHeader().setStretchLastSection(True)
        self.summary_stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.summary_stats_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        summary_stats_layout.addWidget(self.summary_stats_table)
        self.boxplot_widget.addTab(summary_stats_tab, "Summary Statistics")

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
            self.boundary_info_label.setText(
                f"Peak boundary: {line_x:.2f} min (add second boundary)"
            )

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
        start_rt = (
            min(self.peak_start_rt, self.peak_end_rt)
            if self.peak_start_rt and self.peak_end_rt
            else 0
        )
        end_rt = (
            max(self.peak_start_rt, self.peak_end_rt)
            if self.peak_start_rt and self.peak_end_rt
            else 0
        )

        if start_rt >= end_rt:
            return

        # Collect data for boxplot, table, and apex detection
        boxplot_data = {}  # group_name -> list of integrated areas
        table_data = []  # list of tuples (group, sample_name, peak_area)
        apex_rt = None
        apex_intensity = float("-inf")

        separate_groups = self.separate_groups_cb.isChecked()

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
            integrated_area = self._calculate_peak_area_with_boundaries(
                original_rt, intensity, start_rt, end_rt
            )

            # Track apex: highest intensity within the integration window
            for rt_val, intensity_val in zip(original_rt, intensity):
                try:
                    rt_float = float(rt_val)
                    intensity_float = float(intensity_val)
                except (TypeError, ValueError):
                    continue

                if start_rt <= rt_float <= end_rt and not np.isnan(intensity_float):
                    if intensity_float > apex_intensity:
                        apex_intensity = intensity_float
                        apex_rt = rt_float

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
                patch.set_facecolor(
                    (color.red() / 255, color.green() / 255, color.blue() / 255, 0.7)
                )

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
            ax.set_title(
                f"Peak Integration ({start_rt:.2f} - {end_rt:.2f} min, apex {apex_rt:.2f} min)"
            )
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
                self.integration_callback.__self__.update_peak_integration_samples(
                    compound_name, ion_name, sample_data_list
                )

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

        # Set number of rows
        self.peak_area_table.setRowCount(len(table_data))

        # Populate the table
        for row, (group, sample_name, peak_area) in enumerate(table_data):
            # Group column
            group_item = QTableWidgetItem(str(group))
            self.peak_area_table.setItem(row, 0, group_item)

            # Sample name column
            sample_item = QTableWidgetItem(str(sample_name))
            self.peak_area_table.setItem(row, 1, sample_item)

            # Peak area column (formatted to scientific notation)
            area_item = QTableWidgetItem(f"{peak_area:.2e}")
            area_item.setData(
                Qt.ItemDataRole.UserRole, peak_area
            )  # Store actual value for sorting
            self.peak_area_table.setItem(row, 2, area_item)

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

        # Set number of rows
        self.summary_stats_table.setRowCount(len(stats_data))

        # Populate the table
        for row, stats in enumerate(stats_data):
            # Group name
            group_item = QTableWidgetItem(stats["group"])
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
                    item = QTableWidgetItem(f"{value:.2f}")
                else:
                    item = QTableWidgetItem(f"{value:.2e}")
                item.setData(
                    Qt.ItemDataRole.UserRole, value
                )  # Store actual value for sorting
                self.summary_stats_table.setItem(row, col, item)

        # Auto-resize columns to fit content
        self.summary_stats_table.resizeColumnsToContents()

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
            r_header = (
                header.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("%", "pct")
            )
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
            r_header = (
                header.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("%", "pct")
            )
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

    def _calculate_peak_area_with_boundaries(
        self, rt_array, intensity_array, start_rt, end_rt
    ):
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

        # Temporarily disconnect signal to avoid recursion
        self.mz_tolerance_da_spin.valueChanged.disconnect()
        self.mz_tolerance_da_spin.setValue(da_value)
        self.mz_tolerance_da_spin.valueChanged.connect(self.update_mz_tolerance_ppm)

    def update_mz_tolerance_ppm(self):
        """Update ppm value when Da value changes"""
        da_value = self.mz_tolerance_da_spin.value()
        ppm = (da_value * 1e6) / self.target_mz if self.target_mz > 0 else 0

        # Temporarily disconnect signal to avoid recursion
        self.mz_tolerance_ppm_spin.valueChanged.disconnect()
        self.mz_tolerance_ppm_spin.setValue(ppm)
        self.mz_tolerance_ppm_spin.valueChanged.connect(self.update_mz_tolerance_da)

    def create_chart(self) -> InteractiveChartView:
        """Create the chart widget"""
        self.chart = QChart()
        self.chart.setTitle("")

        # Create axes with better formatting
        self.x_axis = QValueAxis()
        self.x_axis.setTitleText("Retention Time (min)")
        self.x_axis.setLabelFormat("%.1f")  # One decimal place
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

        # Store original ranges for reset functionality
        self.original_x_range = None
        self.original_y_range = None

        return chart_view

    def reset_view(self):
        """Reset the chart view to show all data"""
        if self.chart.series():
            # Calculate the full range of all series
            min_x = float("inf")
            max_x = float("-inf")
            min_y = float("inf")
            max_y = float("-inf")

            for series in self.chart.series():
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
                if not self.normalize_cb.isChecked():
                    y_padding = (max_y - min_y) * 0.05
                    y_axis = self.chart.axes(Qt.Orientation.Vertical)[0]
                    y_axis.setRange(max(0, min_y - y_padding), max_y + y_padding)

    def extract_eic_data(self):
        """Extract EIC data in a separate thread"""
        if self.target_mz == 0.0:
            QMessageBox.warning(self, "Warning", "Invalid m/z value!")
            return

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.extract_btn.setEnabled(False)

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

        self.extraction_worker.progress.connect(self.progress_bar.setValue)
        self.extraction_worker.finished.connect(self.on_extraction_finished)
        self.extraction_worker.error.connect(self.on_extraction_error)

        self.extraction_worker.start()

    def on_extraction_finished(self, eic_data: dict):
        """Handle completion of EIC extraction"""
        self.eic_data = eic_data
        self.progress_bar.setVisible(False)
        self.extract_btn.setEnabled(True)

        # Calculate group shifts
        self.calculate_group_shifts()

        # Populate group settings table with extracted groups
        self.populate_group_settings_table()

        # Update plot
        self.update_plot()

        # Update scatter plot if it exists
        if hasattr(self, "scatter_plot_view") and self.scatter_plot_view is not None:
            self.scatter_plot_view.update_scatter_plot()

    def on_extraction_error(self, error_message: str):
        """Handle EIC extraction error"""
        self.progress_bar.setVisible(False)
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

        if (
            hasattr(self.file_manager, "files_data")
            and self.file_manager.files_data is not None
        ):
            # Get all possible group values from the entire sample matrix
            if self.grouping_column in self.file_manager.files_data.columns:
                group_values = (
                    self.file_manager.files_data[self.grouping_column].dropna().unique()
                )
                for value in group_values:
                    groups.add(str(value))

        # Also include groups from current EIC data (in case some aren't in files_data)
        if self.eic_data:
            for data in self.eic_data.values():
                if self.grouping_column in data["metadata"]:
                    group_value = data["metadata"][self.grouping_column]
                    groups.add(
                        str(group_value) if group_value is not None else "Unknown"
                    )

        # Sort groups and assign shifts (using natural sort)
        sorted_groups = natsorted(groups)
        shift_amount = self.rt_shift_spin.value()

        self.group_shifts = {}
        for i, group in enumerate(sorted_groups):
            self.group_shifts[group] = i * shift_amount

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

        # Clear existing series
        self.chart.removeAllSeries()

        # Enable reset view button now that we have data
        self.reset_view_btn.setEnabled(True)

        separate_groups = self.separate_groups_cb.isChecked()
        crop_rt = self.crop_rt_cb.isChecked()
        normalize = self.normalize_cb.isChecked()

        # Get RT window if cropping is enabled
        rt_start = self.compound_data["RT_start_min"] if crop_rt else None
        rt_end = self.compound_data["RT_end_min"] if crop_rt else None

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
                rt_plot = rt + shift

            groups_data[group].append(
                {
                    "rt": rt_plot,
                    "intensity": intensity,
                    "metadata": metadata,
                    "filepath": filepath,
                }
            )

        # Create separate series for each file, but group them for legend display
        # Iterate through groups in the same sorted order as group_shifts
        sorted_groups = natsorted(groups_data.keys())

        for group_name in sorted_groups:
            # Check group settings - skip if group should not be plotted
            group_settings = self.group_settings.get(
                group_name,
                {"scaling": 1.0, "plot": True, "negative": False, "line_width": 1.0},
            )

            # Get group color
            group_color = self._get_group_color(group_name)

            # Always add at least one legend entry per group, even if no data or not plotted
            # This ensures the legend order matches the separation order
            group_files = groups_data.get(group_name, [])

            if not group_settings["plot"] or len(group_files) == 0:
                # Create an invisible/empty series just for the legend entry
                series = QLineSeries()
                if separate_groups:
                    shift = self.group_shifts.get(group_name, 0.0)
                    legend_name = f"{group_name} (+ {shift:.1f} min)"
                else:
                    legend_name = group_name
                series.setName(legend_name)

                # Apply group color (even for invisible series)
                if group_color:
                    color = QColor(group_color)
                    color.setAlpha(180)
                    pen = QPen(color)
                    pen.setWidthF(group_settings.get("line_width", 1.0))
                    series.setPen(pen)

                # Add the series to chart (won't be visible but shows in legend)
                self.chart.addSeries(series)
                series.attachAxis(self.x_axis)
                series.attachAxis(self.y_axis)
                continue  # Skip to next group

            first_file_in_group = True

            for file_data in group_files:
                rt = file_data["rt"]
                intensity = file_data["intensity"]

                # Apply group scaling factor
                intensity = intensity * group_settings["scaling"]

                # Apply negative intensities if enabled for this group
                if group_settings["negative"]:
                    intensity = -intensity

                # Create individual series for each file
                series = QLineSeries()

                # Store the sample filename for hover tooltips
                filepath = file_data["filepath"]
                filename = file_data["metadata"].get(
                    "filename",
                    filepath.split("\\")[-1]
                    if "\\" in filepath
                    else filepath.split("/")[-1]
                    if "/" in filepath
                    else filepath,
                )

                # Store custom property for hover detection
                series.setProperty("sample_filename", filename)

                # Only the first file in each group gets the group name for legend
                if first_file_in_group:
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
                    series.append(float(x), float(y))

                # Apply group color with transparency and line width
                if group_color:
                    color = QColor(group_color)
                    color.setAlpha(
                        180
                    )  # Make lines semi-transparent (0-255, 180 = ~70% opacity)
                    pen = QPen(color)
                    pen.setWidthF(group_settings["line_width"])
                    series.setPen(pen)

                # Add series to chart
                self.chart.addSeries(series)
                series.attachAxis(self.x_axis)
                series.attachAxis(self.y_axis)

        # Add reference lines
        self._add_reference_lines(groups_data, separate_groups)

        # Show legend with better formatting
        legend = self.chart.legend()
        legend.setVisible(True)
        legend.setAlignment(Qt.AlignmentFlag.AlignRight)
        legend.setMarkerShape(legend.MarkerShape.MarkerShapeRectangle)

        # Hide legend markers for series with empty names
        for marker in legend.markers():
            series = marker.series()
            if series.name() == "":
                marker.setVisible(False)

        # Update Y-axis title and range based on normalization status
        if normalize:
            self.y_axis.setTitleText("Normalized Intensity")
            # For normalized data, set explicit range 0-1 with some padding
            # Use a timer to ensure the range is applied after series are fully added
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(50, lambda: self.y_axis.setRange(-0.05, 1.05))
        else:
            self.y_axis.setTitleText("Intensity")
            # For non-normalized data, calculate range from actual data
            self._set_y_axis_from_data()

        # Restore saved view or reset to show all data
        from PyQt6.QtCore import QTimer

        if preserve_view and saved_x_range and saved_y_range:
            # Restore the saved axis ranges
            def restore_ranges():
                x_axis = self.chart.axes(Qt.Orientation.Horizontal)[0]
                y_axis = self.chart.axes(Qt.Orientation.Vertical)[0]
                x_axis.setRange(saved_x_range[0], saved_x_range[1])
                # For normalized data, enforce the normalized range
                if normalize:
                    y_axis.setRange(-0.05, 1.05)
                else:
                    y_axis.setRange(saved_y_range[0], saved_y_range[1])

            QTimer.singleShot(60, restore_ranges)
        else:
            # Automatically reset view to show all data after any changes
            QTimer.singleShot(
                50, self.reset_view
            )  # Small delay to ensure chart is fully updated

        # Update series cache for hover detection
        if hasattr(self.chart_view, "update_series_cache"):
            self.chart_view.update_series_cache()

        # Re-add peak boundary lines if they exist
        self._restore_peak_boundary_lines()

        # Automatically add scatter plot when EIC data is loaded
        if (
            not hasattr(self, "_scatter_plot_auto_added")
            or not self._scatter_plot_auto_added
        ):
            self._scatter_plot_auto_added = True
            # Use a timer to add scatter plot after the EIC chart is fully rendered
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(100, self.auto_add_scatter_plot)

    def _add_reference_lines(self, groups_data, separate_groups):
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

        if compound_rt > 0:
            if separate_groups:
                # Add dashed vertical line for each group at the compound RT (with group shifts)
                for group_name in groups_data.keys():
                    # Apply the same group shift as applied to the data
                    shift = self.group_shifts.get(group_name, 0.0)
                    reference_rt = compound_rt + shift

                    # Create vertical line series
                    vertical_line = QLineSeries()
                    vertical_line.setName("")  # No legend entry

                    # Get Y range for the line (extend beyond visible data)
                    y_min = y_axis.min()
                    y_max = y_axis.max()
                    y_range = y_max - y_min
                    line_y_start = y_min - y_range * 0.1
                    line_y_end = y_max + y_range * 0.1

                    vertical_line.append(reference_rt, line_y_start)
                    vertical_line.append(reference_rt, line_y_end)

                    # Style: dashed grey line
                    vertical_pen = QPen(QColor(128, 128, 128))  # Grey
                    vertical_pen.setWidth(1)
                    vertical_pen.setStyle(Qt.PenStyle.DashLine)
                    vertical_line.setPen(vertical_pen)

                    self.chart.addSeries(vertical_line)
                    vertical_line.attachAxis(x_axis)
                    vertical_line.attachAxis(y_axis)
            else:
                # Add single vertical line at compound RT (no shift when not separating groups)
                vertical_line = QLineSeries()
                vertical_line.setName("")  # No legend entry

                # Get Y range for the line (extend beyond visible data)
                y_min = y_axis.min()
                y_max = y_axis.max()
                y_range = y_max - y_min
                line_y_start = y_min - y_range * 0.1
                line_y_end = y_max + y_range * 0.1

                vertical_line.append(compound_rt, line_y_start)
                vertical_line.append(compound_rt, line_y_end)

                # Style: dashed grey line
                vertical_pen = QPen(QColor(128, 128, 128))  # Grey
                vertical_pen.setWidth(1)
                vertical_pen.setStyle(Qt.PenStyle.DashLine)
                vertical_line.setPen(vertical_pen)

                self.chart.addSeries(vertical_line)
                vertical_line.attachAxis(x_axis)
                vertical_line.attachAxis(y_axis)

    def _set_y_axis_from_data(self):
        """Set Y-axis range based on actual data in the chart"""
        if not self.chart.series():
            return

        # Find Y data range from all series
        min_y, max_y = float("inf"), float("-inf")

        for series in self.chart.series():
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
            padding = y_range * 0.05 if y_range > 0 else max_y * 0.05
            y_min = max(0, min_y - padding)  # Don't go below 0 for intensity
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

    def show_context_menu(self, rt_value: float, position: QPointF):
        """Show context menu at the specified position"""
        context_menu = QMenu(self)

        # Add RT info at the top
        rt_action = QAction(f"RT: {rt_value:.2f} min", self)
        rt_action.setEnabled(False)  # Make it non-clickable header
        context_menu.addAction(rt_action)

        context_menu.addSeparator()

        # Add peak boundary options
        if len(self.peak_boundary_lines) == 0:
            # No boundaries set - offer to add first one
            add_boundary_action = QAction("Add peak boundary", self)
            add_boundary_action.triggered.connect(
                lambda: self.add_peak_boundary(rt_value)
            )
            context_menu.addAction(add_boundary_action)
        elif len(self.peak_boundary_lines) == 1:
            # One boundary set - offer to add second one or remove all
            add_second_boundary_action = QAction("Add second peak boundary", self)
            add_second_boundary_action.triggered.connect(
                lambda: self.add_peak_boundary(rt_value)
            )
            context_menu.addAction(add_second_boundary_action)

            remove_boundary_action = QAction("Remove peak boundaries", self)
            remove_boundary_action.triggered.connect(self.remove_peak_boundaries)
            context_menu.addAction(remove_boundary_action)
        else:  # len == 2
            # Both boundaries set - only offer to remove them
            remove_boundary_action = QAction("Remove peak boundaries", self)
            remove_boundary_action.triggered.connect(self.remove_peak_boundaries)
            context_menu.addAction(remove_boundary_action)

        context_menu.addSeparator()

        # Add MS1 viewing option
        ms1_action = QAction("View MS1 spectra at RT", self)
        ms1_action.triggered.connect(lambda: self.view_ms1_spectra(rt_value))
        context_menu.addAction(ms1_action)

        context_menu.addSeparator()

        # Add MSMS viewing options
        msms_3s_action = QAction("View MSMS (±3 seconds)", self)
        msms_3s_action.triggered.connect(
            lambda: self.view_msms_spectra(rt_value, 3.0 / 60.0)
        )
        context_menu.addAction(msms_3s_action)

        msms_6s_action = QAction("View MSMS (±6 seconds)", self)
        msms_6s_action.triggered.connect(
            lambda: self.view_msms_spectra(rt_value, 6.0 / 60.0)
        )
        context_menu.addAction(msms_6s_action)

        msms_9s_action = QAction("View MSMS (±9 seconds)", self)
        msms_9s_action.triggered.connect(
            lambda: self.view_msms_spectra(rt_value, 9.0 / 60.0)
        )
        context_menu.addAction(msms_9s_action)

        # Show the menu at the clicked position
        global_pos = self.chart_view.mapToGlobal(position.toPoint())
        context_menu.exec(global_pos)

    def view_msms_spectra(self, rt_center: float, rt_window: float):
        """View MSMS spectra within the specified RT window"""
        try:
            # Calculate RT window
            rt_start = rt_center - rt_window
            rt_end = rt_center + rt_window

            # Find MSMS spectra
            msms_spectra = self.find_msms_spectra(rt_start, rt_end)

            if not msms_spectra:
                QMessageBox.information(
                    self,
                    "No MSMS Found",
                    f"No MSMS spectra found for m/z {self.target_mz:.4f} "
                    f"in RT window {rt_center:.2f} ± {rt_window:.1f} min",
                )
                return

            # Open MSMS viewer window
            msms_viewer = MSMSViewerWindow(
                msms_spectra,
                self.target_mz,
                rt_center,
                rt_window,
                self.compound_data["Name"],
                self.adduct,
                self,
            )
            msms_viewer.show()

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to view MSMS spectra: {str(e)}"
            )

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

            # Open MS1 viewer window
            ms1_viewer = MS1ViewerWindow(
                ms1_spectra,
                self.target_mz,
                rt_center,
                self.compound_data["Name"],
                self.adduct,
                self.mz_tolerance_da_spin.value(),
                self.compound_data.get("ChemicalFormula", ""),
                self,
            )
            ms1_viewer.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to view MS1 spectra: {str(e)}")

    def find_ms1_spectra(self, rt_center: float):
        """Find MS1 spectra closest to the specified RT for each file"""
        ms1_spectra = {}  # filepath -> spectrum data

        files_data = self.file_manager.get_files_data()

        for _, row in files_data.iterrows():
            filepath = row["Filepath"]
            filename = row["filename"]
            group = row.get("group", "Unknown")

            try:
                closest_spectrum = None
                min_rt_diff = float("inf")

                # Check if we have cached data (memory mode)
                if (
                    self.file_manager.keep_in_memory
                    and filepath in self.file_manager.cached_data
                ):
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
                                    (
                                        self.polarity.lower()
                                        in ["+", "positive", "pos"]
                                        and spectrum_polarity.lower()
                                        in ["+", "positive", "pos"]
                                    )
                                    or (
                                        self.polarity.lower()
                                        in ["-", "negative", "neg"]
                                        and spectrum_polarity.lower()
                                        in ["-", "negative", "neg"]
                                    )
                                )
                            ):
                                continue

                            # Find closest spectrum to RT
                            rt_diff = abs(spectrum_rt - rt_center)
                            if rt_diff < min_rt_diff:
                                min_rt_diff = rt_diff
                                closest_spectrum = {
                                    "rt": spectrum_rt,
                                    "mz": spectrum_data["mz"],
                                    "intensity": spectrum_data["intensity"],
                                    "polarity": spectrum_polarity,
                                    "filename": filename,
                                    "group": group,
                                }

                else:
                    # Read from file to get MS1 spectra
                    reader = self.file_manager.get_mzml_reader(filepath)

                    for spectrum in reader:
                        if spectrum.ms_level == 1:  # MS1 spectra
                            spectrum_rt = spectrum.scan_time_in_minutes()

                            # Check polarity if available
                            spectrum_polarity = (
                                self.file_manager._get_spectrum_polarity(spectrum)
                            )
                            if (
                                self.polarity
                                and spectrum_polarity
                                and not (
                                    (
                                        self.polarity.lower()
                                        in ["+", "positive", "pos"]
                                        and spectrum_polarity.lower()
                                        in ["+", "positive", "pos"]
                                    )
                                    or (
                                        self.polarity.lower()
                                        in ["-", "negative", "neg"]
                                        and spectrum_polarity.lower()
                                        in ["-", "negative", "neg"]
                                    )
                                )
                            ):
                                continue

                            # Find closest spectrum to RT
                            rt_diff = abs(spectrum_rt - rt_center)
                            if rt_diff < min_rt_diff:
                                min_rt_diff = rt_diff

                                # Extract spectrum data
                                mz_array = spectrum.mz
                                intensity_array = spectrum.i

                                if len(mz_array) > 0:
                                    closest_spectrum = {
                                        "rt": spectrum_rt,
                                        "mz": mz_array,
                                        "intensity": intensity_array,
                                        "polarity": spectrum_polarity,
                                        "filename": filename,
                                        "group": group,
                                    }

                # Add the closest spectrum if found
                if closest_spectrum is not None:
                    ms1_spectra[filepath] = {
                        "filename": filename,
                        "group": group,
                        "spectrum": closest_spectrum,
                    }

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue

        return ms1_spectra

    def view_2d_scatter_plot(self, rt_center: float):
        """Add/remove 2D scatter plot (RT vs m/z) underneath the EIC plot"""
        try:
            if (
                hasattr(self, "scatter_plot_view")
                and self.scatter_plot_view is not None
            ):
                # Remove existing scatter plot
                self.remove_scatter_plot()
            else:
                # Add scatter plot
                self.add_scatter_plot(rt_center)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to toggle 2D scatter plot: {str(e)}"
            )

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
        three_way_splitter.setChildrenCollapsible(
            False
        )  # Prevent completely collapsing widgets

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
        self.scatter_plot_view.setMaximumHeight(
            16777215
        )  # Remove the fixed height constraint

        scatter_layout.addWidget(self.scatter_plot_view)

        # Add the scatter container to the new splitter
        three_way_splitter.addWidget(scatter_container)

        # Set initial splitter sizes (50% EIC, 25% boxplot, 25% scatter plot)
        three_way_splitter.setSizes([500, 250, 250])
        three_way_splitter.setStretchFactor(0, 1)  # EIC chart is stretchable
        three_way_splitter.setStretchFactor(1, 0)  # Boxplot maintains its proportion
        three_way_splitter.setStretchFactor(
            2, 0
        )  # Scatter plot maintains its proportion

        # Add the new splitter to the right panel layout
        current_layout.addWidget(three_way_splitter)

        # Store reference to new splitter for cleanup
        self.chart_scatter_splitter = three_way_splitter

        # Update context menu text
        self.update_context_menu_text("Hide 2D scatter plot")

    def remove_scatter_plot(self):
        """Remove the 2D scatter plot from the EIC window and restore two-way splitter layout"""
        if (
            hasattr(self, "chart_scatter_splitter")
            and self.chart_scatter_splitter is not None
        ):
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
            new_eic_boxplot_splitter.setStretchFactor(
                1, 0
            )  # Boxplot maintains proportion

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

    def find_msms_spectra(self, rt_start: float, rt_end: float):
        """Find MSMS spectra within RT window for the target m/z and polarity"""
        msms_spectra = {}  # filepath -> list of spectra

        # Define precursor tolerance (in Da)
        precursor_tolerance = 0.01  # 10 mDa tolerance for precursor matching

        files_data = self.file_manager.get_files_data()

        for _, row in files_data.iterrows():
            filepath = row["Filepath"]
            filename = row["filename"]

            try:
                file_msms = []

                # Check if we have cached data (memory mode)
                if (
                    self.file_manager.keep_in_memory
                    and filepath in self.file_manager.cached_data
                ):
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
                                (
                                    self.polarity.lower() in ["+", "positive", "pos"]
                                    and spectrum_polarity.lower()
                                    in ["+", "positive", "pos"]
                                )
                                or (
                                    self.polarity.lower() in ["-", "negative", "neg"]
                                    and spectrum_polarity.lower()
                                    in ["-", "negative", "neg"]
                                )
                            ):
                                continue

                            # Extract spectrum data
                            mz_array = spectrum_data["mz"]
                            intensity_array = spectrum_data["intensity"]

                            if len(mz_array) > 0:
                                msms_spectrum = {
                                    "rt": spectrum_rt,
                                    "precursor_mz": precursor_mz,
                                    "precursor_intensity": spectrum_data.get(
                                        "precursor_intensity", 0
                                    ),
                                    "mz": mz_array,
                                    "intensity": intensity_array,
                                    "scan_id": spectrum_data.get(
                                        "scan_id", f"RT_{spectrum_rt:.2f}"
                                    ),
                                    "polarity": spectrum_polarity,
                                }
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
                                precursor_mz = (
                                    spectrum.selected_precursors[0]["mz"]
                                    if spectrum.selected_precursors
                                    else None
                                )
                                if precursor_mz is None:
                                    continue

                                # Check if precursor matches target m/z
                                if (
                                    abs(precursor_mz - self.target_mz)
                                    > precursor_tolerance
                                ):
                                    continue

                                # Check polarity if available
                                spectrum_polarity = (
                                    self.file_manager._get_spectrum_polarity(spectrum)
                                )
                                if (
                                    self.polarity
                                    and spectrum_polarity
                                    and self.polarity != spectrum_polarity
                                ):
                                    continue

                                # Extract spectrum data
                                mz_array = spectrum.mz
                                intensity_array = spectrum.i

                                # Try to get precursor intensity
                                precursor_intensity = 0
                                try:
                                    if (
                                        spectrum.selected_precursors
                                        and len(spectrum.selected_precursors) > 0
                                    ):
                                        precursor_info = spectrum.selected_precursors[0]
                                        precursor_intensity = precursor_info.get(
                                            "intensity", 0
                                        )
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
                                    }
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


class InteractiveMSMSChartView(QChartView):
    """Interactive chart view for MSMS spectra with pan and zoom capabilities"""

    def __init__(self, chart):
        super().__init__(chart)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Spectrum data for popup display
        self.spectrum_data = None
        self.filename = None
        self.group = None

        # Mouse interaction state
        self.is_panning = False
        self.is_zooming = False
        self.last_mouse_pos = QPointF()
        self.pan_start_pos = QPointF()
        self.zoom_start_pos = QPointF()

        # Store chart ranges for interactions
        self.interaction_start_x_range = None
        self.interaction_start_y_range = None

        # Zoom anchor point
        self.zoom_anchor_x = 0.0
        self.zoom_anchor_y = 0.0

        # Disable default rubber band
        self.setRubberBand(QChartView.RubberBand.NoRubberBand)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
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
            # Right click: start zooming
            self.is_zooming = True
            self.zoom_start_pos = event.position()
            self.last_mouse_pos = event.position()

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

            x_range = (
                self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
            )
            y_range = (
                self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
            )
            self.zoom_anchor_x = self.interaction_start_x_range[0] + rel_x * x_range
            self.zoom_anchor_y = self.interaction_start_y_range[0] + rel_y * y_range

    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if self.is_panning:
            self._handle_panning(event)
        elif self.is_zooming:
            self._handle_zooming(event)

        self.last_mouse_pos = event.position()
        super().mouseMoveEvent(event)

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
        # Get zoom factor
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15

        # Get current ranges
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]

        # Calculate zoom center (mouse position in chart coordinates)
        plot_area = self.chart().plotArea()
        rel_x = (event.position().x() - plot_area.left()) / plot_area.width()
        rel_y = (event.position().y() - plot_area.top()) / plot_area.height()

        x_range = x_axis.max() - x_axis.min()
        y_range = y_axis.max() - y_axis.min()
        zoom_center_x = x_axis.min() + rel_x * x_range
        zoom_center_y = y_axis.min() + (1.0 - rel_y) * y_range  # Invert Y

        # Calculate new ranges
        new_x_range = x_range / zoom_factor
        new_y_range = y_range / zoom_factor

        # Calculate new bounds
        new_x_min = zoom_center_x - rel_x * new_x_range
        new_x_max = zoom_center_x + (1.0 - rel_x) * new_x_range
        new_y_min = zoom_center_y - (1.0 - rel_y) * new_y_range
        new_y_max = zoom_center_y + rel_y * new_y_range

        # Apply new ranges
        x_axis.setRange(new_x_min, new_x_max)
        y_axis.setRange(new_y_min, new_y_max)

    def _handle_panning(self, event):
        """Handle panning interaction"""
        if not self.interaction_start_x_range or not self.interaction_start_y_range:
            return

        # Calculate movement delta
        delta_x = event.position().x() - self.pan_start_pos.x()
        delta_y = event.position().y() - self.pan_start_pos.y()

        # Get plot area
        plot_area = self.chart().plotArea()
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]

        # Calculate data range per pixel
        x_range = self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
        y_range = self.interaction_start_y_range[1] - self.interaction_start_y_range[0]

        x_per_pixel = x_range / plot_area.width()
        y_per_pixel = y_range / plot_area.height()

        # Calculate new ranges
        x_offset = -delta_x * x_per_pixel
        y_offset = delta_y * y_per_pixel  # Positive because Y axis is inverted

        new_x_min = self.interaction_start_x_range[0] + x_offset
        new_x_max = self.interaction_start_x_range[1] + x_offset
        new_y_min = self.interaction_start_y_range[0] + y_offset
        new_y_max = self.interaction_start_y_range[1] + y_offset

        # Apply new ranges
        x_axis.setRange(new_x_min, new_x_max)
        y_axis.setRange(new_y_min, new_y_max)

    def _handle_zooming(self, event):
        """Handle zooming interaction"""
        if not self.interaction_start_x_range or not self.interaction_start_y_range:
            return

        # Calculate movement delta
        delta_x = event.position().x() - self.zoom_start_pos.x()
        delta_y = event.position().y() - self.zoom_start_pos.y()

        # Calculate zoom factors
        zoom_sensitivity = 0.005
        x_zoom_factor = 1.0 - (delta_x * zoom_sensitivity)
        y_zoom_factor = 1.0 + (delta_y * zoom_sensitivity)

        # Clamp zoom factors
        x_zoom_factor = max(0.1, min(10.0, x_zoom_factor))
        y_zoom_factor = max(0.1, min(10.0, y_zoom_factor))

        # Calculate new ranges anchored at the zoom point
        original_x_range = (
            self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
        )
        original_y_range = (
            self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
        )

        new_x_range = original_x_range * x_zoom_factor
        new_y_range = original_y_range * y_zoom_factor

        # Calculate new bounds
        anchor_to_left = self.zoom_anchor_x - self.interaction_start_x_range[0]
        anchor_to_right = self.interaction_start_x_range[1] - self.zoom_anchor_x
        anchor_to_bottom = self.zoom_anchor_y - self.interaction_start_y_range[0]
        anchor_to_top = self.interaction_start_y_range[1] - self.zoom_anchor_y

        # Scale distances from anchor
        new_anchor_to_left = anchor_to_left * x_zoom_factor
        new_anchor_to_right = anchor_to_right * x_zoom_factor
        new_anchor_to_bottom = anchor_to_bottom * y_zoom_factor
        new_anchor_to_top = anchor_to_top * y_zoom_factor

        new_x_min = self.zoom_anchor_x - new_anchor_to_left
        new_x_max = self.zoom_anchor_x + new_anchor_to_right
        new_y_min = self.zoom_anchor_y - new_anchor_to_bottom
        new_y_max = self.zoom_anchor_y + new_anchor_to_top

        # Apply new ranges
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]
        x_axis.setRange(new_x_min, new_x_max)
        y_axis.setRange(new_y_min, new_y_max)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click events to show popup window"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.spectrum_data and self.filename and self.group:
                # Create and show popup window
                popup = MSMSPopupWindow(
                    self.spectrum_data, self.filename, self.group, self
                )
                popup.show()
        super().mouseDoubleClickEvent(event)


class MS1ViewerWindow(QWidget):
    """Window for displaying MS1 spectra for the target ion"""

    def __init__(
        self,
        ms1_spectra,
        target_mz,
        rt_center,
        compound_name,
        adduct,
        mz_tolerance,
        formula,
        parent=None,
    ):
        super().__init__(parent)

        # Configure as independent window
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.ms1_spectra = ms1_spectra
        self.target_mz = target_mz
        self.rt_center = rt_center
        self.compound_name = compound_name
        self.adduct = adduct
        self.mz_tolerance = mz_tolerance
        self.formula = formula
        self.show_isotope_pattern = False

        self.init_ui()

    def init_ui(self):
        """Initialize the MS1 viewer UI"""
        self.setWindowTitle(
            f"MS1 Spectra: {self.compound_name} - {self.adduct} "
            f"(RT: {self.rt_center:.2f} min)"
        )
        self.setGeometry(100, 100, 1600, 1000)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Header with information
        total_spectra = len(self.ms1_spectra)
        header_text = (
            f"<b>{self.compound_name} ({self.adduct})</b> | "
            f"m/z: {self.target_mz:.4f} | "
            f"RT: {self.rt_center:.2f} min | "
            f"Files: {total_spectra}"
        )

        if self.formula:
            header_text += f" | Formula: {self.formula}"

        header_label = QLabel(header_text)
        header_label.setStyleSheet(
            "QLabel { margin: 2px; padding: 5px; font-size: 12px; }"
        )
        header_label.setFixedHeight(30)
        layout.addWidget(header_label)

        # Controls
        controls_layout = QHBoxLayout()

        # Show isotope pattern checkbox (only if formula is available)
        if self.formula:
            self.isotope_checkbox = QCheckBox("Show theoretical isotope pattern")
            self.isotope_checkbox.toggled.connect(self.toggle_isotope_pattern)
            controls_layout.addWidget(self.isotope_checkbox)

            # m/z tolerance for isotope matching
            controls_layout.addWidget(QLabel("Isotope m/z tolerance:"))
            self.isotope_tolerance_ppm_spin = QDoubleSpinBox()
            self.isotope_tolerance_ppm_spin.setRange(0.1, 100.0)
            self.isotope_tolerance_ppm_spin.setValue(5.0)  # Default 5 ppm
            self.isotope_tolerance_ppm_spin.setSuffix(" ppm")
            self.isotope_tolerance_ppm_spin.setDecimals(1)
            self.isotope_tolerance_ppm_spin.setSingleStep(0.5)
            self.isotope_tolerance_ppm_spin.valueChanged.connect(
                self.on_isotope_tolerance_changed
            )
            controls_layout.addWidget(self.isotope_tolerance_ppm_spin)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create scroll area for the spectra grid
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        self.grid_layout = QGridLayout(scroll_widget)
        self.grid_layout.setSpacing(5)
        self.grid_layout.setContentsMargins(5, 5, 5, 5)

        # Create charts for each file
        self.chart_views = {}
        row = 0
        col = 0
        max_cols = 2  # Two columns layout

        for filepath, file_data in self.ms1_spectra.items():
            filename = file_data["filename"]
            group = file_data["group"]
            spectrum = file_data["spectrum"]

            # File header
            display_name = filename.split(".")[0] if "." in filename else filename
            file_header_text = (
                f"<b>{display_name}</b> | Group: {group} | RT: {spectrum['rt']:.2f} min"
            )
            file_label = QLabel(file_header_text)
            file_label.setStyleSheet("""
                QLabel { 
                    background-color: #f0f0f0; 
                    padding: 4px; 
                    margin: 1px;
                    border: 1px solid #ccc;
                    border-radius: 2px;
                    font-size: 10px;
                }
            """)
            file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            file_label.setMaximumHeight(25)
            self.grid_layout.addWidget(file_label, row * 2, col)

            # Create MS1 chart
            chart_view = self.create_ms1_chart(spectrum, filename)
            chart_view.setMinimumSize(750, 400)
            self.chart_views[filepath] = chart_view
            self.grid_layout.addWidget(chart_view, row * 2 + 1, col)

            # Move to next position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        layout.addWidget(scroll_area)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def create_ms1_chart(self, spectrum, filename):
        """Create a chart for MS1 spectrum"""
        chart = QChart()
        chart.setTitle(f"MS1 Spectrum - {filename}")
        chart.legend().hide()

        # Create series for the spectrum
        series = QLineSeries()

        mz_array = np.array(spectrum["mz"])
        intensity_array = np.array(spectrum["intensity"])

        # Filter to zoom in around target m/z ± 10 Da, but keep full spectrum for context
        zoom_window = 10.0  # ±10 Da around target
        zoom_min = max(0, self.target_mz - zoom_window)
        zoom_max = self.target_mz + zoom_window

        # Create stick spectrum by adding baseline points between peaks
        for i in range(len(mz_array)):
            mz_val = float(mz_array[i])
            intensity_val = float(intensity_array[i])
            # Add baseline point before peak
            series.append(mz_val, 0.0)
            # Add peak point
            series.append(mz_val, intensity_val)
            # Add baseline point after peak
            series.append(mz_val, 0.0)

        # Style the main spectrum series
        pen = QPen(QColor("#2E86AB"))
        pen.setWidth(1)
        series.setPen(pen)
        chart.addSeries(series)

        # Highlight peaks within EIC extraction window
        highlight_series = QLineSeries()
        eic_window = self.mz_tolerance
        eic_mask = np.abs(mz_array - self.target_mz) <= eic_window

        if np.any(eic_mask):
            highlight_mz = mz_array[eic_mask]
            highlight_intensity = intensity_array[eic_mask]

            # Create stick spectrum for highlighted peaks
            for i in range(len(highlight_mz)):
                mz_val = float(highlight_mz[i])
                intensity_val = float(highlight_intensity[i])
                # Add baseline point before peak
                highlight_series.append(mz_val, 0.0)
                # Add peak point
                highlight_series.append(mz_val, intensity_val)
                # Add baseline point after peak
                highlight_series.append(mz_val, 0.0)

            # Style highlighted series
            highlight_pen = QPen(QColor("#F18F01"))
            highlight_pen.setWidth(3)
            highlight_series.setPen(highlight_pen)
            chart.addSeries(highlight_series)

        # Create axes
        x_axis = QValueAxis()
        y_axis = QValueAxis()

        # Set axis ranges
        if len(mz_array) > 0:
            # Full spectrum range for context
            full_mz_min = float(np.min(mz_array))
            full_mz_max = float(np.max(mz_array))
            full_intensity_max = float(np.max(intensity_array))

            # Set initial zoom to target region but allow full spectrum access
            x_axis.setRange(zoom_min, zoom_max)
            y_axis.setRange(0, full_intensity_max * 1.1)

            x_axis.setTitleText("m/z")
            y_axis.setTitleText("Intensity")

            chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
            chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

            series.attachAxis(x_axis)
            series.attachAxis(y_axis)

            if np.any(eic_mask):
                highlight_series.attachAxis(x_axis)
                highlight_series.attachAxis(y_axis)

        # Create interactive chart view
        chart_view = InteractiveMS1ChartView(
            chart,
            self.target_mz,
            zoom_min,
            zoom_max,
            full_mz_min if len(mz_array) > 0 else 0,
            full_mz_max if len(mz_array) > 0 else 1000,
            mz_array,
            intensity_array,
        )
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Auto-scale Y-axis to initial zoom window
        chart_view.auto_scale_y_axis()

        # Add top-3 signal labels
        chart_view.update_top_signal_labels()

        return chart_view

    def toggle_isotope_pattern(self, checked):
        """Toggle display of theoretical isotope pattern"""
        self.show_isotope_pattern = checked

        if checked and self.formula:
            # Calculate and display isotope pattern
            self.add_isotope_pattern_to_charts()
        else:
            # Remove isotope pattern from charts
            self.remove_isotope_pattern_from_charts()

    def on_isotope_tolerance_changed(self):
        """Handle changes to isotope tolerance"""
        if self.show_isotope_pattern and self.formula:
            # Refresh isotope pattern with new tolerance
            self.add_isotope_pattern_to_charts()

    def add_isotope_pattern_to_charts(self):
        """Add theoretical isotope pattern to all charts as small vertical rectangles"""
        try:
            isotope_pattern = self.calculate_isotope_pattern(self.formula)

            for filepath, chart_view in self.chart_views.items():
                chart = chart_view.chart()

                # Remove existing isotope series
                for series in chart.series():
                    if hasattr(series, "objectName") and series.objectName().startswith(
                        "isotope_"
                    ):
                        chart.removeSeries(series)

                # Add new isotope pattern
                if isotope_pattern:
                    # Get spectrum data for this file
                    spectrum = self.ms1_spectra[filepath]["spectrum"]
                    mz_array = np.array(spectrum["mz"])
                    intensity_array = np.array(spectrum["intensity"])

                    # Get isotope tolerance in Daltons
                    tolerance_ppm = (
                        self.isotope_tolerance_ppm_spin.value()
                        if hasattr(self, "isotope_tolerance_ppm_spin")
                        else 5.0
                    )

                    # Find monoisotopic peak to determine scaling
                    monoisotopic_mz = isotope_pattern[0][
                        0
                    ]  # First isotope is monoisotopic
                    tolerance_da = (tolerance_ppm / 1e6) * monoisotopic_mz

                    mono_mask = np.abs(mz_array - monoisotopic_mz) <= tolerance_da
                    if np.any(mono_mask):
                        # Monoisotopic peak found - use its intensity for scaling
                        mono_intensity = np.max(intensity_array[mono_mask])
                        has_monoisotopic = True
                    else:
                        # No monoisotopic peak - use fallback scaling
                        mono_intensity = np.max(intensity_array) * 0.1
                        has_monoisotopic = False

                    # Create rectangular indicators for each isotope
                    for i, (theo_mz, rel_abundance) in enumerate(isotope_pattern):
                        isotope_series = QLineSeries()
                        isotope_series.setObjectName(f"isotope_{i}")

                        # Calculate theoretical intensity
                        theo_intensity = mono_intensity * rel_abundance

                        # Check if this isotope is actually present in the spectrum
                        tolerance_da_isotope = (tolerance_ppm / 1e6) * theo_mz
                        isotope_mask = (
                            np.abs(mz_array - theo_mz) <= tolerance_da_isotope
                        )

                        if np.any(isotope_mask) or not has_monoisotopic:
                            # Isotope found or no monoisotopic reference - draw full rectangle
                            rect_height = theo_intensity
                            if has_monoisotopic and np.any(isotope_mask):
                                # If isotope is found, use actual intensity for rectangle height
                                actual_intensity = np.max(intensity_array[isotope_mask])
                                rect_height = max(
                                    theo_intensity, actual_intensity * 0.8
                                )  # At least 80% of actual
                        else:
                            # Isotope not found - draw thin line at theoretical position
                            rect_height = theo_intensity

                        # Create rectangle as a series of points
                        # Rectangle width in m/z units (very small)
                        rect_width = tolerance_da_isotope * 0.3  # 30% of tolerance

                        # Rectangle centered at theoretical m/z and intensity
                        center_mz = theo_mz
                        center_intensity = (
                            theo_intensity / 2
                        )  # Center of rectangle height

                        # Define rectangle corners
                        left = center_mz - rect_width
                        right = center_mz + rect_width
                        bottom = center_intensity - (rect_height / 2)
                        top = center_intensity + (rect_height / 2)

                        # Ensure bottom is not below zero
                        if bottom < 0:
                            bottom = 0
                            top = rect_height

                        # Create rectangle outline
                        isotope_series.append(left, bottom)  # Bottom left
                        isotope_series.append(left, top)  # Top left
                        isotope_series.append(right, top)  # Top right
                        isotope_series.append(right, bottom)  # Bottom right
                        isotope_series.append(left, bottom)  # Close rectangle

                        # Style the rectangle
                        if np.any(isotope_mask) and has_monoisotopic:
                            # Isotope found - solid line
                            isotope_pen = QPen(QColor("#A23B72"))  # Purple
                            isotope_pen.setWidth(2)
                            isotope_pen.setStyle(Qt.PenStyle.SolidLine)
                        else:
                            # Isotope not found or no monoisotopic reference - dashed line
                            isotope_pen = QPen(QColor("#A23B72"))  # Purple
                            isotope_pen.setWidth(1)
                            isotope_pen.setStyle(Qt.PenStyle.DashLine)

                        isotope_series.setPen(isotope_pen)
                        isotope_series.setOpacity(0.7)

                        chart.addSeries(isotope_series)

                        # Attach to existing axes
                        chart.addSeries(isotope_series)

                        # Attach to existing axes
                        axes = chart.axes()
                        if len(axes) >= 2:
                            isotope_series.attachAxis(axes[0])  # X axis
                            isotope_series.attachAxis(axes[1])  # Y axis

        except Exception as e:
            QMessageBox.warning(
                self,
                "Isotope Pattern Error",
                f"Could not calculate isotope pattern: {str(e)}",
            )

    def remove_isotope_pattern_from_charts(self):
        """Remove isotope pattern from all charts"""
        for chart_view in self.chart_views.values():
            chart = chart_view.chart()
            # Create a list of series to remove to avoid modifying while iterating
            series_to_remove = []
            for series in chart.series():
                if hasattr(series, "objectName") and series.objectName().startswith(
                    "isotope_"
                ):
                    series_to_remove.append(series)

            # Remove the isotope series
            for series in series_to_remove:
                chart.removeSeries(series)

    def calculate_isotope_pattern(self, formula):
        """Calculate theoretical isotope pattern for a molecular formula"""
        # This is a simplified isotope pattern calculation
        # For a complete implementation, you would use a library like pyMSpec or similar

        try:
            composition = parse_molecular_formula(formula)

            # Simple isotope pattern calculation for common elements
            # This is a basic implementation - for production use, consider a dedicated library

            isotopes = []
            base_mz = self.target_mz

            # M+0 (monoisotopic)
            isotopes.append((base_mz, 1.0))

            # M+1 (mainly 13C)
            if "C" in composition:
                c_count = composition["C"]
                # Probability of one 13C (simplified)
                m1_abundance = c_count * 0.011  # 1.1% natural abundance of 13C
                if m1_abundance > 0.01:  # Only show if significant
                    isotopes.append((base_mz + 1.003355, m1_abundance))

            # M+2 (mainly 13C x2, 34S, 37Cl)
            m2_abundance = 0.0
            if "C" in composition:
                c_count = composition["C"]
                # Probability of two 13C
                m2_abundance += (c_count * (c_count - 1) / 2) * (0.011**2)

            if "S" in composition:
                s_count = composition["S"]
                # 34S natural abundance ~4.2%
                m2_abundance += s_count * 0.042

            if "Cl" in composition:
                cl_count = composition["Cl"]
                # 37Cl natural abundance ~24.2%
                m2_abundance += cl_count * 0.242

            if m2_abundance > 0.01:  # Only show if significant
                isotopes.append((base_mz + 2.006710, m2_abundance))

            return isotopes

        except Exception as e:
            print(f"Error calculating isotope pattern: {e}")
            return []


class InteractiveMS1ChartView(QChartView):
    """Interactive chart view for MS1 spectra with pan and zoom capabilities"""

    def __init__(
        self,
        chart,
        target_mz,
        zoom_min,
        zoom_max,
        full_min,
        full_max,
        mz_array,
        intensity_array,
    ):
        super().__init__(chart)
        self.target_mz = target_mz
        self.zoom_min = zoom_min
        self.zoom_max = zoom_max
        self.full_min = full_min
        self.full_max = full_max

        # Store spectrum data for Y-axis auto-scaling
        self.mz_array = np.array(mz_array)
        self.intensity_array = np.array(intensity_array)

        # Interaction state
        self.is_panning = False
        self.is_zooming = False
        self.last_mouse_pos = None
        self.pan_start_pos = None
        self.zoom_start_pos = None
        self.interaction_start_x_range = None
        self.interaction_start_y_range = None
        self.zoom_anchor_x = 0
        self.zoom_anchor_y = 0

        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)

        # Hover tooltip setup
        self.tooltip_label = None

        # Top signal labels
        self.signal_labels = []  # List to store QLabel widgets for top signals

    def auto_scale_y_axis(self):
        """Auto-scale Y-axis based on visible X-axis range"""
        if len(self.mz_array) == 0:
            return

        # Get current X-axis range
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]

        x_min = x_axis.min()
        x_max = x_axis.max()

        # Find intensities within the visible m/z range
        visible_mask = (self.mz_array >= x_min) & (self.mz_array <= x_max)

        if np.any(visible_mask):
            visible_intensities = self.intensity_array[visible_mask]
            max_visible_intensity = np.max(visible_intensities)

            # Set Y-axis range with some padding (10% above the maximum)
            if max_visible_intensity > 0:
                y_max = max_visible_intensity * 1.1
                y_axis.setRange(0, y_max)
        else:
            # Fallback if no data in visible range
            max_intensity = (
                np.max(self.intensity_array) if len(self.intensity_array) > 0 else 100
            )
            y_axis.setRange(0, max_intensity * 1.1)

        # Update top signal labels after Y-axis change
        self.update_top_signal_labels()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
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
            self.is_zooming = True
            self.zoom_start_pos = event.position()
            self.last_mouse_pos = event.position()

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

            x_range = (
                self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
            )
            y_range = (
                self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
            )
            self.zoom_anchor_x = self.interaction_start_x_range[0] + rel_x * x_range
            self.zoom_anchor_y = self.interaction_start_y_range[1] - rel_y * y_range

            self.setCursor(Qt.CursorShape.SizeAllCursor)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events"""
        if self.is_panning:
            self._handle_panning(event)
        elif self.is_zooming:
            self._handle_zooming(event)

        self.last_mouse_pos = event.position()

        # Handle hover tooltips when not interacting
        if not self.is_panning and not self.is_zooming:
            self._handle_hover_tooltip(event)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
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

            # Get current axis ranges
            x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
            y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]

            current_x_range = x_axis.max() - x_axis.min()
            current_y_range = y_axis.max() - y_axis.min()

            # Calculate new ranges
            new_x_range = current_x_range / zoom_factor
            new_y_range = current_y_range / zoom_factor

            # Calculate anchor point
            anchor_x = x_axis.min() + rel_x * current_x_range

            # Calculate new axis bounds
            new_x_min = anchor_x - rel_x * new_x_range
            new_x_max = anchor_x + (1 - rel_x) * new_x_range
            new_y_min = 0  # Keep Y minimum at 0
            new_y_max = y_axis.max() / zoom_factor

            # Apply constraints
            new_x_min = max(self.full_min, new_x_min)
            new_x_max = min(self.full_max, new_x_max)
            new_y_max = max(new_y_max, y_axis.max() * 0.1)  # Minimum zoom out

            x_axis.setRange(new_x_min, new_x_max)
            y_axis.setRange(new_y_min, new_y_max)

            # Auto-scale Y-axis based on new X-axis range (this will also update labels)
            self.auto_scale_y_axis()

    def _handle_panning(self, event: QMouseEvent):
        """Handle panning interaction"""
        if not self.interaction_start_x_range or not self.interaction_start_y_range:
            return

        delta_x = event.position().x() - self.pan_start_pos.x()
        delta_y = event.position().y() - self.pan_start_pos.y()

        plot_area = self.chart().plotArea()
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]

        x_range = self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
        y_range = self.interaction_start_y_range[1] - self.interaction_start_y_range[0]

        x_per_pixel = x_range / plot_area.width()
        y_per_pixel = y_range / plot_area.height()

        x_offset = -delta_x * x_per_pixel
        y_offset = delta_y * y_per_pixel

        new_x_min = self.interaction_start_x_range[0] + x_offset
        new_x_max = self.interaction_start_x_range[1] + x_offset
        new_y_min = self.interaction_start_y_range[0] + y_offset
        new_y_max = self.interaction_start_y_range[1] + y_offset

        # Apply constraints
        new_x_min = max(self.full_min, new_x_min)
        new_x_max = min(self.full_max, new_x_max)
        new_y_min = max(0, new_y_min)  # Don't pan below 0 intensity

        x_axis.setRange(new_x_min, new_x_max)
        y_axis.setRange(new_y_min, new_y_max)

        # Auto-scale Y-axis based on new X-axis range (this will also update labels)
        self.auto_scale_y_axis()

    def _handle_zooming(self, event: QMouseEvent):
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

        original_x_range = (
            self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
        )
        original_y_range = (
            self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
        )

        new_x_range = original_x_range * x_zoom_factor
        new_y_range = original_y_range * y_zoom_factor

        anchor_to_left = self.zoom_anchor_x - self.interaction_start_x_range[0]
        anchor_to_right = self.interaction_start_x_range[1] - self.zoom_anchor_x
        anchor_to_bottom = self.zoom_anchor_y - self.interaction_start_y_range[0]
        anchor_to_top = self.interaction_start_y_range[1] - self.zoom_anchor_y

        new_x_min = (
            self.zoom_anchor_x - (anchor_to_left / original_x_range) * new_x_range
        )
        new_x_max = (
            self.zoom_anchor_x + (anchor_to_right / original_x_range) * new_x_range
        )
        new_y_min = (
            self.zoom_anchor_y - (anchor_to_bottom / original_y_range) * new_y_range
        )
        new_y_max = (
            self.zoom_anchor_y + (anchor_to_top / original_y_range) * new_y_range
        )

        # Apply constraints
        new_x_min = max(self.full_min, new_x_min)
        new_x_max = min(self.full_max, new_x_max)
        new_y_min = max(0, new_y_min)

        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]

        x_axis.setRange(new_x_min, new_x_max)
        y_axis.setRange(new_y_min, new_y_max)

        # Auto-scale Y-axis based on new X-axis range (this will also update labels)
        self.auto_scale_y_axis()

    def _handle_hover_tooltip(self, event):
        """Handle hover tooltip display for showing m/z values"""
        plot_area = self.chart().plotArea()
        if not plot_area.contains(event.position()):
            self._hide_tooltip()
            return

        # Convert mouse position to chart coordinates
        mouse_x = event.position().x()
        mouse_y = event.position().y()

        # Get chart axes
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]

        # Convert mouse position to data coordinates
        rel_x = (mouse_x - plot_area.left()) / plot_area.width()
        rel_y = (mouse_y - plot_area.top()) / plot_area.height()

        data_x = x_axis.min() + rel_x * (x_axis.max() - x_axis.min())
        data_y = y_axis.max() - rel_y * (y_axis.max() - y_axis.min())  # Y is inverted

        # Find closest signal within hover tolerance (use ppm tolerance from parent window)
        parent_window = self.parent()
        while parent_window and not hasattr(
            parent_window, "isotope_tolerance_ppm_spin"
        ):
            parent_window = parent_window.parent()

        tolerance_ppm = 5.0  # Default tolerance
        if parent_window and hasattr(parent_window, "isotope_tolerance_ppm_spin"):
            tolerance_ppm = parent_window.isotope_tolerance_ppm_spin.value()

        closest_mz, closest_intensity = self._find_closest_signal_ppm(
            data_x, tolerance_ppm
        )

        if closest_mz is not None:
            # Show tooltip
            tooltip_text = f"m/z: {closest_mz:.4f}\nIntensity: {closest_intensity:.2e}"
            self._show_tooltip(event.globalPosition().toPoint(), tooltip_text)
        else:
            self._hide_tooltip()

    def _find_closest_signal_ppm(self, data_x, tolerance_ppm):
        """Find the closest signal to the mouse X position using ppm tolerance"""
        if len(self.mz_array) == 0:
            return None, None

        # Calculate tolerance in Da units
        tolerance_da = (tolerance_ppm / 1e6) * data_x

        # Find signals within ppm tolerance
        mz_mask = np.abs(self.mz_array - data_x) <= tolerance_da
        if not np.any(mz_mask):
            return None, None

        candidate_mz = self.mz_array[mz_mask]
        candidate_intensities = self.intensity_array[mz_mask]

        # Find the closest match by m/z distance
        mz_distances = np.abs(candidate_mz - data_x)
        closest_idx = np.argmin(mz_distances)

        return candidate_mz[closest_idx], candidate_intensities[closest_idx]

    def _show_tooltip(self, global_pos, text):
        """Show tooltip at the specified position"""
        if self.tooltip_label is None:
            from PyQt6.QtWidgets import QLabel

            self.tooltip_label = QLabel(self)
            self.tooltip_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(255, 255, 255, 230);
                    border: 1px solid #666;
                    border-radius: 4px;
                    padding: 4px;
                    font-size: 10px;
                }
            """)
            self.tooltip_label.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents
            )

        self.tooltip_label.setText(text)
        self.tooltip_label.adjustSize()

        # Position tooltip near mouse but keep it visible
        local_pos = self.mapFromGlobal(global_pos)
        tooltip_x = local_pos.x() + 15
        tooltip_y = local_pos.y() - 30

        # Keep tooltip within widget bounds
        if tooltip_x + self.tooltip_label.width() > self.width():
            tooltip_x = local_pos.x() - self.tooltip_label.width() - 5
        if tooltip_y < 0:
            tooltip_y = local_pos.y() + 15

        self.tooltip_label.move(tooltip_x, tooltip_y)
        self.tooltip_label.show()

    def _hide_tooltip(self):
        """Hide the tooltip"""
        if self.tooltip_label is not None:
            self.tooltip_label.hide()

    def leaveEvent(self, event):
        """Hide tooltip when mouse leaves the widget"""
        self._hide_tooltip()
        super().leaveEvent(event)

    def update_top_signal_labels(self):
        """Update labels for the top-3 most abundant signals in the visible range"""
        # Clear existing labels
        for label in self.signal_labels:
            label.deleteLater()
        self.signal_labels.clear()

        if len(self.mz_array) == 0:
            return

        # Get current visible range
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]
        plot_area = self.chart().plotArea()

        x_min = x_axis.min()
        x_max = x_axis.max()

        # Find signals within visible range
        visible_mask = (self.mz_array >= x_min) & (self.mz_array <= x_max)
        if not np.any(visible_mask):
            return

        visible_mz = self.mz_array[visible_mask]
        visible_intensities = self.intensity_array[visible_mask]

        # Find top-3 most abundant signals
        if len(visible_intensities) > 0:
            # Get indices of top intensities (up to 3)
            top_indices = np.argsort(visible_intensities)[-3:][
                ::-1
            ]  # Top 3, highest first

            for i, idx in enumerate(top_indices):
                mz = visible_mz[idx]
                intensity = visible_intensities[idx]

                # Convert data coordinates to widget coordinates
                rel_x = (mz - x_min) / (x_max - x_min)
                rel_y = 1.0 - (intensity - y_axis.min()) / (y_axis.max() - y_axis.min())

                widget_x = plot_area.left() + rel_x * plot_area.width()
                widget_y = plot_area.top() + rel_y * plot_area.height()

                # Create label
                from PyQt6.QtWidgets import QLabel

                label = QLabel(self)
                label.setText(f"{mz:.4f}")
                label.setStyleSheet(f"""
                    QLabel {{
                        background-color: rgba(46, 134, 171, {200 - i * 50});  /* Blue with decreasing transparency */
                        color: white;
                        border-radius: 3px;
                        padding: 2px 4px;
                        font-size: 9px;
                        font-weight: bold;
                    }}
                """)
                label.adjustSize()

                # Position label to the right of the signal
                label_x = widget_x + 5
                label_y = widget_y - label.height() // 2

                # Keep label within widget bounds
                if label_x + label.width() > self.width():
                    label_x = widget_x - label.width() - 5
                if label_y < 0:
                    label_y = 0
                elif label_y + label.height() > self.height():
                    label_y = self.height() - label.height()

                label.move(int(label_x), int(label_y))
                label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                label.show()

                self.signal_labels.append(label)

    def resizeEvent(self, event):
        """Handle resize events to update label positions"""
        super().resizeEvent(event)
        # Update labels after resize
        self.update_top_signal_labels()


class MSMSViewerWindow(QWidget):
    """Window for displaying MSMS spectra in a grid layout"""

    def __init__(
        self,
        msms_spectra,
        target_mz,
        rt_center,
        rt_window,
        compound_name,
        adduct,
        parent=None,
    ):
        super().__init__(parent)

        # Configure as independent window
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.msms_spectra = msms_spectra
        self.target_mz = target_mz
        self.rt_center = rt_center
        self.rt_window = rt_window
        self.compound_name = compound_name
        self.adduct = adduct

        self.init_ui()

    def init_ui(self):
        """Initialize the MSMS viewer UI"""
        self.setWindowTitle(
            f"MSMS Spectra: {self.compound_name} - {self.adduct} "
            f"(RT: {self.rt_center:.2f} ± {self.rt_window:.1f} min)"
        )
        self.setGeometry(
            100, 100, 1800, 1000
        )  # Increased width and height for similarity panels

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Header with information - compact layout
        total_spectra = sum(len(data["spectra"]) for data in self.msms_spectra.values())
        header_label = QLabel(
            f"<b>{self.compound_name} ({self.adduct})</b> | "
            f"m/z: {self.target_mz:.4f} | "
            f"RT: {self.rt_center:.2f} ± {self.rt_window:.1f} min | "
            f"Files: {len(self.msms_spectra)} | "
            f"Spectra: {total_spectra}"
        )
        header_label.setStyleSheet(
            "QLabel { margin: 2px; padding: 5px; font-size: 12px; }"
        )
        header_label.setFixedHeight(30)
        header_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(header_label)

        # Pre-process data: sort by intensity and find global m/z range
        self._preprocess_spectra_data()

        # Create main layout - no splitter needed
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(1, 1, 1, 1)
        main_layout.setSpacing(1)

        # Top: Inter-file similarity overview
        inter_file_widget = self._create_similarity_overview_widget()
        main_layout.addWidget(inter_file_widget)

        # Bottom: Spectra grid
        # Create scroll area for the spectra grid
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setSpacing(1)  # Reduce spacing between widgets
        grid_layout.setContentsMargins(1, 1, 1, 1)

        # Organize files and create charts
        row = 0
        for filepath, file_data in self.processed_data:
            filename = file_data["filename"]
            group = file_data.get("group", "Unknown")
            spectra = file_data["spectra"]

            # File header with filename, group, and similarity statistics
            similarity_info = ""
            if filename in self.intra_file_similarities:
                stats = self.intra_file_similarities[filename]
                similarity_info = (
                    f" | Cosine Similarities: "
                    f"Min:{stats['min']:.3f} "
                    f"10%:{stats['percentile_10']:.3f} "
                    f"Med:{stats['median']:.3f} "
                    f"90%:{stats['percentile_90']:.3f} "
                    f"Max:{stats['max']:.3f}"
                )

            display_name = filename.split(".")[0] if "." in filename else filename
            file_header_text = f"<b>{display_name}</b> | Group: {group} | {len(spectra)} spectra{similarity_info}"
            file_label = QLabel(file_header_text)
            file_label.setStyleSheet("""
                QLabel { 
                    background-color: #f0f0f0; 
                    padding: 4px; 
                    margin: 1px;
                    border: 1px solid #ccc;
                    border-radius: 2px;
                    font-size: 10px;
                }
            """)
            file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            file_label.setMaximumHeight(25)
            grid_layout.addWidget(file_label, row, 0, 1, max(1, len(spectra)))
            row += 1

            # Add spectra horizontally for this file (sorted by intensity)
            for col, spectrum_data in enumerate(spectra):
                chart_widget = self.create_msms_chart(spectrum_data, filename, group)
                grid_layout.addWidget(chart_widget, row, col)

            row += 1

        main_layout.addWidget(scroll_area)
        layout.addWidget(main_widget)

        # Close button - compact
        close_btn = QPushButton("Close")
        close_btn.setMaximumWidth(100)
        close_btn.clicked.connect(self.close)

        # Add close button in a horizontal layout to center it
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_layout.addWidget(close_btn)
        close_layout.addStretch()
        layout.addLayout(close_layout)

    def _preprocess_spectra_data(self):
        """Preprocess spectra data: sort by intensity and find global m/z range"""
        self.global_mz_min = float("inf")
        self.global_mz_max = float("-inf")

        processed_files = []

        for filepath, file_data in self.msms_spectra.items():
            # Sort spectra by precursor intensity (descending)
            sorted_spectra = sorted(
                file_data["spectra"],
                key=lambda x: x.get("precursor_intensity", 0),
                reverse=True,
            )

            # Update global m/z range
            for spectrum in sorted_spectra:
                mz_array = spectrum["mz"]
                if len(mz_array) > 0:
                    self.global_mz_min = min(self.global_mz_min, np.min(mz_array))
                    self.global_mz_max = max(self.global_mz_max, np.max(mz_array))

            # Create processed file data
            processed_file_data = file_data.copy()
            processed_file_data["spectra"] = sorted_spectra
            processed_files.append((filepath, processed_file_data))

        # Sort files by filename for consistent ordering (using natural sort)
        natsort_key = natsort_keygen()
        self.processed_data = sorted(
            processed_files, key=lambda x: natsort_key(x[1]["filename"])
        )

        # Add padding to global m/z range
        if self.global_mz_min != float("inf") and self.global_mz_max != float("-inf"):
            mz_range = self.global_mz_max - self.global_mz_min
            padding = mz_range * 0.05 if mz_range > 0 else 1.0
            self.global_mz_min -= padding
            self.global_mz_max += padding
        else:
            # Fallback if no spectra found
            self.global_mz_min = 0
            self.global_mz_max = 1000

        # Calculate cosine similarities
        self._calculate_cosine_similarities()

    def _calculate_cosine_similarities(self):
        """Calculate cosine similarity scores within and between files"""
        self.intra_file_similarities = {}  # filename -> similarity stats
        self.inter_file_similarities = {}  # (file1, file2) -> list of all similarities

        # Calculate intra-file similarities (within each file)
        for filepath, file_data in self.processed_data:
            filename = file_data["filename"]
            spectra = file_data["spectra"]

            similarities = []
            if len(spectra) > 1:
                for i in range(len(spectra)):
                    for j in range(i + 1, len(spectra)):
                        sim = calculate_cosine_similarity(spectra[i], spectra[j])
                        similarities.append(sim)

            self.intra_file_similarities[filename] = calculate_similarity_statistics(
                similarities
            )

        # Calculate inter-file similarities (between all pairs of files)
        files_list = list(self.processed_data)
        for i in range(len(files_list)):
            for j in range(i + 1, len(files_list)):
                file1_path, file1_data = files_list[i]
                file2_path, file2_data = files_list[j]

                filename1 = file1_data["filename"]
                filename2 = file2_data["filename"]
                spectra1 = file1_data["spectra"]
                spectra2 = file2_data["spectra"]

                similarities = []
                for spec1 in spectra1:
                    for spec2 in spectra2:
                        sim = calculate_cosine_similarity(spec1, spec2)
                        similarities.append(sim)

                key = (filename1, filename2)
                self.inter_file_similarities[key] = (
                    similarities  # Store all similarities, not just stats
                )

    def _create_similarity_overview_widget(self):
        """Create a widget displaying cosine similarity overview"""
        overview_widget = QWidget()
        layout = QVBoxLayout(overview_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Inter-file similarity matrix (top panel)
        inter_file_label = QLabel("<b>Cosine Similarity Matrix (Median Values)</b>")
        inter_file_label.setFixedHeight(20)
        inter_file_label.setStyleSheet("QLabel { font-size: 12px; }")
        layout.addWidget(inter_file_label)

        # Create table for inter-file similarities
        files = [data[1]["filename"] for data in self.processed_data]
        if len(files) > 1:
            inter_table = QTableWidget(len(files), len(files))
            inter_table.setHorizontalHeaderLabels(
                [f.split(".")[0] for f in files]
            )  # Show filename without extension
            inter_table.setVerticalHeaderLabels([f.split(".")[0] for f in files])
            inter_table.setFixedHeight(min(150, 30 + len(files) * 35))
            inter_table.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

            # Set table properties for better display
            inter_table.horizontalHeader().setDefaultSectionSize(100)
            inter_table.verticalHeader().setDefaultSectionSize(20)

            # Fill the table
            for i, file1 in enumerate(files):
                for j, file2 in enumerate(files):
                    if i == j:
                        # Diagonal - show intra-file median similarity with distinct styling
                        if file1 in self.intra_file_similarities:
                            stats = self.intra_file_similarities[file1]
                            median_sim = stats["median"]
                            text = f"{median_sim:.3f}"
                            item = QTableWidgetItem(text)
                            # Use lighter colors for diagonal to distinguish from inter-file
                            if median_sim >= 0.8:
                                item.setBackground(
                                    QColor(76, 175, 80, 100)
                                )  # Light Green
                            elif median_sim >= 0.6:
                                item.setBackground(
                                    QColor(255, 193, 7, 100)
                                )  # Light Amber
                            elif median_sim >= 0.4:
                                item.setBackground(
                                    QColor(255, 152, 0, 100)
                                )  # Light Orange
                            else:
                                item.setBackground(
                                    QColor(244, 67, 54, 100)
                                )  # Light Red
                        else:
                            item = QTableWidgetItem("N/A")
                            item.setBackground(
                                QColor(240, 240, 240)
                            )  # Light gray background
                    else:
                        # Off-diagonal - show inter-file median similarity with color coding
                        key1 = (file1, file2)
                        key2 = (file2, file1)

                        similarities = None
                        if key1 in self.inter_file_similarities:
                            similarities = self.inter_file_similarities[key1]
                        elif key2 in self.inter_file_similarities:
                            similarities = self.inter_file_similarities[key2]

                        if similarities:
                            stats = calculate_similarity_statistics(similarities)
                            median_sim = stats["median"]
                            # Show only median value, use color for indication
                            text = f"{median_sim:.3f}"
                        else:
                            median_sim = 0.0
                            text = "0.000"

                        item = QTableWidgetItem(text)

                        # Use color intensity to indicate median similarity level
                        if median_sim >= 0.8:
                            item.setBackground(QColor(76, 175, 80, 200))  # Strong Green
                        elif median_sim >= 0.6:
                            item.setBackground(QColor(255, 193, 7, 200))  # Strong Amber
                        elif median_sim >= 0.4:
                            item.setBackground(
                                QColor(255, 152, 0, 200)
                            )  # Strong Orange
                        else:
                            item.setBackground(QColor(244, 67, 54, 200))  # Strong Red

                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    # Set smaller font for compact display
                    font = item.font()
                    font.setPointSize(7)
                    item.setFont(font)
                    inter_table.setItem(i, j, item)

            layout.addWidget(inter_table)

            # Add compact legend
            legend_label = QLabel(
                "<small><b>Legend:</b> Values show median cosine similarity | "
                "Diagonal = Intra-file (lighter colors) | Off-diagonal = Inter-file (darker colors)</small>"
            )
            legend_label.setFixedHeight(15)
            legend_label.setStyleSheet("QLabel { font-size: 10px; }")
            layout.addWidget(legend_label)
        else:
            no_comparison_label = QLabel(
                "<i>At least 2 files needed for inter-file comparison</i>"
            )
            layout.addWidget(no_comparison_label)

        return overview_widget

    def create_msms_chart(self, spectrum_data, filename, group):
        """Create a chart widget for a single MSMS spectrum"""
        # Create chart
        chart = QChart()

        # Get precursor intensity for display
        precursor_intensity = spectrum_data.get("precursor_intensity", 0)
        intensity_text = (
            f"{precursor_intensity:.1e}" if precursor_intensity > 0 else "N/A"
        )

        chart.setTitle(
            f"RT: {spectrum_data['rt']:.2f} min\n"
            f"Precursor: {spectrum_data['precursor_mz']:.4f}\n"
            f"Intensity: {intensity_text}"
        )

        # Create series for the spectrum
        series = QLineSeries()

        # Add spectrum data as vertical lines (stick spectrum)
        mz_array = spectrum_data["mz"]
        intensity_array = spectrum_data["intensity"]

        # Normalize intensities to 0-100 for better visualization
        if len(intensity_array) > 0:
            max_intensity = np.max(intensity_array)
            if max_intensity > 0:
                normalized_intensity = (intensity_array / max_intensity) * 100
            else:
                normalized_intensity = intensity_array
        else:
            normalized_intensity = intensity_array

        # Create stick spectrum by adding points at baseline and peak height
        for mz, intensity in zip(mz_array, normalized_intensity):
            # Add baseline point
            series.append(float(mz), 0.0)
            # Add peak point
            series.append(float(mz), float(intensity))
            # Add baseline point after peak
            series.append(float(mz), 0.0)

        # Style the series
        pen = QPen(QColor(0, 100, 200))  # Blue color
        pen.setWidth(1)
        series.setPen(pen)

        chart.addSeries(series)

        # Create and configure axes
        x_axis = QValueAxis()
        x_axis.setTitleText("m/z")
        x_axis.setLabelFormat("%.1f")
        x_axis.setTickCount(6)

        y_axis = QValueAxis()
        y_axis.setTitleText("Intensity (%)")
        y_axis.setLabelFormat("%.0f")
        y_axis.setRange(0, 105)  # 0-100% plus some padding
        y_axis.setTickCount(6)

        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

        series.attachAxis(x_axis)
        series.attachAxis(y_axis)

        # Use global m/z range for consistent x-axis limits
        x_axis.setRange(self.global_mz_min, self.global_mz_max)

        # Create interactive chart view with dynamic sizing
        chart_view = InteractiveMSMSChartView(chart)
        chart_view.setMinimumSize(250, 150)  # Reduced minimum size
        chart_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Store spectrum data for popup display
        chart_view.spectrum_data = spectrum_data
        chart_view.filename = filename
        chart_view.group = group

        # Hide legend since we only have one series
        chart.legend().setVisible(False)

        return chart_view

    def closeEvent(self, event):
        """Clean up when closing the window"""
        if hasattr(self, "extraction_worker") and self.extraction_worker.isRunning():
            self.extraction_worker.quit()
            self.extraction_worker.wait()
        event.accept()


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
                    if (
                        self.file_manager.keep_in_memory
                        and filepath in self.file_manager.cached_data
                    ):
                        # Use cached memory data
                        cached_file_data = self.file_manager.cached_data[filepath]

                        # Handle both old format (list) and new format (dict with ms1/ms2)
                        if (
                            isinstance(cached_file_data, dict)
                            and "ms1" in cached_file_data
                        ):
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
                                    (
                                        self.polarity.lower()
                                        in ["+", "positive", "pos"]
                                        and spectrum_data.get("polarity").lower()
                                        in ["+", "positive", "pos"]
                                    )
                                    or (
                                        self.polarity.lower()
                                        in ["-", "negative", "neg"]
                                        and spectrum_data.get("polarity").lower()
                                        in ["-", "negative", "neg"]
                                    )
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
                                    mz_mask = (mz_array >= mz_min) & (
                                        mz_array <= mz_max
                                    )
                                    if np.any(mz_mask):
                                        filtered_mz = mz_array[mz_mask]
                                        filtered_intensity = intensity_array[mz_mask]

                                        # Add signals to group
                                        if group not in group_signals:
                                            group_signals[group] = []

                                        for mz, intensity in zip(
                                            filtered_mz, filtered_intensity
                                        ):
                                            signal_info = {
                                                "rt": spectrum_rt,
                                                "mz": mz,
                                                "intensity": intensity,
                                                "group": group,
                                                "filename": filename,
                                                "in_eic_window": eic_mz_min
                                                <= mz
                                                <= eic_mz_max,
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
                                    spectrum_polarity = (
                                        self.file_manager._get_spectrum_polarity(
                                            spectrum
                                        )
                                    )
                                    if (
                                        self.polarity
                                        and spectrum_polarity
                                        and not (
                                            (
                                                self.polarity.lower()
                                                in ["+", "positive", "pos"]
                                                and spectrum_polarity.lower()
                                                in ["+", "positive", "pos"]
                                            )
                                            or (
                                                self.polarity.lower()
                                                in ["-", "negative", "neg"]
                                                and spectrum_polarity.lower()
                                                in ["-", "negative", "neg"]
                                            )
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
                                        mz_mask = (mz_array >= mz_min) & (
                                            mz_array <= mz_max
                                        )
                                        if np.any(mz_mask):
                                            filtered_mz = mz_array[mz_mask]
                                            filtered_intensity = intensity_array[
                                                mz_mask
                                            ]

                                            # Add signals to group
                                            if group not in group_signals:
                                                group_signals[group] = []

                                            for mz, intensity in zip(
                                                filtered_mz, filtered_intensity
                                            ):
                                                signal_info = {
                                                    "rt": spectrum_rt,
                                                    "mz": mz,
                                                    "intensity": intensity,
                                                    "group": group,
                                                    "filename": filename,
                                                    "in_eic_window": eic_mz_min
                                                    <= mz
                                                    <= eic_mz_max,
                                                }
                                                group_signals[group].append(signal_info)
                                                self.signal_data.append(signal_info)

                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")
                    continue

            # Create scatter series for each group
            self.create_scatter_series(group_signals)

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
                level_max = min_intensity + (
                    intensity_range * (level + 1) / intensity_levels
                )

                # Calculate transparency (10% for lowest, 100% for highest)
                transparency = int(
                    25 + (230 * level / (intensity_levels - 1))
                )  # 25-255 range

                # Set color with transparency
                color = QColor(base_color)
                color.setAlpha(transparency)
                series.setColor(color)
                series.setBorderColor(color.darker(110))

                # Add points for this intensity level
                for signal in signals:
                    if level_min <= signal["intensity"] <= level_max or (
                        level == intensity_levels - 1
                        and signal["intensity"] >= level_min
                    ):
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
        x_axis.setLabelFormat("%.1f")

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
        from PyQt6.QtCharts import QLineSeries
        from PyQt6.QtGui import QPen, QColor
        from PyQt6.QtCore import Qt

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
                current_mz_tolerance = (
                    self.eic_window.mz_tolerance_da_spin.value() * 3
                )  # 3x for scatter plot
                print(f"Using updated m/z tolerance: {current_mz_tolerance:.6f} Da")
                self.mz_tolerance = current_mz_tolerance

            # Get current RT range from EIC window
            if self.eic_window and hasattr(self.eic_window, "get_rt_range"):
                self.rt_min, self.rt_max = self.eic_window.get_rt_range()
                print(
                    f"Using updated RT range: {self.rt_min:.2f} - {self.rt_max:.2f} min"
                )

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
            import traceback

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
                        rel_x = (
                            event.position().x() - plot_area.left()
                        ) / plot_area.width()
                        rel_y = (
                            event.position().y() - plot_area.top()
                        ) / plot_area.height()

                        data_rt = x_axis.min() + rel_x * (x_axis.max() - x_axis.min())
                        data_mz = y_axis.max() - rel_y * (
                            y_axis.max() - y_axis.min()
                        )  # Y is inverted

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

                x_range = (
                    self.interaction_start_x_range[1]
                    - self.interaction_start_x_range[0]
                )
                y_range = (
                    self.interaction_start_y_range[1]
                    - self.interaction_start_y_range[0]
                )
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
            x_range = (
                self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
            )
            y_range = (
                self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
            )

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

        original_x_range = (
            self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
        )
        original_y_range = (
            self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
        )

        new_x_range = original_x_range * x_zoom_factor
        new_y_range = original_y_range * y_zoom_factor

        anchor_to_left = self.zoom_anchor_x - self.interaction_start_x_range[0]
        anchor_to_right = self.interaction_start_x_range[1] - self.zoom_anchor_x
        anchor_to_bottom = self.zoom_anchor_y - self.interaction_start_y_range[0]
        anchor_to_top = self.interaction_start_y_range[1] - self.zoom_anchor_y

        new_x_min = (
            self.zoom_anchor_x - (anchor_to_left / original_x_range) * new_x_range
        )
        new_x_max = (
            self.zoom_anchor_x + (anchor_to_right / original_x_range) * new_x_range
        )
        new_y_min = (
            self.zoom_anchor_y - (anchor_to_bottom / original_y_range) * new_y_range
        )
        new_y_max = (
            self.zoom_anchor_y + (anchor_to_top / original_y_range) * new_y_range
        )

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
            if (
                abs(signal["rt"] - rt) <= self.hover_tolerance_rt
                and abs(signal["mz"] - mz) <= self.hover_tolerance_mz
            ):
                return signal["group"]
        return None

    def highlight_group(self, group_to_highlight):
        """Highlight all signals from a specific group"""
        if not hasattr(self, "series_by_group"):
            return

        # Restore original colors if we were highlighting another group
        if (
            self.current_hover_group
            and self.current_hover_group in self.original_colors
        ):
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
