"""
EIC (Extracted Ion Chromatogram) window for displaying chromatographic data
"""

import sys
import os
import re
import traceback
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
    QWidgetAction,
    QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QMargins, QRect
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPen, QColor, QPainter, QMouseEvent, QAction, QBrush
from PyQt6.QtWidgets import QSizePolicy, QStyledItemDelegate, QStyleOptionViewItem
from .utils import calculate_cosine_similarity, calculate_similarity_statistics
import numpy as np
from typing import Dict, Tuple, Optional, List
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import seaborn as sns
from .utils import (
    calculate_mz_from_formula,
    format_mz,
    format_retention_time,
    parse_molecular_formula,
)
from natsort import natsorted, natsort_keygen
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .compound_manager import CompoundManager


class ClickableLabel(QLabel):
    """QLabel that emits a *clicked* signal when left-clicked."""

    clicked = pyqtSignal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically using the UserRole value."""

    def __lt__(self, other):
        try:
            v1 = self.data(Qt.ItemDataRole.UserRole)
            v2 = other.data(Qt.ItemDataRole.UserRole)
            if v1 is not None and v2 is not None:
                return float(v1) < float(v2)
        except (TypeError, ValueError):
            pass
        return super().__lt__(other)


class BarDelegate(QStyledItemDelegate):
    """
    Item delegate that paints a horizontal background bar proportional to the
    cell's numeric value relative to the column maximum.

    The bar is drawn in the group colour (passed per-row via UserRole + 1) at
    alpha 0.5, and the text is then drawn on top.
    """

    # Qt.ItemDataRole values we piggy-back on
    BAR_FRAC_ROLE = Qt.ItemDataRole.UserRole + 2  # float  0.0-1.0
    BAR_COLOR_ROLE = Qt.ItemDataRole.UserRole + 3  # QColor

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        # Let the base class draw the standard background (selection highlight etc.)
        super().paint(painter, option, index)

        frac = index.data(self.BAR_FRAC_ROLE)
        color = index.data(self.BAR_COLOR_ROLE)

        if frac is None or color is None or frac <= 0:
            return

        bar_width = int(option.rect.width() * min(frac, 1.0))
        bar_rect = option.rect.adjusted(0, 1, 0, -1)
        bar_rect.setWidth(bar_width)

        if isinstance(color, QColor):
            bar_color = QColor(color)
        else:
            bar_color = QColor(color)
        bar_color.setAlphaF(0.5)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(bar_rect, bar_color)
        painter.restore()

        # Re-draw the text on top so it stays readable
        painter.save()
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if text and False:
            painter.drawText(
                option.rect.adjusted(4, 0, -4, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                str(text),
            )
        painter.restore()


class CenteredBarDelegate(QStyledItemDelegate):
    """
    Item delegate that paints a centered horizontal bar indicating ppm deviation
    from the theoretical m/z value.

    The bar extends from the cell centre toward the left (negative deviation) or
    right (positive deviation).  The half-width of the cell represents the m/z
    tolerance in ppm, so the leftmost edge = -tolerance and the rightmost = +tolerance.
    The bar is drawn in the group colour at alpha 0.5 so the cell text stays readable.
    """

    PPM_DEVIATION_ROLE = Qt.ItemDataRole.UserRole + 4  # float  (ppm deviation)
    PPM_RANGE_ROLE = Qt.ItemDataRole.UserRole + 5  # float  (tolerance ± ppm)
    PPM_BAR_COLOR_ROLE = Qt.ItemDataRole.UserRole + 6  # QColor

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        super().paint(painter, option, index)

        ppm_dev = index.data(self.PPM_DEVIATION_ROLE)
        ppm_range = index.data(self.PPM_RANGE_ROLE)
        color = index.data(self.PPM_BAR_COLOR_ROLE)

        if ppm_dev is None or ppm_range is None or ppm_range <= 0:
            return

        frac = max(-1.0, min(1.0, ppm_dev / ppm_range))  # -1.0 … +1.0
        if abs(frac) < 1e-6:
            return

        rect = option.rect
        center_x = rect.left() + rect.width() / 2
        bar_w = max(1, int(abs(frac) * rect.width() / 2))
        bar_top = rect.top() + 2
        bar_h = max(1, rect.height() - 4)

        if frac > 0:
            bar_rect = QRect(int(center_x), bar_top, bar_w, bar_h)
        else:
            bar_rect = QRect(int(center_x) - bar_w, bar_top, bar_w, bar_h)

        # Negative offset → dodgerblue, positive → firebrick (direction is always clear)
        bar_color = QColor("dodgerblue") if frac < 0 else QColor("firebrick")
        bar_color.setAlphaF(0.5)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(bar_rect, bar_color)
        painter.restore()


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

        scan_id = self.spectrum_data.get("scan_id", "")
        filter_str = self.spectrum_data.get("filter_string", "")
        header_text = (
            f"<b>File:</b> {self.filename}<br>"
            f"<b>Group:</b> {self.group}<br>"
            f"<b>RT:</b> {self.spectrum_data['rt']:.4f} min<br>"
            f"<b>Precursor m/z:</b> {self.spectrum_data['precursor_mz']:.4f}<br>"
            f"<b>Precursor Intensity:</b> {intensity_text}"
        )
        if scan_id:
            header_text += f"<br><b>Scan:</b> {scan_id}"
        if filter_str:
            header_text += f"<br><b>Filter:</b> {filter_str}"

        header_label = QLabel(header_text)
        header_label.setStyleSheet("""
            QLabel { 
                background-color: #f0f0f0; 
                padding: 5px; 
                margin: 2px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        header_label.setMaximumHeight(
            100
        )  # enough room for scan_id + filter_string lines
        layout.addWidget(header_label)

        # Create splitter for table and chart (horizontal layout)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create table for m/z and intensity values (left side)
        self.create_data_table()
        splitter.addWidget(self.table_widget)

        # Create large MSMS chart with interactive capabilities (right side)
        chart = self.create_large_msms_chart()
        self.chart_view = InteractiveMSMSChartView(chart)
        self.chart_view.spectrum_data = self.spectrum_data  # Enable hover tooltip
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

            # Create m/z item (NumericTableWidgetItem sorts by UserRole float)
            mz_item = NumericTableWidgetItem()
            mz_item.setData(Qt.ItemDataRole.DisplayRole, f"{mz_val:.4f}")
            mz_item.setData(Qt.ItemDataRole.UserRole, mz_val)  # Store for selection

            # Create intensity item
            intensity_item = NumericTableWidgetItem()
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

            # Create relative abundance item
            rel_abund_item = NumericTableWidgetItem()
            rel_abund_item.setData(Qt.ItemDataRole.DisplayRole, f"{rel_abund_val:.1f}")
            rel_abund_item.setData(Qt.ItemDataRole.UserRole, rel_abund_val)
            # Bar-delegate data: fraction 0.0-1.0 and colour
            rel_abund_item.setData(BarDelegate.BAR_FRAC_ROLE, rel_abund_val / 100.0)
            rel_abund_item.setData(BarDelegate.BAR_COLOR_ROLE, QColor("steelblue"))

            self.table_widget.setItem(i, 0, mz_item)
            self.table_widget.setItem(i, 1, intensity_item)
            self.table_widget.setItem(i, 2, rel_abund_item)

        # Enable sorting after all data is populated
        self.table_widget.setSortingEnabled(True)

        # Apply bar delegate to the relative abundance column
        self.table_widget.setItemDelegateForColumn(2, BarDelegate(self.table_widget))

        # Connect selection change
        self.table_widget.itemSelectionChanged.connect(self.on_table_selection_changed)

        # Resize columns to content - allow interactive resizing
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        # Set initial column widths
        header.resizeSection(0, 120)
        header.resizeSection(1, 120)
        header.resizeSection(2, 120)

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

        # Clear existing series (also invalidates any hover series in the chart view)
        chart.removeAllSeries()
        self.chart_view._hover_series = None  # Reference is stale after removeAllSeries
        self.chart_view._hover_mz = None

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
            f"RT: {self.spectrum_data['rt']:.4f} min, "
            f"Precursor: {self.spectrum_data['precursor_mz']:.4f}, "
            f"Intensity: {intensity_text}"
            + (
                f"\nScan: {self.spectrum_data['scan_id']}"
                if self.spectrum_data.get("scan_id")
                else ""
            )
            + (
                f" | Filter: {self.spectrum_data['filter_string']}"
                if self.spectrum_data.get("filter_string")
                else ""
            )
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
            self.zoom_anchor_y = self.interaction_start_y_range[1] - rel_y * y_range

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
        settings_callback=None,
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
            self.polarity = polarity
        else:
            # Fallback: Calculate target m/z using compound manager
            try:
                # Use the compound manager to calculate m/z properly
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
        self._syncing_scatter_x_axis = False

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
        self._load_stylesheet()
        self.extract_eic_data()

    def _load_stylesheet(self):
        """Apply the shared application stylesheet so table styles are consistent."""
        stylesheet_path = os.path.join(os.path.dirname(__file__), "style.css")
        if os.path.exists(stylesheet_path):
            with open(stylesheet_path, "r") as f:
                self.setStyleSheet(f.read())

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

        # Optional scatter plot button (lazy-loads scatter data when first shown)
        self.scatter_toggle_btn = QPushButton("Show 2D Scatter Plot")
        self.scatter_toggle_btn.clicked.connect(self.toggle_scatter_plot)
        layout.addWidget(self.scatter_toggle_btn)

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
        """
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)

        # Always show the SMILES string as text
        smiles_label = QLabel(f"<b>SMILES:</b><br><small>{smiles}</small>")
        smiles_label.setWordWrap(True)
        smiles_label.setToolTip(smiles)
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
            img_label.setToolTip(smiles)
            container_layout.addWidget(img_label)

        except ImportError:
            pass  # rdkit not installed — SMILES text already added above
        except Exception:
            pass  # Invalid SMILES or rendering error — SMILES text already added above

        return container

    def create_control_panel(self) -> QGroupBox:
        """Create the control panel"""
        group = QGroupBox("Extraction Parameters")
        layout = QFormLayout(group)

        # EIC calculation method
        self.eic_method_combo = QComboBox()
        self.eic_method_combo.addItems(["Sum of all signals", "Most intensive signal"])
        default_eic_method = self.defaults.get("eic_method", "Sum of all signals")
        idx = self.eic_method_combo.findText(default_eic_method)
        self.eic_method_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.eic_method_combo.currentTextChanged.connect(self.update_plot)
        self.eic_method_combo.currentTextChanged.connect(
            lambda v: self._notify_setting("eic_method", v)
        )
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
        self.mz_tolerance_ppm_spin.valueChanged.connect(
            lambda v: self._notify_setting("mz_tolerance_ppm", v)
        )
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
        if (
            "separate_groups" in self.defaults
            and "separation_mode" not in self.defaults
        ):
            default_mode = "By group" if self.defaults["separate_groups"] else "None"
        idx = self.separation_mode_combo.findText(default_mode)
        if idx >= 0:
            self.separation_mode_combo.setCurrentIndex(idx)
        self.separation_mode_combo.currentTextChanged.connect(self.update_plot)
        layout.addRow("Separation:", self.separation_mode_combo)

        # RT shift for group separation (more flexible range)
        self.rt_shift_spin = QDoubleSpinBox()
        self.rt_shift_spin.setRange(0.0, 60.0)  # Allow up to 60 minutes
        self.rt_shift_spin.setValue(self.defaults["rt_shift_min"])  # Use default
        self.rt_shift_spin.setSuffix(" min")
        self.rt_shift_spin.setDecimals(1)
        self.rt_shift_spin.setEnabled(True)  # Always enabled
        self.rt_shift_spin.valueChanged.connect(self.update_plot)
        self.rt_shift_spin.valueChanged.connect(
            lambda v: self._notify_setting("rt_shift_min", v)
        )
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
        self.legend_position_combo.currentTextChanged.connect(
            lambda v: self._notify_setting("legend_position", v)
        )
        layout.addRow("Legend:", self.legend_position_combo)

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
        self.group_settings_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.group_settings_table.horizontalHeader().setStretchLastSection(False)
        self.group_settings_table.verticalHeader().setVisible(True)
        self.group_settings_table.setMaximumHeight(800)
        self.group_settings_table.setMinimumHeight(340)

        # Set initial column widths
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
                self.calculate_file_shifts()
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
        self.boxplot_widget.addTab(self.boxplot_canvas, "Boxplot peak areas")

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
        self.peak_area_table.horizontalHeader().setStretchLastSection(False)
        self.peak_area_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.peak_area_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

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
        self.summary_stats_table.horizontalHeader().setStretchLastSection(False)
        self.summary_stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.summary_stats_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

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
        self.mz_sample_table.setHorizontalHeaderLabels(
            ["Group", "Sample Name", "Mean m/z", "m/z P10", "m/z P90", "EIC Width (Da)"]
        )
        self.mz_sample_table.setAlternatingRowColors(True)
        self.mz_sample_table.setSortingEnabled(True)
        self.mz_sample_table.horizontalHeader().setStretchLastSection(False)
        self.mz_sample_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.mz_sample_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        # Compact appearance — stylesheet forces pixel font on every cell
        self.mz_sample_table.verticalHeader().setDefaultSectionSize(20)
        self.mz_sample_table.verticalHeader().setMinimumSectionSize(16)

        mz_sample_layout.addWidget(self.mz_sample_table)
        # Centred bar delegate for ppm deviation visualisation (always shown)
        self._mz_ppm_bar_delegate_sample = CenteredBarDelegate(self.mz_sample_table)
        for _col in (2, 3, 4):
            self.mz_sample_table.setItemDelegateForColumn(
                _col, self._mz_ppm_bar_delegate_sample
            )
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
        self.mz_group_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.mz_group_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        # Compact appearance — stylesheet forces pixel font on every cell
        self.mz_group_table.verticalHeader().setDefaultSectionSize(20)
        self.mz_group_table.verticalHeader().setMinimumSectionSize(16)

        mz_group_layout.addWidget(self.mz_group_table)
        # Centred bar delegate for ppm deviation visualisation (always shown)
        self._mz_ppm_bar_delegate_group = CenteredBarDelegate(self.mz_group_table)
        for _col in (1, 2, 3, 5, 6):
            self.mz_group_table.setItemDelegateForColumn(
                _col, self._mz_ppm_bar_delegate_group
            )
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
        self.rt_sample_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.rt_sample_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.rt_sample_table.verticalHeader().setDefaultSectionSize(20)
        self.rt_sample_table.verticalHeader().setMinimumSectionSize(16)
        rt_sample_layout.addWidget(self.rt_sample_table)

        # CenteredBarDelegate for RT-position columns (apex, left, right); BarDelegate for width
        self._rt_sample_centered_delegate = CenteredBarDelegate(self.rt_sample_table)
        for _col in (2, 3, 4):
            self.rt_sample_table.setItemDelegateForColumn(
                _col, self._rt_sample_centered_delegate
            )
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
        self.rt_group_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.rt_group_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.rt_group_table.verticalHeader().setDefaultSectionSize(20)
        self.rt_group_table.verticalHeader().setMinimumSectionSize(16)
        rt_group_layout.addWidget(self.rt_group_table)

        # CenteredBarDelegate for Mean Apex RT (col 1); BarDelegate for the rest (cols 2–6)
        self._rt_group_centered_delegate = CenteredBarDelegate(self.rt_group_table)
        self.rt_group_table.setItemDelegateForColumn(
            1, self._rt_group_centered_delegate
        )
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
        self.copy_similarity_excel_btn.clicked.connect(
            self._copy_similarity_table_excel
        )
        similarity_buttons_layout.addWidget(self.copy_similarity_excel_btn)

        self.copy_similarity_r_btn = QPushButton("Copy as R matrix")
        self.copy_similarity_r_btn.clicked.connect(self._copy_similarity_table_r)
        similarity_buttons_layout.addWidget(self.copy_similarity_r_btn)

        similarity_buttons_layout.addStretch()
        similarity_layout.addLayout(similarity_buttons_layout)

        # Peak shape similarity table (as matrix)
        self.similarity_table = QTableWidget()
        self.similarity_table.setAlternatingRowColors(True)
        self.similarity_table.setSortingEnabled(
            False
        )  # Disable sorting for matrix view
        self.similarity_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.similarity_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

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
        self.pca_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
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
        self.regression_model_combo.currentTextChanged.connect(
            self._update_calibration_plot
        )
        calibration_controls.addWidget(self.regression_model_combo)

        # Axis transformation selection
        calibration_controls.addWidget(QLabel("Axis Scale:"))
        self.axis_transform_combo = QComboBox()
        self.axis_transform_combo.addItems(["Linear", "Log2/Log2", "Log10/Log10"])
        self.axis_transform_combo.currentTextChanged.connect(
            self._update_calibration_plot
        )
        calibration_controls.addWidget(self.axis_transform_combo)

        # Injection volume / dilution normalization
        self.normalize_peak_area_checkbox = QCheckBox(
            "Normalize peak areas by injection volume × dilution"
        )
        self.normalize_peak_area_checkbox.setChecked(False)
        self.normalize_peak_area_checkbox.setToolTip(
            "When checked, each peak area is multiplied by the sample's injection volume "
            "and dilution factor before being used in regression fitting and prediction."
        )
        self.normalize_peak_area_checkbox.stateChanged.connect(
            lambda _: self._update_calibration_table(self._all_peak_data)
            if hasattr(self, "_all_peak_data")
            else None
        )
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
        self.calibration_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.calibration_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.calibration_table.verticalHeader().setDefaultSectionSize(20)
        self.calibration_table.verticalHeader().setMinimumSectionSize(16)
        self.calibration_table.itemChanged.connect(self._on_calibration_table_changed)
        calibration_splitter.addWidget(self.calibration_table)

        # Calibration plot
        self.calibration_figure = Figure(figsize=(8, 6))
        self.calibration_canvas = FigureCanvas(self.calibration_figure)
        self.calibration_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.calibration_toolbar = NavigationToolbar(
            self.calibration_canvas, calibration_tab
        )

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
        self.copy_abundances_excel_btn.clicked.connect(
            self._copy_abundances_table_excel
        )
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
        self.calculated_abundances_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.calculated_abundances_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.calculated_abundances_table.verticalHeader().setDefaultSectionSize(20)
        self.calculated_abundances_table.verticalHeader().setMinimumSectionSize(16)
        calculated_abundances_layout.addWidget(self.calculated_abundances_table)
        self.boxplot_widget.addTab(calculated_abundances_tab, "Calculated Abundances")

        # Tab 10: Quantification Group Summaries
        quant_summary_tab = QWidget()
        quant_summary_layout = QVBoxLayout(quant_summary_tab)

        quant_summary_buttons = QHBoxLayout()
        self.copy_quant_summary_excel_btn = QPushButton("Copy as Excel Tab-delimited")
        self.copy_quant_summary_excel_btn.clicked.connect(
            self._copy_quant_summary_table_excel
        )
        quant_summary_buttons.addWidget(self.copy_quant_summary_excel_btn)
        self.copy_quant_summary_r_btn = QPushButton("Copy as R dataframe")
        self.copy_quant_summary_r_btn.clicked.connect(self._copy_quant_summary_table_r)
        quant_summary_buttons.addWidget(self.copy_quant_summary_r_btn)
        quant_summary_buttons.addStretch()
        quant_summary_layout.addLayout(quant_summary_buttons)

        self.quant_summary_table = QTableWidget()
        self.quant_summary_table.setColumnCount(7)
        self.quant_summary_table.setHorizontalHeaderLabels(
            ["Group", "Min", "Perc_10", "Median", "Mean", "Perc_90", "Max"]
        )
        self.quant_summary_table.setAlternatingRowColors(True)
        self.quant_summary_table.setSortingEnabled(True)
        self.quant_summary_table.horizontalHeader().setStretchLastSection(False)
        self.quant_summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.quant_summary_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
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
            integrated_area = self._calculate_peak_area_with_boundaries(
                original_rt, intensity, start_rt, end_rt
            )

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
                            fwhm_left = win_rt[i] + (half_max - win_int[i]) / dI * (
                                win_rt[i + 1] - win_rt[i]
                            )
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
                            fwhm_right = win_rt[i - 1] + (
                                half_max - win_int[i - 1]
                            ) / dI * (win_rt[i] - win_rt[i - 1])
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
        col_maxima = {
            key: max((s[key] for s in stats_data), default=0)
            for key in stat_keys_for_bar
        }

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
                gc = QColor(group_color)
                gc.setAlphaF(0.5)
                group_item.setBackground(gc)
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

            stats = self.file_manager.get_mz_stats_in_rt_window(
                filepath, self.target_mz, mz_tolerance, start_rt, end_rt, polarity
            )
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
        headers = [
            tbl.horizontalHeaderItem(c).text().replace(" ", "_").replace("/", "_")
            for c in range(tbl.columnCount())
        ]
        lines = [
            f"df <- data.frame(",
            "  " + ",\n  ".join(f"{h} = c()" for h in headers),
            ")",
        ]
        rows_data = []
        for r in range(tbl.rowCount()):
            row_vals = [
                tbl.item(r, c).text() if tbl.item(r, c) else ""
                for c in range(tbl.columnCount())
            ]
            rows_data.append(row_vals)
        col_lines = []
        for ci, h in enumerate(headers):
            vals = [
                f'"{rows_data[ri][ci]}"' if ci < 2 else rows_data[ri][ci]
                for ri in range(len(rows_data))
            ]
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
        headers = [
            tbl.horizontalHeaderItem(c).text().replace(" ", "_").replace("/", "_")
            for c in range(tbl.columnCount())
        ]
        rows_data = []
        for r in range(tbl.rowCount()):
            rows_data.append(
                [
                    tbl.item(r, c).text() if tbl.item(r, c) else ""
                    for c in range(tbl.columnCount())
                ]
            )
        col_lines = []
        for ci, h in enumerate(headers):
            vals = [
                f'"{rows_data[ri][ci]}"' if ci == 0 else rows_data[ri][ci]
                for ri in range(len(rows_data))
            ]
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
        sample_rows = sorted(
            rt_data, key=lambda x: (natsort_key(x[0]), natsort_key(x[1]))
        )

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
                    "std_apex": float(np.std(a, ddof=1))
                    if len(a) > 1
                    else (0.0 if len(a) == 1 else None),
                    "mean_fwhm": float(np.mean(f)) if len(f) > 0 else None,
                    "std_fwhm": float(np.std(f, ddof=1))
                    if len(f) > 1
                    else (0.0 if len(f) == 1 else None),
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
                _gc = QColor(_grp_color)
                _gc.setAlphaF(0.5)
                grp_cell.setBackground(_gc)
            self.rt_group_table.setItem(row_idx, 0, grp_cell)

            for col_idx, key in enumerate(grp_col_keys, start=1):
                val = gr[key]
                if val is not None:
                    item = NumericTableWidgetItem(f"{val:.4f}")
                    item.setData(Qt.ItemDataRole.UserRole, val)
                    if key == "mean_apex":
                        # Centred bar: offset from reference RT
                        item.setData(
                            CenteredBarDelegate.PPM_DEVIATION_ROLE, val - ref_rt
                        )
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
            row_vals = [
                tbl.item(r, c).text() if tbl.item(r, c) else ""
                for c in range(tbl.columnCount())
            ]
            lines.append("\t".join(row_vals))
        QApplication.clipboard().setText("\n".join(lines))

    def _copy_rt_sample_table_r(self):
        """Copy RT per-sample table as R dataframe code."""
        tbl = self.rt_sample_table
        if tbl.rowCount() == 0:
            return
        headers = [
            tbl.horizontalHeaderItem(c).text().replace(" ", "_").replace("/", "_")
            for c in range(tbl.columnCount())
        ]
        rows_data = [
            [
                tbl.item(r, c).text() if tbl.item(r, c) else ""
                for c in range(tbl.columnCount())
            ]
            for r in range(tbl.rowCount())
        ]
        text_cols = {0, 1}
        col_lines = []
        for ci, h in enumerate(headers):
            vals = [
                f'"{rows_data[ri][ci]}"' if ci in text_cols else rows_data[ri][ci]
                for ri in range(len(rows_data))
            ]
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
            row_vals = [
                tbl.item(r, c).text() if tbl.item(r, c) else ""
                for c in range(tbl.columnCount())
            ]
            lines.append("\t".join(row_vals))
        QApplication.clipboard().setText("\n".join(lines))

    def _copy_rt_group_table_r(self):
        """Copy RT per-group table as R dataframe code."""
        tbl = self.rt_group_table
        if tbl.rowCount() == 0:
            return
        headers = [
            tbl.horizontalHeaderItem(c).text().replace(" ", "_").replace("/", "_")
            for c in range(tbl.columnCount())
        ]
        rows_data = [
            [
                tbl.item(r, c).text() if tbl.item(r, c) else ""
                for c in range(tbl.columnCount())
            ]
            for r in range(tbl.rowCount())
        ]
        col_lines = []
        for ci, h in enumerate(headers):
            vals = [
                f'"{rows_data[ri][ci]}"' if ci == 0 else rows_data[ri][ci]
                for ri in range(len(rows_data))
            ]
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
        for row_idx, (group, sample_name, mean_mz, p10, p90, eic_w) in enumerate(
            self._mz_stats_sample_rows
        ):
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
                    item.setData(
                        CenteredBarDelegate.PPM_BAR_COLOR_ROLE, QColor(grp_color)
                    )
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
                _gc = QColor(_grp_color_g)
                _gc.setAlphaF(0.5)
                grp_cell.setBackground(_gc)
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
                        item.setData(
                            CenteredBarDelegate.PPM_BAR_COLOR_ROLE, QColor(_grp_color_g)
                        )
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
        r_code_lines.append(
            f"rownames(similarity_matrix) <- c({', '.join(sample_names_r)})"
        )
        r_code_lines.append(
            f"colnames(similarity_matrix) <- c({', '.join(sample_names_r)})"
        )

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
        if (
            len(clean_intensity1) < 2
            or np.std(clean_intensity1) == 0
            or np.std(clean_intensity2) == 0
        ):
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
            sample_group[sample_name] = (
                str(group_value) if group_value is not None else "Unknown"
            )

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
        corr_matrix = np.ones(
            (n_samples, n_samples)
        )  # Diagonal is 1.0 (self-correlation)

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
        self.similarity_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )

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
        self.similarity_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed
        )
        for col in range(n_samples):
            self.similarity_table.setColumnWidth(col, 70)
        # Restore Interactive so user can still resize manually afterwards
        self.similarity_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )

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
                scatter = ax.scatter(
                    x, y, c=[color], s=100, alpha=0.7, edgecolors="black", linewidth=1.5
                )

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
                        distance = np.sqrt(
                            (event.xdata - x) ** 2 + (event.ydata - y) ** 2
                        )

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

        self._pca_hover_cid = self.pca_canvas.mpl_connect(
            "motion_notify_event", on_hover
        )

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
                quant_data = self.file_manager.get_quantification_data(
                    filepath, compound_name
                )

                if quant_data is not None:
                    true_abundance, unit = quant_data
                    dilution = self.file_manager.get_dilution_factor(filepath)
                    inj_vol = self.file_manager.get_injection_volume(filepath)
                    correction_factor = inj_vol * dilution
                    # The stored abundance IS the in-vial concentration.
                    # The actual sample concentration = vial_abundance * dilution.
                    vial_abundance = true_abundance
                    # Optionally normalize peak area by injection volume and dilution
                    corrected_area = (
                        peak_area * correction_factor
                        if self.normalize_peak_area_checkbox.isChecked()
                        else peak_area
                    )
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
            checkbox_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            )
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
                        _mf = files_data_plot[
                            files_data_plot["filename"] == sample_name
                        ]
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
            equation = (
                f"y = {coeffs[0]:.4g}x\u00b2 + {coeffs[1]:.4g}x + {coeffs[2]:.4g}"
            )

        # Store calibration info
        self.calibration_info = {
            "coeffs": coeffs,
            "model_type": model_type,
            "transform": transform,
            "unit": self.calibration_table.item(0, 4).text()
            if self.calibration_table.rowCount() > 0
            else "",
        }

        # Regression line spanning ALL samples (calibration + unknown)
        all_x_raw = calib_checked_x + calib_excl_x + unknown_x_raw
        all_x_t = (
            [apply_transform(v) for v in all_x_raw] if all_x_raw else list(x_fit_pts)
        )
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
                quant_data = self.file_manager.get_quantification_data(
                    filepath, compound_name
                )

            correction_factor = inj_vol * dilution
            # Optionally correct peak area by injection volume and dilution before prediction
            corrected_pa = (
                peak_area * correction_factor
                if self.normalize_peak_area_checkbox.isChecked()
                else peak_area
            )
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
            type_item.setBackground(
                CALIB_BG if data["type"] == "Calibration standard" else UNKNOWN_BG
            )
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

            self.calculated_abundances_table.setItem(
                i, 3, QTableWidgetItem(f"{data['peak_area']:.2e}")
            )
            self.calculated_abundances_table.setItem(
                i, 4, QTableWidgetItem(data["actual_abundance"])
            )
            self.calculated_abundances_table.setItem(
                i, 5, QTableWidgetItem(f"{data['predicted_abundance']:.4g}")
            )
            self.calculated_abundances_table.setItem(
                i, 6, QTableWidgetItem(data["unit"])
            )
            self.calculated_abundances_table.setItem(
                i, 7, QTableWidgetItem(f"{data['dilution']:.4g}")
            )
            self.calculated_abundances_table.setItem(
                i, 8, QTableWidgetItem(f"{data['inj_vol']:.4g}")
            )
            self.calculated_abundances_table.setItem(
                i, 9, QTableWidgetItem(f"{data['correction_factor']:.4g}")
            )

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
            header.append(
                self.calculated_abundances_table.horizontalHeaderItem(col).text()
            )
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
        col_maxima = {
            k: max((abs(s[k]) for s in stats_data), default=0) for k in stat_keys
        }

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
        header = [
            self.quant_summary_table.horizontalHeaderItem(c).text()
            for c in range(self.quant_summary_table.columnCount())
        ]
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
            col_name = (
                self.quant_summary_table.horizontalHeaderItem(col)
                .text()
                .replace(" ", "_")
            )
            values = []
            for row in range(self.quant_summary_table.rowCount()):
                item = self.quant_summary_table.item(row, col)
                text = item.text() if item else ""
                values.append(f'"{text}"' if col == 0 else text)
            lines.append(f"{col_name} = c({', '.join(values)})")
        QApplication.clipboard().setText(f"data.frame(\n  {',\n  '.join(lines)}\n)")

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
        self.calculate_file_shifts()

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
            group_col = (
                self.grouping_column if self.grouping_column in df.columns else "group"
            )
            if group_col not in df.columns:
                group_col = None

            work_df = df.assign(_inj_order=inj_order)
            if group_col is not None:
                group_vals = work_df[group_col].astype(str)
                all_groups = natsorted(group_vals.unique())
                group_rank = group_vals.map({g: i for i, g in enumerate(all_groups)})
                work_df = work_df.assign(_group_rank=group_rank)
                sorted_df = work_df.sort_values(
                    ["_group_rank", "_inj_order"], na_position="last"
                )
            else:
                sorted_df = work_df.sort_values("_inj_order", na_position="last")

            for rank, (_, row) in enumerate(sorted_df.iterrows()):
                self.file_shifts[row["Filepath"]] = rank * shift_amount
        else:
            # "By injection order": sort globally by injection_order
            sorted_df = df.assign(_inj_order=inj_order).sort_values(
                "_inj_order", na_position="last"
            )
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

        # Enable reset view button now that we have data
        self.reset_view_btn.setEnabled(True)

        sep_mode = self._separation_mode()
        separate_groups = sep_mode == "By group"
        separate_injection = sep_mode in (
            "By injection order",
            "By group, then injection order",
        )
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
                if separate_injection:
                    # Per-file legend entry showing its individual shift
                    file_shift = self.file_shifts.get(filepath, 0.0)
                    name_without_ext = (
                        filename.rsplit(".", 1)[0] if "." in filename else filename
                    )
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
        self._add_reference_lines(
            groups_data, separate_groups, separate_injection, sep_mode
        )

        # Show legend with better formatting
        legend = self.chart.legend()
        legend_pos = (
            self.legend_position_combo.currentText()
            if hasattr(self, "legend_position_combo")
            else "Right"
        )
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

    def _add_reference_lines(
        self, groups_data, separate_groups, separate_injection=False, sep_mode="None"
    ):
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

        def _add_vline(reference_rt):
            vertical_line = QLineSeries()
            vertical_line.setName("")  # No legend entry
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
                    group_shift = (
                        self.group_shifts.get(group_name, 0.0)
                        if separate_groups
                        else 0.0
                    )
                    for file_data in group_files:
                        total_shift = group_shift + self.file_shifts.get(
                            file_data["filepath"], 0.0
                        )
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
            if not (
                self.file_manager.keep_in_memory
                and filepath in self.file_manager.cached_data
            ):
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

        # Add MSMS viewing options (unfiltered)
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

        # Add per-type MSMS submenus when a filter regex is configured
        filter_types = self._get_msms_filter_types_at_rt(rt_value, 9.0 / 60.0)
        if filter_types:
            context_menu.addSeparator()
            type_header = QAction("— by filter-string type —", self)
            type_header.setEnabled(False)
            context_menu.addAction(type_header)
            for ftype in filter_types:
                sub = context_menu.addMenu(f"MSMS: {ftype}")
                for secs, secs_min in (
                    (3, 3.0 / 60.0),
                    (6, 6.0 / 60.0),
                    (9, 9.0 / 60.0),
                ):
                    action = sub.addAction(f"±{secs} seconds")
                    action.triggered.connect(
                        lambda checked=False,
                        ft=ftype,
                        sw=secs_min: self.view_msms_spectra(
                            rt_value, sw, filter_type=ft
                        )
                    )

        # Show the menu at the clicked position
        global_pos = self.chart_view.mapToGlobal(position.toPoint())
        context_menu.exec(global_pos)

    def view_msms_spectra(
        self, rt_center: float, rt_window: float, filter_type: Optional[str] = None
    ):
        """View MSMS spectra within the specified RT window.

        If *filter_type* is given, only spectra whose filter-string matches the
        configured regex and produces that type label are shown.
        """
        try:
            # Calculate RT window
            rt_start = rt_center - rt_window
            rt_end = rt_center + rt_window

            # Find MSMS spectra (optionally restricted to one filter-string type)
            msms_spectra = self.find_msms_spectra(
                rt_start, rt_end, filter_type=filter_type
            )

            if not msms_spectra:
                type_hint = f" [{filter_type}]" if filter_type else ""
                QMessageBox.information(
                    self,
                    "No MSMS Found",
                    f"No MSMS spectra{type_hint} found for m/z {self.target_mz:.4f} "
                    f"in RT window {rt_center:.2f} ± {rt_window * 60:.0f} s",
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
                file_manager=self.file_manager,
                filter_type=filter_type,
                defaults=self.defaults,
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
                                    "scan_id": spectrum_data.get("scan_id"),
                                    "filter_string": spectrum_data.get(
                                        "filter_string", "NA"
                                    ),
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
                                        "scan_id": spectrum.ID,
                                        "filter_string": spectrum.get(
                                            "MS:1000512", "NA"
                                        ),
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

        # Keep both charts in lockstep horizontally
        self._connect_scatter_x_axis_sync()

        # Update context menu text
        self.update_context_menu_text("Hide 2D scatter plot")
        if hasattr(self, "scatter_toggle_btn"):
            self.scatter_toggle_btn.setText("Hide 2D Scatter Plot")

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
        if hasattr(self, "scatter_toggle_btn"):
            self.scatter_toggle_btn.setText("Show 2D Scatter Plot")

    def toggle_scatter_plot(self):
        """Show/hide the optional RT vs m/z scatter plot."""
        if self.scatter_plot_view is not None:
            self.remove_scatter_plot()
            return

        rt_min, rt_max = self.get_rt_range()
        self.add_scatter_plot((rt_min + rt_max) / 2.0)

    def _get_scatter_x_axis(self):
        """Return the scatter chart x-axis if available."""
        if self.scatter_plot_view is None or not hasattr(
            self.scatter_plot_view, "chart"
        ):
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

        if (
            abs(scatter_x_axis.min() - minimum) < 1e-12
            and abs(scatter_x_axis.max() - maximum) < 1e-12
        ):
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

        if (
            abs(self.x_axis.min() - minimum) < 1e-12
            and abs(self.x_axis.max() - maximum) < 1e-12
        ):
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

    def find_msms_spectra(
        self, rt_start: float, rt_end: float, filter_type: Optional[str] = None
    ):
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
                                    "filter_string": spectrum_data.get("filter_string"),
                                }
                                # Apply filter-type restriction
                                if filter_type is not None:
                                    parsed = self._parse_filter_string_type(
                                        msms_spectrum["filter_string"] or ""
                                    )
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
                                        "filter_string": self.file_manager._get_filter_string(
                                            spectrum
                                        ),
                                    }
                                    # Apply filter-type restriction
                                    if filter_type is not None:
                                        parsed = self._parse_filter_string_type(
                                            spectrum_data["filter_string"] or ""
                                        )
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

        # Hover tooltip label for m/z values
        self.hover_label = QLabel(self)
        self.hover_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 220);
                border: 1px solid #555;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 11px;
            }
        """)
        self.hover_label.hide()
        self.hover_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hover_mz = None
        self._hover_series = None  # Firebrick stick drawn over the hovered peak

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

        # Update hover tooltip when not interacting
        if not self.is_panning and not self.is_zooming:
            self._update_hover_tooltip(event)

        super().mouseMoveEvent(event)

    def _update_hover_tooltip(self, event):
        """Show m/z tooltip for the peak closest to the cursor in Euclidean pixel space"""
        if self.spectrum_data is None:
            self.hover_label.hide()
            return

        plot_area = self.chart().plotArea()
        if not plot_area.contains(event.position()):
            self.hover_label.hide()
            if self._hover_series is not None:
                self.chart().removeSeries(self._hover_series)
                self._hover_series = None
            self._hover_mz = None
            return

        mz_array = self.spectrum_data.get("mz", [])
        intensity_array = self.spectrum_data.get("intensity", [])
        if len(mz_array) == 0:
            self.hover_label.hide()
            return

        max_intensity = max((float(v) for v in intensity_array), default=1.0)
        if max_intensity <= 0:
            max_intensity = 1.0

        # Find peak whose stick is closest to the cursor in pixel-space Euclidean distance
        PIXEL_THRESHOLD = 20
        mx = event.position().x()
        my = event.position().y()

        best_mz = None
        best_norm_int = 0.0
        best_dist = float("inf")

        for mz_val, int_val in zip(mz_array, intensity_array):
            mz_f = float(mz_val)
            norm_int = (float(int_val) / max_intensity) * 100.0
            tip = self.chart().mapToPosition(QPointF(mz_f, norm_int))
            base = self.chart().mapToPosition(QPointF(mz_f, 0.0))
            tx, ty, bx, by = tip.x(), tip.y(), base.x(), base.y()
            dx, dy = bx - tx, by - ty
            seg_len_sq = dx * dx + dy * dy
            t = (
                max(0.0, min(1.0, ((mx - tx) * dx + (my - ty) * dy) / seg_len_sq))
                if seg_len_sq > 0
                else 0.0
            )
            dist = ((mx - tx - t * dx) ** 2 + (my - ty - t * dy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_mz = mz_f
                best_norm_int = norm_int

        if best_mz is not None and best_dist <= PIXEL_THRESHOLD:
            if best_mz != self._hover_mz:
                self._hover_mz = best_mz

                if self._hover_series is not None:
                    self.chart().removeSeries(self._hover_series)
                    self._hover_series = None

                hover_series = QLineSeries()
                hover_pen = QPen(QColor(178, 34, 34))  # Firebrick
                hover_pen.setWidth(3)
                hover_series.setPen(hover_pen)
                hover_series.append(best_mz, 0.0)
                hover_series.append(best_mz, best_norm_int)
                hover_series.append(best_mz, 0.0)
                self.chart().addSeries(hover_series)
                x_axes = self.chart().axes(Qt.Orientation.Horizontal)
                y_axes = self.chart().axes(Qt.Orientation.Vertical)
                if x_axes and y_axes:
                    hover_series.attachAxis(x_axes[0])
                    hover_series.attachAxis(y_axes[0])
                self._hover_series = hover_series
                self.chart().legend().setVisible(False)

                self.hover_label.setText(f"m/z: {best_mz:.4f}")
                self.hover_label.adjustSize()

            # Position label centered above the peak tip
            tip_pos = self.chart().mapToPosition(QPointF(best_mz, best_norm_int))
            lx = int(tip_pos.x()) - self.hover_label.width() // 2
            ly = int(tip_pos.y()) - self.hover_label.height() - 6
            lx = max(0, min(lx, self.width() - self.hover_label.width()))
            if ly < 0:
                ly = int(tip_pos.y()) + 6
            self.hover_label.move(lx, ly)
            self.hover_label.show()
        else:
            if self._hover_series is not None:
                self.chart().removeSeries(self._hover_series)
                self._hover_series = None
            self._hover_mz = None
            self.hover_label.hide()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_zooming = False

        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        """Hide hover tooltip when the mouse leaves the chart"""
        if self._hover_series is not None:
            self.chart().removeSeries(self._hover_series)
            self._hover_series = None
        self.hover_label.hide()
        self._hover_mz = None
        super().leaveEvent(event)

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

        # Number of annotated signals spinner
        controls_layout.addWidget(QLabel("Annotated signals:"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(0, 20)
        self.top_n_spin.setValue(3)
        self.top_n_spin.setFixedWidth(55)
        self.top_n_spin.valueChanged.connect(self.update_all_top_signal_labels)
        controls_layout.addWidget(self.top_n_spin)

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

            # File header (clickable → opens single-spectrum detail window)
            display_name = filename.split(".")[0] if "." in filename else filename
            scan_id = spectrum.get("scan_id", "")
            filter_str = spectrum.get("filter_string", "")
            file_header_text = (
                f"<b>{display_name}</b> | Group: {group} | RT: {spectrum['rt']:.4f} min"
                " ·  <i>click to open single view</i>"
            )
            if scan_id:
                file_header_text += f" | Scan: {scan_id}"
            if filter_str:
                file_header_text += f" | Filter: {filter_str}"
            file_label = ClickableLabel(file_header_text)
            file_label.setStyleSheet("""
                QLabel { 
                    background-color: #e8f4fd; 
                    padding: 4px; 
                    margin: 1px;
                    border: 1px solid #7ab3d4;
                    border-radius: 2px;
                }
                QLabel:hover {
                    background-color: #c5e4f5;
                }
            """)
            file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            file_label.setMaximumHeight(25)

            # Capture loop variables explicitly via default-argument binding
            def _open_single(fp=filepath, fd=file_data):
                win = MS1SingleSpectrumWindow(
                    spectrum_data=fd["spectrum"],
                    filename=fd["filename"],
                    group=fd["group"],
                    target_mz=self.target_mz,
                    compound_name=self.compound_name,
                    adduct=self.adduct,
                    mz_tolerance=self.mz_tolerance,
                    formula=self.formula,
                    parent=self,
                )
                win.show()

            file_label.clicked.connect(_open_single)
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

    def update_all_top_signal_labels(self):
        """Propagate the top-N spinner value to every chart view and refresh labels"""
        n = self.top_n_spin.value()
        for chart_view in self.chart_views.values():
            chart_view.top_n = n
            chart_view.update_top_signal_labels()

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

        # Hover overlay state
        self.tooltip_label = None
        self._hover_mz = None
        self._hover_series = None  # Firebrick stick drawn over the hovered peak

        # Top signal labels
        self.signal_labels = []  # List to store QLabel widgets for top signals
        self.top_n = 3  # Number of top signals to annotate

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
        """Handle hover: highlight the peak closest to the cursor in Euclidean pixel space"""
        plot_area = self.chart().plotArea()
        if not plot_area.contains(event.position()):
            self._hide_tooltip()
            return

        if len(self.mz_array) == 0:
            self._hide_tooltip()
            return

        # Find peak whose stick is closest to the cursor in pixel-space Euclidean distance
        PIXEL_THRESHOLD = 20
        mx = event.position().x()
        my = event.position().y()

        best_mz = None
        best_intensity = 0.0
        best_dist = float("inf")

        for mz_f, int_f in zip(self.mz_array, self.intensity_array):
            mz_f = float(mz_f)
            int_f = float(int_f)
            tip = self.chart().mapToPosition(QPointF(mz_f, int_f))
            base = self.chart().mapToPosition(QPointF(mz_f, 0.0))
            tx, ty, bx, by = tip.x(), tip.y(), base.x(), base.y()
            dx, dy = bx - tx, by - ty
            seg_len_sq = dx * dx + dy * dy
            t = (
                max(0.0, min(1.0, ((mx - tx) * dx + (my - ty) * dy) / seg_len_sq))
                if seg_len_sq > 0
                else 0.0
            )
            dist = ((mx - tx - t * dx) ** 2 + (my - ty - t * dy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_mz = mz_f
                best_intensity = int_f

        if best_mz is not None and best_dist <= PIXEL_THRESHOLD:
            if best_mz != self._hover_mz:
                self._hover_mz = best_mz

                # Remove old firebrick series
                if self._hover_series is not None:
                    self.chart().removeSeries(self._hover_series)
                    self._hover_series = None

                # Add new firebrick series for this peak
                hover_s = QLineSeries()
                hover_pen = QPen(QColor(178, 34, 34))  # Firebrick
                hover_pen.setWidth(3)
                hover_s.setPen(hover_pen)
                hover_s.append(best_mz, 0.0)
                hover_s.append(best_mz, best_intensity)
                hover_s.append(best_mz, 0.0)
                self.chart().addSeries(hover_s)
                x_axes = self.chart().axes(Qt.Orientation.Horizontal)
                y_axes = self.chart().axes(Qt.Orientation.Vertical)
                if x_axes and y_axes:
                    hover_s.attachAxis(x_axes[0])
                    hover_s.attachAxis(y_axes[0])
                self._hover_series = hover_s
                self.chart().legend().hide()

                if self.tooltip_label is None:
                    self.tooltip_label = QLabel(self)
                    self.tooltip_label.setStyleSheet("""
                        QLabel {
                            background-color: rgba(255, 255, 255, 220);
                            border: 1px solid #555;
                            border-radius: 3px;
                            padding: 2px 6px;
                            font-size: 11px;
                        }
                    """)
                    self.tooltip_label.setAttribute(
                        Qt.WidgetAttribute.WA_TransparentForMouseEvents
                    )
                self.tooltip_label.setText(f"m/z: {best_mz:.4f}")
                self.tooltip_label.adjustSize()

            # Re-position label centered above the peak tip every move
            tip_pos = self.chart().mapToPosition(QPointF(best_mz, best_intensity))
            lx = int(tip_pos.x()) - self.tooltip_label.width() // 2
            ly = int(tip_pos.y()) - self.tooltip_label.height() - 6
            lx = max(0, min(lx, self.width() - self.tooltip_label.width()))
            if ly < 0:
                ly = int(tip_pos.y()) + 6
            self.tooltip_label.move(lx, ly)
            self.tooltip_label.show()
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
        """Hide the tooltip label and remove any hover series"""
        if self.tooltip_label is not None:
            self.tooltip_label.hide()
        if self._hover_series is not None:
            self.chart().removeSeries(self._hover_series)
            self._hover_series = None
        self._hover_mz = None

    def leaveEvent(self, event):
        """Hide tooltip when mouse leaves the widget"""
        self._hide_tooltip()
        super().leaveEvent(event)

    def update_top_signal_labels(self):
        """Update labels for the top-N most abundant signals in the visible range"""
        # Clear existing labels
        for label in self.signal_labels:
            label.deleteLater()
        self.signal_labels.clear()

        if len(self.mz_array) == 0 or self.top_n <= 0:
            return

        # Get current visible range
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]

        x_min = x_axis.min()
        x_max = x_axis.max()

        # Find signals within visible range
        visible_mask = (self.mz_array >= x_min) & (self.mz_array <= x_max)
        if not np.any(visible_mask):
            return

        visible_mz = self.mz_array[visible_mask]
        visible_intensities = self.intensity_array[visible_mask]

        # Find top-N most abundant signals
        if len(visible_intensities) > 0:
            n = min(self.top_n, len(visible_intensities))
            top_indices = np.argsort(visible_intensities)[-n:][
                ::-1
            ]  # Top N, highest first

            for idx in top_indices:
                mz = visible_mz[idx]
                intensity = visible_intensities[idx]

                # Use mapToPosition for accurate pixel position (consistent with hover)
                tip_pos = self.chart().mapToPosition(
                    QPointF(float(mz), float(intensity))
                )
                widget_x = tip_pos.x()
                widget_y = tip_pos.y()

                # Create label — uniform style, alpha 0.5 (128 in Qt's 0-255 range)
                label = QLabel(self)
                label.setText(f"{mz:.4f}")
                label.setStyleSheet("""
                    QLabel {
                        background-color: rgba(46, 134, 171, 128);
                        color: white;
                        border-radius: 3px;
                        padding: 2px 4px;
                        font-weight: bold;
                    }
                """)
                label.adjustSize()

                # Center label horizontally above the peak tip
                label_x = int(widget_x) - label.width() // 2
                label_y = int(widget_y) - label.height() - 6

                # Keep label within widget bounds
                label_x = max(0, min(label_x, self.width() - label.width()))
                if label_y < 0:
                    label_y = int(widget_y) + 6

                label.move(label_x, label_y)
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
        file_manager=None,
        filter_type=None,
        defaults=None,
    ):
        super().__init__(parent)

        # Configure as independent window
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.msms_spectra = msms_spectra
        self.target_mz = target_mz
        self.rt_center = rt_center
        self.rt_window = rt_window
        self.compound_name = compound_name
        self.adduct = adduct
        self.file_manager = file_manager
        self.filter_type = filter_type
        self.defaults = defaults or {}

        self.init_ui()

    def _group_color_for(self, group: str) -> QColor | None:
        """Return a QColor for the given group, or None."""
        if self.file_manager is None:
            return None
        color = self.file_manager.get_group_color(group)
        if color:
            return QColor(color) if not isinstance(color, QColor) else color
        return None

    def init_ui(self):
        """Initialize the MSMS viewer UI"""
        total_spectra = sum(len(data["spectra"]) for data in self.msms_spectra.values())
        type_tag = f" | Filter: {self.filter_type}" if self.filter_type else ""
        self.setWindowTitle(
            f"MSMS: {self.compound_name} ({self.adduct}) | "
            f"m/z {self.target_mz:.4f} | "
            f"RT {self.rt_center:.2f}\u00b1{self.rt_window * 60:.0f} s"
            f"{type_tag} | "
            f"{len(self.msms_spectra)} files | {total_spectra} spectra"
        )
        self.setGeometry(100, 100, 1800, 1000)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Pre-process data: sort by intensity and find global m/z range
        self._preprocess_spectra_data()

        # Create main splitter for vertical resizing (top: heatmap, bottom: spectra grid)
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: Inter-file similarity overview (heatmap)
        inter_file_widget = self._create_similarity_overview_widget()
        main_splitter.addWidget(inter_file_widget)

        # Bottom: Spectra grid
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setSpacing(1)
        grid_layout.setContentsMargins(1, 1, 1, 1)
        grid_layout.setVerticalSpacing(0)

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
                    f" | Sim: "
                    f"Med:{stats['median']:.3f} "
                    f"90%:{stats['percentile_90']:.3f}"
                )

            display_name = filename.split(".")[0] if "." in filename else filename
            file_header_text = f"<b>{display_name}</b> | {group} | {len(spectra)} spectra{similarity_info}"
            file_label = QLabel(file_header_text)

            # Colour header by group
            grp_color = self._group_color_for(group)
            if grp_color:
                c = QColor(grp_color)
                c.setAlphaF(0.5)
                r, g, b, a = c.red(), c.green(), c.blue(), c.alpha()
                bg_css = f"rgba({r},{g},{b},{a})"
            else:
                bg_css = "#f0f0f0"

            file_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {bg_css};
                    padding: 2px 4px;
                    margin: 0px;
                    border-bottom: 1px solid #ccc;
                }}
            """)
            file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            file_label.setMaximumHeight(18)
            grid_layout.addWidget(file_label, row, 0, 1, max(1, len(spectra)))
            row += 1

            # Add spectra horizontally for this file (sorted by intensity)
            for col, spectrum_data in enumerate(spectra):
                chart_widget = self.create_msms_chart(spectrum_data, filename, group)
                grid_layout.addWidget(chart_widget, row, col)

            row += 1

        # Add scroll area to splitter
        main_splitter.addWidget(scroll_area)

        # Total height ~1000, give 200 to heatmap, 800 to spectra grid
        main_splitter.setSizes([200, 800])

        layout.addWidget(main_splitter)

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

        # Sort files by group then filename for consistent ordering
        natsort_key = natsort_keygen()
        self.processed_data = sorted(
            processed_files,
            key=lambda x: (
                natsort_key(x[1].get("group", "Unknown")),
                natsort_key(x[1]["filename"]),
            ),
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

    def _get_pair_mz_tolerance(self, spec1, spec2) -> float:
        """Return the appropriate m/z tolerance (Da) for a pair of spectra.

        Looks up per–filter-type-group overrides from *self.defaults*; falls
        back to the configured default tolerance when no match is found.
        """
        defaults = self.defaults
        default_tol = defaults.get("msms_similarity_default_tolerance", 0.1)
        group_tols = {
            entry["filter_type"]: float(entry["mz_tolerance"])
            for entry in defaults.get("msms_similarity_group_tolerances", [])
        }
        if not group_tols:
            return default_tol

        pattern = defaults.get("msms_filter_regex", "")
        replacement = defaults.get("msms_filter_replacement", "")
        if not pattern:
            return default_tol

        try:
            compiled = re.compile(pattern)
        except re.error:
            return default_tol

        def parse_type(fs):
            if not fs:
                return None
            m = compiled.search(fs)
            return m.expand(replacement) if m else None

        type1 = parse_type(spec1.get("filter_string") or "")
        type2 = parse_type(spec2.get("filter_string") or "")

        # Prefer exact match when both spectra share the same filter type
        if type1 is not None and type1 == type2 and type1 in group_tols:
            return group_tols[type1]
        for ft in (type1, type2):
            if ft and ft in group_tols:
                return group_tols[ft]
        return default_tol

    def _calculate_cosine_similarities(self):
        """Calculate cosine similarity scores within and between files"""
        self.intra_file_similarities = {}  # filename -> similarity stats
        self.inter_file_similarities = {}  # (file1, file2) -> list of all similarities

        method = self.defaults.get("msms_similarity_method", "CosineHungarian")

        # Calculate intra-file similarities (within each file)
        for filepath, file_data in self.processed_data:
            filename = file_data["filename"]
            spectra = file_data["spectra"]

            similarities = []
            if len(spectra) > 1:
                for i in range(len(spectra)):
                    for j in range(i + 1, len(spectra)):
                        tol = self._get_pair_mz_tolerance(spectra[i], spectra[j])
                        sim = calculate_cosine_similarity(
                            spectra[i], spectra[j], mz_tolerance=tol, method=method
                        )
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
                        tol = self._get_pair_mz_tolerance(spec1, spec2)
                        sim = calculate_cosine_similarity(
                            spec1, spec2, mz_tolerance=tol, method=method
                        )
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
        file_group = {
            data[1]["filename"]: data[1].get("group", "Unknown")
            for data in self.processed_data
        }
        if len(files) > 1:
            inter_table = QTableWidget(len(files), len(files))

            # Set coloured header items per group
            for idx, fname in enumerate(files):
                grp = file_group.get(fname, "Unknown")
                short = fname.split(".")[0] if "." in fname else fname
                grp_color = self._group_color_for(grp)

                for make_h in (True, False):
                    hi = QTableWidgetItem(short)
                    if grp_color:
                        c = QColor(grp_color)
                        c.setAlphaF(0.5)
                        hi.setBackground(c)
                    fnt = hi.font()
                    # fnt.setPointSize(7)
                    hi.setFont(fnt)
                    if make_h:
                        inter_table.setHorizontalHeaderItem(idx, hi)
                    else:
                        inter_table.setVerticalHeaderItem(idx, hi)

            inter_table.setMinimumHeight(min(120, 24 + len(files) * 20))
            inter_table.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )

            # Set table properties for better display
            inter_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Interactive
            )
            inter_table.horizontalHeader().setDefaultSectionSize(70)
            inter_table.verticalHeader().setDefaultSectionSize(18)

            # Enable context menu
            inter_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            inter_table.customContextMenuRequested.connect(
                lambda pos: self._show_similarity_context_menu(inter_table, pos)
            )

            # Single click opens the two spectra in individual viewer windows
            inter_table.cellClicked.connect(self._on_similarity_cell_clicked)

            # Store reference to inter_table for context menu handler
            self.inter_table = inter_table

            # Fill the table
            for i, file1 in enumerate(files):
                for j, file2 in enumerate(files):
                    similarities = []  # Initialize similarities for all cases

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
                    # Compact font
                    font = item.font()
                    # font.setPointSize(6)
                    item.setFont(font)

                    # Store metadata for context menu (file indices and similarity data)
                    item.setData(
                        Qt.ItemDataRole.UserRole,
                        {
                            "file1_idx": i,
                            "file2_idx": j,
                            "file1_name": file1,
                            "file2_name": file2,
                            "similarities": similarities,
                            "is_diagonal": i == j,
                        },
                    )

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
            f"RT: {spectrum_data['rt']:.2f} min / {spectrum_data['rt'] / 60.0:.1f} sec /  | "
            f"Precursor: {spectrum_data['precursor_mz']:.4f} | "
            f"Intensity: {intensity_text}"
            + (
                f"\nScan: {spectrum_data['scan_id']}"
                if spectrum_data.get("scan_id")
                else ""
            )
            + (
                f" | Filter: {spectrum_data['filter_string']}"
                if spectrum_data.get("filter_string")
                else ""
            )
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

    def _show_similarity_context_menu(self, table, pos):
        """Show context menu for similarity table cells"""
        item = table.itemAt(pos)
        if not item:
            return

        # Get stored metadata
        metadata = item.data(Qt.ItemDataRole.UserRole)
        if not metadata:
            return

        file1_idx = metadata["file1_idx"]
        file2_idx = metadata["file2_idx"]
        file1_name = metadata["file1_name"]
        file2_name = metadata["file2_name"]
        similarities = metadata["similarities"]
        is_diagonal = metadata["is_diagonal"]

        # Get spectrum data for the two files
        file1_data = None
        file2_data = None
        for filepath, data in self.processed_data:
            if data["filename"] == file1_name:
                file1_data = data
            if data["filename"] == file2_name:
                file2_data = data

        if not file1_data or not file2_data:
            return

        # Count number of signals in each file
        num_signals_file1 = sum(len(spec["mz"]) for spec in file1_data["spectra"])
        num_signals_file2 = sum(len(spec["mz"]) for spec in file2_data["spectra"])

        # Get precursor information (use first spectrum from each file as representative)
        file1_spectra = file1_data["spectra"]
        file2_spectra = file2_data["spectra"]

        # Create context menu
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: white;
                border: 1px solid #ccc;
                padding: 5px;
            }
            QMenu::item {
                padding: 5px 20px;
            }
            QMenu::item:selected {
                background-color: #e0e0e0;
            }
        """)

        # Add information section
        info_section = QLabel()
        info_text = f"<b>Comparison Information</b><br>"
        info_text += f"File A: {file1_name}<br>"
        info_text += f"File B: {file2_name}<br>"
        info_text += f"<br>"
        info_text += f"<b>Signals:</b><br>"
        info_text += f"  Spectrum A: {num_signals_file1} signals across {len(file1_spectra)} spectra<br>"
        info_text += f"  Spectrum B: {num_signals_file2} signals across {len(file2_spectra)} spectra<br>"
        info_text += f"<br>"

        # Add precursor info if available
        if file1_spectra and file2_spectra:
            info_text += f"<b>Representative Precursor Info:</b><br>"
            info_text += f"  Spectrum A: m/z = {file1_spectra[0]['precursor_mz']:.4f}, "
            info_text += (
                f"Intensity = {file1_spectra[0].get('precursor_intensity', 0):.2e}<br>"
            )
            info_text += f"  Spectrum B: m/z = {file2_spectra[0]['precursor_mz']:.4f}, "
            info_text += (
                f"Intensity = {file2_spectra[0].get('precursor_intensity', 0):.2e}<br>"
            )

        info_section.setText(info_text)
        info_section.setStyleSheet("padding: 10px; background-color: #f9f9f9;")
        info_section.setWordWrap(True)

        # Add the info label as a widget action
        info_action = QAction(self)
        info_action.setEnabled(False)
        menu.addAction(info_action)

        # Create a custom widget to hold the label
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.addWidget(info_section)
        info_layout.setContentsMargins(0, 0, 0, 0)

        widget_action = QWidgetAction(self)
        widget_action.setDefaultWidget(info_widget)
        menu.addAction(widget_action)

        menu.addSeparator()

        # Add mirror plot action (only for off-diagonal or if there are multiple spectra)
        if not is_diagonal or len(file1_spectra) >= 2:
            mirror_action = QAction("Show Mirror Plot", self)
            mirror_action.triggered.connect(
                lambda: self._show_mirror_plot(file1_data, file2_data, is_diagonal)
            )
            menu.addAction(mirror_action)

        # Show menu at cursor position
        menu.exec(table.viewport().mapToGlobal(pos))

    def _show_mirror_plot(self, file1_data, file2_data, is_diagonal):
        """Show mirror plot dialog for two files or two spectra from same file"""
        if is_diagonal and len(file1_data["spectra"]) >= 2:
            # For diagonal, compare first two spectra from the same file
            spectrum_a = file1_data["spectra"][0]
            spectrum_b = file1_data["spectra"][1]
            title_a = f"{file1_data['filename']} - RT: {spectrum_a['rt']:.2f} min"
            title_b = f"{file1_data['filename']} - RT: {spectrum_b['rt']:.2f} min"
        else:
            # For off-diagonal, compare first spectrum from each file
            spectrum_a = file1_data["spectra"][0] if file1_data["spectra"] else None
            spectrum_b = file2_data["spectra"][0] if file2_data["spectra"] else None

            if not spectrum_a or not spectrum_b:
                QMessageBox.warning(
                    self, "No Spectra", "No spectra available for comparison."
                )
                return

            title_a = f"{file1_data['filename']} - RT: {spectrum_a['rt']:.2f} min"
            title_b = f"{file2_data['filename']} - RT: {spectrum_b['rt']:.2f} min"

        # Calculate cosine similarity between these two spectra
        tol = self._get_pair_mz_tolerance(spectrum_a, spectrum_b)
        method = self.defaults.get("msms_similarity_method", "CosineHungarian")
        similarity = calculate_cosine_similarity(
            spectrum_a, spectrum_b, mz_tolerance=tol, method=method
        )

        # Open enhanced mirror plot window
        if not hasattr(self, "_open_popups"):
            self._open_popups = []
        window = EnhancedMirrorPlotWindow(
            spectrum_a,
            spectrum_b,
            title_a,
            title_b,
            similarity,
            mz_tolerance=tol,
            method=method,
        )
        window.destroyed.connect(
            lambda: self._open_popups.remove(window)
            if window in self._open_popups
            else None
        )
        self._open_popups.append(window)
        window.show()

    def _on_similarity_cell_clicked(self, row: int, col: int):
        """Open the two representative spectra for the clicked similarity cell."""
        if not hasattr(self, "inter_table"):
            return
        item = self.inter_table.item(row, col)
        if not item:
            return
        metadata = item.data(Qt.ItemDataRole.UserRole)
        if not metadata:
            return

        file1_name = metadata["file1_name"]
        file2_name = metadata["file2_name"]
        is_diagonal = metadata["is_diagonal"]

        # Locate the spectrum lists from processed_data
        file1_data = next(
            (d for _, d in self.processed_data if d["filename"] == file1_name), None
        )
        file2_data = next(
            (d for _, d in self.processed_data if d["filename"] == file2_name), None
        )
        if not file1_data or not file2_data:
            return

        spectra1 = file1_data["spectra"]
        spectra2 = file2_data["spectra"]

        if is_diagonal:
            # Show the two most intense spectra from the same file
            if len(spectra1) < 2:
                QMessageBox.information(
                    self,
                    "Single spectrum",
                    f"{file1_name} has only one spectrum — nothing to compare.",
                )
                return
            spec_a, spec_b = spectra1[0], spectra1[1]
            name_a = name_b = file1_data["filename"]
            group_a = group_b = file1_data.get("group", "")
        else:
            if not spectra1 or not spectra2:
                QMessageBox.warning(self, "No Spectra", "No spectra available.")
                return
            spec_a = spectra1[0]
            spec_b = spectra2[0]
            name_a = file1_data["filename"]
            name_b = file2_data["filename"]
            group_a = file1_data.get("group", "")
            group_b = file2_data.get("group", "")

        title_a = f"{name_a} — RT: {spec_a['rt']:.2f} min"
        title_b = f"{name_b} — RT: {spec_b['rt']:.2f} min"

        tol = self._get_pair_mz_tolerance(spec_a, spec_b)
        method = self.defaults.get("msms_similarity_method", "CosineHungarian")
        similarity = calculate_cosine_similarity(
            spec_a, spec_b, mz_tolerance=tol, method=method
        )

        if not hasattr(self, "_open_popups"):
            self._open_popups = []
        window = EnhancedMirrorPlotWindow(
            spec_a,
            spec_b,
            title_a,
            title_b,
            similarity,
            mz_tolerance=tol,
            method=method,
        )
        window.destroyed.connect(
            lambda: self._open_popups.remove(window)
            if window in self._open_popups
            else None
        )
        self._open_popups.append(window)
        window.show()

    def closeEvent(self, event):
        """Clean up when closing the window"""
        if hasattr(self, "extraction_worker") and self.extraction_worker.isRunning():
            self.extraction_worker.quit()
            self.extraction_worker.wait()
        event.accept()


class EnhancedMirrorPlotWindow(QWidget):
    """Mirror plot + fragment comparison table for two MSMS spectra.

    Spectrum A is drawn with positive intensities (blue), spectrum B with
    negative intensities (red).  Clicking any row in the fragment table
    highlights the corresponding peak(s) in firebrick colour.

    Matched fragments are placed in the same row; fragments present in only
    one spectrum leave the other spectrum's columns blank.
    """

    _MATCH_BG = QColor(220, 235, 252)  # light blue  – matched pair
    _ONLY_A_BG = QColor(255, 235, 230)  # light peach – only in A
    _ONLY_B_BG = QColor(255, 235, 230)  # light rose  – only in B
    # custom Qt data role used to store the original row-index so sorting works:
    _ROW_IDX_ROLE = Qt.ItemDataRole.UserRole + 1

    def __init__(
        self,
        spectrum_a,
        spectrum_b,
        title_a,
        title_b,
        similarity,
        mz_tolerance=0.1,
        method="CosineHungarian",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.spectrum_a = spectrum_a
        self.spectrum_b = spectrum_b
        self.title_a = title_a
        self.title_b = title_b
        self.similarity = similarity
        self.mz_tolerance = mz_tolerance
        self.method = method

        self._mz_a = np.array(spectrum_a["mz"], dtype=float)
        self._int_a = np.array(spectrum_a["intensity"], dtype=float)
        self._mz_b = np.array(spectrum_b["mz"], dtype=float)
        self._int_b = np.array(spectrum_b["intensity"], dtype=float)

        _max_a = (
            float(np.max(self._int_a))
            if len(self._int_a) > 0 and np.max(self._int_a) > 0
            else 1.0
        )
        _max_b = (
            float(np.max(self._int_b))
            if len(self._int_b) > 0 and np.max(self._int_b) > 0
            else 1.0
        )
        self._max_a = _max_a
        self._max_b = _max_b
        self._rel_a = self._int_a / _max_a * 100.0
        self._rel_b = self._int_b / _max_b * 100.0

        self._rows: list = self._compute_fragment_table()
        self._highlight_a: set = set()
        self._highlight_b: set = set()

        self.setWindowTitle(f"Mirror Plot — {title_a}  vs  {title_b}")
        self.resize(1420, 720)
        self._init_ui()

    # ------------------------------------------------------------------
    def _compute_fragment_table(self) -> list:
        """Greedy intensity-weighted peak matching (mirrors CosineGreedy logic).

        Returns a list of row dicts, sorted by ascending m/z.  Keys:
        idx_a, mz_a, int_a  (None if this row is unmatched from B only)
        idx_b, mz_b, int_b  (None if this row is unmatched from A only)
        delta_mz, delta_ppm (None if not matched)
        """
        mz_a, int_a = self._mz_a, self._int_a
        mz_b, int_b = self._mz_b, self._int_b
        tol = self.mz_tolerance

        # All candidate pairs within tolerance, sorted by product of intensities
        candidates = [
            (i, j, float(int_a[i]) * float(int_b[j]))
            for i in range(len(mz_a))
            for j in range(len(mz_b))
            if abs(float(mz_a[i]) - float(mz_b[j])) <= tol
        ]
        candidates.sort(key=lambda x: -x[2])

        used_a: set = set()
        used_b: set = set()
        pairs: list = []
        for i, j, _ in candidates:
            if i not in used_a and j not in used_b:
                pairs.append((i, j))
                used_a.add(i)
                used_b.add(j)

        rows: list = []
        for i, j in pairs:
            delta = float(mz_a[i]) - float(mz_b[j])
            rows.append(
                {
                    "idx_a": i,
                    "mz_a": float(mz_a[i]),
                    "int_a": float(int_a[i]),
                    "idx_b": j,
                    "mz_b": float(mz_b[j]),
                    "int_b": float(int_b[j]),
                    "delta_mz": delta,
                    "delta_ppm": delta / float(mz_b[j]) * 1e6
                    if float(mz_b[j]) != 0.0
                    else 0.0,
                }
            )
        for i in range(len(mz_a)):
            if i not in used_a:
                rows.append(
                    {
                        "idx_a": i,
                        "mz_a": float(mz_a[i]),
                        "int_a": float(int_a[i]),
                        "idx_b": None,
                        "mz_b": None,
                        "int_b": None,
                        "delta_mz": None,
                        "delta_ppm": None,
                    }
                )
        for j in range(len(mz_b)):
            if j not in used_b:
                rows.append(
                    {
                        "idx_a": None,
                        "mz_a": None,
                        "int_a": None,
                        "idx_b": j,
                        "mz_b": float(mz_b[j]),
                        "int_b": float(int_b[j]),
                        "delta_mz": None,
                        "delta_ppm": None,
                    }
                )

        rows.sort(key=lambda r: r["mz_a"] if r["mz_a"] is not None else r["mz_b"])
        return rows

    # ------------------------------------------------------------------
    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 8)
        outer.setSpacing(4)

        # ---- header block (fixed height — must not stretch vertically) ----
        n_matched = sum(
            1 for r in self._rows if r["idx_a"] is not None and r["idx_b"] is not None
        )
        n_only_a = sum(1 for r in self._rows if r["idx_b"] is None)
        n_only_b = sum(1 for r in self._rows if r["idx_a"] is None)

        def _spec_info_html(spectrum, title, label):
            """Build one info line for a spectrum."""
            prec_mz = spectrum.get("precursor_mz", None)
            prec_int = spectrum.get("precursor_intensity", None)
            scan_id = (
                spectrum.get("id") or spectrum.get("scan_id") or spectrum.get("index")
            )
            fs = spectrum.get("filter_string") or ""
            parts = [f"<b>{label}: {title}</b>"]
            if prec_mz is not None:
                parts.append(f"precursor m/z: <b>{float(prec_mz):.4f}</b>")
            if prec_int is not None:
                parts.append(f"precursor int.: <b>{float(prec_int):.3e}</b>")
            if scan_id is not None:
                parts.append(f"scan: <b>{scan_id}</b>")
            if fs:
                parts.append(f"<span style='color:#555;font-size:10px'>{fs}</span>")
            return " &nbsp;|&nbsp; ".join(parts)

        score_line = (
            f"{self.method} score: <b>{self.similarity:.4f}</b>"
            f" &nbsp;|&nbsp; tolerance: {self.mz_tolerance:.4f} Da"
            f" &nbsp;|&nbsp; matched: <b>{n_matched}</b>"
            f" &nbsp; only-A: <b>{n_only_a}</b>"
            f" &nbsp; only-B: <b>{n_only_b}</b>"
        )
        hdr_html = (
            f"{_spec_info_html(self.spectrum_a, self.title_a, 'A')}<br>"
            f"{_spec_info_html(self.spectrum_b, self.title_b, 'B')}<br>"
            f"<span style='color:#444'>{score_line}</span>"
        )
        hdr = QLabel(hdr_html)
        hdr.setStyleSheet("font-size: 11px; padding: 3px 4px;")
        hdr.setWordWrap(True)
        hdr.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        outer.addWidget(hdr, 0)  # stretch=0 → never grows vertically

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)  # stretch=1 → takes all remaining space

        # ---- left: mirror plot ----
        plot_w = QWidget()
        plot_l = QVBoxLayout(plot_w)
        plot_l.setContentsMargins(0, 0, 0, 0)
        plot_l.setSpacing(0)
        self._fig = Figure(figsize=(7, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        toolbar = NavigationToolbar(self._canvas, plot_w)
        plot_l.addWidget(toolbar)
        plot_l.addWidget(self._canvas)
        splitter.addWidget(plot_w)

        # ---- right: fragment table ----
        tbl_w = QWidget()
        tbl_l = QVBoxLayout(tbl_w)
        tbl_l.setContentsMargins(4, 0, 0, 0)
        tbl_l.setSpacing(2)

        hint = QLabel(
            "<small>Click a row to highlight that peak in the mirror plot.&nbsp;&nbsp;"
            "<span style='background:#dceafc;padding:1px 4px'>Blue background</span> = matched &nbsp;"
            "<span style='background:#ffebee;padding:1px 4px'>Peach background</span> = only in A or B</small>"
        )
        hint.setWordWrap(True)
        hint.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        tbl_l.addWidget(hint, 0)

        COL_HEADERS = [
            "m/z",
            "A m/z",
            "A Intensity",
            "A Rel.%",
            "B m/z",
            "B Intensity",
            "B Rel.%",
            "\u0394m/z (Da)",
            "\u0394m/z (ppm)",
        ]
        self._table = QTableWidget(len(self._rows), len(COL_HEADERS))
        self._table.setHorizontalHeaderLabels(COL_HEADERS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # NOTE: keep sorting disabled until after population — enabling it first
        # causes Qt to re-sort after every setItem(), scattering values into
        # wrong rows and leaving most cells empty.
        self._table.setSortingEnabled(False)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(True)
        self._table.verticalHeader().setDefaultSectionSize(20)

        RA = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        for row_idx, row in enumerate(self._rows):
            is_matched = row["idx_a"] is not None and row["idx_b"] is not None
            has_a = row["idx_a"] is not None
            has_b = row["idx_b"] is not None
            bg = (
                self._MATCH_BG
                if is_matched
                else self._ONLY_B_BG
                if not has_a
                else self._ONLY_A_BG
            )

            # Start with empty placeholder items so every cell has a background
            items = [QTableWidgetItem("") for _ in range(len(COL_HEADERS))]

            # col 0: mean m/z (average when matched, single value otherwise)
            mean_mz = (
                (row["mz_a"] + row["mz_b"]) / 2.0
                if is_matched
                else row["mz_a"]
                if has_a
                else row["mz_b"]
            )
            items[0] = NumericTableWidgetItem(f"{mean_mz:.4f}")
            items[0].setData(Qt.ItemDataRole.UserRole, mean_mz)

            if has_a:
                rel_a = row["int_a"] / self._max_a * 100.0
                items[1] = NumericTableWidgetItem(f"{row['mz_a']:.4f}")
                items[1].setData(Qt.ItemDataRole.UserRole, row["mz_a"])
                items[2] = NumericTableWidgetItem(f"{row['int_a']:.4e}")
                items[2].setData(Qt.ItemDataRole.UserRole, row["int_a"])
                items[3] = NumericTableWidgetItem(f"{rel_a:.1f}")
                items[3].setData(Qt.ItemDataRole.UserRole, rel_a)

            if has_b:
                rel_b = row["int_b"] / self._max_b * 100.0
                items[4] = NumericTableWidgetItem(f"{row['mz_b']:.4f}")
                items[4].setData(Qt.ItemDataRole.UserRole, row["mz_b"])
                items[5] = NumericTableWidgetItem(f"{row['int_b']:.4e}")
                items[5].setData(Qt.ItemDataRole.UserRole, row["int_b"])
                items[6] = NumericTableWidgetItem(f"{rel_b:.1f}")
                items[6].setData(Qt.ItemDataRole.UserRole, rel_b)

            if is_matched:
                items[7] = NumericTableWidgetItem(f"{row['delta_mz']:.4f}")
                items[7].setData(Qt.ItemDataRole.UserRole, abs(row["delta_mz"]))
                items[8] = NumericTableWidgetItem(f"{row['delta_ppm']:.2f}")
                items[8].setData(Qt.ItemDataRole.UserRole, abs(row["delta_ppm"]))

            # Store original row index on every item in this row so that after
            # the user enables sorting, _on_row_clicked can look up the row data
            # from any column-0 item in the visually-reordered table.
            for col_idx, it in enumerate(items):
                it.setData(self._ROW_IDX_ROLE, row_idx)
                it.setBackground(bg)
                it.setTextAlignment(RA)
                self._table.setItem(row_idx, col_idx, it)

        # Enable sorting only after all rows are fully populated
        self._table.setSortingEnabled(True)
        self._table.cellClicked.connect(self._on_row_clicked)
        tbl_l.addWidget(self._table, 1)  # stretch=1 → fills remaining height
        splitter.addWidget(tbl_w)
        splitter.setSizes([680, 540])

        self._draw_plot()

    # ------------------------------------------------------------------
    def _draw_plot(self):
        """Redraw the mirror plot, highlighting selected peaks in firebrick."""
        ax = self._ax
        ax.clear()

        _NORMAL_COLOR = "#1565c0"
        _HL_COLOR = "firebrick"

        for i in range(len(self._mz_a)):
            hl = i in self._highlight_a
            ax.vlines(
                float(self._mz_a[i]),
                0.0,
                float(self._rel_a[i]),
                colors=_HL_COLOR if hl else _NORMAL_COLOR,
                linewidth=2.4 if hl else 1.4,
                alpha=0.95 if hl else 0.85,
            )

        for i in range(len(self._mz_b)):
            hl = i in self._highlight_b
            ax.vlines(
                float(self._mz_b[i]),
                0.0,
                -float(self._rel_b[i]),
                colors=_HL_COLOR if hl else _NORMAL_COLOR,
                linewidth=2.4 if hl else 1.4,
                alpha=0.95 if hl else 0.85,
            )

        ax.axhline(0.0, color="black", linewidth=0.8)

        ticks = np.arange(-100, 101, 25)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(abs(int(t))) for t in ticks])
        ax.set_ylim(-110, 110)
        ax.set_xlabel("m/z", fontsize=11)
        ax.set_ylabel("Relative Intensity (%)", fontsize=11)
        ax.set_title(f"{self.method}  ·  score: {self.similarity:.4f}", fontsize=11)
        ax.grid(True, alpha=0.2)

        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor=_NORMAL_COLOR, label=f"↑ {self.title_a}"),
            Patch(facecolor=_NORMAL_COLOR, label=f"↓ {self.title_b}"),
        ]
        if self._highlight_a or self._highlight_b:
            legend_handles.append(Patch(facecolor=_HL_COLOR, label="selected"))
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_row_clicked(self, row: int, col: int):
        """Highlight the peaks corresponding to the clicked table row."""
        # The table may be sorted; retrieve original row index from any item in
        # this visual row (we stored _ROW_IDX_ROLE on all columns during build).
        item = self._table.item(row, col) or self._table.item(row, 0)
        if item is None:
            return
        orig_idx = item.data(self._ROW_IDX_ROLE)
        if orig_idx is None:
            # fallback: scan all columns
            for c in range(self._table.columnCount()):
                it = self._table.item(row, c)
                if it is not None:
                    orig_idx = it.data(self._ROW_IDX_ROLE)
                    if orig_idx is not None:
                        break
        if orig_idx is None:
            return
        r = self._rows[int(orig_idx)]
        self._highlight_a = {r["idx_a"]} if r["idx_a"] is not None else set()
        self._highlight_b = {r["idx_b"]} if r["idx_b"] is not None else set()
        self._draw_plot()


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


# ─────────────────────────────────────────────────────────────────────────────
# MS1 single-spectrum detail window
# ─────────────────────────────────────────────────────────────────────────────


def _make_vline():
    """Return a thin vertical separator QFrame widget."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setFixedWidth(10)
    return line


class InteractiveMS1SingleChartView(InteractiveMS1ChartView):
    """
    Extends InteractiveMS1ChartView for the single-spectrum detail window.
    Adds relative-m/z mode (stores raw m/z separately), an enriched hover
    tooltip that shows both actual m/z and Δm/z, and Δm/z-formatted top-N
    signal labels when in relative mode.
    """

    def __init__(
        self,
        chart,
        target_mz,
        zoom_min,
        zoom_max,
        full_min,
        full_max,
        raw_mz_array,
        intensity_array,
        relative_mode=False,
        reference_mz=None,
    ):
        raw_mz = np.asarray(raw_mz_array, dtype=float)
        ref = float(reference_mz) if reference_mz is not None else float(target_mz)
        plot_mz = raw_mz - ref if relative_mode else raw_mz.copy()

        super().__init__(
            chart,
            0.0 if relative_mode else float(target_mz),
            zoom_min,
            zoom_max,
            full_min,
            full_max,
            plot_mz,
            np.asarray(intensity_array, dtype=float),
        )
        self.raw_mz_array = raw_mz
        self.relative_mode = relative_mode
        self.reference_mz = ref

    # ------------------------------------------------------------------
    # Hover tooltip
    # ------------------------------------------------------------------
    def _handle_hover_tooltip(self, event):
        """Highlight closest peak; show actual m/z, Δm/z and intensity."""
        plot_area = self.chart().plotArea()
        if not plot_area.contains(event.position()):
            self._hide_tooltip()
            return

        if len(self.mz_array) == 0:
            self._hide_tooltip()
            return

        PIXEL_THRESHOLD = 20
        mx = event.position().x()
        my = event.position().y()

        best_idx = None
        best_dist = float("inf")

        for i, (pmz, intf) in enumerate(zip(self.mz_array, self.intensity_array)):
            pmz = float(pmz)
            intf = float(intf)
            tip = self.chart().mapToPosition(QPointF(pmz, intf))
            base = self.chart().mapToPosition(QPointF(pmz, 0.0))
            tx, ty, bx, by = tip.x(), tip.y(), base.x(), base.y()
            dx, dy = bx - tx, by - ty
            seg_len_sq = dx * dx + dy * dy
            t = (
                max(0.0, min(1.0, ((mx - tx) * dx + (my - ty) * dy) / seg_len_sq))
                if seg_len_sq > 0
                else 0.0
            )
            dist = ((mx - tx - t * dx) ** 2 + (my - ty - t * dy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None and best_dist <= PIXEL_THRESHOLD:
            best_plot_mz = float(self.mz_array[best_idx])
            best_intensity = float(self.intensity_array[best_idx])

            if best_plot_mz != self._hover_mz:
                self._hover_mz = best_plot_mz

                if self._hover_series is not None:
                    self.chart().removeSeries(self._hover_series)
                    self._hover_series = None

                hover_s = QLineSeries()
                hover_pen = QPen(QColor(178, 34, 34))
                hover_pen.setWidth(3)
                hover_s.setPen(hover_pen)
                hover_s.append(best_plot_mz, 0.0)
                hover_s.append(best_plot_mz, best_intensity)
                hover_s.append(best_plot_mz, 0.0)
                self.chart().addSeries(hover_s)
                x_axes = self.chart().axes(Qt.Orientation.Horizontal)
                y_axes = self.chart().axes(Qt.Orientation.Vertical)
                if x_axes and y_axes:
                    hover_s.attachAxis(x_axes[0])
                    hover_s.attachAxis(y_axes[0])
                self._hover_series = hover_s
                self.chart().legend().hide()

                if self.tooltip_label is None:
                    self.tooltip_label = QLabel(self)
                    self.tooltip_label.setStyleSheet("""
                        QLabel {
                            background-color: rgba(255, 255, 255, 220);
                            border: 1px solid #555;
                            border-radius: 3px;
                            padding: 2px 6px;
                            font-size: 11px;
                        }
                    """)
                    self.tooltip_label.setAttribute(
                        Qt.WidgetAttribute.WA_TransparentForMouseEvents
                    )

                # Tooltip text
                if self.relative_mode:
                    raw_mz = float(self.raw_mz_array[best_idx])
                    tooltip_text = (
                        f"Δm/z: {best_plot_mz:+.4f} Da  |  "
                        f"m/z: {raw_mz:.4f}  |  "
                        f"Int: {best_intensity:.4g}"
                    )
                else:
                    tooltip_text = (
                        f"m/z: {best_plot_mz:.4f}  |  Int: {best_intensity:.4g}"
                    )
                self.tooltip_label.setText(tooltip_text)
                self.tooltip_label.adjustSize()

            tip_pos = self.chart().mapToPosition(QPointF(best_plot_mz, best_intensity))
            lx = int(tip_pos.x()) - self.tooltip_label.width() // 2
            ly = int(tip_pos.y()) - self.tooltip_label.height() - 6
            lx = max(0, min(lx, self.width() - self.tooltip_label.width()))
            if ly < 0:
                ly = int(tip_pos.y()) + 6
            self.tooltip_label.move(lx, ly)
            self.tooltip_label.show()
        else:
            self._hide_tooltip()

    # ------------------------------------------------------------------
    # Top-N signal labels
    # ------------------------------------------------------------------
    def update_top_signal_labels(self):
        """Show top-N labels; use Δm/z notation when in relative mode."""
        for label in self.signal_labels:
            label.deleteLater()
        self.signal_labels.clear()

        if len(self.mz_array) == 0 or self.top_n <= 0:
            return

        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
        x_min, x_max = x_axis.min(), x_axis.max()

        visible_mask = (self.mz_array >= x_min) & (self.mz_array <= x_max)
        if not np.any(visible_mask):
            return

        visible_plot_mz = self.mz_array[visible_mask]
        visible_intensities = self.intensity_array[visible_mask]
        visible_raw_mz = self.raw_mz_array[visible_mask]

        n = min(self.top_n, len(visible_intensities))
        top_indices = np.argsort(visible_intensities)[-n:][::-1]

        for idx in top_indices:
            plot_mz = float(visible_plot_mz[idx])
            intensity = float(visible_intensities[idx])

            tip_pos = self.chart().mapToPosition(QPointF(plot_mz, intensity))

            if self.relative_mode:
                label_text = f"Δ{plot_mz:+.4f}"
            else:
                label_text = f"{plot_mz:.4f}"

            label = QLabel(self)
            label.setText(label_text)
            label.setStyleSheet("""
                QLabel {
                    background-color: rgba(46, 134, 171, 128);
                    color: white;
                    border-radius: 3px;
                    padding: 2px 4px;
                    font-weight: bold;
                }
            """)
            label.adjustSize()

            lx = int(tip_pos.x()) - label.width() // 2
            ly = int(tip_pos.y()) - label.height() - 6
            lx = max(0, min(lx, self.width() - label.width()))
            if ly < 0:
                ly = int(tip_pos.y()) + 6

            label.move(lx, ly)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            label.show()
            self.signal_labels.append(label)


class MS1SingleSpectrumWindow(QWidget):
    """
    Detail window for a single MS1 spectrum, opened by clicking a plot title
    in MS1ViewerWindow (the MS1 overview grid).

    Features
    --------
    * Interactive stick spectrum with left-drag pan, right-drag zoom and
      mouse-wheel zoom.
    * Theoretical isotope-pattern overlay (rectangle markers) with a
      configurable ppm tolerance.
    * Relative m/z mode: shows Δm/z = observed − reference_mz on the x-axis.
      The reference defaults to the theoretical ion m/z; the user can edit it.
    * Top-N most-abundant signal labels (showing Δm/z when in relative mode).
    * Euclidean-pixel hover highlighting with tooltip showing m/z, Δm/z and
      intensity.
    """

    def __init__(
        self,
        spectrum_data,
        filename,
        group,
        target_mz,
        compound_name,
        adduct,
        mz_tolerance,
        formula,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.spectrum_data = spectrum_data
        self.filename = filename
        self.group = group
        self.target_mz = float(target_mz)
        self.compound_name = compound_name
        self.adduct = adduct
        self.mz_tolerance = float(mz_tolerance)
        self.formula = formula

        self.raw_mz = np.array(spectrum_data.get("mz", []), dtype=float)
        self.intensity_arr = np.array(spectrum_data.get("intensity", []), dtype=float)

        # State
        self.relative_mode = False
        self.reference_mz = self.target_mz
        self.show_isotope = False

        # Widget references
        self.chart_view = None
        self._chart_container_layout = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        display_name = (
            self.filename.split(".")[0] if "." in self.filename else self.filename
        )
        scan_id = self.spectrum_data.get("scan_id", "")
        filter_str = self.spectrum_data.get("filter_string", "")
        rt_val = self.spectrum_data.get("rt", 0.0)

        self.setWindowTitle(
            f"MS1 Detail – {self.compound_name} ({self.adduct}) | {display_name}"
        )
        self.setGeometry(150, 150, 1400, 850)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Header ────────────────────────────────────────────────────
        header_parts = [
            f"<b>{display_name}</b>",
            f"Group: {self.group}",
            f"RT: {rt_val:.4f} min",
            f"Compound: <b>{self.compound_name}</b>",
            f"Adduct: {self.adduct}",
            f"Target m/z: {self.target_mz:.4f}",
        ]
        if self.formula:
            header_parts.append(f"Formula: {self.formula}")
        if scan_id:
            header_parts.append(f"Scan: {scan_id}")
        if filter_str:
            header_parts.append(f"Filter: {filter_str}")
        header_label = QLabel("  |  ".join(header_parts))
        header_label.setStyleSheet(
            "QLabel { margin: 2px; padding: 5px; font-size: 12px; "
            "background: #f0f4f8; border: 1px solid #c0ccd8; border-radius: 3px; }"
        )
        header_label.setFixedHeight(32)
        root.addWidget(header_label)

        # ── Controls ──────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        # Isotope pattern
        if self.formula:
            self.isotope_chk = QCheckBox("Isotope pattern")
            self.isotope_chk.setToolTip(
                "Overlay theoretical isotope pattern (rectangle markers)"
            )
            self.isotope_chk.toggled.connect(self._on_isotope_toggled)
            ctrl.addWidget(self.isotope_chk)

            ctrl.addWidget(QLabel("Tolerance:"))
            self.iso_tol_spin = QDoubleSpinBox()
            self.iso_tol_spin.setRange(0.1, 100.0)
            self.iso_tol_spin.setValue(5.0)
            self.iso_tol_spin.setSuffix(" ppm")
            self.iso_tol_spin.setDecimals(1)
            self.iso_tol_spin.setSingleStep(0.5)
            self.iso_tol_spin.setFixedWidth(90)
            self.iso_tol_spin.valueChanged.connect(self._on_iso_tol_changed)
            ctrl.addWidget(self.iso_tol_spin)

            ctrl.addWidget(_make_vline())

        # Top-N signals
        ctrl.addWidget(QLabel("Top signals:"))
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(0, 30)
        self.top_n_spin.setValue(5)
        self.top_n_spin.setFixedWidth(55)
        self.top_n_spin.valueChanged.connect(self._on_top_n_changed)
        ctrl.addWidget(self.top_n_spin)

        ctrl.addWidget(_make_vline())

        # Relative m/z mode
        self.rel_chk = QCheckBox("Relative m/z")
        self.rel_chk.setToolTip(
            "Show Δm/z = observed − reference m/z on the x-axis (in Da)"
        )
        self.rel_chk.toggled.connect(self._on_relative_toggled)
        ctrl.addWidget(self.rel_chk)

        ctrl.addWidget(QLabel("Reference m/z:"))
        self.ref_mz_spin = QDoubleSpinBox()
        self.ref_mz_spin.setRange(0.0, 100000.0)
        self.ref_mz_spin.setDecimals(4)
        self.ref_mz_spin.setSingleStep(0.0001)
        self.ref_mz_spin.setFixedWidth(120)
        self.ref_mz_spin.setValue(self.target_mz)
        self.ref_mz_spin.setToolTip(
            "Reference m/z for relative mode (defaults to theoretical ion m/z)"
        )
        self.ref_mz_spin.valueChanged.connect(self._on_ref_mz_changed)
        ctrl.addWidget(self.ref_mz_spin)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # ── Chart container ───────────────────────────────────────────
        chart_container = QWidget()
        self._chart_container_layout = QVBoxLayout(chart_container)
        self._chart_container_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(chart_container, stretch=1)

        # ── Close button ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        # Build the initial chart
        self._rebuild_chart()

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------
    def _on_isotope_toggled(self, checked):
        self.show_isotope = checked
        if checked and self.formula:
            self._add_isotope_overlay()
        else:
            self._remove_isotope_overlay()

    def _on_iso_tol_changed(self):
        if self.show_isotope and self.formula:
            self._add_isotope_overlay()

    def _on_top_n_changed(self, n):
        if self.chart_view is not None:
            self.chart_view.top_n = n
            self.chart_view.update_top_signal_labels()

    def _on_relative_toggled(self, checked):
        self.relative_mode = checked
        if checked:
            self.reference_mz = self.ref_mz_spin.value()
        self._rebuild_chart()

    def _on_ref_mz_changed(self, value):
        self.reference_mz = value
        if self.relative_mode:
            self._rebuild_chart()

    # ------------------------------------------------------------------
    # Chart construction / rebuilding
    # ------------------------------------------------------------------
    def _get_plot_mz(self):
        """Return m/z in plot coordinates (shifted when in relative mode)."""
        if self.relative_mode:
            return self.raw_mz - self.reference_mz
        return self.raw_mz.copy()

    def _rebuild_chart(self):
        """Remove the existing chart view and build a fresh one."""
        if self.chart_view is not None:
            self._chart_container_layout.removeWidget(self.chart_view)
            self.chart_view.deleteLater()
            self.chart_view = None

        if len(self.raw_mz) == 0:
            placeholder = QLabel("No spectrum data available.")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._chart_container_layout.addWidget(placeholder)
            return

        plot_mz = self._get_plot_mz()

        zoom_window = 10.0
        if self.relative_mode:
            zoom_min = -zoom_window
            zoom_max = zoom_window
        else:
            zoom_min = max(float(plot_mz.min()), self.target_mz - zoom_window)
            zoom_max = self.target_mz + zoom_window

        full_min = float(plot_mz.min())
        full_max = float(plot_mz.max())

        chart = self._create_qchart(plot_mz, zoom_min, zoom_max)

        self.chart_view = InteractiveMS1SingleChartView(
            chart,
            target_mz=self.target_mz,
            zoom_min=zoom_min,
            zoom_max=zoom_max,
            full_min=full_min,
            full_max=full_max,
            raw_mz_array=self.raw_mz,
            intensity_array=self.intensity_arr,
            relative_mode=self.relative_mode,
            reference_mz=self.reference_mz,
        )
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.top_n = self.top_n_spin.value()
        self._chart_container_layout.addWidget(self.chart_view)

        self.chart_view.auto_scale_y_axis()
        self.chart_view.update_top_signal_labels()

        if self.show_isotope and self.formula:
            self._add_isotope_overlay()

    def _create_qchart(self, plot_mz, zoom_min, zoom_max):
        """Build a QChart with stick-spectrum series."""
        chart = QChart()
        display_name = (
            self.filename.split(".")[0] if "." in self.filename else self.filename
        )
        chart.setTitle(f"MS1 – {display_name}  |  {self.compound_name} ({self.adduct})")
        chart.legend().hide()

        # Main spectrum series
        series = QLineSeries()
        pen = QPen(QColor("#2E86AB"))
        pen.setWidth(1)
        series.setPen(pen)
        for pmz, intf in zip(plot_mz, self.intensity_arr):
            pmz = float(pmz)
            intf = float(intf)
            series.append(pmz, 0.0)
            series.append(pmz, intf)
            series.append(pmz, 0.0)
        chart.addSeries(series)

        # EIC-window highlight series
        eic_mask = np.abs(self.raw_mz - self.target_mz) <= self.mz_tolerance
        if np.any(eic_mask):
            hl = QLineSeries()
            hl_pen = QPen(QColor("#F18F01"))
            hl_pen.setWidth(3)
            hl.setPen(hl_pen)
            for pmz, intf in zip(plot_mz[eic_mask], self.intensity_arr[eic_mask]):
                pmz = float(pmz)
                intf = float(intf)
                hl.append(pmz, 0.0)
                hl.append(pmz, intf)
                hl.append(pmz, 0.0)
            chart.addSeries(hl)

        # Axes
        x_axis = QValueAxis()
        y_axis = QValueAxis()
        x_axis.setTitleText("Δm/z (Da)" if self.relative_mode else "m/z")
        y_axis.setTitleText("Intensity")
        x_axis.setRange(zoom_min, zoom_max)
        max_int = (
            float(self.intensity_arr.max()) if len(self.intensity_arr) > 0 else 100.0
        )
        y_axis.setRange(0.0, max_int * 1.1)
        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        for s in chart.series():
            s.attachAxis(x_axis)
            s.attachAxis(y_axis)

        return chart

    # ------------------------------------------------------------------
    # Isotope pattern overlay
    # ------------------------------------------------------------------
    def _calculate_isotope_pattern(self):
        """
        Return [(theoretical_mz, relative_abundance), …].
        First entry is the monoisotopic peak (rel. abundance = 1.0).
        Abundances are relative to the monoisotopic peak.
        """
        try:
            composition = parse_molecular_formula(self.formula)
            base = self.target_mz
            isotopes = [(base, 1.0)]

            # M+1  (13C dominant)
            if "C" in composition:
                m1 = composition["C"] * 0.011
                if m1 > 0.01:
                    isotopes.append((base + 1.003355, m1))

            # M+2  (13C×2, 34S, 37Cl, 18O)
            m2 = 0.0
            if "C" in composition:
                c = composition["C"]
                m2 += c * (c - 1) / 2 * (0.011**2)
            if "S" in composition:
                m2 += composition["S"] * 0.0421
            if "Cl" in composition:
                m2 += composition["Cl"] * 0.2424
            if "O" in composition:
                m2 += composition["O"] * 0.00205
            if m2 > 0.01:
                isotopes.append((base + 2.00671, m2))

            # M+3  (estimate)
            m3 = 0.0
            if "C" in composition:
                c = composition["C"]
                m3 += c * (c - 1) * (c - 2) / 6 * (0.011**3)
            if "S" in composition and "C" in composition:
                m3 += composition["S"] * composition["C"] * 0.011 * 0.0421
            if m3 > 0.005:
                isotopes.append((base + 3.010065, m3))

            return isotopes
        except Exception as exc:
            print(f"Isotope pattern calculation error: {exc}")
            return []

    def _add_isotope_overlay(self):
        """Add (or refresh) theoretical isotope rectangle series on the chart."""
        if self.chart_view is None:
            return
        self._remove_isotope_overlay()

        isotope_pattern = self._calculate_isotope_pattern()
        if not isotope_pattern:
            return

        tol_ppm = self.iso_tol_spin.value() if hasattr(self, "iso_tol_spin") else 5.0
        chart = self.chart_view.chart()
        axes = chart.axes()
        if len(axes) < 2:
            return

        # Scale based on observed monoisotopic peak intensity
        mono_mz = isotope_pattern[0][0]
        tol_da_mono = (tol_ppm / 1e6) * mono_mz
        mono_mask = np.abs(self.raw_mz - mono_mz) <= tol_da_mono
        if np.any(mono_mask):
            mono_intensity = float(np.max(self.intensity_arr[mono_mask]))
        else:
            mono_intensity = (
                float(np.max(self.intensity_arr)) * 0.1
                if len(self.intensity_arr) > 0
                else 1.0
            )

        for i, (theo_mz, rel_abund) in enumerate(isotope_pattern):
            theo_intensity = mono_intensity * rel_abund
            tol_da = (tol_ppm / 1e6) * theo_mz

            isotope_mask = np.abs(self.raw_mz - theo_mz) <= tol_da
            found = np.any(isotope_mask)

            # Plot coordinate for center of the rectangle
            plot_center = (
                (theo_mz - self.reference_mz) if self.relative_mode else theo_mz
            )
            rect_half_w = tol_da * 0.5

            # Rectangle spans the theoretical intensity
            left = plot_center - rect_half_w
            right = plot_center + rect_half_w

            iso_s = QLineSeries()
            iso_s.setObjectName(f"isotope_{i}")
            iso_s.append(left, 0.0)
            iso_s.append(left, theo_intensity)
            iso_s.append(right, theo_intensity)
            iso_s.append(right, 0.0)
            iso_s.append(left, 0.0)

            iso_pen = QPen(QColor("#A23B72"))
            iso_pen.setWidth(2 if found else 1)
            iso_pen.setStyle(Qt.PenStyle.SolidLine if found else Qt.PenStyle.DashLine)
            iso_s.setPen(iso_pen)
            iso_s.setOpacity(0.75)

            chart.addSeries(iso_s)
            iso_s.attachAxis(axes[0])
            iso_s.attachAxis(axes[1])

        chart.legend().hide()

    def _remove_isotope_overlay(self):
        """Remove all isotope-pattern series from the chart."""
        if self.chart_view is None:
            return
        chart = self.chart_view.chart()
        to_remove = [
            s
            for s in chart.series()
            if hasattr(s, "objectName") and s.objectName().startswith("isotope_")
        ]
        for s in to_remove:
            chart.removeSeries(s)
