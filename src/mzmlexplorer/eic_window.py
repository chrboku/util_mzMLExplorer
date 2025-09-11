"""
EIC (Extracted Ion Chromatogram) window for displaying chromatographic data
"""
import sys
import pandas as pd
import time
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QCheckBox, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QFormLayout, QMessageBox, QProgressBar,
                             QSplitter, QComboBox, QMenu, QScrollArea, QGridLayout,
                             QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPen, QColor, QPainter, QMouseEvent, QAction
from PyQt6.QtWidgets import QSizePolicy
from .utils import calculate_cosine_similarity, calculate_similarity_statistics
import numpy as np
from typing import Dict, Tuple, Optional
from .utils import calculate_mz_from_formula, format_mz, format_retention_time
from natsort import natsorted


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
        precursor_intensity = self.spectrum_data.get('precursor_intensity', 0)
        intensity_text = f"{precursor_intensity:.1e}" if precursor_intensity > 0 else "N/A"
        
        header_text = (f"<b>File:</b> {self.filename}<br>"
                      f"<b>Group:</b> {self.group}<br>"
                      f"<b>RT:</b> {self.spectrum_data['rt']:.2f} min<br>"
                      f"<b>Precursor m/z:</b> {self.spectrum_data['precursor_mz']:.4f}<br>"
                      f"<b>Precursor Intensity:</b> {intensity_text}")
        
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
        mz_array = np.array(self.spectrum_data['mz'])
        intensity_array = np.array(self.spectrum_data['intensity'])
        
        # Calculate relative abundance (% of base peak)
        max_intensity = np.max(intensity_array) if len(intensity_array) > 0 else 1
        relative_abundance = (intensity_array / max_intensity * 100) if max_intensity > 0 else np.zeros_like(intensity_array)
        
        # Setup table
        num_rows = len(mz_array)
        self.table_widget.setRowCount(num_rows)
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(['m/z', 'Intensity', 'Rel. Abundance (%)'])
        
        # Disable sorting during population to prevent data loss
        self.table_widget.setSortingEnabled(False)
        
        # Set selection behavior
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        
        # Create custom table items that sort numerically
        for i, (mz, intensity, rel_abund) in enumerate(zip(mz_array, intensity_array, relative_abundance)):
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
                intensity_item.setData(Qt.ItemDataRole.DisplayRole, f"{intensity_val:.2e}")
            else:
                intensity_item.setData(Qt.ItemDataRole.DisplayRole, f"{intensity_val:.2f}")
            intensity_item.setData(Qt.ItemDataRole.UserRole, intensity_val)  # Store for selection
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
            mz_item = selected_items[0] if selected_items[0].column() == 0 else selected_items[1]
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
        mz_array = self.spectrum_data['mz']
        intensity_array = self.spectrum_data['intensity']
        
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
            is_highlighted = (self.selected_mz is not None and 
                            abs(mz - self.selected_mz) < tolerance)
            
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
        x_axis = chart.axes(Qt.Orientation.Horizontal)[0] if chart.axes(Qt.Orientation.Horizontal) else None
        y_axis = chart.axes(Qt.Orientation.Vertical)[0] if chart.axes(Qt.Orientation.Vertical) else None
        
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
        precursor_intensity = self.spectrum_data.get('precursor_intensity', 0)
        intensity_text = f"{precursor_intensity:.1e}" if precursor_intensity > 0 else "N/A"
        
        chart.setTitle(f"MSMS Spectrum\n"
                      f"RT: {self.spectrum_data['rt']:.2f} min, "
                      f"Precursor: {self.spectrum_data['precursor_mz']:.4f}, "
                      f"Intensity: {intensity_text}")
        
        # Create series for the spectrum
        series = QLineSeries()
        
        # Add spectrum data as vertical lines (stick spectrum)
        mz_array = self.spectrum_data['mz']
        intensity_array = self.spectrum_data['intensity']
        
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
            self.zoom_anchor_y = self.interaction_start_y_range[1] - rel_y * y_range  # Y is inverted
            
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events"""
        # Check if we're tracking a potential right-click for context menu
        if (self.right_click_pending and self.mouse_press_pos is not None and 
            event.buttons() & Qt.MouseButton.RightButton):
            
            current_pos = event.position().toPoint()
            distance = ((current_pos.x() - self.mouse_press_pos.x()) ** 2 + 
                       (current_pos.y() - self.mouse_press_pos.y()) ** 2) ** 0.5
            
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
        closest_distance = float('inf')
        closest_sample_name = None
        
        # Check all series for proximity
        for series in self.chart().series():
            if isinstance(series, QLineSeries):
                # Get series data from cache or create it
                series_id = id(series)
                if series_id not in self.series_data_cache:
                    self._cache_series_data(series)
                
                series_data = self.series_data_cache.get(series_id, {})
                points = series_data.get('points', [])
                sample_name = series_data.get('name', 'Unknown')
                
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
        self.series_data_cache[series_id] = {
            'points': points,
            'name': sample_name
        }
    
    def _find_closest_distance_to_series(self, mouse_x, mouse_y, points):
        """Find the closest distance from mouse to the series line"""
        if len(points) < 2:
            return float('inf')
        
        # Get axes to normalize distances
        x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0] if self.chart().axes(Qt.Orientation.Horizontal) else None
        y_axis = self.chart().axes(Qt.Orientation.Vertical)[0] if self.chart().axes(Qt.Orientation.Vertical) else None
        
        if not x_axis or not y_axis:
            return float('inf')
        
        # Get axis ranges for normalization
        x_range = x_axis.max() - x_axis.min()
        y_range = y_axis.max() - y_axis.min()
        
        # Avoid division by zero
        if x_range <= 0 or y_range <= 0:
            return float('inf')
        
        # Normalize mouse coordinates
        norm_mouse_x = (mouse_x - x_axis.min()) / x_range
        norm_mouse_y = (mouse_y - y_axis.min()) / y_range
        
        min_distance = float('inf')
        
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
                norm_mouse_x, norm_mouse_y,
                norm_x1, norm_y1, norm_x2, norm_y2
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
                distance = ((release_pos.x() - self.mouse_press_pos.x()) ** 2 + 
                           (release_pos.y() - self.mouse_press_pos.y()) ** 2) ** 0.5
                
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
    
    def __init__(self, file_manager, target_mz, mz_tolerance, rt_start, rt_end, eic_method="Sum of all signals", adduct=None):
        super().__init__()
        self.file_manager = file_manager
        self.target_mz = target_mz
        self.mz_tolerance = mz_tolerance
        self.rt_start = rt_start
        self.rt_end = rt_end
        self.eic_method = eic_method
        self.adduct = adduct
        
        # Determine polarity from adduct
        self.polarity = self._determine_polarity(adduct)
    
    def _determine_polarity(self, adduct):
        """Determine polarity from adduct string"""
        if not adduct:
            return None
        
        # Check for explicit polarity markers
        if '+' in adduct:
            return '+'
        elif '-' in adduct:
            return '-'
        else:
            return None
    
    def run(self):
        try:
            files_data = self.file_manager.get_files_data()
            total_files = len(files_data)
            eic_data = {}
            
            for i, (_, row) in enumerate(files_data.iterrows()):
                filepath = row['Filepath']
                
                # Extract EIC with polarity consideration
                rt, intensity = self.file_manager.extract_eic(
                    filepath, self.target_mz, self.mz_tolerance, 
                    self.rt_start, self.rt_end, self.eic_method, self.polarity
                )
                
                eic_data[filepath] = {
                    'rt': rt,
                    'intensity': intensity,
                    'metadata': row.to_dict()
                }
                
                # Update progress
                progress_value = int((i + 1) / total_files * 100)
                self.progress.emit(progress_value)
            
            self.finished.emit(eic_data)
            
        except Exception as e:
            self.error.emit(str(e))


class EICWindow(QWidget):
    """Window for displaying extracted ion chromatograms"""
    
    def __init__(self, compound_data: dict, adduct: str, file_manager, mz_value=None, polarity=None, defaults=None, parent=None):
        super().__init__(parent)
        
        # Configure as independent window
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint | 
                           Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        self.compound_data = compound_data
        self.adduct = adduct
        self.file_manager = file_manager
        self.eic_data = {}
        self.group_shifts = {}
        
        # Store defaults (use application defaults if none provided)
        self.defaults = defaults if defaults is not None else {
            'mz_tolerance_ppm': 5.0,
            'separate_groups': True,
            'rt_shift_min': 1.0,
            'crop_rt_window': False,
            'normalize_samples': False
        }
        
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
                    compound_data['Name'], 
                    adduct
                )
                
                if self.target_mz is None:
                    raise ValueError("Could not calculate m/z value")
                    
                self.polarity = None  # Polarity not available in fallback mode
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to calculate m/z: {str(e)}")
                self.target_mz = 0.0
                self.polarity = None
        
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
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel with the chart"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Chart
        self.chart_view = self.create_chart()
        layout.addWidget(self.chart_view)
        
        return panel
    
    def create_compound_info_group(self) -> QGroupBox:
        """Create the compound information group"""
        group = QGroupBox("Compound Information")
        layout = QVBoxLayout(group)
        
        # Determine compound info display
        formula_info = ""
        if 'ChemicalFormula' in self.compound_data and self.compound_data['ChemicalFormula']:
            formula_info = f"<b>Formula:</b> {self.compound_data['ChemicalFormula']}<br>"
        elif 'Mass' in self.compound_data and self.compound_data['Mass']:
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
        self.mz_tolerance_ppm_spin.setValue(self.defaults['mz_tolerance_ppm'])  # Use default
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
        
        # Group separation
        self.separate_groups_cb = QCheckBox("Separate by groups")
        self.separate_groups_cb.setChecked(self.defaults['separate_groups'])  # Use default
        self.separate_groups_cb.stateChanged.connect(self.update_plot)
        layout.addRow(self.separate_groups_cb)
        
        # RT shift for group separation (more flexible range)
        self.rt_shift_spin = QDoubleSpinBox()
        self.rt_shift_spin.setRange(0.0, 60.0)  # Allow up to 60 minutes
        self.rt_shift_spin.setValue(self.defaults['rt_shift_min'])  # Use default
        self.rt_shift_spin.setSuffix(" min")
        self.rt_shift_spin.setDecimals(1)
        self.rt_shift_spin.setEnabled(True)  # Always enabled
        self.rt_shift_spin.valueChanged.connect(self.update_plot)
        layout.addRow("Group RT Shift:", self.rt_shift_spin)
        
        # RT cropping option
        self.crop_rt_cb = QCheckBox("Crop to RT Window")
        self.crop_rt_cb.setChecked(self.defaults['crop_rt_window'])  # Use default
        self.crop_rt_cb.stateChanged.connect(self.update_plot)
        layout.addRow(self.crop_rt_cb)
        
        # Normalization option
        self.normalize_cb = QCheckBox("Normalize to Max per Sample")
        self.normalize_cb.setChecked(self.defaults['normalize_samples'])  # Use default
        self.normalize_cb.stateChanged.connect(self.update_plot)
        layout.addRow(self.normalize_cb)
        
        return group
    
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
        self.chart.setTitle("Extracted Ion Chromatogram")
        
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
            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')
            
            for series in self.chart.series():
                if series.count() > 0:
                    for i in range(series.count()):
                        point = series.at(i)
                        min_x = min(min_x, point.x())
                        max_x = max(max_x, point.x())
                        min_y = min(min_y, point.y())
                        max_y = max(max_y, point.y())
            
            if min_x != float('inf'):
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
            self.adduct  # Pass adduct for polarity determination
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
        
        # Update plot
        self.update_plot()
    
    def on_extraction_error(self, error_message: str):
        """Handle EIC extraction error"""
        self.progress_bar.setVisible(False)
        self.extract_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"EIC extraction failed: {error_message}")
    
    def calculate_group_shifts(self):
        """Calculate RT shifts for group separation"""
        if not self.eic_data:
            return
        
        # Get unique groups
        groups = set()
        for data in self.eic_data.values():
            if 'group' in data['metadata']:
                groups.add(data['metadata']['group'])
        
        # Sort groups and assign shifts
        sorted_groups = sorted(groups)
        shift_amount = self.rt_shift_spin.value()
        
        self.group_shifts = {}
        for i, group in enumerate(sorted_groups):
            self.group_shifts[group] = i * shift_amount
    
    def update_plot(self):
        """Update the EIC plot"""
        if not self.eic_data:
            return
        
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
        rt_start = self.compound_data['RT_start_min'] if crop_rt else None
        rt_end = self.compound_data['RT_end_min'] if crop_rt else None
        
        # Organize data by groups first
        groups_data = {}
        for filepath, data in self.eic_data.items():
            rt = data['rt']
            intensity = data['intensity']
            metadata = data['metadata']
            
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
            
            group = metadata.get('group', 'Unknown')
            if group not in groups_data:
                groups_data[group] = []
            
            # Apply group shift only if separation is enabled
            rt_plot = rt.copy()
            if separate_groups:
                shift = self.group_shifts.get(group, 0.0)
                rt_plot = rt + shift
            
            groups_data[group].append({
                'rt': rt_plot,
                'intensity': intensity,
                'metadata': metadata,
                'filepath': filepath
            })
        
        # Create separate series for each file, but group them for legend display
        for group_name, group_files in groups_data.items():
            first_file_in_group = True
            
            # Get group color and make it transparent
            group_color = self.file_manager.get_group_color(group_name)
            
            for file_data in group_files:
                rt = file_data['rt']
                intensity = file_data['intensity']
                
                # Create individual series for each file
                series = QLineSeries()
                
                # Store the sample filename for hover tooltips
                filepath = file_data['filepath']
                filename = file_data['metadata'].get('filename', 
                          filepath.split('\\')[-1] if '\\' in filepath else 
                          filepath.split('/')[-1] if '/' in filepath else filepath)
                
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
                
                # Apply group color with transparency
                if group_color:
                    color = QColor(group_color)
                    color.setAlpha(180)  # Make lines semi-transparent (0-255, 180 = ~70% opacity)
                    pen = QPen(color)
                    pen.setWidth(2)
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
        legend.setAlignment(Qt.AlignmentFlag.AlignTop)
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
            self.y_axis.setRange(-0.05, 1.05)  # Slight padding for better visualization
        else:
            self.y_axis.setTitleText("Intensity")
            # For non-normalized data, calculate range from actual data
            self._set_y_axis_from_data()
        
        # Automatically reset view to show all data after any changes
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(50, self.reset_view)  # Small delay to ensure chart is fully updated
        
        # Update series cache for hover detection
        if hasattr(self.chart_view, 'update_series_cache'):
            self.chart_view.update_series_cache()
    
    def _add_reference_lines(self, groups_data, separate_groups):
        """Add reference lines to the chart"""
        # Get the current axis ranges to draw lines across the full chart
        x_axis = self.chart.axes(Qt.Orientation.Horizontal)[0]
        y_axis = self.chart.axes(Qt.Orientation.Vertical)[0]
        
        # Calculate overall RT range for reference lines (using the final shifted values)
        min_rt, max_rt = float('inf'), float('-inf')
        for group_files in groups_data.values():
            for file_data in group_files:
                rt_values = file_data['rt']
                if len(rt_values) > 0:
                    min_rt = min(min_rt, np.min(rt_values))
                    max_rt = max(max_rt, np.max(rt_values))
        
        if min_rt == float('inf'):
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
        compound_rt = self.compound_data.get('RT_min', 0.0)
        
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
        min_y, max_y = float('inf'), float('-inf')
        
        for series in self.chart.series():
            if hasattr(series, 'pointsVector'):
                points = series.pointsVector()
                if points:
                    y_values = [p.y() for p in points if not np.isnan(p.y())]
                    if y_values:
                        min_y = min(min_y, min(y_values))
                        max_y = max(max_y, max(y_values))
        
        # Set Y range with padding if we found valid data
        if min_y != float('inf') and max_y != float('-inf'):
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
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        for series in self.chart.series():
            if hasattr(series, 'pointsVector'):
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
        if min_x != float('inf') and max_x != float('-inf'):
            x_range = max_x - min_x
            padding = x_range * 0.05
            x_min, x_max = min_x - padding, max_x + padding
            self.x_axis.setRange(x_min, x_max)
            
            # Store original range for reset functionality (only if not set or forced)
            if self.original_x_range is None or force_x_auto_zoom:
                self.original_x_range = (x_min, x_max)
        
        # Set Y ranges with some padding (always update if forced or no original range)
        if min_y != float('inf') and max_y != float('-inf'):
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
        
        # Add MSMS viewing options
        msms_3s_action = QAction("View MSMS (±3 seconds)", self)
        msms_3s_action.triggered.connect(lambda: self.view_msms_spectra(rt_value, 3.0 / 60.))
        context_menu.addAction(msms_3s_action)
        
        msms_6s_action = QAction("View MSMS (±6 seconds)", self)
        msms_6s_action.triggered.connect(lambda: self.view_msms_spectra(rt_value, 6.0 / 60.))
        context_menu.addAction(msms_6s_action)
        
        msms_9s_action = QAction("View MSMS (±9 seconds)", self)
        msms_9s_action.triggered.connect(lambda: self.view_msms_spectra(rt_value, 9.0 / 60.))
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
                    f"in RT window {rt_center:.2f} ± {rt_window:.1f} min"
                )
                return
            
            # Open MSMS viewer window
            msms_viewer = MSMSViewerWindow(
                msms_spectra, 
                self.target_mz, 
                rt_center, 
                rt_window,
                self.compound_data['Name'],
                self.adduct,
                self
            )
            msms_viewer.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to view MSMS spectra: {str(e)}")
    
    def find_msms_spectra(self, rt_start: float, rt_end: float):
        """Find MSMS spectra within RT window for the target m/z and polarity"""
        msms_spectra = {}  # filepath -> list of spectra
        
        # Define precursor tolerance (in Da)
        precursor_tolerance = 0.01  # 10 mDa tolerance for precursor matching
        
        files_data = self.file_manager.get_files_data()
        
        for _, row in files_data.iterrows():
            filepath = row['Filepath']
            filename = row['filename']
            
            try:
                file_msms = []
                
                # Check if we have cached data (memory mode)
                if self.file_manager.keep_in_memory and filepath in self.file_manager.cached_data:
                    cached_file_data = self.file_manager.cached_data[filepath]
                    
                    # Handle both old format (list) and new format (dict with ms1/ms2)
                    if isinstance(cached_file_data, dict) and 'ms2' in cached_file_data:
                        ms2_spectra = cached_file_data['ms2']
                        
                        for spectrum_data in ms2_spectra:
                            spectrum_rt = spectrum_data['scan_time']
                            precursor_mz = spectrum_data.get('precursor_mz')
                            spectrum_polarity = spectrum_data.get('polarity')

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
                            if not ( (self.polarity.lower() in ["+", "positive", "pos"] and spectrum_polarity.lower() in ["+", "positive", "pos"]) or (self.polarity.lower() in ["-", "negative", "neg"] and spectrum_polarity.lower() in ["-", "negative", "neg"]) ):
                                continue
                            
                            # Extract spectrum data
                            mz_array = spectrum_data['mz']
                            intensity_array = spectrum_data['intensity']

                            if len(mz_array) > 0:
                                msms_spectrum = {
                                    'rt': spectrum_rt,
                                    'precursor_mz': precursor_mz,
                                    'precursor_intensity': spectrum_data.get('precursor_intensity', 0),
                                    'mz': mz_array,
                                    'intensity': intensity_array,
                                    'scan_id': spectrum_data.get('scan_id', f"RT_{spectrum_rt:.2f}"),
                                    'polarity': spectrum_polarity
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
                                precursor_mz = spectrum.selected_precursors[0]['mz'] if spectrum.selected_precursors else None
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
                                        precursor_intensity = precursor_info.get('intensity', 0)
                                        if precursor_intensity is None:
                                            precursor_intensity = 0
                                except:
                                    precursor_intensity = 0
                                
                                if len(mz_array) > 0:
                                    spectrum_data = {
                                        'rt': spectrum_rt,
                                        'precursor_mz': precursor_mz,
                                        'precursor_intensity': precursor_intensity,
                                        'mz': mz_array,
                                        'intensity': intensity_array,
                                        'scan_id': spectrum.ID,
                                        'polarity': spectrum_polarity
                                    }
                                    file_msms.append(spectrum_data)
                                    
                            except Exception as e:
                                print(f"Error processing spectrum in {filename}: {e}")
                                continue
                
                if file_msms:
                    # Sort by RT
                    file_msms.sort(key=lambda x: x['rt'])
                    msms_spectra[filepath] = {
                        'filename': filename,
                        'group': row.get('group', 'Unknown'),
                        'spectra': file_msms,
                        'metadata': row.to_dict()
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
            
            x_range = self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
            y_range = self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
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
        original_x_range = self.interaction_start_x_range[1] - self.interaction_start_x_range[0]
        original_y_range = self.interaction_start_y_range[1] - self.interaction_start_y_range[0]
        
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
                popup = MSMSPopupWindow(self.spectrum_data, self.filename, self.group, self)
                popup.show()
        super().mouseDoubleClickEvent(event)


class MSMSViewerWindow(QWidget):
    """Window for displaying MSMS spectra in a grid layout"""
    
    def __init__(self, msms_spectra, target_mz, rt_center, rt_window, compound_name, adduct, parent=None):
        super().__init__(parent)
        
        # Configure as independent window
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint | 
                           Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)
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
        self.setGeometry(100, 100, 1800, 1000)  # Increased width and height for similarity panels
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Header with information - compact layout
        total_spectra = sum(len(data['spectra']) for data in self.msms_spectra.values())
        header_label = QLabel(
            f"<b>{self.compound_name} ({self.adduct})</b> | "
            f"m/z: {self.target_mz:.4f} | "
            f"RT: {self.rt_center:.2f} ± {self.rt_window:.1f} min | "
            f"Files: {len(self.msms_spectra)} | "
            f"Spectra: {total_spectra}"
        )
        header_label.setStyleSheet("QLabel { margin: 2px; padding: 5px; font-size: 12px; }")
        header_label.setFixedHeight(30)
        header_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
            filename = file_data['filename']
            group = file_data.get('group', 'Unknown')
            spectra = file_data['spectra']
            
            # File header with filename, group, and similarity statistics
            similarity_info = ""
            if filename in self.intra_file_similarities:
                stats = self.intra_file_similarities[filename]
                similarity_info = (f" | Cosine Similarities: "
                                 f"Min:{stats['min']:.3f} "
                                 f"10%:{stats['percentile_10']:.3f} "
                                 f"Med:{stats['median']:.3f} "
                                 f"90%:{stats['percentile_90']:.3f} "
                                 f"Max:{stats['max']:.3f}")
            
            display_name = filename.split('.')[0] if '.' in filename else filename
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
        self.global_mz_min = float('inf')
        self.global_mz_max = float('-inf')
        
        processed_files = []
        
        for filepath, file_data in self.msms_spectra.items():
            # Sort spectra by precursor intensity (descending)
            sorted_spectra = sorted(file_data['spectra'], 
                                  key=lambda x: x.get('precursor_intensity', 0), 
                                  reverse=True)
            
            # Update global m/z range
            for spectrum in sorted_spectra:
                mz_array = spectrum['mz']
                if len(mz_array) > 0:
                    self.global_mz_min = min(self.global_mz_min, np.min(mz_array))
                    self.global_mz_max = max(self.global_mz_max, np.max(mz_array))
            
            # Create processed file data
            processed_file_data = file_data.copy()
            processed_file_data['spectra'] = sorted_spectra
            processed_files.append((filepath, processed_file_data))
        
        # Sort files by filename for consistent ordering
        self.processed_data = sorted(processed_files, key=lambda x: x[1]['filename'])
        
        # Add padding to global m/z range
        if self.global_mz_min != float('inf') and self.global_mz_max != float('-inf'):
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
            filename = file_data['filename']
            spectra = file_data['spectra']
            
            similarities = []
            if len(spectra) > 1:
                for i in range(len(spectra)):
                    for j in range(i + 1, len(spectra)):
                        sim = calculate_cosine_similarity(spectra[i], spectra[j])
                        similarities.append(sim)
            
            self.intra_file_similarities[filename] = calculate_similarity_statistics(similarities)
        
        # Calculate inter-file similarities (between all pairs of files)
        files_list = list(self.processed_data)
        for i in range(len(files_list)):
            for j in range(i + 1, len(files_list)):
                file1_path, file1_data = files_list[i]
                file2_path, file2_data = files_list[j]
                
                filename1 = file1_data['filename']
                filename2 = file2_data['filename']
                spectra1 = file1_data['spectra']
                spectra2 = file2_data['spectra']
                
                similarities = []
                for spec1 in spectra1:
                    for spec2 in spectra2:
                        sim = calculate_cosine_similarity(spec1, spec2)
                        similarities.append(sim)
                
                key = (filename1, filename2)
                self.inter_file_similarities[key] = similarities  # Store all similarities, not just stats
    
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
        files = [data[1]['filename'] for data in self.processed_data]
        if len(files) > 1:
            inter_table = QTableWidget(len(files), len(files))
            inter_table.setHorizontalHeaderLabels([f.split('.')[0] for f in files])  # Show filename without extension
            inter_table.setVerticalHeaderLabels([f.split('.')[0] for f in files])
            inter_table.setFixedHeight(min(150, 30 + len(files) * 35))
            inter_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            
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
                            median_sim = stats['median']
                            text = f"{median_sim:.3f}"
                            item = QTableWidgetItem(text)
                            # Use lighter colors for diagonal to distinguish from inter-file
                            if median_sim >= 0.8:
                                item.setBackground(QColor(76, 175, 80, 100))  # Light Green
                            elif median_sim >= 0.6:
                                item.setBackground(QColor(255, 193, 7, 100))  # Light Amber
                            elif median_sim >= 0.4:
                                item.setBackground(QColor(255, 152, 0, 100))  # Light Orange
                            else:
                                item.setBackground(QColor(244, 67, 54, 100))  # Light Red
                        else:
                            item = QTableWidgetItem("N/A")
                            item.setBackground(QColor(240, 240, 240))  # Light gray background
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
                            median_sim = stats['median']
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
                            item.setBackground(QColor(255, 152, 0, 200))  # Strong Orange
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
            no_comparison_label = QLabel("<i>At least 2 files needed for inter-file comparison</i>")
            layout.addWidget(no_comparison_label)
        
        return overview_widget
    

    
    def create_msms_chart(self, spectrum_data, filename, group):
        """Create a chart widget for a single MSMS spectrum"""
        # Create chart
        chart = QChart()
        
        # Get precursor intensity for display
        precursor_intensity = spectrum_data.get('precursor_intensity', 0)
        intensity_text = f"{precursor_intensity:.1e}" if precursor_intensity > 0 else "N/A"
        
        chart.setTitle(f"RT: {spectrum_data['rt']:.2f} min\n"
                      f"Precursor: {spectrum_data['precursor_mz']:.4f}\n"
                      f"Intensity: {intensity_text}")
        
        # Create series for the spectrum
        series = QLineSeries()
        
        # Add spectrum data as vertical lines (stick spectrum)
        mz_array = spectrum_data['mz']
        intensity_array = spectrum_data['intensity']
        
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
        chart_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Store spectrum data for popup display
        chart_view.spectrum_data = spectrum_data
        chart_view.filename = filename
        chart_view.group = group
        
        # Hide legend since we only have one series
        chart.legend().setVisible(False)
        
        return chart_view

    def closeEvent(self, event):
        """Clean up when closing the window"""
        if hasattr(self, 'extraction_worker') and self.extraction_worker.isRunning():
            self.extraction_worker.quit()
            self.extraction_worker.wait()
        event.accept()
