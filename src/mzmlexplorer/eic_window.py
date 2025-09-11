"""
EIC (Extracted Ion Chromatogram) window for displaying chromatographic data
"""
import sys
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QCheckBox, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QFormLayout, QMessageBox, QProgressBar,
                             QSplitter, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPen, QColor, QPainter, QMouseEvent
import numpy as np
from typing import Dict, Tuple, Optional
from .utils import calculate_mz_from_formula, format_mz, format_retention_time


class InteractiveChartView(QChartView):
    """Custom chart view with interactive mouse controls"""
    
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
            # Right click: start zooming
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
        if self.is_panning:
            self._handle_panning(event)
        elif self.is_zooming:
            self._handle_zooming(event)
        
        self.last_mouse_pos = event.position()
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
        elif event.button() == Qt.MouseButton.RightButton:
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
    
    def closeEvent(self, event):
        """Clean up when closing the window"""
        if hasattr(self, 'extraction_worker') and self.extraction_worker.isRunning():
            self.extraction_worker.quit()
            self.extraction_worker.wait()
        event.accept()
