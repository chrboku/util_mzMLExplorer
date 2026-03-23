"""
MS1 viewer windows: MS1ViewerWindow, InteractiveMS1ChartView,
InteractiveMS1SingleChartView, MS1SingleSpectrumWindow.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QScrollArea,
    QGridLayout,
    QFrame,
)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPen, QColor, QPainter, QMouseEvent
from .window_shared import ClickableLabel
from .utils import parse_molecular_formula


def _make_vline():
    """Return a thin vertical separator QFrame widget."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setFixedWidth(10)
    return line


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
