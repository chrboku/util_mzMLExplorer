"""
Multi-adduct EIC window for displaying multiple adduct chromatograms
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
    QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QMargins
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPen, QColor, QPainter, QMouseEvent, QAction, QBrush
from PyQt6.QtWidgets import QSizePolicy
from .utils import calculate_cosine_similarity, calculate_similarity_statistics
import numpy as np
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
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


class ClickableLabel(QLabel):
    """Custom QLabel that emits a clicked signal when left-clicked"""

    clicked = pyqtSignal()

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class InteractiveEICWidget(QWidget):
    """Interactive EIC plot widget for a single adduct"""

    def __init__(
        self,
        compound,
        adduct,
        file_manager,
        mz_value=None,
        polarity=None,
        defaults=None,
        parent=None,
    ):
        super().__init__(parent)
        self.compound = compound
        self.adduct = adduct
        self.file_manager = file_manager
        self.mz_value = mz_value
        self.polarity = polarity
        self.defaults = defaults or {}

        # Set up the plot
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        """Setup the UI for this EIC widget"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # Header with adduct information (clickable)
        compound_name = self.compound.get("Name", "Unknown")
        if self.mz_value is not None:
            header_text = f"{self.adduct} (m/z: {self.mz_value:.4f})"
        else:
            header_text = f"{self.adduct} (m/z: not calculated)"

        self.header_label = ClickableLabel(header_text)
        self.header_label.setStyleSheet("""
            QLabel { 
                background-color: #f0f0f0; 
                padding: 3px; 
                margin: 1px;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-size: 9px;
                font-weight: bold;
            }
            QLabel:hover {
                background-color: #e0e0e0;
                border: 1px solid #999;
            }
        """)
        self.header_label.setMaximumHeight(25)
        self.header_label.setToolTip("Click to open individual EIC viewer")

        # Connect click signal to open individual EIC viewer
        self.header_label.clicked.connect(self._open_individual_eic_viewer)

        layout.addWidget(self.header_label)

        # Create matplotlib figure with smaller size for matrix layout
        self.figure = Figure(figsize=(4, 2.5), dpi=80)
        self.figure.patch.set_facecolor("white")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)

        # Add navigation toolbar for interactivity
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setMaximumHeight(25)
        self.toolbar.setStyleSheet("""
            QToolBar { 
                border: none; 
                background-color: #f8f8f8;
                spacing: 2px;
            }
            QToolBar QToolButton { 
                border: 1px solid #ccc;
                border-radius: 2px;
                padding: 2px;
                margin: 1px;
                background-color: white;
            }
            QToolBar QToolButton:hover {
                background-color: #e8e8e8;
            }
        """)

        # Only show essential navigation tools
        # Hide some actions to save space
        actions_to_hide = ["Configure subplots", "Save", "Forward", "Back"]
        for action in self.toolbar.actions():
            if action.text() in actions_to_hide:
                action.setVisible(False)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Set size constraints for matrix layout
        self.setMinimumSize(300, 250)  # Increased slightly for toolbar
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def load_data(self):
        """Load and plot EIC data"""
        if self.file_manager.get_files_data().empty:
            self.plot_empty("No files loaded")
            return

        if self.mz_value is None:
            self.plot_empty("No m/z value")
            return

        try:
            # Get EIC data for this adduct
            eic_data = self._extract_eic_data()
            self.plot_eic(eic_data)
        except Exception as e:
            print(f"Error loading EIC data for {self.adduct}: {str(e)}")
            import traceback

            traceback.print_exc()
            self.plot_empty(f"Error: {str(e)}")

    def _extract_eic_data(self):
        """Extract EIC data for this adduct"""
        # Get defaults - convert ppm to Da, but make it more generous
        mz_tolerance_ppm = self.defaults.get("mz_tolerance_ppm", 5.0)
        mz_tolerance_da = (self.mz_value * mz_tolerance_ppm) / 1e6

        # Make tolerance more generous for testing
        mz_tolerance_da = max(mz_tolerance_da, 0.01)  # At least 0.01 Da

        calculation_method = self.defaults.get(
            "calculation_method", "Sum of all signals"
        )

        files_data = self.file_manager.get_files_data()
        eic_results = {}

        for idx, file_row in files_data.iterrows():
            filename = file_row["filename"]
            file_path = file_row["Filepath"]  # Use capital F as in the file_manager

            try:
                # Use the file_manager's extract_eic method
                rt_values, intensity_values = self.file_manager.extract_eic(
                    filepath=file_path,
                    target_mz=self.mz_value,
                    mz_tolerance=mz_tolerance_da,
                    rt_start=None,
                    rt_end=None,
                    calculation_method=calculation_method,
                    polarity=self.polarity,
                )

                if len(rt_values) > 0 and len(intensity_values) > 0:
                    # Check if we have any non-zero intensities
                    max_intensity = (
                        intensity_values.max() if len(intensity_values) > 0 else 0
                    )
                    non_zero_points = np.sum(intensity_values > 0)

                    # Store the data with metadata including group info
                    eic_results[filename] = {
                        "rt": rt_values,
                        "intensity": intensity_values,
                        "metadata": file_row.to_dict(),  # Include all file metadata
                    }

            except Exception as e:
                print(f"ERROR processing file {filename}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

        return eic_results

    def plot_eic(self, eic_data):
        """Plot EIC data with auto-zoom to compound RT range and intelligent y-scaling"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not eic_data:
            ax.text(
                0.5,
                0.5,
                "No data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )
        else:
            # Plot each file's EIC with group-based colors
            plots_made = 0
            all_rt_values = []

            # Organize data by groups for consistent coloring
            groups_data = {}
            for filename, data in eic_data.items():
                metadata = data.get("metadata", {})
                group = metadata.get("group", "Unknown")
                if group not in groups_data:
                    groups_data[group] = []
                groups_data[group].append((filename, data))

            # Plot by groups
            for group_name, group_files in groups_data.items():
                # Get group color
                group_color = self.file_manager.get_group_color(group_name)

                if group_color:
                    # Convert hex color to RGB tuple for matplotlib
                    color_obj = QColor(group_color)
                    color_rgb = (
                        color_obj.red() / 255.0,
                        color_obj.green() / 255.0,
                        color_obj.blue() / 255.0,
                        0.7,
                    )  # Add alpha for transparency
                else:
                    # Fallback to a default color if no group color is defined
                    color_rgb = (0.5, 0.5, 0.5, 0.7)

                for filename, data in group_files:
                    if len(data["rt"]) > 0 and len(data["intensity"]) > 0:
                        all_rt_values.extend(data["rt"])

                        ax.plot(
                            data["rt"],
                            data["intensity"],
                            label=filename,
                            color=color_rgb,
                            linewidth=1,
                        )
                        plots_made += 1

            if plots_made == 0:
                ax.text(
                    0.5,
                    0.5,
                    "Data extracted but no intensities > 0",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            else:
                ax.set_xlabel("Retention Time (min)", fontsize=8)
                ax.set_ylabel("Intensity", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)

                # Force y-axis to use scientific notation
                ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

                # Only show legend if there are multiple files and space allows
                if len(eic_data) > 1 and len(eic_data) <= 4:
                    ax.legend(fontsize=6, loc="upper right")

                # Auto-zoom to compound RT range and set intelligent y-limits
                self._set_auto_zoom_with_y_scaling(ax, eic_data, all_rt_values)

        # Tight layout with small margins
        self.figure.tight_layout(pad=0.5)
        self.canvas.draw()

    def _set_auto_zoom_with_y_scaling(self, ax, eic_data, all_rt_values):
        """Set automatic zoom to compound RT range with intelligent y-axis scaling"""
        # Get compound RT range from compound data
        rt_start = self.compound.get("RT_start_min")
        rt_end = self.compound.get("RT_end_min")
        rt_center = self.compound.get("RT_min")

        # Check if using default full range (0-100 min) - treat as no RT info
        is_full_range = rt_start == 0.0 and rt_end == 100.0

        # Determine RT window for zoom
        if rt_start is not None and rt_end is not None and not is_full_range:
            # Use defined RT window
            rt_margin = (rt_end - rt_start) * 0.1  # Add 10% margin
            zoom_start = max(0, rt_start - rt_margin)
            zoom_end = rt_end + rt_margin
            rt_window_start = rt_start
            rt_window_end = rt_end

        elif rt_center is not None and not is_full_range:
            # Use RT center with default window
            window_width = 2.0  # Default 2-minute window around center
            zoom_start = max(0, rt_center - window_width)
            zoom_end = rt_center + window_width
            rt_window_start = (
                rt_center - 1.0
            )  # 1 minute around center for intensity calc
            rt_window_end = rt_center + 1.0

        else:
            # No RT info available or using full range, try to find a reasonable zoom based on data
            if all_rt_values:
                data_rt_min = min(all_rt_values)
                data_rt_max = max(all_rt_values)
                rt_range = data_rt_max - data_rt_min

                if (
                    rt_range > 5
                ):  # If data spans more than 5 minutes, zoom to middle portion
                    center = (data_rt_min + data_rt_max) / 2
                    zoom_start = center - 2.5
                    zoom_end = center + 2.5
                    rt_window_start = center - 2.5
                    rt_window_end = center + 2.5
                else:
                    # Use full data range with small margin
                    margin = max(0.5, rt_range * 0.1)
                    zoom_start = max(0, data_rt_min - margin)
                    zoom_end = data_rt_max + margin
                    rt_window_start = data_rt_min
                    rt_window_end = data_rt_max
            else:
                # No data, don't zoom
                return

        # Find maximum intensity within the RT window for intelligent y-scaling
        max_intensity_in_window = 0
        min_intensity_in_window = float("inf")

        for filename, data in eic_data.items():
            if len(data["rt"]) > 0 and len(data["intensity"]) > 0:
                # Filter data to RT window
                rt_mask = (data["rt"] >= rt_window_start) & (
                    data["rt"] <= rt_window_end
                )
                if np.any(rt_mask):
                    intensities_in_window = data["intensity"][rt_mask]
                    if len(intensities_in_window) > 0:
                        window_max = intensities_in_window.max()
                        window_min = intensities_in_window.min()
                        max_intensity_in_window = max(
                            max_intensity_in_window, window_max
                        )
                        if window_min < min_intensity_in_window:
                            min_intensity_in_window = window_min

        # Apply the zoom
        try:
            ax.set_xlim(zoom_start, zoom_end)

            # Set intelligent y-limits based on intensity in RT window
            if max_intensity_in_window > 0:
                # Add 20% margin above max intensity, start from 0 or slightly below min
                y_margin = max_intensity_in_window * 0.2
                y_max = max_intensity_in_window + y_margin

                # Set y_min to 0 or slightly below minimum if there are negative values
                if min_intensity_in_window != float("inf"):
                    y_min = (
                        min(0, min_intensity_in_window * 1.1)
                        if min_intensity_in_window < 0
                        else 0
                    )
                else:
                    y_min = 0

                ax.set_ylim(y_min, y_max)

            # Add vertical lines to show the compound's expected RT range
            if rt_start is not None and rt_end is not None:
                ax.axvspan(
                    rt_start,
                    rt_end,
                    alpha=0.1,
                    color="gray",
                    label="Expected RT range",
                    zorder=0,
                )
            elif rt_center is not None:
                ax.axvline(
                    rt_center,
                    alpha=0.3,
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    label="Expected RT",
                    zorder=0,
                )

        except Exception as e:
            print(f"    Error setting zoom: {e}")

    def _open_individual_eic_viewer(self):
        """Open the individual EIC viewer for this adduct"""
        try:
            # Import here to avoid circular imports
            from .eic_window import EICWindow

            # Create and show the individual EIC window
            eic_window = EICWindow(
                compound_data=self.compound,  # Use compound_data parameter name
                adduct=self.adduct,
                file_manager=self.file_manager,
                mz_value=self.mz_value,  # Pass the m/z value
                polarity=self.polarity,  # Pass the polarity
                defaults=self.defaults,
                parent=self.parent(),
            )
            eic_window.show()

        except Exception as e:
            print(f"Error opening individual EIC viewer: {e}")
            # Show a message box if there's an error
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self, "Error", f"Could not open individual EIC viewer:\n{str(e)}"
            )

    def _set_auto_zoom(self, ax, all_rt_values):
        """Set automatic zoom to compound RT range (legacy method, replaced by _set_auto_zoom_with_y_scaling)"""
        # This method is kept for compatibility but is no longer used
        pass

    def reset_zoom(self):
        """Reset zoom to show all data"""
        if hasattr(self, "toolbar"):
            self.toolbar.home()  # Use toolbar's home function

    def plot_empty(self, message="No data or invalid m/z"):
        """Plot empty placeholder"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.set_xlabel("Retention Time (min)", fontsize=8)
        ax.set_ylabel("Intensity", fontsize=8)
        ax.tick_params(labelsize=7)
        self.figure.tight_layout(pad=0.5)
        self.canvas.draw()


class MultiAdductWindow(QWidget):
    """Window for displaying multiple adduct EICs"""

    def __init__(
        self,
        compound,
        adducts_data,
        file_manager,
        defaults=None,
        show_predefined_only=True,
        parent=None,
    ):
        super().__init__(parent)
        self.compound = compound
        self.adducts_data = adducts_data  # List of tuples: (adduct, mz_value, polarity)
        self.file_manager = file_manager
        self.defaults = defaults or {}
        self.show_predefined_only = show_predefined_only

        compound_name = compound.get("Name", "Unknown")
        window_type = "Predefined Adducts" if show_predefined_only else "All Adducts"
        self.setWindowTitle(f"Multi-Adduct EIC - {compound_name} ({window_type})")
        self.setWindowFlags(Qt.WindowType.Window)

        # Set initial size for matrix layout - wider to accommodate 3 columns
        self.resize(1200, 900)

        self.setup_ui()

    def _calculate_max_intensity_in_rt_window(self, adduct, mz_value, polarity):
        """Calculate maximum intensity for an adduct within the compound's RT window"""
        if mz_value is None:
            return 0

        try:
            # Get defaults for EIC extraction
            mz_tolerance_ppm = self.defaults.get("mz_tolerance_ppm", 5.0)
            mz_tolerance_da = (mz_value * mz_tolerance_ppm) / 1e6
            mz_tolerance_da = max(mz_tolerance_da, 0.01)  # At least 0.01 Da
            calculation_method = self.defaults.get(
                "calculation_method", "Sum of all signals"
            )

            # Get compound RT range
            rt_start = self.compound.get("RT_start_min")
            rt_end = self.compound.get("RT_end_min")
            rt_center = self.compound.get("RT_min")

            # Determine RT window for intensity calculation
            if rt_start is not None and rt_end is not None:
                rt_window_start = rt_start
                rt_window_end = rt_end
            elif rt_center is not None:
                rt_window_start = rt_center - 1.0  # 1 minute around center
                rt_window_end = rt_center + 1.0
            else:
                # No RT info, return 0 (will be sorted last)
                return 0

            max_intensity = 0
            files_data = self.file_manager.get_files_data()

            for idx, file_row in files_data.iterrows():
                file_path = file_row["Filepath"]

                try:
                    # Extract EIC data
                    rt_values, intensity_values = self.file_manager.extract_eic(
                        filepath=file_path,
                        target_mz=mz_value,
                        mz_tolerance=mz_tolerance_da,
                        rt_start=None,
                        rt_end=None,
                        calculation_method=calculation_method,
                        polarity=polarity,
                    )

                    if len(rt_values) > 0 and len(intensity_values) > 0:
                        # Filter to RT window
                        rt_mask = (rt_values >= rt_window_start) & (
                            rt_values <= rt_window_end
                        )
                        if np.any(rt_mask):
                            intensities_in_window = intensity_values[rt_mask]
                            if len(intensities_in_window) > 0:
                                window_max = intensities_in_window.max()
                                max_intensity = max(max_intensity, window_max)

                    # Try without polarity filter if no data found
                    if max_intensity == 0 and polarity is not None:
                        rt_values_no_pol, intensity_values_no_pol = (
                            self.file_manager.extract_eic(
                                filepath=file_path,
                                target_mz=mz_value,
                                mz_tolerance=mz_tolerance_da,
                                rt_start=None,
                                rt_end=None,
                                calculation_method=calculation_method,
                                polarity=None,
                            )
                        )

                        if (
                            len(rt_values_no_pol) > 0
                            and len(intensity_values_no_pol) > 0
                        ):
                            rt_mask = (rt_values_no_pol >= rt_window_start) & (
                                rt_values_no_pol <= rt_window_end
                            )
                            if np.any(rt_mask):
                                intensities_in_window = intensity_values_no_pol[rt_mask]
                                if len(intensities_in_window) > 0:
                                    window_max = intensities_in_window.max()
                                    max_intensity = max(max_intensity, window_max)

                except Exception as e:
                    continue

            return max_intensity

        except Exception as e:
            print(f"Error calculating max intensity for {adduct}: {e}")
            return 0

    def setup_ui(self):
        """Setup the window UI"""
        layout = QVBoxLayout(self)

        # Header with compound information
        compound_name = self.compound.get("Name", "Unknown")
        rt_min = self.compound.get("RT_min", "N/A")
        header_text = (
            f"<b>Compound:</b> {compound_name}<br><b>Expected RT:</b> {rt_min} min"
        )

        header_label = QLabel(header_text)
        header_label.setStyleSheet("""
            QLabel { 
                background-color: #e8f4f8; 
                padding: 8px; 
                margin: 5px;
                border: 2px solid #4a90e2;
                border-radius: 5px;
                font-size: 12px;
            }
        """)
        layout.addWidget(header_label)

        # Scroll area for EIC plots in matrix layout
        scroll_area = QScrollArea()
        scroll_widget = QWidget()

        # Create a grid layout with 3 columns
        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setSpacing(10)

        # Calculate max intensities for all adducts and sort by descending abundance
        adducts_with_intensity = []
        for adduct, mz_value, polarity in self.adducts_data:
            if mz_value is not None:  # Only process adducts with valid m/z
                max_intensity = self._calculate_max_intensity_in_rt_window(
                    adduct, mz_value, polarity
                )
                adducts_with_intensity.append(
                    (adduct, mz_value, polarity, max_intensity)
                )

        # Sort by maximum intensity in descending order
        adducts_with_intensity.sort(key=lambda x: x[3], reverse=True)

        # Add EIC widgets for each adduct in order of descending abundance
        row = 0
        col = 0
        valid_adducts_count = 0

        for adduct, mz_value, polarity, max_intensity in adducts_with_intensity:
            eic_widget = InteractiveEICWidget(
                self.compound,
                adduct,
                self.file_manager,
                mz_value,
                polarity,
                self.defaults,
                self,
            )

            # Set fixed size for matrix layout (increased height for navigation toolbar)
            eic_widget.setMinimumSize(350, 280)
            eic_widget.setMaximumSize(500, 380)

            grid_layout.addWidget(eic_widget, row, col)

            valid_adducts_count += 1
            col += 1

            # Move to next row after 3 columns
            if col >= 3:
                col = 0
                row += 1

        # Set scroll area properties
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(scroll_area)

        # Status bar
        if valid_adducts_count > 0:
            status_text = f"Showing {valid_adducts_count} adducts in {row + 1} rows (sorted by descending abundance in RT window)"
        else:
            status_text = "No adducts available"

        status_label = QLabel(status_text)
        status_label.setStyleSheet("""
            QLabel { 
                background-color: #f9f9f9; 
                padding: 5px; 
                border-top: 1px solid #ccc;
                font-size: 10px;
                color: #666;
            }
        """)
        layout.addWidget(status_label)
