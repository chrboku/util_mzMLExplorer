"""
Multi-adduct EIC window for displaying multiple adduct chromatograms
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QProgressDialog,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def extract_eic_for_target(file_manager, files_data, mz_value, polarity, mz_tolerance_ppm, calculation_method):
    """Extract EIC data (keyed by filename) for a given m/z / polarity across all loaded files.

    Shared helper so both the per-adduct and per-sample widgets, as well as the
    parallel adduct-extraction pass in ``MultiAdductWindow``, use identical
    extraction logic.
    """
    mz_tolerance_da = (mz_value * mz_tolerance_ppm) / 1e6
    eic_results = {}

    for idx, file_row in files_data.iterrows():
        filename = file_row["filename"]
        file_path = file_row["Filepath"]  # Use capital F as in the file_manager

        try:
            rt_values, intensity_values = file_manager.extract_eic(
                filepath=file_path,
                target_mz=mz_value,
                mz_tolerance=mz_tolerance_da,
                rt_start=None,
                rt_end=None,
                calculation_method=calculation_method,
                polarity=polarity,
            )

            if len(rt_values) > 0 and len(intensity_values) > 0:
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
        compound_update_callback=None,
        adducts_df=None,
        latest_compound_callback=None,
        eic_data=None,
    ):
        super().__init__(parent)
        self.compound = compound
        self.adduct = adduct
        self.file_manager = file_manager
        self.mz_value = mz_value
        self.polarity = polarity
        self.defaults = defaults or {}
        self.compound_update_callback = compound_update_callback
        self.adducts_df = adducts_df
        self.latest_compound_callback = latest_compound_callback
        # Optional precomputed EIC data (e.g. extracted in parallel ahead of
        # widget creation) to avoid re-extracting it synchronously here.
        self._precomputed_eic_data = eic_data

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
            # Use precomputed EIC data if it was extracted ahead of time
            # (e.g. in parallel across adducts), otherwise extract it now.
            if self._precomputed_eic_data is not None:
                eic_data = self._precomputed_eic_data
            else:
                eic_data = self._extract_eic_data()
            self.plot_eic(eic_data)
        except Exception as e:
            print(f"Error loading EIC data for {self.adduct}: {str(e)}")
            import traceback

            traceback.print_exc()
            self.plot_empty(f"Error: {str(e)}")

    def _extract_eic_data(self):
        """Extract EIC data for this adduct"""
        # Use the same ppm→Da conversion as the single EIC window so both
        # visualisations rely on an identical extraction tolerance.
        mz_tolerance_ppm = self.defaults.get("mz_tolerance_ppm", 5.0)
        calculation_method = self.defaults.get("calculation_method", "Sum of all signals")
        files_data = self.file_manager.get_files_data()

        return extract_eic_for_target(self.file_manager, files_data, self.mz_value, self.polarity, mz_tolerance_ppm, calculation_method)

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
            rt_window_start = rt_center - 1.0  # 1 minute around center for intensity calc
            rt_window_end = rt_center + 1.0

        else:
            # No RT info available or using full range - use entire RT range from all EICs
            if all_rt_values:
                data_rt_min = min(all_rt_values)
                data_rt_max = max(all_rt_values)
                rt_range = data_rt_max - data_rt_min

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
                rt_mask = (data["rt"] >= rt_window_start) & (data["rt"] <= rt_window_end)
                if np.any(rt_mask):
                    intensities_in_window = data["intensity"][rt_mask]
                    if len(intensities_in_window) > 0:
                        window_max = intensities_in_window.max()
                        window_min = intensities_in_window.min()
                        max_intensity_in_window = max(max_intensity_in_window, window_max)
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
                    y_min = min(0, min_intensity_in_window * 1.1) if min_intensity_in_window < 0 else 0
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
            from .windows import EICWindow

            # Ensure the standalone EIC window starts with the same visual baseline
            # as the multi-adduct view (no group time shifting by default).
            if isinstance(self.defaults, dict) and self.defaults:
                eic_defaults = self.defaults.copy()
            else:
                eic_defaults = {}

            # Ensure required keys exist and align the view defaults with the matrix view
            eic_defaults.setdefault("mz_tolerance_ppm", 5.0)
            eic_defaults.setdefault("rt_shift_min", 1.0)
            eic_defaults.setdefault("eic_view_mode", "Show Entire EIC")
            eic_defaults.setdefault("normalize_samples", False)
            eic_defaults.setdefault("normalize_mode", "No normalization")
            eic_defaults["separate_groups"] = False

            # Create and show the individual EIC window
            eic_window = EICWindow(
                compound_data=self.compound,  # Use compound_data parameter name
                adduct=self.adduct,
                file_manager=self.file_manager,
                mz_value=self.mz_value,  # Pass the m/z value
                polarity=self.polarity,  # Pass the polarity
                defaults=eic_defaults,
                parent=self.parent(),
                compound_update_callback=self.compound_update_callback,
                adducts_data=self.adducts_df,
                latest_compound_callback=self.latest_compound_callback,
            )
            eic_window.show()

        except Exception as e:
            print(f"Error opening individual EIC viewer: {e}")
            # Show a message box if there's an error
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Error", f"Could not open individual EIC viewer:\n{str(e)}")

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


class SampleEICWidget(QWidget):
    """EIC plot widget for a single sample showing all adducts overlaid"""

    def __init__(
        self,
        compound,
        sample_filename,
        sample_filepath,
        adducts_data,
        file_manager,
        defaults=None,
        parent=None,
        adduct_results=None,
    ):
        super().__init__(parent)
        self.compound = compound
        self.sample_filename = sample_filename
        self.sample_filepath = sample_filepath
        self.adducts_data = adducts_data  # list of (adduct, mz_value, polarity)
        self.file_manager = file_manager
        self.defaults = defaults or {}
        # Optional precomputed per-adduct EIC results for this sample (e.g.
        # extracted in parallel across adducts ahead of widget creation).
        self._precomputed_adduct_results = adduct_results

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        """Setup the UI for this sample EIC widget"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # Header with sample name
        self.header_label = QLabel(self.sample_filename)
        self.header_label.setStyleSheet("""
            QLabel { 
                background-color: #f0f4f0; 
                padding: 3px; 
                margin: 1px;
                border: 1px solid #aaccaa;
                border-radius: 3px;
                font-weight: bold;
            }
        """)
        self.header_label.setMaximumHeight(25)
        layout.addWidget(self.header_label)

        # Create matplotlib figure
        self.figure = Figure(figsize=(4, 2.5), dpi=80)
        self.figure.patch.set_facecolor("white")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)

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
        for action in self.toolbar.actions():
            if action.text() in ["Configure subplots", "Save", "Forward", "Back"]:
                action.setVisible(False)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setMinimumSize(300, 250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def load_data(self):
        """Load and plot EIC data for all adducts"""
        try:
            if self._precomputed_adduct_results is not None:
                eic_data = self._precomputed_adduct_results
            else:
                eic_data = self._extract_eic_data()
            self.plot_eic(eic_data)
        except Exception as e:
            print(f"Error loading sample EIC data for {self.sample_filename}: {str(e)}")
            import traceback

            traceback.print_exc()
            self.plot_empty(f"Error: {str(e)}")

    def _extract_eic_data(self):
        """Extract EIC for each adduct from this sample"""
        mz_tolerance_ppm = self.defaults.get("mz_tolerance_ppm", 5.0)
        calculation_method = self.defaults.get("calculation_method", "Sum of all signals")

        adduct_results = {}
        for adduct, mz_value, polarity in self.adducts_data:
            if mz_value is None:
                continue
            mz_tolerance_da = (mz_value * mz_tolerance_ppm) / 1e6
            try:
                rt_values, intensity_values = self.file_manager.extract_eic(
                    filepath=self.sample_filepath,
                    target_mz=mz_value,
                    mz_tolerance=mz_tolerance_da,
                    rt_start=None,
                    rt_end=None,
                    calculation_method=calculation_method,
                    polarity=polarity,
                )
                if len(rt_values) > 0 and len(intensity_values) > 0:
                    adduct_results[adduct] = {
                        "rt": rt_values,
                        "intensity": intensity_values,
                        "mz_value": mz_value,
                    }
            except Exception as e:
                print(f"ERROR extracting {adduct} from {self.sample_filename}: {str(e)}")
                continue
        return adduct_results

    def plot_eic(self, eic_data):
        """Plot EICs of all adducts overlaid for this sample"""
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
            # Use tab10 + tab20b colours for up to 20 adducts
            base_colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20b.colors)
            all_rt_values = []
            plots_made = 0

            for idx, (adduct, data) in enumerate(eic_data.items()):
                color = base_colors[idx % len(base_colors)]
                if len(data["rt"]) > 0 and len(data["intensity"]) > 0:
                    all_rt_values.extend(data["rt"])
                    ax.plot(
                        data["rt"],
                        data["intensity"],
                        label=adduct,
                        color=color,
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
                ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
                ax.legend(fontsize=6, loc="upper right")
                self._set_auto_zoom(ax, eic_data, all_rt_values)

        self.figure.tight_layout(pad=0.5)
        self.canvas.draw()

    def _set_auto_zoom(self, ax, eic_data, all_rt_values):
        """Set automatic zoom to compound RT range with intelligent y-axis scaling"""
        rt_start = self.compound.get("RT_start_min")
        rt_end = self.compound.get("RT_end_min")
        rt_center = self.compound.get("RT_min")

        is_full_range = rt_start == 0.0 and rt_end == 100.0

        if rt_start is not None and rt_end is not None and not is_full_range:
            rt_margin = (rt_end - rt_start) * 0.1
            zoom_start = max(0, rt_start - rt_margin)
            zoom_end = rt_end + rt_margin
            rt_window_start = rt_start
            rt_window_end = rt_end
        elif rt_center is not None and not is_full_range:
            window_width = 2.0
            zoom_start = max(0, rt_center - window_width)
            zoom_end = rt_center + window_width
            rt_window_start = rt_center - 1.0
            rt_window_end = rt_center + 1.0
        else:
            if all_rt_values:
                data_rt_min = min(all_rt_values)
                data_rt_max = max(all_rt_values)
                rt_range = data_rt_max - data_rt_min
                margin = max(0.5, rt_range * 0.1)
                zoom_start = max(0, data_rt_min - margin)
                zoom_end = data_rt_max + margin
                rt_window_start = data_rt_min
                rt_window_end = data_rt_max
            else:
                return

        max_intensity_in_window = 0
        min_intensity_in_window = float("inf")

        for adduct, data in eic_data.items():
            if len(data["rt"]) > 0 and len(data["intensity"]) > 0:
                rt_mask = (data["rt"] >= rt_window_start) & (data["rt"] <= rt_window_end)
                if np.any(rt_mask):
                    intensities = data["intensity"][rt_mask]
                    if len(intensities) > 0:
                        max_intensity_in_window = max(max_intensity_in_window, intensities.max())
                        min_intensity_in_window = min(min_intensity_in_window, intensities.min())

        try:
            ax.set_xlim(zoom_start, zoom_end)
            if max_intensity_in_window > 0:
                y_margin = max_intensity_in_window * 0.2
                y_max = max_intensity_in_window + y_margin
                if min_intensity_in_window != float("inf"):
                    y_min = min(0, min_intensity_in_window * 1.1) if min_intensity_in_window < 0 else 0
                else:
                    y_min = 0
                ax.set_ylim(y_min, y_max)

            if rt_start is not None and rt_end is not None and not is_full_range:
                ax.axvspan(rt_start, rt_end, alpha=0.1, color="gray", zorder=0)
            elif rt_center is not None and not is_full_range:
                ax.axvline(rt_center, alpha=0.3, color="red", linestyle="--", linewidth=1, zorder=0)
        except Exception as e:
            print(f"Error setting zoom: {e}")

    def plot_empty(self, message="No data"):
        """Plot empty placeholder"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=9)
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
        compound_update_callback=None,
        adducts_df=None,
        latest_compound_callback=None,
    ):
        super().__init__(parent)
        self.compound = compound
        self.adducts_data = adducts_data  # List of tuples: (adduct, mz_value, polarity)
        self.file_manager = file_manager
        self.defaults = defaults or {}
        self.show_predefined_only = show_predefined_only
        self.compound_update_callback = compound_update_callback
        self.adducts_df = adducts_df  # Adducts DataFrame for m/z calculation in child EIC windows
        self.latest_compound_callback = latest_compound_callback  # Look up newest saved RT window

        compound_name = compound.get("Name", "Unknown")
        window_type = "Predefined Adducts" if show_predefined_only else "All Adducts"
        self.setWindowTitle(f"Multi-Adduct EIC - {compound_name} ({window_type})")
        self.setWindowFlags(Qt.WindowType.Window)

        # Set initial size for matrix layout - wider to accommodate 3 columns
        self.resize(1200, 900)

        self.setup_ui()

    def _get_rt_window(self):
        """Return (rt_window_start, rt_window_end) for the compound, or (None, None)."""
        rt_start = self.compound.get("RT_start_min")
        rt_end = self.compound.get("RT_end_min")
        rt_center = self.compound.get("RT_min")

        if rt_start is not None and rt_end is not None:
            return rt_start, rt_end
        elif rt_center is not None:
            return rt_center - 1.0, rt_center + 1.0
        return None, None

    @staticmethod
    def _max_intensity_in_window(eic_data, rt_window_start, rt_window_end):
        """Compute the maximum intensity across all files within an RT window,
        given already-extracted EIC data (as returned by ``extract_eic_for_target``)."""
        max_intensity = 0
        for data in eic_data.values():
            rt_values = data.get("rt")
            intensity_values = data.get("intensity")
            if rt_values is None or len(rt_values) == 0:
                continue
            rt_mask = (rt_values >= rt_window_start) & (rt_values <= rt_window_end)
            if np.any(rt_mask):
                intensities_in_window = intensity_values[rt_mask]
                if len(intensities_in_window) > 0:
                    max_intensity = max(max_intensity, intensities_in_window.max())
        return max_intensity

    def _process_adduct(self, adduct, mz_value, polarity):
        """Extract EIC data for one adduct and compute its maximum intensity within
        the compound's RT window. Safe to run inside a worker thread - only reads
        from ``file_manager`` and performs no Qt calls."""
        if mz_value is None:
            return adduct, mz_value, polarity, {}, 0

        mz_tolerance_ppm = self.defaults.get("mz_tolerance_ppm", 5.0)
        calculation_method = self.defaults.get("calculation_method", "Sum of all signals")
        files_data = self.file_manager.get_files_data()

        eic_data = extract_eic_for_target(self.file_manager, files_data, mz_value, polarity, mz_tolerance_ppm, calculation_method)

        rt_window_start, rt_window_end = self._get_rt_window()
        max_intensity = 0
        if rt_window_start is not None and rt_window_end is not None:
            max_intensity = self._max_intensity_in_window(eic_data, rt_window_start, rt_window_end)

            # Fall back to an unfiltered-polarity extraction purely for sorting
            # purposes if nothing was found in-window with the given polarity.
            if max_intensity == 0 and polarity is not None:
                fallback_data = extract_eic_for_target(self.file_manager, files_data, mz_value, None, mz_tolerance_ppm, calculation_method)
                max_intensity = self._max_intensity_in_window(fallback_data, rt_window_start, rt_window_end)

        return adduct, mz_value, polarity, eic_data, max_intensity

    def setup_ui(self):
        """Setup the window UI"""
        layout = QVBoxLayout(self)

        # Header with compound information
        compound_name = self.compound.get("Name", "Unknown")
        rt_min = self.compound.get("RT_min", "N/A")
        header_text = f"<b>Compound:</b> {compound_name}<br><b>Expected RT:</b> {rt_min} min"

        header_label = QLabel(header_text)
        header_label.setStyleSheet("""
            QLabel { 
                background-color: #e8f4f8; 
                padding: 1px; 
                margin: 1px;
                border: 2px solid #4a90e2;
                border-radius: 5px;
            }
        """)
        layout.addWidget(header_label)

        # Calculate max intensities for all adducts and sort by descending abundance
        # Show a progress dialog during the (potentially slow) extraction step
        valid_adducts = [(a, mz, pol) for a, mz, pol in self.adducts_data if mz is not None]
        files_data = self.file_manager.get_files_data()
        n_adducts = len(valid_adducts)
        n_samples = len(files_data)
        total = n_adducts
        progress = QProgressDialog("Extracting EIC data…", None, 0, max(total, 1), self.parent())
        progress.setWindowTitle("Multi-Adduct EIC – Please Wait")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setAutoReset(False)
        progress.setValue(0)
        QApplication.processEvents()

        adducts_with_intensity = []
        eic_data_by_adduct = {}
        parallel_tasks = max(1, int(self.defaults.get("parallel_tasks", 4)))
        completed = 0
        with ThreadPoolExecutor(max_workers=min(parallel_tasks, max(n_adducts, 1))) as executor:
            futures = {executor.submit(self._process_adduct, adduct, mz_value, polarity): adduct for adduct, mz_value, polarity in valid_adducts}
            for future in as_completed(futures):
                adduct, mz_value, polarity, eic_data, max_intensity = future.result()
                eic_data_by_adduct[adduct] = eic_data
                adducts_with_intensity.append((adduct, mz_value, polarity, max_intensity))
                completed += 1
                progress.setValue(completed)
                progress.setLabelText(f"Processed adduct {completed}/{total}: {adduct}")
                QApplication.processEvents()

        progress.setValue(total)

        # Sort by maximum intensity in descending order
        adducts_with_intensity.sort(key=lambda x: x[3], reverse=True)
        sorted_adducts = [(a, mz, pol) for a, mz, pol, _ in adducts_with_intensity]

        # ── Scroll area that holds both sections ──────────────────────────────
        scroll_area = QScrollArea()
        outer_widget = QWidget()
        outer_layout = QVBoxLayout(outer_widget)
        outer_layout.setSpacing(8)

        # ── Section 1: EICs per adduct ────────────────────────────────────────
        adduct_section_label = QLabel("<b>EICs per Adduct</b> (all samples overlaid, sorted by abundance)")
        adduct_section_label.setStyleSheet("""
            QLabel {
                background-color: #dce8f4;
                padding: 4px 6px;
                border-left: 4px solid #4a90e2;
                font-size: 11px;
            }
        """)
        outer_layout.addWidget(adduct_section_label)

        adduct_grid_widget = QWidget()
        grid_layout = QGridLayout(adduct_grid_widget)
        grid_layout.setSpacing(10)

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
                compound_update_callback=self.compound_update_callback,
                adducts_df=self.adducts_df,
                latest_compound_callback=self.latest_compound_callback,
                eic_data=eic_data_by_adduct.get(adduct),
            )
            eic_widget.setMinimumSize(350, 280)
            eic_widget.setMaximumSize(500, 380)
            grid_layout.addWidget(eic_widget, row, col)
            valid_adducts_count += 1
            col += 1
            if col >= 3:
                col = 0
                row += 1

        outer_layout.addWidget(adduct_grid_widget)

        # ── Section 2: EICs per sample (optional) ─────────────────────────────
        show_sample_eics = self.defaults.get("show_multi_adduct_sample_eics", True)
        sample_count = 0

        if show_sample_eics:
            # ── Separator ─────────────────────────────────────────────────────
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            outer_layout.addWidget(sep)

            sample_section_label = QLabel("<b>EICs per Sample</b> (all adducts overlaid)")
            sample_section_label.setStyleSheet("""
                QLabel {
                    background-color: #dcf4e4;
                    padding: 4px 6px;
                    border-left: 4px solid #4a90e2;
                    font-size: 11px;
                }
            """)
            outer_layout.addWidget(sample_section_label)

            sample_grid_widget = QWidget()
            sample_grid_layout = QGridLayout(sample_grid_widget)
            sample_grid_layout.setSpacing(10)

            s_row = 0
            s_col = 0

            # Reshape the already-extracted per-adduct EIC data into a
            # per-sample structure so SampleEICWidget doesn't need to
            # re-extract anything from the mzML files.
            adduct_mz_lookup = {a: mz for a, mz, _pol in sorted_adducts}
            eic_data_by_sample = {}
            for adduct, mz_value, _pol in sorted_adducts:
                for filename, data in eic_data_by_adduct.get(adduct, {}).items():
                    eic_data_by_sample.setdefault(filename, {})[adduct] = {
                        "rt": data["rt"],
                        "intensity": data["intensity"],
                        "mz_value": adduct_mz_lookup.get(adduct, mz_value),
                    }

            for _, file_row in files_data.iterrows():
                filename = file_row["filename"]
                filepath = file_row["Filepath"]

                sample_widget = SampleEICWidget(
                    self.compound,
                    filename,
                    filepath,
                    sorted_adducts,
                    self.file_manager,
                    self.defaults,
                    self,
                    adduct_results=eic_data_by_sample.get(filename, {}),
                )
                sample_widget.setMinimumSize(350, 280)
                sample_widget.setMaximumSize(500, 380)
                sample_grid_layout.addWidget(sample_widget, s_row, s_col)
                sample_count += 1
                s_col += 1
                if s_col >= 3:
                    s_col = 0
                    s_row += 1

            outer_layout.addWidget(sample_grid_widget)

        outer_layout.addStretch(1)

        # Set scroll area properties
        scroll_area.setWidget(outer_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(scroll_area)

        # Status bar
        if valid_adducts_count > 0:
            if show_sample_eics:
                status_text = f"Showing {valid_adducts_count} adducts (sorted by descending abundance) and {sample_count} samples"
            else:
                status_text = f"Showing {valid_adducts_count} adducts (sorted by descending abundance)"
        else:
            status_text = "No adducts available"

        status_label = QLabel(status_text)
        status_label.setStyleSheet("""
            QLabel { 
                background-color: #f9f9f9; 
                padding: 1px; 
                border-top: 1px solid #ccc;
                color: #666;
            }
        """)
        layout.addWidget(status_label)
