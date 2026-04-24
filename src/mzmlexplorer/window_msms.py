"""
MSMS viewer windows: MSMSPopupWindow, InteractiveMSMSChartView,
MSMSViewerWindow, EnhancedMirrorPlotWindow.
"""

import sys
import os
import re
import traceback
import time
import numpy as np
from typing import Dict, Tuple, Optional, List
from natsort import natsorted, natsort_keygen
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: F401 (kept for compatibility)
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar  # noqa: F401
from matplotlib.figure import Figure  # noqa: F401
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QSplitter,
    QMenu,
    QScrollArea,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QApplication,
    QWidgetAction,
    QSizePolicy,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QCheckBox,
    QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QMargins
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPen, QColor, QPainter, QMouseEvent, QAction, QBrush
from .window_shared import CollapsibleBox, ANNOTATION_COLOR_PRESETS, BarDelegate, NumericTableWidgetItem, NoScrollSpinBox, NoScrollDoubleSpinBox
from .utils import calculate_cosine_similarity, calculate_similarity_statistics, make_usi
from .FormulaTools import FragmentAnnotator


class FragmentEICWorker(QThread):
    """Background worker that extracts per-fragment EIC traces.

    Searches ALL MS2 spectra in the file that share the same precursor m/z
    and polarity, so the resulting traces span the full chromatographic run
    rather than just the small RT window shown in MSMSViewerWindow.

    Priority order for data source:
      1. file_manager in-memory cache (fast, no I/O).
      2. Direct pymzml file read (only when cache not available).
      3. Fallback: the pre-built *all_similar_spectra* list.
    """

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        fragment_mz_list,
        all_similar_spectra=None,
        mz_tolerance_ppm=10.0,
        file_manager=None,
        filepath=None,
        precursor_mz=None,
        polarity=None,
        precursor_tolerance=0.01,
        parent=None,
    ):
        super().__init__(parent)
        self.fragment_mz_list = np.asarray(fragment_mz_list, dtype=float)
        self.all_similar_spectra = list(all_similar_spectra) if all_similar_spectra else []
        self.mz_tolerance_ppm = float(mz_tolerance_ppm)
        self.file_manager = file_manager
        self.filepath = filepath
        self.precursor_mz = precursor_mz
        self.polarity = polarity
        self.precursor_tolerance = float(precursor_tolerance)

    def _gather_spectra(self):
        """Return a list of {rt, mz, intensity} dicts for all matching MS2 scans."""
        spectra = []

        # --- 1. Try in-memory cache first (thread-safe read) ---
        if self.file_manager is not None and self.filepath and self.file_manager.keep_in_memory:
            cached = self.file_manager.cached_data.get(self.filepath)
            if isinstance(cached, dict) and "ms2" in cached:
                for spec in cached["ms2"]:
                    pmz = spec.get("precursor_mz")
                    if pmz is None:
                        continue
                    if self.precursor_mz is not None and abs(pmz - self.precursor_mz) > self.precursor_tolerance:
                        continue
                    spec_pol = spec.get("polarity")
                    if self.polarity and spec_pol and spec_pol != self.polarity:
                        continue
                    spectra.append(
                        {
                            "rt": float(spec["scan_time"]),
                            "mz": spec["mz"],
                            "intensity": spec["intensity"],
                        }
                    )
                return spectra

        # --- 2. Read directly from file in background thread ---
        if self.filepath and os.path.isfile(self.filepath):
            try:
                import pymzml  # local import – only needed here

                reader = pymzml.run.Reader(self.filepath)
                for spectrum in reader:
                    if spectrum.ms_level != 2:
                        continue
                    try:
                        pmz = spectrum.selected_precursors[0]["mz"] if spectrum.selected_precursors else None
                        if pmz is None:
                            continue
                        if self.precursor_mz is not None and abs(pmz - self.precursor_mz) > self.precursor_tolerance:
                            continue
                        if self.file_manager is not None:
                            spec_pol = self.file_manager._get_spectrum_polarity(spectrum)
                        else:
                            spec_pol = None
                        if self.polarity and spec_pol and spec_pol != self.polarity:
                            continue
                        spectra.append(
                            {
                                "rt": spectrum.scan_time_in_minutes(),
                                "mz": spectrum.mz,
                                "intensity": spectrum.i,
                            }
                        )
                    except Exception:
                        continue
                return spectra
            except Exception as exc:
                print(f"FragmentEICWorker: file read failed ({exc}), using fallback")

        # --- 3. Fallback: the pre-built list (may only cover the RT window) ---
        return self.all_similar_spectra

    def run(self):
        try:
            spectra = self._gather_spectra()
            if not spectra:
                self.finished.emit({})
                return

            result = {}
            for frag_mz in self.fragment_mz_list:
                rt_list = []
                int_list = []
                for spec in spectra:
                    spec_rt = float(spec.get("rt", spec.get("scan_time", 0.0)))
                    spec_mz = np.asarray(spec["mz"], dtype=float)
                    spec_int = np.asarray(spec["intensity"], dtype=float)
                    tol_da = frag_mz * self.mz_tolerance_ppm / 1e6
                    mask = np.abs(spec_mz - frag_mz) <= tol_da
                    intensity = float(np.max(spec_int[mask])) if np.any(mask) else 0.0
                    rt_list.append(spec_rt)
                    int_list.append(intensity)

                if rt_list:
                    order = np.argsort(rt_list)
                    result[float(frag_mz)] = (
                        np.array(rt_list)[order],
                        np.array(int_list)[order],
                    )

            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
class InteractiveEICChartView(QChartView):
    """QChartView with left-drag pan, right-drag zoom, wheel zoom and
    double-click-to-reset — suitable for the fragment EIC panel."""

    def __init__(self, chart, parent=None):
        super().__init__(chart, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRubberBand(QChartView.RubberBand.NoRubberBand)
        self.setMouseTracking(True)

        self._panning = False
        self._zooming = False
        self._pan_start = QPointF()
        self._zoom_start = QPointF()
        self._start_x_range = None
        self._start_y_range = None
        self._zoom_anchor_x = 0.0
        self._zoom_anchor_y = 0.0
        self._full_x_range = None  # set by _draw_fragment_eic_plot for reset
        self._full_y_range = None

    def set_full_range(self, x_min, x_max, y_min, y_max):
        """Store the default range so double-click can restore it."""
        self._full_x_range = (x_min, x_max)
        self._full_y_range = (y_min, y_max)

    def _x_axis(self):
        axes = self.chart().axes(Qt.Orientation.Horizontal)
        return axes[0] if axes else None

    def _y_axis(self):
        axes = self.chart().axes(Qt.Orientation.Vertical)
        return axes[0] if axes else None

    def mousePressEvent(self, event):
        xa, ya = self._x_axis(), self._y_axis()
        if xa is None or ya is None:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._pan_start = event.position()
            self._start_x_range = (xa.min(), xa.max())
            self._start_y_range = (ya.min(), ya.max())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self._zooming = True
            self._zoom_start = event.position()
            self._start_x_range = (xa.min(), xa.max())
            self._start_y_range = (ya.min(), ya.max())
            pa = self.chart().plotArea()
            rx = max(0.0, min(1.0, (event.position().x() - pa.left()) / pa.width()))
            ry = max(0.0, min(1.0, (event.position().y() - pa.top()) / pa.height()))
            self._zoom_anchor_x = self._start_x_range[0] + rx * (self._start_x_range[1] - self._start_x_range[0])
            self._zoom_anchor_y = self._start_y_range[0] + (1.0 - ry) * (self._start_y_range[1] - self._start_y_range[0])
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        xa, ya = self._x_axis(), self._y_axis()
        if xa is None or ya is None:
            return
        if self._panning and self._start_x_range:
            pa = self.chart().plotArea()
            dx = event.position().x() - self._pan_start.x()
            dy = event.position().y() - self._pan_start.y()
            ox = -dx * (self._start_x_range[1] - self._start_x_range[0]) / pa.width()
            oy = dy * (self._start_y_range[1] - self._start_y_range[0]) / pa.height()
            xa.setRange(self._start_x_range[0] + ox, self._start_x_range[1] + ox)
            ya.setRange(self._start_y_range[0] + oy, self._start_y_range[1] + oy)
        elif self._zooming and self._start_x_range:
            dx = event.position().x() - self._zoom_start.x()
            dy = event.position().y() - self._zoom_start.y()
            s = 0.005
            xf = max(0.1, min(10.0, 1.0 - dx * s))
            yf = max(0.1, min(10.0, 1.0 + dy * s))
            al = self._zoom_anchor_x - self._start_x_range[0]
            ar = self._start_x_range[1] - self._zoom_anchor_x
            ab = self._zoom_anchor_y - self._start_y_range[0]
            at = self._start_y_range[1] - self._zoom_anchor_y
            xa.setRange(self._zoom_anchor_x - al * xf, self._zoom_anchor_x + ar * xf)
            ya.setRange(self._zoom_anchor_y - ab * yf, self._zoom_anchor_y + at * yf)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self._zooming = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Reset to the full data range stored by set_full_range()."""
        if event.button() == Qt.MouseButton.LeftButton and self._full_x_range and self._full_y_range:
            xa, ya = self._x_axis(), self._y_axis()
            if xa and ya:
                xa.setRange(*self._full_x_range)
                ya.setRange(*self._full_y_range)
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        xa, ya = self._x_axis(), self._y_axis()
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


def _format_collision_energy(ce, separator: str = " | ") -> str:
    """Format a ``collision_energy`` field (dict or legacy float) into a display string.

    Returns an empty string when *ce* is ``None``.
    """
    if ce is None:
        return ""
    if isinstance(ce, dict):
        val = ce.get("value")
        unit = ce.get("unit", "eV")
        method = ce.get("method")
        parts = []
        if val is not None:
            parts.append(f"CE: {val:.1f} {unit}")
        if method:
            parts.append(method)
        return (separator + separator.join(parts)) if parts else ""
    # Legacy float
    try:
        return f"{separator}CE: {float(ce):.1f} eV"
    except (TypeError, ValueError):
        return ""


class MSMSPopupWindow(QWidget):
    """Popup window for displaying a single MSMS spectrum in a larger view"""

    def __init__(
        self,
        spectrum_data,
        filename,
        group,
        parent=None,
        compound_formula=None,
        adduct=None,
        adduct_info=None,
        compound_smiles=None,
        filepath=None,
        file_manager=None,
        all_similar_spectra=None,
    ):
        super().__init__(parent)
        self.spectrum_data = spectrum_data
        self.filename = filename
        self.group = group
        self.compound_formula = compound_formula
        self.adduct = adduct
        self.adduct_info = adduct_info  # dict/Series with ElementsAdded, ElementsLost, Charge, …
        self.compound_smiles = compound_smiles
        self.filepath = filepath
        self.file_manager = file_manager
        self.all_similar_spectra = list(all_similar_spectra) if all_similar_spectra else []
        self.selected_mz = None  # Track selected m/z for highlighting

        # Fragment EIC state
        self._eic_data = {}  # {frag_mz: (rt_arr, int_arr)}
        self._eic_series = {}  # {frag_mz: QLineSeries}
        self._eic_frag_colors = {}  # {frag_mz: QColor}
        self._eic_worker = None

        self._usi = make_usi(self.spectrum_data, self.filename)
        self.setWindowTitle(f"MSMS Spectrum — {self.filename} | Group: {self.group} | {self._usi}")
        self.setWindowFlags(Qt.WindowType.Window)  # Make it a separate window
        self.resize(1400, 800)

        self.setup_ui()

    def setup_ui(self):
        """Setup the popup window UI"""
        layout = QVBoxLayout(self)

        # Main horizontal splitter: table | MSMS chart | fragment EIC panel
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create table for m/z and intensity values (left side)
        self.create_data_table()
        splitter.addWidget(self.table_widget)

        # Create large MSMS chart with interactive capabilities (centre)
        chart = self.create_large_msms_chart()
        self.chart_view = InteractiveMSMSChartView(chart)
        self.chart_view.spectrum_data = self.spectrum_data  # Enable hover tooltip
        self.chart_view._usi = self._usi
        self.chart_view._open_comparison_fn = self._open_comparison_window
        self.chart_view.setMinimumSize(400, 300)
        splitter.addWidget(self.chart_view)

        # Fragment EIC panel (right, ~20 % of window width)
        self._eic_panel = self._build_eic_panel()
        splitter.addWidget(self._eic_panel)

        # Proportions: table 25 %, MSMS chart 55 %, EIC panel 20 %
        splitter.setSizes([300, 820, 280])

        layout.addWidget(splitter)

        # Kick off EIC extraction if we have the necessary data
        if self.all_similar_spectra or self.filepath:
            self._start_fragment_eic_extraction()

        # Annotation controls and close button
        button_layout = QHBoxLayout()
        ppm_label = QLabel("Tolerance:")
        self.ppm_spinbox = NoScrollDoubleSpinBox()
        self.ppm_spinbox.setRange(0.1, 2000.0)
        self.ppm_spinbox.setValue(5.0)
        self.ppm_spinbox.setSuffix(" ppm")
        self.ppm_spinbox.setDecimals(1)
        self.ppm_spinbox.setFixedWidth(110)
        self.ppm_spinbox.setToolTip(
            "Mass tolerance in ppm.\nThe absolute Da window is fixed to precursor_m/z × ppm / 1e6,\nand displayed ppm errors are also expressed relative to the precursor m/z."
        )
        extra_elem_label = QLabel("Extra elements:")
        self.extra_elements_edit = QLineEdit()
        self.extra_elements_edit.setPlaceholderText("e.g. Na or C2H4O")
        self.extra_elements_edit.setToolTip(
            "Additional element budget added on top of the compound formula\nand adduct when searching for fragment annotations.\nEnter a sum formula, e.g. 'Na', 'K', or 'C2H4O'."
        )
        self.extra_elements_edit.setFixedWidth(110)
        self.annotate_button = QPushButton("Annotate Fragments")
        self.annotate_button.clicked.connect(self._annotate_fragments)
        if not self.compound_formula:
            self.annotate_button.setEnabled(False)
            self.annotate_button.setToolTip("No compound formula available for annotation")
        button_layout.addWidget(ppm_label)
        button_layout.addWidget(self.ppm_spinbox)
        button_layout.addWidget(extra_elem_label)
        button_layout.addWidget(self.extra_elements_edit)
        button_layout.addWidget(self.annotate_button)
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        # Copy-to-clipboard buttons
        copy_layout = QHBoxLayout()
        copy_lbl = QLabel("Copy:")
        copy_lbl.setStyleSheet("color: #5f6368; font-size: 10px;")
        copy_layout.addWidget(copy_lbl)

        copy_frag_tsv_btn = QPushButton("Copy Fragment table to TSV")
        copy_frag_tsv_btn.setToolTip("Copy the fragment table (as currently displayed) as a tab-separated values table\nsuitable for pasting into Excel.")
        copy_frag_tsv_btn.clicked.connect(self._copy_fragment_table_tsv)
        copy_layout.addWidget(copy_frag_tsv_btn)

        copy_frag_r_btn = QPushButton("Copy Fragment table to R")
        copy_frag_r_btn.setToolTip(
            "Copy the fragment table as an R data.frame() expression. \nNote: paste it into a file and source the file, otherwise \nthe REP with the limit of 4096 characters will fail \nto correctly parse the command."
        )
        copy_frag_r_btn.clicked.connect(self._copy_fragment_table_r)
        copy_layout.addWidget(copy_frag_r_btn)

        copy_eic_tsv_btn = QPushButton("Copy EIC data to TSV")
        copy_eic_tsv_btn.setToolTip("Copy the fragment EIC data (as currently shown in the right panel)\nas a long-format tab-separated table suitable for pasting into Excel.")
        copy_eic_tsv_btn.clicked.connect(self._copy_eic_tsv)
        copy_layout.addWidget(copy_eic_tsv_btn)

        copy_eic_r_btn = QPushButton("Copy EIC data to R")
        copy_eic_r_btn.setToolTip(
            "Copy the fragment EIC data as an R data.frame() expression. \nNote: paste it into a file and source the file, otherwise \nthe REP with the limit of 4096 characters will fail \nto correctly parse the command."
        )
        copy_eic_r_btn.clicked.connect(self._copy_eic_r)
        copy_layout.addWidget(copy_eic_r_btn)

        copy_layout.addStretch()
        layout.addLayout(copy_layout)

    def _render_structure(self):
        """Render self.compound_smiles into self._structure_label using RDKit.

        Uses SVG output so the background is genuinely transparent.
        Silent no-op when RDKit / QtSvg is not installed or the SMILES is invalid.
        """
        smiles = self.compound_smiles
        if not smiles:
            return
        smiles = str(smiles).strip()
        if not smiles or smiles.lower() in ("nan", "none", ""):
            return
        try:
            from rdkit import Chem
            from rdkit.Chem.Draw import rdMolDraw2D
            from PyQt6.QtSvg import QSvgRenderer
            from PyQt6.QtCore import QByteArray
            from PyQt6.QtGui import QPixmap, QPainter

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return
            w, h = self._structure_label.width(), self._structure_label.height()
            drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
            drawer.drawOptions().clearBackground = False
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg_bytes = QByteArray(drawer.GetDrawingText().encode())

            pixmap = QPixmap(w, h)
            pixmap.fill(Qt.GlobalColor.transparent)
            renderer = QSvgRenderer(svg_bytes)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            self._structure_label.setPixmap(pixmap)
        except Exception:
            pass

    def create_data_table(self):
        """Create a sortable table with m/z and intensity data"""
        self.table_widget = QTableWidget()

        # Get data and ensure they're numpy arrays
        mz_array = np.array(self.spectrum_data["mz"])
        intensity_array = np.array(self.spectrum_data["intensity"])

        # Calculate relative abundance (% of base peak)
        max_intensity = np.max(intensity_array) if len(intensity_array) > 0 else 1
        relative_abundance = (intensity_array / max_intensity * 100) if max_intensity > 0 else np.zeros_like(intensity_array)

        # Setup table
        num_rows = len(mz_array)
        self.table_widget.setRowCount(num_rows)
        self.table_widget.setColumnCount(5)
        self.table_widget.setHorizontalHeaderLabels(["m/z", "Intensity", "Rel. Abundance (%)", "Annotation", "Neutral Loss"])

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

            # Create m/z item (NumericTableWidgetItem sorts by UserRole float)
            mz_item = NumericTableWidgetItem()
            mz_item.setData(Qt.ItemDataRole.DisplayRole, f"{mz_val:.4f}")
            mz_item.setData(Qt.ItemDataRole.UserRole, mz_val)  # Store for selection

            # Create intensity item
            intensity_item = NumericTableWidgetItem()
            if intensity_val >= 1000:
                intensity_item.setData(Qt.ItemDataRole.DisplayRole, f"{intensity_val:.2e}")
            else:
                intensity_item.setData(Qt.ItemDataRole.DisplayRole, f"{intensity_val:.2f}")
            intensity_item.setData(Qt.ItemDataRole.UserRole, intensity_val)  # Store for selection

            # Create relative abundance item
            rel_abund_item = NumericTableWidgetItem()
            rel_abund_item.setData(Qt.ItemDataRole.DisplayRole, f"{rel_abund_val:.1f}")
            rel_abund_item.setData(Qt.ItemDataRole.UserRole, rel_abund_val)
            # Bar-delegate data: fraction 0.0-1.0 and colour
            rel_abund_item.setData(BarDelegate.BAR_FRAC_ROLE, rel_abund_val / 100.0)
            rel_abund_item.setData(BarDelegate.BAR_COLOR_ROLE, QColor("steelblue"))

            # Create empty annotation placeholders
            annot_item = QTableWidgetItem("")
            annot_item.setFlags(annot_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            nl_item = QTableWidgetItem("")
            nl_item.setFlags(nl_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self.table_widget.setItem(i, 0, mz_item)
            self.table_widget.setItem(i, 1, intensity_item)
            self.table_widget.setItem(i, 2, rel_abund_item)
            self.table_widget.setItem(i, 3, annot_item)
            self.table_widget.setItem(i, 4, nl_item)

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
        header.resizeSection(3, 200)
        header.resizeSection(4, 200)

        # Set minimum height
        self.table_widget.setMinimumHeight(200)

    def _annotate_fragments(self):
        """Run fragment formula annotation and populate Annotation/Neutral Loss columns."""
        if not self.compound_formula:
            QMessageBox.warning(self, "No Formula", "No compound formula available for annotation.")
            return

        ppm = self.ppm_spinbox.value()
        ion_mode = "negative" if self.adduct and self.adduct.strip().endswith("-") else "positive"

        # Compute absolute Da tolerance relative to the precursor m/z so that
        # the tolerance is not artificially tightened for small fragments.
        precursor_mz = float(self.spectrum_data.get("precursor_mz", 0) or 0)
        tol_da = (precursor_mz * ppm / 1e6) if precursor_mz > 0 else None

        # Optional extra element budget entered by the user
        extra_formula = self.extra_elements_edit.text().strip() or None

        mz_array = np.array(self.spectrum_data["mz"], dtype=float)

        try:
            annotator = FragmentAnnotator()
            annotation_results = annotator.annotate(
                self.compound_formula,
                mz_array,
                ppm=ppm,
                ion_mode=ion_mode,
                adduct_info=self.adduct_info,
                extra_formula=extra_formula,
                tol_da=tol_da,
            )
        except Exception as e:
            QMessageBox.critical(self, "Annotation Error", f"Fragment annotation failed:\n{str(e)}")
            return

        # Build lookup: mz_value -> (frag_display, frag_tooltip, nl_display, nl_tooltip)
        annotation_lookup = {
            r["mz"]: (
                *self._format_annotation_result(r["fragment_formulas"]),
                *self._format_annotation_result(r["neutral_loss_formulas"]),
            )
            for r in annotation_results
        }

        self.table_widget.setSortingEnabled(False)
        for row in range(self.table_widget.rowCount()):
            mz_item = self.table_widget.item(row, 0)
            if mz_item is None:
                continue
            mz_val = mz_item.data(Qt.ItemDataRole.UserRole)
            if mz_val in annotation_lookup:
                frag_str, frag_tip, nl_str, nl_tip = annotation_lookup[mz_val]
                frag_cell = QTableWidgetItem(frag_str)
                frag_cell.setFlags(frag_cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if frag_tip:
                    frag_cell.setToolTip(frag_tip)
                nl_cell = QTableWidgetItem(nl_str)
                nl_cell.setFlags(nl_cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if nl_tip:
                    nl_cell.setToolTip(nl_tip)
                self.table_widget.setItem(row, 3, frag_cell)
                self.table_widget.setItem(row, 4, nl_cell)
        self.table_widget.setSortingEnabled(True)

    @staticmethod
    def _format_annotation_result(candidates):
        """Format annotation candidates for compact table display.

        Returns a tuple ``(display_str, tooltip_str)``.

        * The first (best) hit is always shown in full.
        * When there are additional hits a ``[+N]`` badge is appended to the
          display string, and *tooltip_str* lists all additional formulas (one
          per line) so they can be inspected on hover.
        * *tooltip_str* is an empty string when there is only one hit.
        """
        if not candidates:
            return "", ""
        best_formula, _mass, best_ppm = candidates[0]
        display = f"{best_formula} ({best_ppm:+.1f} ppm)"
        extras = candidates[1:]
        tooltip = ""
        if extras:
            display += f"  [+{len(extras)}]"
            tooltip = "Additional candidates:\n" + "\n".join(f"  {formula} ({ppm_err:+.1f} ppm)" for formula, _m, ppm_err in extras)
        return display, tooltip

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

        # Also update EIC highlighting
        self._update_eic_highlight()

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
            is_highlighted = self.selected_mz is not None and abs(mz - self.selected_mz) < tolerance

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

    # ------------------------------------------------------------------
    # Fragment EIC panel
    # ------------------------------------------------------------------

    def _build_eic_panel(self):
        """Create the right-hand fragment EIC panel widget."""
        panel = QWidget()
        panel.setMinimumWidth(120)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        # ---- Controls row 1: normalize + interpolate zeros ----
        ctrl_row1 = QHBoxLayout()
        self._eic_normalize_cb = QCheckBox("Normalize")
        self._eic_normalize_cb.setChecked(True)
        self._eic_normalize_cb.setToolTip("When checked each fragment EIC is normalized to its most\nabundant signal within the shown RT window.")
        self._eic_normalize_cb.stateChanged.connect(self._draw_fragment_eic_plot)
        ctrl_row1.addWidget(self._eic_normalize_cb)

        self._eic_interp_zeros_cb = QCheckBox("Interp. zeros")
        self._eic_interp_zeros_cb.setChecked(False)
        self._eic_interp_zeros_cb.setToolTip(
            "When checked, zero-intensity points that lie between two non-zero\n"
            "points are replaced by linearly interpolated values.\n"
            "Leading/trailing zeros (before the first or after the last detection)\n"
            "are kept as-is."
        )
        self._eic_interp_zeros_cb.stateChanged.connect(self._draw_fragment_eic_plot)
        ctrl_row1.addWidget(self._eic_interp_zeros_cb)
        ctrl_row1.addStretch()
        vbox.addLayout(ctrl_row1)

        # ---- Controls row 2: RT window + top-N ----
        ctrl_row2 = QHBoxLayout()
        rt_lbl = QLabel("RT ±")
        self._eic_rt_window_sb = NoScrollDoubleSpinBox()
        self._eic_rt_window_sb.setRange(0.05, 30.0)
        self._eic_rt_window_sb.setValue(1.0)
        self._eic_rt_window_sb.setSingleStep(0.25)
        self._eic_rt_window_sb.setDecimals(2)
        self._eic_rt_window_sb.setSuffix(" min")
        self._eic_rt_window_sb.setFixedWidth(90)
        self._eic_rt_window_sb.setToolTip("Half-width of the retention time window centred on this\nspectrum's RT that is shown in the fragment EIC plot.")
        self._eic_rt_window_sb.valueChanged.connect(self._draw_fragment_eic_plot)
        ctrl_row2.addWidget(rt_lbl)
        ctrl_row2.addWidget(self._eic_rt_window_sb)

        topn_lbl = QLabel("Top N:")
        self._eic_top_n_sb = NoScrollSpinBox()
        self._eic_top_n_sb.setRange(0, 999)
        self._eic_top_n_sb.setValue(0)
        self._eic_top_n_sb.setSpecialValueText("all")
        self._eic_top_n_sb.setFixedWidth(60)
        self._eic_top_n_sb.setToolTip("Show only the N most-abundant fragment EICs\n(ranked by peak intensity in the un-normalized EIC).\nSet to 0 to show all fragments.")
        self._eic_top_n_sb.valueChanged.connect(self._draw_fragment_eic_plot)
        ctrl_row2.addWidget(topn_lbl)
        ctrl_row2.addWidget(self._eic_top_n_sb)

        ppm_lbl = QLabel("Bin ±")
        self._eic_ppm_sb = NoScrollDoubleSpinBox()
        self._eic_ppm_sb.setRange(0.1, 500.0)
        self._eic_ppm_sb.setValue(10.0)
        self._eic_ppm_sb.setSingleStep(1.0)
        self._eic_ppm_sb.setDecimals(1)
        self._eic_ppm_sb.setSuffix(" ppm")
        self._eic_ppm_sb.setFixedWidth(90)
        self._eic_ppm_sb.setToolTip("m/z tolerance used when searching for a fragment signal\nin each MS² scan.  Changing this re-triggers the extraction.")
        self._eic_ppm_sb.valueChanged.connect(self._start_fragment_eic_extraction)
        ctrl_row2.addWidget(ppm_lbl)
        ctrl_row2.addWidget(self._eic_ppm_sb)
        ctrl_row2.addStretch()
        vbox.addLayout(ctrl_row2)

        # ---- Progress / status label ----
        self._eic_status_label = QLabel("Extracting fragment EICs…")
        self._eic_status_label.setStyleSheet("color: #5f6368; font-size: 10px;")
        vbox.addWidget(self._eic_status_label)

        # ---- Qt Chart ----
        self._eic_chart = QChart()
        self._eic_chart.setMargins(QMargins(0, 0, 0, 0))
        self._eic_chart.layout().setContentsMargins(0, 0, 0, 0)
        self._eic_chart.legend().setVisible(False)
        self._eic_chart.setTitle("")

        self._eic_x_axis = QValueAxis()
        self._eic_x_axis.setTitleText("RT (min)")
        self._eic_x_axis.setLabelFormat("%.2f")
        self._eic_x_axis.setTickCount(5)
        self._eic_chart.addAxis(self._eic_x_axis, Qt.AlignmentFlag.AlignBottom)

        self._eic_y_axis = QValueAxis()
        self._eic_y_axis.setTitleText("Intensity")
        self._eic_y_axis.setLabelFormat("%.2e")
        self._eic_y_axis.setTickCount(5)
        self._eic_y_axis.setRange(0, 100)
        self._eic_chart.addAxis(self._eic_y_axis, Qt.AlignmentFlag.AlignLeft)

        self._eic_chart_view = InteractiveEICChartView(self._eic_chart)
        self._eic_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._eic_chart_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._eic_chart_view.setToolTip("Left-drag: pan  |  Right-drag: zoom  |  Scroll: zoom  |  Double-click: reset view")
        vbox.addWidget(self._eic_chart_view, stretch=1)

        return panel

    def _start_fragment_eic_extraction(self):
        """Start background extraction of EIC traces for all fragments."""
        mz_array = np.array(self.spectrum_data["mz"], dtype=float)
        if len(mz_array) == 0:
            self._eic_status_label.setText("No fragments to show.")
            return

        precursor_mz = float(self.spectrum_data.get("precursor_mz") or 0.0)
        polarity = self.spectrum_data.get("polarity")

        self._eic_status_label.setText(f"Extracting EICs for {len(mz_array)} fragments…")

        if self._eic_worker is not None:
            self._eic_worker.quit()
            self._eic_worker.wait()

        self._eic_worker = FragmentEICWorker(
            mz_array,
            all_similar_spectra=self.all_similar_spectra,
            mz_tolerance_ppm=self._eic_ppm_sb.value(),
            file_manager=self.file_manager,
            filepath=self.filepath,
            precursor_mz=precursor_mz if precursor_mz > 0 else None,
            polarity=polarity,
            precursor_tolerance=0.01,
            parent=self,
        )
        self._eic_worker.finished.connect(self._on_fragment_eic_ready)
        self._eic_worker.error.connect(lambda msg: self._eic_status_label.setText(f"Error: {msg}"))
        self._eic_worker.start()

    def _on_fragment_eic_ready(self, result):
        """Receive extracted EIC data and render the plot."""
        self._eic_data = result
        n_frags = len(result)
        # Count unique RT points (proxy for number of spectra scanned)
        n_pts = len(next(iter(result.values()))[0]) if result else 0
        if n_frags:
            self._eic_status_label.setText(f"{n_frags} fragment EICs | {n_pts} spectra")
        else:
            self._eic_status_label.setText("No matching fragments found in file.")
        self._draw_fragment_eic_plot()

    # Tab-20 palette as QColor objects (matches matplotlib's default colour cycle)
    _EIC_PALETTE = [
        QColor(0x1F, 0x77, 0xB4),
        QColor(0xAE, 0xC7, 0xE8),
        QColor(0xFF, 0x7F, 0x0E),
        QColor(0xFF, 0xBB, 0x78),
        QColor(0x2C, 0xA0, 0x2C),
        QColor(0x98, 0xDF, 0x8A),
        QColor(0xD6, 0x27, 0x28),
        QColor(0xFF, 0x98, 0x96),
        QColor(0x94, 0x67, 0xBD),
        QColor(0xC5, 0xB0, 0xD5),
        QColor(0x8C, 0x56, 0x4B),
        QColor(0xC4, 0x9C, 0x94),
        QColor(0xE3, 0x77, 0xC2),
        QColor(0xF7, 0xB6, 0xD2),
        QColor(0x7F, 0x7F, 0x7F),
        QColor(0xC7, 0xC7, 0xC7),
        QColor(0xBC, 0xBD, 0x22),
        QColor(0xDB, 0xDB, 0x8D),
        QColor(0x17, 0xBE, 0xCF),
        QColor(0x9E, 0xDA, 0xE5),
    ]

    def _get_eic_colors(self, n):
        """Return *n* QColor objects cycling through the tab-20 palette."""
        palette = self._EIC_PALETTE
        return [palette[i % len(palette)] for i in range(n)]

    @staticmethod
    def _interpolate_zeros(rt_arr, int_arr):
        """Linearly interpolate zero-intensity gaps that lie between two non-zero
        detections.  Leading and trailing zeros (no detection on that side) are
        left untouched.
        """
        arr = int_arr.astype(float).copy()
        nonzero_idx = np.where(arr > 0)[0]
        if len(nonzero_idx) < 2:
            return arr
        first_nz, last_nz = nonzero_idx[0], nonzero_idx[-1]
        inner = np.where((arr == 0.0) & (np.arange(len(arr)) > first_nz) & (np.arange(len(arr)) < last_nz))[0]
        if len(inner):
            arr[inner] = np.interp(rt_arr[inner], rt_arr[nonzero_idx], arr[nonzero_idx])
        return arr

    def _draw_fragment_eic_plot(self):
        """Rebuild the Qt EIC chart from the current _eic_data."""
        if not self._eic_data:
            return

        normalize = self._eic_normalize_cb.isChecked()
        interp_zeros = self._eic_interp_zeros_cb.isChecked()
        rt_center = float(self.spectrum_data.get("rt", 0.0))
        rt_half = self._eic_rt_window_sb.value()
        rt_min = rt_center - rt_half
        rt_max = rt_center + rt_half
        top_n = self._eic_top_n_sb.value()  # 0 means "show all"

        # Remove all old series (axes remain attached to the chart)
        self._eic_chart.removeAllSeries()
        self._eic_series = {}
        self._eic_frag_colors = {}

        sorted_mz = sorted(self._eic_data.keys())

        # --- top-N filtering (by peak intensity in the un-normalized full EIC) ---
        if top_n > 0 and len(sorted_mz) > top_n:
            peak_intensities = {mz: float(np.max(self._eic_data[mz][1])) if len(self._eic_data[mz][1]) else 0.0 for mz in sorted_mz}
            sorted_mz = sorted(sorted_mz, key=lambda m: peak_intensities[m], reverse=True)[:top_n]
            sorted_mz = sorted(sorted_mz)  # restore m/z order for colour consistency

        colors = self._get_eic_colors(len(sorted_mz))

        y_max_global = 0.0
        series_info = []  # (frag_mz, series, base_color, is_selected)

        for base_color, frag_mz in zip(colors, sorted_mz):
            rt_arr, int_arr = self._eic_data[frag_mz]

            # Restrict to the visible RT window
            mask = (rt_arr >= rt_min) & (rt_arr <= rt_max)
            rt_win = rt_arr[mask]
            int_win = int_arr[mask]

            if len(rt_win) == 0:
                continue

            # Optional zero-interpolation
            if interp_zeros:
                int_win = self._interpolate_zeros(rt_win, int_win)

            if normalize:
                peak = float(np.max(int_win)) if np.max(int_win) > 0 else 1.0
                int_win = int_win / peak * 100.0

            y_max_global = max(y_max_global, float(np.max(int_win)))

            is_selected = self.selected_mz is not None and abs(frag_mz - self.selected_mz) < 0.02

            series = QLineSeries()
            series.setName(f"{frag_mz:.4f}")

            for rt_val, int_val in zip(rt_win, int_win):
                series.append(float(rt_val), float(int_val))

            series_info.append((frag_mz, series, QColor(base_color), is_selected))

        # Add non-selected series first so the selected one renders on top
        y_top = y_max_global * 1.1 if y_max_global > 0 else 100.0

        for frag_mz, series, base_color, is_selected in series_info:
            if is_selected:
                continue
            draw_color = QColor(0, 0, 0, 60)  # transparent black
            pen = QPen(draw_color)
            pen.setWidth(1)
            series.setPen(pen)
            self._eic_chart.addSeries(series)
            series.attachAxis(self._eic_x_axis)
            series.attachAxis(self._eic_y_axis)
            self._eic_series[frag_mz] = series
            self._eic_frag_colors[frag_mz] = base_color

        # Now add the selected series last (on top)
        for frag_mz, series, base_color, is_selected in series_info:
            if not is_selected:
                continue
            draw_color = QColor(0, 0, 0, 255)  # solid black
            pen = QPen(draw_color)
            pen.setWidth(3)
            series.setPen(pen)
            self._eic_chart.addSeries(series)
            series.attachAxis(self._eic_x_axis)
            series.attachAxis(self._eic_y_axis)
            self._eic_series[frag_mz] = series
            self._eic_frag_colors[frag_mz] = base_color

        # Vertical marker at the current spectrum's RT
        rt_marker = QLineSeries()
        marker_pen = QPen(QColor(0, 0, 0, 140))
        marker_pen.setWidth(1)
        marker_pen.setStyle(Qt.PenStyle.DashLine)
        rt_marker.setPen(marker_pen)
        rt_marker.append(rt_center, 0.0)
        rt_marker.append(rt_center, y_top)
        self._eic_chart.addSeries(rt_marker)
        rt_marker.attachAxis(self._eic_x_axis)
        rt_marker.attachAxis(self._eic_y_axis)

        # Update axis ranges
        self._eic_x_axis.setRange(rt_min, rt_max)
        self._eic_y_axis.setRange(0, y_top)
        self._eic_y_axis.setTitleText("Rel. Intensity (%)" if normalize else "Intensity")
        lbl_fmt = "%.0f" if normalize else "%.2e"
        self._eic_y_axis.setLabelFormat(lbl_fmt)

        # Store full range so the interactive view can reset on double-click
        self._eic_chart_view.set_full_range(rt_min, rt_max, 0, y_top)

        # Legend only when there are few enough fragments to be legible
        self._eic_chart.legend().setVisible(0 < len(self._eic_series) <= 10)

    def _update_eic_highlight(self):
        """Redraw the EIC chart with the current selection highlighted on top."""
        self._draw_fragment_eic_plot()

    # ------------------------------------------------------------------
    # Helpers shared by copy methods
    # ------------------------------------------------------------------

    def _fragment_table_rows(self):
        """Return (headers, rows) for the fragment table in current display order.

        *headers* is a list of column-header strings.
        *rows* is a list of lists of display strings, one inner list per row.
        """
        tw = self.table_widget
        col_count = tw.columnCount()
        headers = [tw.horizontalHeaderItem(c).text() for c in range(col_count)]
        vh = tw.verticalHeader()
        rows = []
        for visual_row in range(tw.rowCount()):
            logical_row = vh.logicalIndex(visual_row)
            row_data = []
            for col in range(col_count):
                item = tw.item(logical_row, col)
                row_data.append(item.text() if item else "")
            rows.append(row_data)
        return headers, rows

    def _compute_visible_eic_rows(self):
        """Return a list of (frag_mz, rt, intensity) tuples reflecting exactly
        what is currently displayed in the EIC panel:
        RT-window filter, top-N filter, zero-interpolation and normalisation
        are all applied.
        """
        if not self._eic_data:
            return []

        normalize = self._eic_normalize_cb.isChecked()
        interp_zeros = self._eic_interp_zeros_cb.isChecked()
        rt_center = float(self.spectrum_data.get("rt", 0.0))
        rt_half = self._eic_rt_window_sb.value()
        rt_min = rt_center - rt_half
        rt_max = rt_center + rt_half
        top_n = self._eic_top_n_sb.value()

        sorted_mz = sorted(self._eic_data.keys())
        if top_n > 0 and len(sorted_mz) > top_n:
            peak_intensities = {mz: float(np.max(self._eic_data[mz][1])) if len(self._eic_data[mz][1]) else 0.0 for mz in sorted_mz}
            sorted_mz = sorted(sorted(sorted_mz, key=lambda m: peak_intensities[m], reverse=True)[:top_n])

        rows = []
        for frag_mz in sorted_mz:
            rt_arr, int_arr = self._eic_data[frag_mz]
            mask = (rt_arr >= rt_min) & (rt_arr <= rt_max)
            rt_win = rt_arr[mask]
            int_win = int_arr[mask]
            if len(rt_win) == 0:
                continue
            if interp_zeros:
                int_win = self._interpolate_zeros(rt_win, int_win)
            if normalize:
                peak = float(np.max(int_win)) if np.max(int_win) > 0 else 1.0
                int_win = int_win / peak * 100.0
            for rt_val, int_val in zip(rt_win, int_win):
                rows.append((float(frag_mz), float(rt_val), float(int_val)))
        return rows

    # ------------------------------------------------------------------
    # Copy-to-clipboard methods
    # ------------------------------------------------------------------

    @staticmethod
    def _r_escape(s):
        """Escape a Python string for use inside an R double-quoted string."""
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def _copy_fragment_table_tsv(self):
        """Copy the fragment table (as displayed) to clipboard as TSV."""
        headers, rows = self._fragment_table_rows()
        lines = ["\t".join(headers)]
        for row in rows:
            lines.append("\t".join(row))
        QApplication.clipboard().setText("\n".join(lines))

    def _copy_fragment_table_r(self):
        """Copy the fragment table (as displayed) to clipboard as an R data.frame."""
        headers, rows = self._fragment_table_rows()
        if not rows:
            QApplication.clipboard().setText("# No fragment data available.")
            return

        col_count = len(headers)

        # Decide type per column: numeric if every non-empty value parses as float
        def is_numeric_col(col_idx):
            for row in rows:
                val = row[col_idx].strip()
                if val == "":
                    continue
                try:
                    float(val)
                except ValueError:
                    return False
            return True

        numeric_flags = [is_numeric_col(c) for c in range(col_count)]

        col_lines = []
        for c, (hdr, is_num) in enumerate(zip(headers, numeric_flags)):
            col_name = hdr.replace("/", "").replace(".", "").replace(" ", "_").replace(".", ".").replace("(", "").replace(")", "").replace("%", "pct")
            values = [row[c] for row in rows]
            if is_num:
                r_vals = ", ".join(v if v.strip() else "NA" for v in values)
            else:
                r_vals = ", ".join(f'"{self._r_escape(v)}"' for v in values)
            col_lines.append(f"  {col_name} = c({r_vals})")

        r_code = "fragments <- data.frame(\n" + ",\n".join(col_lines) + ",\n  stringsAsFactors = FALSE\n)"
        QApplication.clipboard().setText(r_code)

    def _copy_eic_tsv(self):
        """Copy the EIC data (as currently displayed) to clipboard as long-format TSV."""
        rows = self._compute_visible_eic_rows()
        normalize = self._eic_normalize_cb.isChecked()
        intensity_header = "rel_intensity_pct" if normalize else "intensity"
        lines = [f"mz\trt_min\t{intensity_header}"]
        for frag_mz, rt_val, int_val in rows:
            lines.append(f"{frag_mz:.6f}\t{rt_val:.6f}\t{int_val:.6g}")
        if len(lines) == 1:
            lines.append("# No EIC data available.")
        QApplication.clipboard().setText("\n".join(lines))

    def _copy_eic_r(self):
        """Copy the EIC data (as currently displayed) to clipboard as R data.frame."""
        rows = self._compute_visible_eic_rows()
        if not rows:
            QApplication.clipboard().setText("# No EIC data available.")
            return
        normalize = self._eic_normalize_cb.isChecked()
        intensity_col = "rel_intensity_pct" if normalize else "intensity"
        mz_vals = ", ".join(f"{r[0]:.6f}" for r in rows)
        rt_vals = ", ".join(f"{r[1]:.6f}" for r in rows)
        int_vals = ", ".join(f"{r[2]:.6g}" for r in rows)
        r_code = f"eic <- data.frame(\n  mz = c({mz_vals}),\n  rt_min = c({rt_vals}),\n  {intensity_col} = c({int_vals}),\n  stringsAsFactors = FALSE\n)"
        QApplication.clipboard().setText(r_code)

    # ------------------------------------------------------------------

    def _open_comparison_window(self, second_spectrum=None, second_filename=None):
        """Open the USI Spectrum Comparator window pre-populated with this spectrum."""
        if not hasattr(self, "_comparison_windows"):
            self._comparison_windows = []
        win = USISpectrumComparisonWindow(
            spectrum_a=self.spectrum_data,
            filename_a=self.filename,
            spectrum_b=second_spectrum,
            filename_b=second_filename,
            file_manager=self.file_manager,
        )
        self._comparison_windows.append(win)
        win.destroyed.connect(lambda _, w=win: self._comparison_windows.remove(w) if w in self._comparison_windows else None)
        win.show()

    def create_large_msms_chart(self):
        """Create a large MSMS spectrum chart"""
        chart = QChart()

        # Get precursor intensity for display
        precursor_intensity = self.spectrum_data.get("precursor_intensity", 0)
        intensity_text = f"{precursor_intensity:.1e}" if precursor_intensity > 0 else "N/A"

        ce = self.spectrum_data.get("collision_energy")
        ce_text = _format_collision_energy(ce)
        chart.setTitle(
            f"MSMS Spectrum — {self._usi}\nRT: {self.spectrum_data['rt']:.4f} min, Precursor: {self.spectrum_data['precursor_mz']:.4f}, Intensity: {intensity_text}{ce_text}"
            + (f"\nScan: {self.spectrum_data['scan_id']}" if self.spectrum_data.get("scan_id") else "")
            + (f" | Filter: {self.spectrum_data['filter_string']}" if self.spectrum_data.get("filter_string") else "")
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


class InteractiveMSMSChartView(QChartView):
    """Interactive chart view for MSMS spectra with pan and zoom capabilities"""

    def __init__(self, chart):
        super().__init__(chart)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Spectrum data for popup display
        self.spectrum_data = None
        self.filename = None
        self.group = None
        self.compound_formula = None
        self.adduct = None
        self.adduct_info = None
        self.compound_smiles = None
        self.filepath = None
        self.file_manager = None
        self.all_similar_spectra = None

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

        # Keep references to popup windows so they are not garbage-collected
        self._popup_windows = []

        # Hover tooltip label for m/z values
        self.hover_label = QLabel(self)
        self.hover_label.setStyleSheet("""
            QLabel {
                background-color: rgba(60, 64, 67, 220);
                color: #ffffff;
                border: none;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 11px;
            }
        """)
        self.hover_label.hide()
        self.hover_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hover_mz = None
        self._hover_norm_int = 0.0
        self._hover_series = None  # Firebrick stick drawn over the hovered peak
        self._press_pos = None
        self._pinned_annotations: list = []  # [(mz, norm_int, color), ...]
        self._ann_labels: list = []
        self._right_press_start = None
        self._right_click_mz: float | None = None
        self._right_click_norm_int: float = 0.0

    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Left click: start panning
            self.is_panning = True
            self.pan_start_pos = event.position()
            self._press_pos = event.position()
            self.last_mouse_pos = event.position()

            # Store current ranges
            x_axis = self.chart().axes(Qt.Orientation.Horizontal)[0]
            y_axis = self.chart().axes(Qt.Orientation.Vertical)[0]
            self.interaction_start_x_range = (x_axis.min(), x_axis.max())
            self.interaction_start_y_range = (y_axis.min(), y_axis.max())

            self.setCursor(Qt.CursorShape.ClosedHandCursor)

        elif event.button() == Qt.MouseButton.RightButton:
            # Right click: start zooming
            self._right_press_start = event.position()
            self._right_click_mz = self._hover_mz
            self._right_click_norm_int = self._hover_norm_int
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

        # Update hover tooltip when not interacting
        if not self.is_panning and not self.is_zooming:
            self._update_hover_tooltip(event)

        if self._pinned_annotations:
            self._redraw_annotation_labels()
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
            t = max(0.0, min(1.0, ((mx - tx) * dx + (my - ty) * dy) / seg_len_sq)) if seg_len_sq > 0 else 0.0
            dist = ((mx - tx - t * dx) ** 2 + (my - ty - t * dy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_mz = mz_f
                best_norm_int = norm_int

        if best_mz is not None and best_dist <= PIXEL_THRESHOLD:
            if best_mz != self._hover_mz:
                self._hover_mz = best_mz
                self._hover_norm_int = best_norm_int

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
            if self._press_pos is not None and self._hover_mz is not None:
                dp = event.position() - self._press_pos
                if dp.x() ** 2 + dp.y() ** 2 < 25.0:
                    self._toggle_pin(self._hover_mz, self._hover_norm_int, "#0064c8")
            self._press_pos = None
            self.is_panning = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_zooming = False
            if self._right_press_start is not None:
                dp = event.position() - self._right_press_start
                if dp.x() ** 2 + dp.y() ** 2 < 25.0:
                    self._show_annotate_context_menu(event, self._right_click_mz, self._right_click_norm_int)
            self._right_press_start = None

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pinned_annotations:
            self._redraw_annotation_labels()

    def _toggle_pin(self, mz: float, norm_int: float, color: str = "#0064c8"):
        """Pin or unpin a permanent m/z label at the given peak."""
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

    def _show_annotate_context_menu(self, event, mz: float | None, norm_int: float):
        """Show context menu. Annotate submenu is only shown when a peak is highlighted."""
        menu = QMenu(self)

        # -- Annotate section (only when a peak is under the cursor) --
        if mz is not None:
            ann_menu = menu.addMenu("Annotate")
            for name, color_str in ANNOTATION_COLOR_PRESETS:
                act = ann_menu.addAction(name)
                act.triggered.connect(lambda _c=False, m=mz, n=norm_int, c=color_str: self._toggle_pin(m, n, c))

        # -- USI section --
        usi = getattr(self, "_usi", None)
        if usi:
            menu.addSeparator()
            usi_action = menu.addAction(f"USI: {usi}")
            usi_action.setEnabled(False)
            copy_usi_action = menu.addAction("Copy USI to Clipboard")
            copy_usi_action.triggered.connect(lambda _c=False, u=usi: QApplication.clipboard().setText(u))

        # -- Open comparison window --
        open_comp_fn = getattr(self, "_open_comparison_fn", None)
        if open_comp_fn is not None:
            menu.addSeparator()
            comp_action = menu.addAction("Open in Spectrum Comparator")
            comp_action.triggered.connect(lambda _c=False: open_comp_fn())

        menu.exec(event.globalPosition().toPoint())

    def _redraw_annotation_labels(self):
        """Reposition all pinned annotation labels to match current chart coordinates."""
        x_axes = self.chart().axes(Qt.Orientation.Horizontal)
        x_axis = x_axes[0] if x_axes else None
        for lbl, (mz, norm_int, _color) in zip(self._ann_labels, self._pinned_annotations):
            tip_pos = self.chart().mapToPosition(QPointF(mz, norm_int))
            lx = int(tip_pos.x()) - lbl.width() // 2
            ly = int(tip_pos.y()) - lbl.height() - 6
            lx = max(0, min(lx, self.width() - lbl.width()))
            if ly < 0:
                ly = int(tip_pos.y()) + 6
            lbl.move(lx, ly)
            if x_axis is not None and x_axis.min() <= mz <= x_axis.max():
                lbl.show()
            else:
                lbl.hide()

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

        if self._pinned_annotations:
            self._redraw_annotation_labels()

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
                # Create as a top-level window (parent=None) so it is not
                # hidden/closed when the MSMS overview window is minimised
                # or closed.
                popup = MSMSPopupWindow(
                    self.spectrum_data,
                    self.filename,
                    self.group,
                    None,
                    compound_formula=self.compound_formula,
                    adduct=self.adduct,
                    adduct_info=self.adduct_info,
                    compound_smiles=self.compound_smiles,
                    filepath=self.filepath,
                    file_manager=self.file_manager,
                    all_similar_spectra=self.all_similar_spectra,
                )
                self._popup_windows.append(popup)
                popup.destroyed.connect(lambda _, p=popup: self._popup_windows.remove(p) if p in self._popup_windows else None)
                popup.show()
        super().mouseDoubleClickEvent(event)


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
        compound_formula=None,
        adduct_info=None,
        compound_smiles=None,
    ):
        super().__init__(parent)

        # Configure as independent window
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.msms_spectra = msms_spectra
        self.target_mz = target_mz
        self.rt_center = rt_center
        self.rt_window = rt_window
        self.compound_name = compound_name
        self.adduct = adduct
        self.compound_formula = compound_formula
        self.adduct_info = adduct_info  # dict with ElementsAdded / ElementsLost / Charge / Multiplier
        self.compound_smiles = compound_smiles
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
            f"MSMS: {self.compound_name} ({self.adduct}) | m/z {self.target_mz:.5f} | RT {self.rt_center:.2f}\u00b1{self.rt_window * 60:.0f} s{type_tag} | {len(self.msms_spectra)} files | {total_spectra} spectra"
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

        scroll_vbox = QVBoxLayout(scroll_widget)
        scroll_vbox.setSpacing(2)
        scroll_vbox.setContentsMargins(2, 2, 2, 2)

        # Organize files by group first
        groups_dict = {}
        for filepath, file_data in self.processed_data:
            group = file_data.get("group", "Unknown")
            if group not in groups_dict:
                groups_dict[group] = []
            groups_dict[group].append((filepath, file_data))

        # Create a collapsible box per group
        for group, group_files in groups_dict.items():
            grp_color = self._group_color_for(group)
            # Build group header title with file count
            group_box = CollapsibleBox(f"{group}  ({len(group_files)} file(s))")
            # Color the header button background
            if grp_color:
                c = QColor(grp_color)
                c.setAlphaF(0.3)
                r, g, b, a = c.red(), c.green(), c.blue(), c.alpha()
                bg_css = f"rgba({r},{g},{b},{a})"
            else:
                bg_css = "#e8eaed"
            group_box.toggle_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg_css};
                    border: none;
                    padding: 4px 6px;
                    text-align: left;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {bg_css};
                    border: 1px solid #aaa;
                }}
            """)
            group_box.set_expanded(True)  # expanded by default

            # Inner grid for files within this group
            inner_widget = QWidget()
            grid_layout = QGridLayout(inner_widget)
            grid_layout.setSpacing(1)
            grid_layout.setContentsMargins(1, 1, 1, 1)
            grid_layout.setVerticalSpacing(0)

            row = 0

            # Separate files with 1 spectrum from files with multiple spectra
            multi_spectra_files = [(fp, fd) for fp, fd in group_files if len(fd["spectra"]) > 1]
            single_spectra_files = [(fp, fd) for fp, fd in group_files if len(fd["spectra"]) == 1]

            SINGLE_SPECTRA_PER_ROW = 3

            def _add_file_to_grid(filepath, file_data, start_row, start_col, col_span):
                """Add one file's header label and spectrum charts to the grid layout.

                Args:
                    filepath: Absolute path to the source file.
                    file_data: Dict with 'filename' and 'spectra' keys.
                    start_row: Grid row for the header; charts go in start_row + 1.
                    start_col: Starting grid column for this file's content.
                    col_span: Number of columns the header label should span.
                """
                filename = file_data["filename"]
                spectra = file_data["spectra"]

                similarity_info = ""
                if filename in self.intra_file_similarities:
                    stats = self.intra_file_similarities[filename]
                    similarity_info = f" | Sim: Med:{stats['median']:.3f} 90%:{stats['percentile_90']:.3f}"

                display_name = filename.split(".")[0] if "." in filename else filename
                file_header_text = f"<b>{display_name}</b> | {len(spectra)} spectra{similarity_info}"
                file_label = QLabel(file_header_text)

                if grp_color:
                    c = QColor(grp_color)
                    c.setAlphaF(0.35)
                    r, g, b, a = c.red(), c.green(), c.blue(), c.alpha()
                    file_bg_css = f"rgba({r},{g},{b},{a})"
                else:
                    file_bg_css = "#f1f3f4"

                file_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {file_bg_css};
                        padding: 2px 4px;
                        margin: 0px;
                        border-bottom: 1px solid #dadce0;
                    }}
                """)
                file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
                file_label.setMaximumHeight(18)
                grid_layout.addWidget(file_label, start_row, start_col, 1, col_span)

                for col_offset, spectrum_data in enumerate(spectra):
                    chart_widget = self.create_msms_chart(spectrum_data, filename, group, filepath=filepath)
                    chart_widget.all_similar_spectra = spectra
                    grid_layout.addWidget(chart_widget, start_row + 1, start_col + col_offset)

            # Add multi-spectrum files (one per row)
            for filepath, file_data in multi_spectra_files:
                n_spectra = len(file_data["spectra"])
                _add_file_to_grid(filepath, file_data, row, 0, n_spectra)
                row += 2

            # Add single-spectrum files in groups of SINGLE_SPECTRA_PER_ROW
            for batch_start in range(0, len(single_spectra_files), SINGLE_SPECTRA_PER_ROW):
                batch = single_spectra_files[batch_start: batch_start + SINGLE_SPECTRA_PER_ROW]
                for col_offset, (filepath, file_data) in enumerate(batch):
                    _add_file_to_grid(filepath, file_data, row, col_offset, 1)
                row += 2

            group_box.add_widget(inner_widget)
            scroll_vbox.addWidget(group_box)

        scroll_vbox.addStretch()

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
        group_tols = {entry["filter_type"]: float(entry["mz_tolerance"]) for entry in defaults.get("msms_similarity_group_tolerances", [])}
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
                        sim = calculate_cosine_similarity(spectra[i], spectra[j], mz_tolerance=tol, method=method)
                        similarities.append(sim)

            self.intra_file_similarities[filename] = calculate_similarity_statistics(similarities)

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
                        sim = calculate_cosine_similarity(spec1, spec2, mz_tolerance=tol, method=method)
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
        files = [data[1]["filename"] for data in self.processed_data]
        file_group = {data[1]["filename"]: data[1].get("group", "Unknown") for data in self.processed_data}
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
                        c.setAlphaF(0.85)
                        hi.setBackground(c)
                        # Choose white or black text based on background luminance
                        lum = 0.299 * c.redF() + 0.587 * c.greenF() + 0.114 * c.blueF()
                        text_color = QColor("black") if lum > 0.5 else QColor("white")
                        hi.setForeground(text_color)
                    fnt = hi.font()
                    fnt.setBold(True)
                    hi.setFont(fnt)
                    if make_h:
                        inter_table.setHorizontalHeaderItem(idx, hi)
                    else:
                        inter_table.setVerticalHeaderItem(idx, hi)

            inter_table.setMinimumHeight(min(120, 24 + len(files) * 20))
            inter_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            # Set table properties for better display
            inter_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            inter_table.horizontalHeader().setDefaultSectionSize(70)
            inter_table.verticalHeader().setDefaultSectionSize(18)

            # Enable context menu
            inter_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            inter_table.customContextMenuRequested.connect(lambda pos: self._show_similarity_context_menu(inter_table, pos))

            # Single click opens the two spectra in individual viewer windows
            inter_table.cellClicked.connect(self._on_similarity_cell_clicked)

            # Store reference to inter_table for context menu handler
            self.inter_table = inter_table

            # Fill the table
            for i, file1 in enumerate(files):
                for j, file2 in enumerate(files):
                    similarities = []  # Initialize similarities for all cases

                    if i == j:
                        # Diagonal - same file vs itself: always 1.000 (perfect match)
                        item = QTableWidgetItem("1.000")
                        item.setBackground(QColor(76, 175, 80, 200))  # Strong Green
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
                            item.setBackground(QColor(255, 152, 0, 200))  # Strong Orange
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
            legend_label = QLabel("<small><b>Legend:</b> Values show median cosine similarity | Diagonal = Same file (1.00) | Off-diagonal = Inter-file similarity</small>")
            legend_label.setFixedHeight(15)
            legend_label.setStyleSheet("QLabel { font-size: 10px; }")
            layout.addWidget(legend_label)
        else:
            no_comparison_label = QLabel("<i>At least 2 files needed for inter-file comparison</i>")
            layout.addWidget(no_comparison_label)

        return overview_widget

    def create_msms_chart(self, spectrum_data, filename, group, filepath=None):
        """Create a chart widget for a single MSMS spectrum"""
        # Create chart
        chart = QChart()

        # Get precursor intensity for display
        precursor_intensity = spectrum_data.get("precursor_intensity", 0)
        intensity_text = f"{precursor_intensity:.1e}" if precursor_intensity > 0 else "N/A"

        ce = spectrum_data.get("collision_energy")
        ce_text = _format_collision_energy(ce)
        usi = make_usi(spectrum_data, filename)
        chart.setTitle(
            f"{usi}\nRT: {spectrum_data['rt']:.2f} min | Precursor: {spectrum_data['precursor_mz']:.4f} | Intensity: {intensity_text}{ce_text}"
            + (f"\nScan: {spectrum_data['scan_id']}" if spectrum_data.get("scan_id") else "")
            + (f" | Filter: {spectrum_data['filter_string']}" if spectrum_data.get("filter_string") else "")
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
        chart_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Store spectrum data for popup display
        chart_view.spectrum_data = spectrum_data
        chart_view.filename = filename
        chart_view.group = group
        chart_view.compound_formula = self.compound_formula
        chart_view.adduct = self.adduct
        chart_view.adduct_info = self.adduct_info
        chart_view.compound_smiles = self.compound_smiles
        chart_view.filepath = filepath
        chart_view.file_manager = self.file_manager
        chart_view._usi = usi
        chart_view._open_comparison_fn = lambda s=spectrum_data, fn=filename: self._open_comparison_from_viewer(s, fn)
        # all_similar_spectra will be set by the caller
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
                background-color: #ffffff;
                border: 1px solid #dadce0;
                padding: 4px 0;
            }
            QMenu::item {
                padding: 5px 20px 5px 12px;
                color: #202124;
            }
            QMenu::item:selected {
                background-color: #e8f0fe;
                color: #1a73e8;
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
            info_text += f"Intensity = {file1_spectra[0].get('precursor_intensity', 0):.2e}<br>"
            info_text += f"  Spectrum B: m/z = {file2_spectra[0]['precursor_mz']:.4f}, "
            info_text += f"Intensity = {file2_spectra[0].get('precursor_intensity', 0):.2e}<br>"

        info_section.setText(info_text)
        info_section.setStyleSheet("padding: 10px; background-color: #f3f3f3; color: #202124;")
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
            mirror_action.triggered.connect(lambda: self._show_mirror_plot(file1_data, file2_data, is_diagonal))
            menu.addAction(mirror_action)

        # Show menu at cursor position
        menu.exec(table.viewport().mapToGlobal(pos))

    def _show_mirror_plot(self, file1_data, file2_data, is_diagonal):
        """Show mirror plot dialog for two files or two spectra from same file"""
        if is_diagonal and len(file1_data["spectra"]) >= 2:
            # For diagonal, compare first two spectra from the same file
            spectrum_a = file1_data["spectra"][0]
            spectrum_b = file1_data["spectra"][1]
        else:
            # For off-diagonal, compare first spectrum from each file
            spectrum_a = file1_data["spectra"][0] if file1_data["spectra"] else None
            spectrum_b = file2_data["spectra"][0] if file2_data["spectra"] else None

            if not spectrum_a or not spectrum_b:
                QMessageBox.warning(self, "No Spectra", "No spectra available for comparison.")
                return

        if not hasattr(self, "_open_popups"):
            self._open_popups = []
        win = USISpectrumComparisonWindow(
            spectrum_a=spectrum_a,
            filename_a=file1_data["filename"],
            spectrum_b=spectrum_b,
            filename_b=file2_data["filename"],
            file_manager=self.file_manager,
            mz_tolerance=self._get_pair_mz_tolerance(spectrum_a, spectrum_b),
            method=self.defaults.get("msms_similarity_method", "CosineHungarian"),
        )
        self._open_popups.append(win)
        win.destroyed.connect(lambda: self._open_popups.remove(win) if win in self._open_popups else None)
        win.show()

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
        file1_data = next((d for _, d in self.processed_data if d["filename"] == file1_name), None)
        file2_data = next((d for _, d in self.processed_data if d["filename"] == file2_name), None)
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
        else:
            if not spectra1 or not spectra2:
                QMessageBox.warning(self, "No Spectra", "No spectra available.")
                return
            spec_a = spectra1[0]
            spec_b = spectra2[0]
            name_a = file1_data["filename"]
            name_b = file2_data["filename"]

        if not hasattr(self, "_open_popups"):
            self._open_popups = []
        win = USISpectrumComparisonWindow(
            spectrum_a=spec_a,
            filename_a=name_a,
            spectrum_b=spec_b,
            filename_b=name_b,
            file_manager=self.file_manager,
            mz_tolerance=self._get_pair_mz_tolerance(spec_a, spec_b),
            method=self.defaults.get("msms_similarity_method", "CosineHungarian"),
        )
        self._open_popups.append(win)
        win.destroyed.connect(lambda: self._open_popups.remove(win) if win in self._open_popups else None)
        win.show()

    def _open_comparison_from_viewer(self, spectrum_data, filename):
        """Open a USISpectrumComparisonWindow pre-populated with one spectrum."""
        if not hasattr(self, "_open_popups"):
            self._open_popups = []
        win = USISpectrumComparisonWindow(
            spectrum_a=spectrum_data,
            filename_a=filename,
            file_manager=self.file_manager,
        )
        self._open_popups.append(win)
        win.destroyed.connect(lambda: self._open_popups.remove(win) if win in self._open_popups else None)
        win.show()

    def closeEvent(self, event):
        """Clean up when closing the window"""
        if hasattr(self, "extraction_worker") and self.extraction_worker.isRunning():
            self.extraction_worker.quit()
            self.extraction_worker.wait()
        event.accept()


class MirrorPlotChartView(InteractiveEICChartView):
    """Interactive chart view for the mirror (pairwise) MSMS plot.

    Extends InteractiveEICChartView with:
    - Hover: highlights the closest signal (A or B), shows its m/z label
      above the peak tip, and selects the matching table row.
    - Left-click on a peak: pins/unpins a permanent m/z annotation.
    - Right-click on a peak: colour submenu (like InteractiveMSMSChartView).
    - Pinned label for matched pairs shows both A and B m/z values.
    """

    _ROW_IDX_ROLE = Qt.ItemDataRole.UserRole + 1
    _PIXEL_THRESHOLD = 20

    def __init__(self, chart, parent=None):
        super().__init__(chart, parent)

        # Spectrum arrays — set by EnhancedMirrorPlotWindow after construction
        self._mz_a: np.ndarray = np.array([], dtype=float)
        self._rel_a: np.ndarray = np.array([], dtype=float)  # positive  (0–100)
        self._mz_b: np.ndarray = np.array([], dtype=float)
        self._rel_b: np.ndarray = np.array([], dtype=float)  # positive; displayed as negative

        # Fragment rows and table widget refs
        self._rows: list = []
        self._table = None  # QTableWidget

        # Hover state
        self._hover_mz: float | None = None
        self._hover_chart_y: float = 0.0
        self._hover_series = None
        self._press_pos: QPointF | None = None
        self._right_press_start: QPointF | None = None
        self._right_click_mz: float | None = None
        self._right_click_chart_y: float = 0.0

        # Hover tooltip label
        self.hover_label = QLabel(self)
        self.hover_label.setStyleSheet("QLabel {  background-color: rgba(60,64,67,220);  color: #ffffff;  border: none; border-radius: 3px;  padding: 2px 6px; font-size: 11px;}")
        self.hover_label.hide()
        self.hover_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Pinned annotations: list of (mz, chart_y, color, label_text)
        self._pinned_annotations: list = []
        self._ann_labels: list = []

        # USI strings — set by EnhancedMirrorPlotWindow after construction
        self._usi_a: str | None = None
        self._usi_b: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_closest_peak(self, mx: float, my: float):
        """Return (mz, chart_y, from_a) for peak whose stick is closest
        to pixel (mx, my), or (None, 0.0, True) when nothing is within
        _PIXEL_THRESHOLD pixels."""
        best_mz = None
        best_chart_y = 0.0
        best_from_a = True
        best_dist = float("inf")

        for mz_arr, rel_arr, positive in (
            (self._mz_a, self._rel_a, True),
            (self._mz_b, self._rel_b, False),
        ):
            for mz_v, rel_v in zip(mz_arr, rel_arr):
                chart_y = float(rel_v) if positive else -float(rel_v)
                tip = self.chart().mapToPosition(QPointF(float(mz_v), chart_y))
                base = self.chart().mapToPosition(QPointF(float(mz_v), 0.0))
                tx, ty, bx, by = tip.x(), tip.y(), base.x(), base.y()
                dx, dy = bx - tx, by - ty
                seg_sq = dx * dx + dy * dy
                t = max(0.0, min(1.0, ((mx - tx) * dx + (my - ty) * dy) / seg_sq)) if seg_sq > 0 else 0.0
                dist = ((mx - tx - t * dx) ** 2 + (my - ty - t * dy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_mz = float(mz_v)
                    best_chart_y = chart_y
                    best_from_a = positive

        if best_mz is not None and best_dist <= self._PIXEL_THRESHOLD:
            return best_mz, best_chart_y, best_from_a
        return None, 0.0, True

    def _row_for_peak(self, mz: float, from_a: bool):
        """Return (row_dict, orig_row_idx) for the matching _rows entry."""
        key = "mz_a" if from_a else "mz_b"
        for idx, r in enumerate(self._rows):
            row_mz = r.get(key)
            if row_mz is not None and abs(float(row_mz) - mz) < 1e-6:
                return r, idx
        return None, -1

    def _visual_row_for_orig(self, orig_idx: int) -> int:
        """Find the visual (possibly sorted) table row with the given orig_idx."""
        if self._table is None:
            return -1
        for vrow in range(self._table.rowCount()):
            it = self._table.item(vrow, 0)
            if it is not None and it.data(self._ROW_IDX_ROLE) == orig_idx:
                return vrow
        return -1

    def _label_text(self, mz: float, row_dict, from_a: bool) -> str:
        """Build label text; for matched pairs includes both A and B m/z."""
        if row_dict is not None:
            mz_a = row_dict.get("mz_a")
            mz_b = row_dict.get("mz_b")
            if mz_a is not None and mz_b is not None:
                return f"{float(mz_a):.4f}\n{float(mz_b):.4f}"
        side = "\u2191" if from_a else "\u2193"
        return f"{side} {mz:.4f}"

    def _place_label(self, lbl: QLabel, mz: float, chart_y: float):
        """Position *lbl* just above the tip of the signal in screen space."""
        tip = self.chart().mapToPosition(QPointF(mz, chart_y))
        lx = int(tip.x()) - lbl.width() // 2
        ly = int(tip.y()) - lbl.height() - 6
        lx = max(0, min(lx, self.width() - lbl.width()))
        if ly < 0:
            ly = int(tip.y()) + 6
        lbl.move(lx, ly)

    # ------------------------------------------------------------------
    # Mouse events (override base class where needed)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position()
        elif event.button() == Qt.MouseButton.RightButton:
            self._right_press_start = event.position()
            self._right_click_mz = self._hover_mz
            self._right_click_chart_y = self._hover_chart_y
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)  # pan / zoom from base
        if not self._panning and not self._zooming:
            self._update_hover(event)
        if self._pinned_annotations:
            self._redraw_annotation_labels()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._press_pos is not None and self._hover_mz is not None:
                dp = event.position() - self._press_pos
                if dp.x() ** 2 + dp.y() ** 2 < 25.0:
                    from_a = self._hover_chart_y >= 0
                    row_dict, _ = self._row_for_peak(self._hover_mz, from_a)
                    lbl_text = self._label_text(self._hover_mz, row_dict, from_a)
                    self._toggle_pin(self._hover_mz, self._hover_chart_y, "#0064c8", lbl_text)
            self._press_pos = None
        elif event.button() == Qt.MouseButton.RightButton:
            if self._right_press_start is not None:
                dp = event.position() - self._right_press_start
                if dp.x() ** 2 + dp.y() ** 2 < 25.0:
                    self._show_context_menu(event, self._right_click_mz, self._right_click_chart_y)
            self._right_press_start = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._clear_hover()
        super().leaveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pinned_annotations:
            self._redraw_annotation_labels()

    # ------------------------------------------------------------------
    # Hover logic
    # ------------------------------------------------------------------

    def _update_hover(self, event):
        """Highlight the nearest peak, update the m/z label, select table row."""
        plot_area = self.chart().plotArea()
        if not plot_area.contains(event.position()):
            self._clear_hover()
            return

        mz, chart_y, from_a = self._find_closest_peak(event.position().x(), event.position().y())
        if mz is None:
            self._clear_hover()
            return

        if mz != self._hover_mz:
            self._hover_mz = mz
            self._hover_chart_y = chart_y

            # Replace hover series (firebrick stick)
            if self._hover_series is not None:
                self.chart().removeSeries(self._hover_series)
                self._hover_series = None
            s = QLineSeries()
            pen = QPen(QColor(178, 34, 34))
            pen.setWidth(3)
            s.setPen(pen)
            s.append(mz, 0.0)
            s.append(mz, chart_y)
            s.append(mz, 0.0)
            self.chart().addSeries(s)
            x_axes = self.chart().axes(Qt.Orientation.Horizontal)
            y_axes = self.chart().axes(Qt.Orientation.Vertical)
            if x_axes and y_axes:
                s.attachAxis(x_axes[0])
                s.attachAxis(y_axes[0])
            # Hide the hover series from the legend
            markers = self.chart().legend().markers(s)
            if markers:
                markers[0].setVisible(False)
            self._hover_series = s

            # Label text
            row_dict, orig_idx = self._row_for_peak(mz, from_a)
            self.hover_label.setText(self._label_text(mz, row_dict, from_a))
            self.hover_label.adjustSize()

            # Select matching table row
            if orig_idx >= 0:
                vrow = self._visual_row_for_orig(orig_idx)
                if vrow >= 0 and self._table is not None:
                    self._table.selectRow(vrow)
                    self._table.scrollTo(self._table.model().index(vrow, 0))

        self._place_label(self.hover_label, mz, chart_y)
        self.hover_label.show()

    def _clear_hover(self):
        """Remove the hover highlight and hide the tooltip label."""
        if self._hover_series is not None:
            self.chart().removeSeries(self._hover_series)
            self._hover_series = None
        self._hover_mz = None
        self.hover_label.hide()

    # ------------------------------------------------------------------
    # Pinned annotations
    # ------------------------------------------------------------------

    def _toggle_pin(self, mz: float, chart_y: float, color: str, label_text: str | None = None):
        """Pin or unpin a permanent coloured m/z label at the peak."""
        for i, (m, *_) in enumerate(self._pinned_annotations):
            if abs(m - mz) < 1e-9:
                self._pinned_annotations.pop(i)
                lbl = self._ann_labels.pop(i)
                lbl.deleteLater()
                return
        if label_text is None:
            from_a = chart_y >= 0
            row_dict, _ = self._row_for_peak(mz, from_a)
            label_text = self._label_text(mz, row_dict, from_a)
        self._pinned_annotations.append((mz, chart_y, color, label_text))
        lbl = QLabel(label_text, self)
        lbl.setStyleSheet(f"color: {color}; background-color: rgba(255,255,255,200); border: none; border-radius: 2px; font-size: 11px; font-weight: bold; padding: 1px 4px;")
        lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lbl.adjustSize()
        self._ann_labels.append(lbl)
        lbl.show()
        self._redraw_annotation_labels()

    def _show_context_menu(self, event, mz: float | None, chart_y: float):
        """Context menu. Annotate submenu is only shown when a peak is highlighted."""
        menu = QMenu(self)

        # -- Annotate section (only when a peak is under the cursor) --
        if mz is not None:
            from_a = chart_y >= 0
            row_dict, _ = self._row_for_peak(mz, from_a)
            label_text = self._label_text(mz, row_dict, from_a)
            ann_menu = menu.addMenu("Annotate")
            for name, color_str in ANNOTATION_COLOR_PRESETS:
                act = ann_menu.addAction(name)
                act.triggered.connect(lambda _c=False, m=mz, cy=chart_y, c=color_str, lt=label_text: self._toggle_pin(m, cy, c, lt))

        # -- USI section --
        if self._usi_a or self._usi_b:
            menu.addSeparator()
        if self._usi_a:
            lbl_a = menu.addAction(f"USI A: {self._usi_a}")
            lbl_a.setEnabled(False)
            copy_a = menu.addAction("Copy USI A to Clipboard")
            copy_a.triggered.connect(lambda _c=False, u=self._usi_a: QApplication.clipboard().setText(u))
        if self._usi_b:
            lbl_b = menu.addAction(f"USI B: {self._usi_b}")
            lbl_b.setEnabled(False)
            copy_b = menu.addAction("Copy USI B to Clipboard")
            copy_b.triggered.connect(lambda _c=False, u=self._usi_b: QApplication.clipboard().setText(u))

        menu.exec(event.globalPosition().toPoint())

    def _redraw_annotation_labels(self):
        """Reposition all pinned labels to match current chart coordinates."""
        x_axes = self.chart().axes(Qt.Orientation.Horizontal)
        x_axis = x_axes[0] if x_axes else None
        for lbl, (mz, chart_y, _color, _text) in zip(self._ann_labels, self._pinned_annotations):
            self._place_label(lbl, mz, chart_y)
            if x_axis is not None and x_axis.min() <= mz <= x_axis.max():
                lbl.show()
            else:
                lbl.hide()


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
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)
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

        _max_a = float(np.max(self._int_a)) if len(self._int_a) > 0 and np.max(self._int_a) > 0 else 1.0
        _max_b = float(np.max(self._int_b)) if len(self._int_b) > 0 and np.max(self._int_b) > 0 else 1.0
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
        candidates = [(i, j, float(int_a[i]) * float(int_b[j])) for i in range(len(mz_a)) for j in range(len(mz_b)) if abs(float(mz_a[i]) - float(mz_b[j])) <= tol]
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
                    "delta_ppm": delta / float(mz_b[j]) * 1e6 if float(mz_b[j]) != 0.0 else 0.0,
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
        n_matched = sum(1 for r in self._rows if r["idx_a"] is not None and r["idx_b"] is not None)
        n_only_a = sum(1 for r in self._rows if r["idx_b"] is None)
        n_only_b = sum(1 for r in self._rows if r["idx_a"] is None)

        def _spec_info_html(spectrum, title, label):
            """Build one info line for a spectrum."""
            prec_mz = spectrum.get("precursor_mz", None)
            prec_int = spectrum.get("precursor_intensity", None)
            scan_id = spectrum.get("id") or spectrum.get("scan_id") or spectrum.get("index")
            fs = spectrum.get("filter_string") or ""
            parts = [f"<b>{label}: {title}</b>"]
            if prec_mz is not None:
                parts.append(f"precursor m/z: <b>{float(prec_mz):.4f}</b>")
            if prec_int is not None:
                parts.append(f"precursor int.: <b>{float(prec_int):.3e}</b>")
            if scan_id is not None:
                parts.append(f"scan: <b>{scan_id}</b>")
            if fs:
                parts.append(f"<span style='color:#5f6368;font-size:10px'>{fs}</span>")
            return " &nbsp;|&nbsp; ".join(parts)

        score_line = f"{self.method} score: <b>{self.similarity:.4f}</b> &nbsp;|&nbsp; tolerance: {self.mz_tolerance:.4f} Da &nbsp;|&nbsp; matched: <b>{n_matched}</b> &nbsp; only-A: <b>{n_only_a}</b> &nbsp; only-B: <b>{n_only_b}</b>"
        hdr_html = f"{_spec_info_html(self.spectrum_a, self.title_a, 'A')}<br>{_spec_info_html(self.spectrum_b, self.title_b, 'B')}<br><span style='color:#444'>{score_line}</span>"
        hdr = QLabel(hdr_html)
        hdr.setStyleSheet("font-size: 11px; padding: 3px 4px;")
        hdr.setWordWrap(True)
        hdr.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        outer.addWidget(hdr, 0)  # stretch=0 → never grows vertically

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)  # stretch=1 → takes all remaining space

        # ---- left: fragment table ----
        tbl_w = QWidget()
        tbl_l = QVBoxLayout(tbl_w)
        tbl_l.setContentsMargins(0, 0, 4, 0)
        tbl_l.setSpacing(2)

        hint = QLabel(
            "<small>Click a row to highlight that peak in the mirror plot.&nbsp;&nbsp;<span style='background:#dceafc;padding:1px 4px'>Blue background</span> = matched &nbsp;<span style='background:#ffebee;padding:1px 4px'>Peach background</span> = only in A or B</small>"
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
        self._table.setStyleSheet("QTableWidget::item:selected { background-color: firebrick; color: #ffffff; }")
        # NOTE: keep sorting disabled until after population — enabling it first
        # causes Qt to re-sort after every setItem(), scattering values into
        # wrong rows and leaving most cells empty.
        self._table.setSortingEnabled(False)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(True)
        self._table.verticalHeader().setDefaultSectionSize(20)

        RA = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        for row_idx, row in enumerate(self._rows):
            is_matched = row["idx_a"] is not None and row["idx_b"] is not None
            has_a = row["idx_a"] is not None
            has_b = row["idx_b"] is not None
            bg = self._MATCH_BG if is_matched else self._ONLY_B_BG if not has_a else self._ONLY_A_BG

            # Start with empty placeholder items so every cell has a background
            items = [QTableWidgetItem("") for _ in range(len(COL_HEADERS))]

            # col 0: mean m/z (average when matched, single value otherwise)
            mean_mz = (row["mz_a"] + row["mz_b"]) / 2.0 if is_matched else row["mz_a"] if has_a else row["mz_b"]
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

        # ---- right: mirror plot (Qt Chart) ----
        self._chart = QChart()
        self._chart.setMargins(QMargins(0, 0, 0, 0))
        self._chart.legend().setVisible(True)
        self._chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._chart_view = MirrorPlotChartView(self._chart)
        self._chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Bind spectrum data so hover/annotation know about the peaks
        self._chart_view._mz_a = self._mz_a
        self._chart_view._rel_a = self._rel_a
        self._chart_view._mz_b = self._mz_b
        self._chart_view._rel_b = self._rel_b
        self._chart_view._rows = self._rows
        self._chart_view._table = self._table
        self._chart_view._usi_a = self.title_a
        self._chart_view._usi_b = self.title_b
        splitter.addWidget(self._chart_view)

        splitter.setSizes([540, 680])

        self._build_chart()

    # ------------------------------------------------------------------
    def _build_chart(self):
        """Rebuild the Qt Chart mirror plot, highlighting selected peaks."""
        chart = self._chart
        chart.removeAllSeries()
        for ax in list(chart.axes()):
            chart.removeAxis(ax)

        _COL_A = QColor("#1565c0")  # blue for spectrum A
        _COL_B = QColor("#0d7a2f")  # green for spectrum B
        _COL_HL = QColor("firebrick")

        series_a_norm = QLineSeries()
        series_a_norm.setName(f"↑ {self.title_a}")
        series_a_norm.setPen(QPen(_COL_A, 1.5))

        series_a_hl = QLineSeries()
        series_a_hl.setName("selected")
        series_a_hl.setPen(QPen(_COL_HL, 2.5))

        series_b_norm = QLineSeries()
        series_b_norm.setName(f"↓ {self.title_b}")
        series_b_norm.setPen(QPen(_COL_B, 1.5))

        series_b_hl = QLineSeries()
        series_b_hl.setName("")
        series_b_hl.setPen(QPen(_COL_HL, 2.5))

        all_mz: list = []

        for i in range(len(self._mz_a)):
            mz = float(self._mz_a[i])
            rel_i = float(self._rel_a[i])
            all_mz.append(mz)
            s = series_a_hl if i in self._highlight_a else series_a_norm
            s.append(mz, 0.0)
            s.append(mz, rel_i)
            s.append(mz, 0.0)

        for i in range(len(self._mz_b)):
            mz = float(self._mz_b[i])
            rel_i = -float(self._rel_b[i])
            all_mz.append(mz)
            s = series_b_hl if i in self._highlight_b else series_b_norm
            s.append(mz, 0.0)
            s.append(mz, rel_i)
            s.append(mz, 0.0)

        # Baseline at y = 0
        baseline = QLineSeries()
        baseline.setName("")
        baseline.setPen(QPen(QColor("#202124"), 0.8))
        if all_mz:
            baseline.append(min(all_mz) - 2.0, 0.0)
            baseline.append(max(all_mz) + 2.0, 0.0)

        for s in [series_a_norm, series_b_norm, series_a_hl, series_b_hl, baseline]:
            chart.addSeries(s)

        x_axis = QValueAxis()
        x_axis.setTitleText("m/z")
        y_axis = QValueAxis()
        y_axis.setRange(-110.0, 110.0)
        y_axis.setTitleText("Relative intensity (%)")
        y_axis.setLabelFormat("%.0f")
        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

        for s in [series_a_norm, series_b_norm, series_a_hl, series_b_hl, baseline]:
            s.attachAxis(x_axis)
            s.attachAxis(y_axis)

        if all_mz:
            margin = max((max(all_mz) - min(all_mz)) * 0.05, 1.0)
            x_axis.setRange(min(all_mz) - margin, max(all_mz) + margin)

        chart.setTitle(f"{self.method}  ·  score: {self.similarity:.4f}")
        # store full range for double-click reset
        if all_mz:
            self._chart_view.set_full_range(x_axis.min(), x_axis.max(), -110.0, 110.0)

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
        # Clear the transient hover series before removeAllSeries() in _build_chart
        self._chart_view._clear_hover()
        self._build_chart()
        # Re-position any pinned annotation labels after the chart rebuild
        if self._chart_view._pinned_annotations:
            self._chart_view._redraw_annotation_labels()


# ===========================================================================
# USI-based arbitrary MSMS Spectrum Comparison Window
# ===========================================================================


class USISpectrumComparisonWindow(QWidget):
    """Compare two arbitrary MSMS spectra side-by-side via a mirror plot.

    Can be opened with zero, one, or two pre-populated spectra.  The user can
    also paste USI strings to look up spectra inside the loaded files (when a
    ``file_manager`` is provided).

    The mirror plot and fragment table are reused from ``EnhancedMirrorPlotWindow``.
    """

    def __init__(
        self,
        spectrum_a=None,
        filename_a=None,
        spectrum_b=None,
        filename_b=None,
        file_manager=None,
        mz_tolerance: float = 0.02,
        method: str = "CosineHungarian",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._spectrum_a = spectrum_a
        self._filename_a = filename_a or ""
        self._spectrum_b = spectrum_b
        self._filename_b = filename_b or ""
        self._file_manager = file_manager
        self._mz_tolerance = mz_tolerance
        self._method = method

        self._usi_a = make_usi(spectrum_a, filename_a) if spectrum_a else ""
        self._usi_b = make_usi(spectrum_b, filename_b) if spectrum_b else ""

        # Keep child windows alive
        self._child_windows: list = []

        self.setWindowTitle("Spectrum Comparator")
        self.resize(1500, 780)
        self._init_ui()

    # ------------------------------------------------------------------
    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 8)
        outer.setSpacing(6)

        # ---- USI input row ----
        usi_row = QHBoxLayout()
        usi_row.setSpacing(8)

        usi_row.addWidget(QLabel("Spectrum A (USI):"))
        self._usi_a_edit = QLineEdit(self._usi_a)
        self._usi_a_edit.setPlaceholderText("USI or leave empty")
        usi_row.addWidget(self._usi_a_edit, 1)

        copy_a_btn = QPushButton("⎘ Copy A")
        copy_a_btn.setToolTip("Copy USI A to clipboard")
        copy_a_btn.setFixedWidth(75)
        copy_a_btn.clicked.connect(lambda: QApplication.clipboard().setText(self._usi_a_edit.text().strip()))
        usi_row.addWidget(copy_a_btn)

        usi_row.addWidget(QLabel("Spectrum B (USI):"))
        self._usi_b_edit = QLineEdit(self._usi_b)
        self._usi_b_edit.setPlaceholderText("USI or leave empty")
        usi_row.addWidget(self._usi_b_edit, 1)

        copy_b_btn = QPushButton("⎘ Copy B")
        copy_b_btn.setToolTip("Copy USI B to clipboard")
        copy_b_btn.setFixedWidth(75)
        copy_b_btn.clicked.connect(lambda: QApplication.clipboard().setText(self._usi_b_edit.text().strip()))
        usi_row.addWidget(copy_b_btn)

        tol_lbl = QLabel("Tol (Da):")
        usi_row.addWidget(tol_lbl)
        self._tol_spin = NoScrollDoubleSpinBox()
        self._tol_spin.setRange(0.001, 2.0)
        self._tol_spin.setDecimals(4)
        self._tol_spin.setSingleStep(0.005)
        self._tol_spin.setValue(self._mz_tolerance)
        self._tol_spin.setFixedWidth(90)
        usi_row.addWidget(self._tol_spin)

        compare_btn = QPushButton("Compare")
        compare_btn.clicked.connect(self._on_compare)
        usi_row.addWidget(compare_btn)

        outer.addLayout(usi_row)

        # ---- placeholder for the mirror plot ----
        self._plot_container = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_container)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._plot_container, 1)

        # If both spectra were already provided, show the mirror plot immediately
        if self._spectrum_a and self._spectrum_b:
            self._show_mirror()

    # ------------------------------------------------------------------
    def _on_compare(self):
        """Triggered by the Compare button — look up spectra from USI text if needed."""
        usi_a_text = self._usi_a_edit.text().strip()
        usi_b_text = self._usi_b_edit.text().strip()

        # If the USI in the text boxes differs from what we already have,
        # try to look up the spectrum from the loaded files.
        if usi_a_text and usi_a_text != self._usi_a:
            result = self._lookup_by_usi(usi_a_text)
            if result is not None:
                self._spectrum_a, self._filename_a = result
                self._usi_a = usi_a_text
            else:
                QMessageBox.warning(
                    self,
                    "Spectrum not found",
                    f"Could not locate spectrum A from USI:\n{usi_a_text}\n\nMake sure the file is loaded in the main window.",
                )
                return

        if usi_b_text and usi_b_text != self._usi_b:
            result = self._lookup_by_usi(usi_b_text)
            if result is not None:
                self._spectrum_b, self._filename_b = result
                self._usi_b = usi_b_text
            else:
                QMessageBox.warning(
                    self,
                    "Spectrum not found",
                    f"Could not locate spectrum B from USI:\n{usi_b_text}\n\nMake sure the file is loaded in the main window.",
                )
                return

        self._mz_tolerance = self._tol_spin.value()

        if not self._spectrum_a or not self._spectrum_b:
            QMessageBox.warning(self, "Missing spectra", "Please provide both spectrum A and spectrum B.")
            return

        self._show_mirror()

    # ------------------------------------------------------------------
    def _show_mirror(self):
        """Build / rebuild the EnhancedMirrorPlotWindow-equivalent inside this window."""
        # Remove previous plot widget if any
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        spec_a = self._spectrum_a
        spec_b = self._spectrum_b
        tol = self._mz_tolerance
        method = self._method
        similarity = calculate_cosine_similarity(spec_a, spec_b, mz_tolerance=tol, method=method)

        title_a = make_usi(spec_a, self._filename_a)
        title_b = make_usi(spec_b, self._filename_b)

        self.setWindowTitle(f"Spectrum Comparator — {title_a}  vs  {title_b}")

        # Reuse the EnhancedMirrorPlotWindow as an embedded widget (no parent window)
        mirror = EnhancedMirrorPlotWindow(
            spec_a,
            spec_b,
            title_a,
            title_b,
            similarity,
            mz_tolerance=tol,
            method=method,
            parent=None,  # will be re-parented below
        )
        # Embed by changing window flags so it acts as a plain widget
        mirror.setWindowFlags(Qt.WindowType.Widget)
        mirror.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self._plot_layout.addWidget(mirror)
        mirror.show()
        self._child_windows.append(mirror)

    # ------------------------------------------------------------------
    def _lookup_by_usi(self, usi: str):
        """Try to find a spectrum matching *usi* in the loaded files.

        The lookup is best-effort: it parses the filename segment, RT, and
        scan_id from the USI and searches the file_manager's cached data.

        Returns ``(spectrum_dict, filename)`` or ``None``.
        """
        if self._file_manager is None:
            return None

        parts = usi.split(":")
        if len(parts) < 2:
            return None

        target_filename = parts[0].strip()
        # Collect all remaining parts for secondary matching
        usi_lower = usi.lower()

        # Parse RT hint
        rt_hint = None
        for p in parts[1:]:
            if p.startswith("RT") and "min" in p:
                try:
                    rt_hint = float(p[2:].replace("min", ""))
                except ValueError:
                    pass

        # Parse scan_id hint
        scan_hint = None
        for p in parts:
            if p.startswith("scan="):
                scan_hint = p[5:]

        try:
            files_data = self._file_manager.get_files_data()
        except Exception:
            return None

        for _, row in files_data.iterrows():
            filepath = row.get("Filepath", "")
            filename = row.get("filename", "")

            # Match by filename or filepath basename
            basename = os.path.basename(str(filepath))
            if target_filename not in (str(filename), basename, os.path.splitext(str(filename))[0], os.path.splitext(basename)[0]):
                continue

            # Try in-memory cache
            cached = None
            if self._file_manager.keep_in_memory:
                cached = self._file_manager.cached_data.get(filepath)

            spectra_pool = []
            if cached and isinstance(cached, dict):
                for level_key in ("ms2", "ms1"):
                    for s in cached.get(level_key, []):
                        spectra_pool.append((s, str(filename)))

            if not spectra_pool:
                # Skip file-based reading (too slow from UI thread)
                continue

            best = None
            best_score = float("inf")
            for s, fn in spectra_pool:
                # Match by scan_id first
                if scan_hint is not None:
                    sid = str(s.get("scan_id") or s.get("id") or "")
                    if sid == scan_hint:
                        return (s, fn)
                # Otherwise match by RT proximity
                if rt_hint is not None:
                    s_rt = float(s.get("scan_time") or s.get("rt") or 0)
                    diff = abs(s_rt - rt_hint)
                    if diff < best_score:
                        best_score = diff
                        best = (s, fn)

            if best is not None and best_score < 0.05:  # within 0.05 min (~3 seconds)
                return best

        return None
