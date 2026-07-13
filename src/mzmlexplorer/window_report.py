"""
Comprehensive compound report feature.

Contains:
- ``ReportOptionsDialog``: dialog for choosing groups/samples, the EIC
  extension window, and which adducts to process before generating a report.
- ``CompoundReportWindow``: the scrollable, multi-page report viewer with a
  table-of-contents tree for quick navigation.

Peak picking is performed automatically for the report using each compound's
fixed ``RT_start_min``/``RT_end_min`` window: the apex is the most intense
point inside that window, and the peak area is the trapezoidal integral over
the same fixed window. This is a simpler convention than the manual
click-to-integrate workflow used elsewhere in the app (there is currently no
automatic peak detection in the codebase), chosen so that the report can be
generated without user interaction.
"""

from __future__ import annotations

import hashlib
import traceback

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from natsort import natsorted
from PyQt6.QtCharts import (
    QAreaSeries,
    QCategoryAxis,
    QChart,
    QLineSeries,
    QLogValueAxis,
    QScatterSeries,
    QValueAxis,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QCursor, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressDialog,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .file_manager import ACQUISITION_DATETIME_COLUMN
from .utils import format_mz, format_retention_time
from .window_eic import InteractiveChartView

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _make_selection_list() -> QListWidget:
    """A plain (non-checkbox) multi-selection list with a white background,
    used for the group/sample/adduct pickers in ``ReportOptionsDialog``.
    Selected rows represent the "included" items.
    """
    list_widget = QListWidget()
    list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    list_widget.setStyleSheet("QListWidget { background-color: #ffffff; }")
    return list_widget


def _selected_texts(list_widget: QListWidget) -> list:
    return [item.text() for item in list_widget.selectedItems()]


def _selected_data(list_widget: QListWidget) -> list:
    return [item.data(Qt.ItemDataRole.UserRole) for item in list_widget.selectedItems()]


def _render_smiles_pixmap(smiles: str, size: int = 180) -> QPixmap | None:
    """Render a SMILES string to a QPixmap using RDKit, or None on failure."""
    if not smiles or not str(smiles).strip() or str(smiles).strip().lower() in ("nan", "none"):
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        drawer.drawOptions().clearBackground = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        pixmap = QPixmap()
        pixmap.loadFromData(png_bytes, "PNG")
        return pixmap
    except Exception:
        return None


def _extract_eic_with_mz(file_manager, filepath, target_mz, mz_tolerance, rt_start=None, rt_end=None, calculation_method="Sum of all signals", polarity=None):
    """Like ``FileManager.extract_eic`` but also returns, per scan, the
    intensity-weighted mean m/z of the matched peaks (needed to determine the
    peak apex's actual measured m/z for mass-deviation reporting).

    Returns (rt_array, intensity_array, mz_array).
    """
    if polarity is not None and str(polarity).lower() in ("positive", "pos", "pos.", "+"):
        polarity = "+"
    elif polarity is not None and str(polarity).lower() in ("negative", "neg", "neg.", "-"):
        polarity = "-"
    else:
        polarity = None

    rt_list, intensity_list, mz_list = [], [], []

    def _handle_scan(rt, mz_array, intensity_array):
        if rt_start is not None and rt < rt_start:
            return
        if rt_end is not None and rt > rt_end:
            return
        if len(mz_array) > 0:
            mz_mask = np.abs(mz_array - target_mz) <= mz_tolerance
            if np.any(mz_mask):
                matched_mz = mz_array[mz_mask]
                matched_int = intensity_array[mz_mask]
                if calculation_method == "Sum of all signals":
                    total_intensity = float(np.sum(matched_int))
                    weight_sum = float(np.sum(matched_int))
                    apex_mz = float(np.average(matched_mz, weights=matched_int)) if weight_sum > 0 else float(np.mean(matched_mz))
                else:
                    best = int(np.argmax(matched_int))
                    total_intensity = float(matched_int[best])
                    apex_mz = float(matched_mz[best])
                rt_list.append(rt)
                intensity_list.append(total_intensity)
                mz_list.append(apex_mz)
                return
        rt_list.append(rt)
        intensity_list.append(0.0)
        mz_list.append(np.nan)

    try:
        if file_manager.keep_in_memory and filepath in file_manager.cached_data:
            cached_file_data = file_manager.cached_data[filepath]
            spectra_data = cached_file_data["ms1"] if isinstance(cached_file_data, dict) and "ms1" in cached_file_data else cached_file_data
            for spectrum_data in spectra_data:
                if polarity is not None and spectrum_data["polarity"] is not None and polarity != spectrum_data["polarity"]:
                    continue
                _handle_scan(spectrum_data["scan_time"], spectrum_data["mz"], spectrum_data["intensity"])
        else:
            reader = file_manager.get_mzml_reader(filepath)
            for spectrum in reader:
                if spectrum.ms_level != 1:
                    continue
                if polarity is not None:
                    spectrum_polarity = file_manager._get_spectrum_polarity(spectrum)
                    if spectrum_polarity is not None and spectrum_polarity != polarity:
                        continue
                _handle_scan(spectrum.scan_time_in_minutes(), spectrum.mz, spectrum.i)
    except Exception as e:
        print(f"Error extracting EIC+mz from {filepath}: {e}")
        return np.array([]), np.array([]), np.array([])

    return np.array(rt_list), np.array(intensity_list), np.array(mz_list)


def _pick_peak(rt_arr, intensity_arr, mz_arr, rt_start, rt_end):
    """Automatic peak picking within the fixed [rt_start, rt_end] window.

    Apex = most intense point inside the window. Area = trapezoidal
    integration over the (fixed) window. Returns None if no data point falls
    inside the window.
    """
    if rt_arr is None or len(rt_arr) == 0:
        return None
    mask = (rt_arr >= rt_start) & (rt_arr <= rt_end)
    if not np.any(mask):
        return None
    idxs = np.where(mask)[0]
    window_rt = rt_arr[idxs]
    window_int = intensity_arr[idxs]
    apex_local = int(np.argmax(window_int))
    apex_idx = idxs[apex_local]
    apex_rt = float(rt_arr[apex_idx])
    apex_intensity = float(intensity_arr[apex_idx])
    apex_mz = None
    if mz_arr is not None and len(mz_arr) > apex_idx:
        candidate = mz_arr[apex_idx]
        if not (isinstance(candidate, float) and np.isnan(candidate)):
            apex_mz = float(candidate)
    area = float(np.trapz(window_int, window_rt)) if len(window_rt) > 1 else float(window_int[0]) if len(window_rt) == 1 else 0.0
    return {"apex_rt": apex_rt, "apex_intensity": apex_intensity, "apex_mz": apex_mz, "area": area}


# ---------------------------------------------------------------------------
# Options dialog
# ---------------------------------------------------------------------------


class ReportOptionsDialog(QDialog):
    """Dialog to configure a compound report before it is generated."""

    def __init__(self, compound_count: int, groups: list, samples: list, fallback_adduct_choices: list, parent=None):
        """
        Args:
            compound_count: number of compounds that will be processed (info only)
            groups: list of experimental group names
            samples: list of (filepath, filename, group) tuples
            fallback_adduct_choices: list of all available adduct strings
        """
        super().__init__(parent)
        self.setWindowTitle("Compound Report Options")
        self.resize(760, 640)

        layout = QVBoxLayout(self)

        info_label = QLabel(f"<b>{compound_count}</b> compound(s) will be processed.")
        layout.addWidget(info_label)

        lists_layout = QHBoxLayout()

        groups_box = QGroupBox("Experimental Groups")
        gbl = QVBoxLayout(groups_box)
        self.groups_list = _make_selection_list()
        for g in groups:
            item = QListWidgetItem(g)
            self.groups_list.addItem(item)
        gbl.addWidget(self.groups_list)
        lists_layout.addWidget(groups_box)

        samples_box = QGroupBox("Samples")
        sbl = QVBoxLayout(samples_box)
        self.samples_list = _make_selection_list()
        for filepath, filename, group in samples:
            item = QListWidgetItem(f"{filename}  [{group}]")
            item.setData(Qt.ItemDataRole.UserRole, filepath)
            item.setData(Qt.ItemDataRole.UserRole + 1, group)
            self.samples_list.addItem(item)
        sbl.addWidget(self.samples_list)
        lists_layout.addWidget(samples_box)

        layout.addLayout(lists_layout, stretch=1)

        # Selecting/deselecting a group selects/deselects all of its samples.
        self.groups_list.itemSelectionChanged.connect(self._sync_samples_to_groups)

        form = QFormLayout()
        self.extension_spin = QDoubleSpinBox()
        self.extension_spin.setRange(0.0, 20.0)
        self.extension_spin.setSingleStep(0.1)
        self.extension_spin.setDecimals(2)
        self.extension_spin.setValue(1.0)
        self.extension_spin.setToolTip(
            "The EIC extraction window is extended before RT_start_min and after RT_end_min\nby this factor times the compound's peak width (RT_end_min - RT_start_min)."
        )
        form.addRow("EIC extension (\u00d7 peak width):", self.extension_spin)
        layout.addLayout(form)

        self.common_adducts_checkbox = QCheckBox("Include each compound's common adducts (Common_adducts)")
        self.common_adducts_checkbox.setChecked(True)
        layout.addWidget(self.common_adducts_checkbox)

        fallback_box = QGroupBox("Fallback adducts (always included for every compound, in addition to the above)")
        fbl = QVBoxLayout(fallback_box)
        self.fallback_list = _make_selection_list()
        for a in fallback_adduct_choices:
            item = QListWidgetItem(a)
            self.fallback_list.addItem(item)
        self.fallback_list.clearSelection()
        fbl.addWidget(self.fallback_list)
        layout.addWidget(fallback_box, stretch=1)

        # All groups/samples included by default; no fallback adducts by default.
        self.groups_list.selectAll()

        button_box = QDialogButtonBox()
        process_button = button_box.addButton("Process", QDialogButtonBox.ButtonRole.AcceptRole)
        process_button.setDefault(True)
        button_box.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _sync_samples_to_groups(self):
        """Selecting/deselecting a group in the Groups list automatically
        selects/deselects all samples belonging to that group.
        """
        selected_groups = set(_selected_texts(self.groups_list))
        self.samples_list.blockSignals(True)
        for i in range(self.samples_list.count()):
            item = self.samples_list.item(i)
            item.setSelected(item.data(Qt.ItemDataRole.UserRole + 1) in selected_groups)
        self.samples_list.blockSignals(False)

    def get_result(self) -> dict:
        return {
            "selected_groups": set(_selected_texts(self.groups_list)),
            "selected_samples": set(_selected_data(self.samples_list)),
            "extension_factor": self.extension_spin.value(),
            "include_common_adducts": self.common_adducts_checkbox.isChecked(),
            "fallback_adducts": _selected_texts(self.fallback_list),
        }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

_ALPHA_EIC = 0.40
_LINE_EXTENT = 1.0e15
_FIREBRICK = QColor(178, 34, 34)
_ANTHRACITE = "#5A6268"
_MZ_BAND_COLORS = [QColor(178, 34, 34), QColor(238, 44, 44), QColor(205, 38, 38)]  # firebrick, firebrick2, firebrick3
_RT_BAND_COLORS = [QColor(178, 34, 34), QColor(238, 44, 44), QColor(205, 38, 38), QColor(139, 26, 26)]  # + firebrick4
_BOX_WIDTH = 0.6


def _make_chart() -> QChart:
    chart = QChart()
    chart.legend().setVisible(False)
    return chart


def _keep_alive(chart: QChart, *objs) -> None:
    """Keep a strong Python reference to ``objs`` for as long as ``chart`` lives.

    ``QAreaSeries(upper, lower)`` does not itself keep the Python wrapper
    objects for its upper/lower ``QLineSeries`` alive: unless those series are
    also added to the chart via ``chart.addSeries(...)``, nothing holds a
    Python reference to them once the enclosing function returns, so Python's
    garbage collector can free them while the C++ ``QAreaSeries`` still holds
    a raw pointer to them. That produces a silent, unrecoverable native access
    violation (no Python traceback) later, e.g. when the chart is rendered or
    axes are attached. Stashing the objects on the chart instance itself
    avoids this.
    """
    refs = getattr(chart, "_report_keep_alive_refs", None)
    if refs is None:
        refs = []
        chart._report_keep_alive_refs = refs
    refs.extend(objs)


def _add_axes(chart: QChart, x_title: str, y_title: str, x_min, x_max, y_min, y_max, y_log: bool = False, y_format: str | None = None):
    x_axis = QValueAxis()
    x_axis.setTitleText(x_title)
    x_axis.setRange(x_min, x_max)
    if y_log:
        y_axis = QLogValueAxis()
        y_axis.setBase(10.0)
        safe_min = y_min if y_min > 0 else 1.0
        safe_max = y_max if y_max > safe_min else safe_min * 10
        y_axis.setRange(safe_min, safe_max)
        y_axis.setLabelFormat(y_format or "%.1e")
    else:
        y_axis = QValueAxis()
        y_axis.setRange(y_min, y_max)
        if y_format:
            y_axis.setLabelFormat(y_format)
    y_axis.setTitleText(y_title)
    chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
    chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
    return x_axis, y_axis


def _chart_view(chart: QChart, x_range=None, y_range=None, height: int = 420) -> InteractiveChartView:
    view = InteractiveChartView(chart)
    view.setMinimumHeight(height)
    view.setMaximumHeight(height)
    view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    view.setStyleSheet(f"border: 2px solid {_ANTHRACITE};")

    if x_range is not None and y_range is not None:

        def _reset_view():
            try:
                x_axes = chart.axes(Qt.Orientation.Horizontal)
                y_axes = chart.axes(Qt.Orientation.Vertical)
                if x_axes:
                    x_axes[0].setRange(*x_range)
                if y_axes:
                    y_axes[0].setRange(*y_range)
            except Exception:
                # Never let an exception escape a Qt signal callback.
                traceback.print_exc()

        view.doubleClickRequested.connect(_reset_view)

    return view


def _attach_hover(series, points: list):
    """points: list of (x, y, label) in the same order as appended to the series."""

    def _on_hover(point, state):
        try:
            if not state:
                QToolTip.hideText()
                return
            if not points:
                return
            best = min(points, key=lambda p: (p[0] - point.x()) ** 2 + (p[1] - point.y()) ** 2)
            QToolTip.showText(QCursor.pos(), f"{best[2]}\nx={point.x():.4g}, y={point.y():.4g}")
        except Exception:
            # Never let an exception escape a Qt signal callback: PyQt6 can
            # abort the whole process (silently, before anything reaches the
            # console) if a slot invoked from C++ raises unexpectedly.
            traceback.print_exc()

    series.hovered.connect(_on_hover)


def _add_reference_bands(chart: QChart, reference_value: float, bands: list, x_min: float = 0.0, x_max: float = 1.0):
    """bands: list of (delta, QColor). Widest band is drawn first so narrower bands sit on top.

    Unlike the peak-boundary vertical lines (plain ``QLineSeries`` strokes, safe
    even with an astronomically large extent), these bands are filled
    ``QAreaSeries``. Rasterizing a filled polygon that spans coordinates many
    orders of magnitude beyond the chart's actual axis range can hang/crash
    Qt's paint engine, so the horizontal extent here is only padded modestly
    beyond the visible x-range instead of using a huge fixed constant.
    """
    span = max(x_max - x_min, 1.0)
    x_left = x_min - span * 10
    x_right = x_max + span * 10

    for delta, color in sorted(bands, key=lambda b: -b[0]):
        upper = QLineSeries()
        upper.append(x_left, reference_value + delta)
        upper.append(x_right, reference_value + delta)
        lower = QLineSeries()
        lower.append(x_left, reference_value - delta)
        lower.append(x_right, reference_value - delta)
        area = QAreaSeries(upper, lower)
        band_color = QColor(color)
        band_color.setAlphaF(0.18)
        area.setBrush(band_color)
        area.setPen(QPen(Qt.PenStyle.NoPen))
        chart.addSeries(area)
        _keep_alive(chart, upper, lower)

    ref_line = QLineSeries()
    ref_line.append(x_left, reference_value)
    ref_line.append(x_right, reference_value)
    pen = ref_line.pen()
    pen.setColor(QColor("#202124"))
    pen.setStyle(Qt.PenStyle.SolidLine)
    pen.setWidth(1)
    ref_line.setPen(pen)
    ref_line.setProperty("no_hover_tooltip", True)
    chart.addSeries(ref_line)


def _deterministic_jitter(label: str, spread: float = 0.28) -> float:
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    frac = (int(digest, 16) % 10000) / 10000.0
    return (frac * 2 - 1) * spread


def build_eic_overlay_widget(sample_traces: list, scale_to_apex: bool = False, peak_bounds=None) -> InteractiveChartView:
    """sample_traces: list of (label, rt_array, intensity_array, QColor)."""
    chart = _make_chart()
    all_rt, all_int = [], []
    for label, rt, intensity, color in sample_traces:
        if len(rt) == 0:
            continue
        y = intensity
        if scale_to_apex:
            peak = np.max(intensity) if len(intensity) else 0.0
            y = intensity / peak if peak > 0 else intensity
        series = QLineSeries()
        line_color = QColor(color)
        line_color.setAlphaF(_ALPHA_EIC)
        pen = series.pen()
        pen.setColor(line_color)
        pen.setWidth(1)
        series.setPen(pen)
        series.setName(label)
        series.setProperty("sample_filename", label)
        for x_val, y_val in zip(rt, y):
            series.append(float(x_val), float(y_val))
        chart.addSeries(series)
        all_rt.append(rt)
        all_int.append(y)

    if all_rt:
        x_min = min(float(np.min(r)) for r in all_rt)
        x_max = max(float(np.max(r)) for r in all_rt)
        y_min = 0.0
        y_max = max(float(np.max(i)) for i in all_int) * 1.05 if any(len(i) for i in all_int) else 1.0
    else:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0

    if peak_bounds is not None:
        for x_val in peak_bounds:
            boundary = QLineSeries()
            boundary.append(float(x_val), -_LINE_EXTENT)
            boundary.append(float(x_val), _LINE_EXTENT)
            pen = boundary.pen()
            pen.setColor(_FIREBRICK)
            pen.setStyle(Qt.PenStyle.SolidLine)
            pen.setWidth(1)
            boundary.setPen(pen)
            boundary.setProperty("no_hover_tooltip", True)
            chart.addSeries(boundary)

    y_format = None if scale_to_apex else "%.1e"
    x_axis, y_axis = _add_axes(chart, "RT (min)", "Scaled Intensity" if scale_to_apex else "Intensity", x_min, x_max, y_min, y_max, y_format=y_format)
    for series in chart.series():
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)

    return _chart_view(chart, x_range=(x_min, x_max), y_range=(y_min, y_max))


def build_eic_offset_widget(entries: list, shift_width: float) -> InteractiveChartView:
    """entries: list of (label, rt_array, intensity_array, QColor, offset_index).

    Each trace's RT axis is shifted by ``offset_index * shift_width`` so that
    multiple groups/samples can be shown side-by-side in a single "waterfall"
    style chart instead of many separate small charts.
    """
    chart = _make_chart()
    all_x, all_int = [], []
    for label, rt, intensity, color, offset_idx in entries:
        if len(rt) == 0:
            continue
        shifted_rt = np.asarray(rt, dtype=float) + offset_idx * shift_width
        series = QLineSeries()
        line_color = QColor(color)
        line_color.setAlphaF(_ALPHA_EIC)
        pen = series.pen()
        pen.setColor(line_color)
        pen.setWidth(1)
        series.setPen(pen)
        series.setName(label)
        series.setProperty("sample_filename", label)
        for x_val, y_val in zip(shifted_rt, intensity):
            series.append(float(x_val), float(y_val))
        chart.addSeries(series)
        all_x.append(shifted_rt)
        all_int.append(intensity)

    if all_x:
        x_min = min(float(np.min(x)) for x in all_x)
        x_max = max(float(np.max(x)) for x in all_x)
        y_max = max(float(np.max(i)) for i in all_int) * 1.05 if any(len(i) for i in all_int) else 1.0
    else:
        x_min, x_max, y_max = 0.0, 1.0, 1.0
    y_min = 0.0

    x_axis, y_axis = _add_axes(chart, "RT (min, offset per group/sample)", "Intensity", x_min, x_max, y_min, y_max, y_format="%.1e")
    for series in chart.series():
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)

    return _chart_view(chart, x_range=(x_min, x_max), y_range=(y_min, y_max))


def _build_value_axis(
    y_title: str,
    y_min: float,
    y_max: float,
    log_y: bool = False,
    y_format: str | None = None,
    reference_value: float | None = None,
    bands: list | None = None,
    tick_format: str = "%.3f",
):
    """Build the value (non-category) axis for a chart.

    When ``reference_value``/``bands`` are given (and the axis is not
    logarithmic), a ``QCategoryAxis`` is used instead of a plain
    ``QValueAxis`` so that the axis always shows fixed labels for the
    expected/nominal value and each +/- tolerance ("band") delta - these
    labels stay put at their actual data values regardless of panning or
    zooming, instead of the usual auto-computed "nice number" ticks that
    shift around as the visible range changes.
    """
    if reference_value is not None and bands and not log_y:
        axis = QCategoryAxis()
        axis.setLabelsPosition(QCategoryAxis.AxisLabelsPosition.AxisLabelsPositionOnValue)
        axis.setRange(y_min, y_max)
        axis.setGridLineVisible(True)
        tick_values = sorted({reference_value} | {reference_value + delta for delta, _ in bands} | {reference_value - delta for delta, _ in bands})
        for value in tick_values:
            delta = value - reference_value
            if abs(delta) < 1e-12:
                label = tick_format % value
            else:
                label = ("+" if delta > 0 else "-") + (tick_format % abs(delta))
            axis.append(label, value)
    elif log_y:
        axis = QLogValueAxis()
        axis.setBase(10.0)
        safe_min = y_min if y_min > 0 else 1.0
        safe_max = y_max if y_max > safe_min else safe_min * 10
        axis.setRange(safe_min, safe_max)
        axis.setLabelFormat(y_format or "%.1e")
    else:
        axis = QValueAxis()
        axis.setRange(y_min, y_max)
        if y_format:
            axis.setLabelFormat(y_format)
    axis.setTitleText(y_title)
    return axis


def build_box_jitter_widget(
    group_values: dict,
    group_colors: dict,
    y_title: str,
    reference_value: float | None = None,
    bands: list | None = None,
    log_y: bool = False,
    tick_format: str = "%.3f",
) -> InteractiveChartView:
    """group_values: dict[group_name] -> list of (value, sample_label) tuples."""
    chart = _make_chart()

    def _valid(v):
        if np.isnan(v):
            return False
        return v > 0 if log_y else True

    groups_present = [g for g in natsorted(group_values.keys()) if any(_valid(v) for v, _ in group_values[g])]

    half = _BOX_WIDTH / 2.0
    n_groups = len(groups_present)
    x_min, x_max = 0.0, max(n_groups, 1)

    # Draw the reference/tolerance bands first so they sit behind the
    # box-and-whisker plots and jittered points drawn afterwards.
    if bands is not None and reference_value is not None:
        _add_reference_bands(chart, reference_value, bands, x_min=x_min, x_max=x_max)

    all_low, all_high = [], []

    for gi, group in enumerate(groups_present):
        entries = [(v, lbl) for v, lbl in group_values[group] if _valid(v)]
        if not entries:
            continue
        center = gi + 0.5
        values = np.array([v for v, _ in entries], dtype=float)
        q1, med, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
        vmin, vmax = float(np.min(values)), float(np.max(values))
        color = QColor(group_colors.get(group, "#888888"))

        whisker_low = QLineSeries()
        whisker_low.append(center, vmin)
        whisker_low.append(center, q1)
        whisker_high = QLineSeries()
        whisker_high.append(center, q3)
        whisker_high.append(center, vmax)
        for whisker in (whisker_low, whisker_high):
            pen = whisker.pen()
            pen.setColor(color)
            pen.setWidth(1)
            whisker.setPen(pen)
            whisker.setProperty("no_hover_tooltip", True)
            chart.addSeries(whisker)

        upper = QLineSeries()
        upper.append(center - half, q3)
        upper.append(center + half, q3)
        lower = QLineSeries()
        lower.append(center - half, q1)
        lower.append(center + half, q1)
        box_area = QAreaSeries(upper, lower)
        box_color = QColor(color)
        box_color.setAlphaF(0.35)
        box_area.setBrush(box_color)
        box_area.setPen(QPen(color))
        chart.addSeries(box_area)
        _keep_alive(chart, upper, lower)

        median_line = QLineSeries()
        median_line.append(center - half, med)
        median_line.append(center + half, med)
        pen = median_line.pen()
        pen.setColor(QColor("#202124"))
        pen.setWidth(2)
        median_line.setPen(pen)
        median_line.setProperty("no_hover_tooltip", True)
        chart.addSeries(median_line)

        jitter = QScatterSeries()
        jitter.setMarkerSize(7.0)
        jitter.setColor(color)
        jitter.setBorderColor(QColor("#202124"))
        jitter_points = []
        for v, lbl in entries:
            x_val = center + _deterministic_jitter(lbl)
            jitter.append(x_val, v)
            jitter_points.append((x_val, v, lbl))
        chart.addSeries(jitter)
        _attach_hover(jitter, jitter_points)

        all_low.append(vmin)
        all_high.append(vmax)

    if all_low and all_high:
        y_min, y_max = min(all_low), max(all_high)
        pad = (y_max - y_min) * 0.1 or 1.0
        y_min -= pad
        y_max += pad
    else:
        y_min, y_max = 0.0, 1.0

    if bands is not None and reference_value is not None and not log_y:
        max_delta = max(delta for delta, _ in bands)
        y_min = min(y_min, reference_value - max_delta)
        y_max = max(y_max, reference_value + max_delta)

    if log_y:
        y_min = max(y_min, 1e-6)
    y_axis = _build_value_axis(y_title, y_min, y_max, log_y=log_y, reference_value=reference_value, bands=bands, tick_format=tick_format)

    category_axis = QCategoryAxis()
    category_axis.setLabelsPosition(QCategoryAxis.AxisLabelsPosition.AxisLabelsPositionCenter)
    category_axis.setRange(x_min, x_max)
    category_axis.setGridLineVisible(False)
    for gi, group in enumerate(groups_present):
        category_axis.append(group, gi + 1.0)

    chart.addAxis(category_axis, Qt.AlignmentFlag.AlignBottom)
    chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
    for series in chart.series():
        series.attachAxis(category_axis)
        series.attachAxis(y_axis)

    return _chart_view(chart, x_range=(x_min, x_max), y_range=(y_min, y_max), height=390)


def build_scatter_by_group_widget(
    points: list,
    group_colors: dict,
    y_title: str,
    reference_value: float | None = None,
    bands: list | None = None,
    log_y: bool = False,
    tick_format: str = "%.3f",
) -> InteractiveChartView:
    """points: list of (x_index, value, group, sample_label) sorted by desired display order."""
    chart = _make_chart()
    if log_y:
        points = [p for p in points if p[1] > 0]
    groups_present = natsorted({p[2] for p in points})
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]

    if x_values:
        x_min, x_max = min(x_values) - 0.5, max(x_values) + 0.5
    else:
        x_min, x_max = -0.5, 0.5

    # Draw the reference/tolerance bands first so they sit behind the
    # scatter points drawn afterwards.
    if bands is not None and reference_value is not None:
        _add_reference_bands(chart, reference_value, bands, x_min=x_min, x_max=x_max)

    for group in groups_present:
        series = QScatterSeries()
        series.setMarkerSize(9.0)
        color = QColor(group_colors.get(group, "#888888"))
        series.setColor(color)
        series.setBorderColor(QColor(Qt.GlobalColor.transparent))
        group_points = [(x_val, y_val, label) for x_val, y_val, g, label in points if g == group]
        for x_val, y_val, _label in group_points:
            series.append(float(x_val), float(y_val))
        chart.addSeries(series)
        _attach_hover(series, group_points)

    if log_y:
        y_min = (min(y_values) * 0.5) if y_values else 1.0
        y_max = (max(y_values) * 2.0) if y_values else 10.0
    elif y_values:
        y_min, y_max = min(y_values), max(y_values)
        pad = (y_max - y_min) * 0.1 or 1.0
        y_min -= pad
        y_max += pad
    else:
        y_min, y_max = 0.0, 1.0

    if bands is not None and reference_value is not None and not log_y:
        max_delta = max(delta for delta, _ in bands)
        y_min = min(y_min, reference_value - max_delta)
        y_max = max(y_max, reference_value + max_delta)

    x_axis = QValueAxis()
    x_axis.setTitleText("Sample (exp. group, measurement time ordered)")
    x_axis.setRange(x_min, x_max)
    y_axis = _build_value_axis(y_title, y_min, y_max, log_y=log_y, y_format="%.1e" if log_y else None, reference_value=reference_value, bands=bands, tick_format=tick_format)

    chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
    chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
    for series in chart.series():
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)

    return _chart_view(chart, x_range=(x_min, x_max), y_range=(y_min, y_max), height=390)


def _heatmap_canvas(matrix: np.ndarray, row_labels: list, col_labels: list, title: str, cmap: str, vmin=None, vmax=None, cbar_label: str = "") -> FigureCanvas:
    n_rows = max(len(row_labels), 1)
    n_cols = max(len(col_labels), 1)
    fig = Figure(figsize=(max(6.0, 0.35 * n_cols), max(4.5, 0.42 * n_rows)))
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(int(max(450, 36 * n_rows)))
    canvas.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    canvas.setStyleSheet(f"border: 2px solid {_ANTHRACITE};")
    return canvas


# ---------------------------------------------------------------------------
# Report window
# ---------------------------------------------------------------------------


class CompoundReportWindow(QWidget):
    """Scrollable multi-page compound report with a table-of-contents tree."""

    def __init__(self, compounds: list, options: dict, compound_manager, file_manager, eic_defaults: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compound Report")
        self.resize(1400, 900)
        self.compound_manager = compound_manager
        self.file_manager = file_manager
        self.eic_defaults = eic_defaults
        self.options = options
        self._page_records = []  # list of (label, widget, toc_item)

        self.setStyleSheet("background-color: #ffffff;")

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.floating_header = QLabel("")
        self.floating_header.setStyleSheet("background-color: #ffffff; font-weight: bold; font-size: 13px; padding: 6px 10px; border-bottom: 1px solid #dadce0;")
        outer_layout.addWidget(self.floating_header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_layout.addWidget(splitter, stretch=1)

        self.toc_tree = QTreeWidget()
        self.toc_tree.setHeaderHidden(True)
        self.toc_tree.setMinimumWidth(220)
        self.toc_tree.setMaximumWidth(420)
        self.toc_tree.itemClicked.connect(self._on_toc_item_clicked)
        splitter.addWidget(self.toc_tree)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { background-color: #ffffff; border: none; }")
        self.scroll_area.viewport().setStyleSheet("background-color: #ffffff;")
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: #ffffff;")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(24)
        self.scroll_area.setWidget(self.content_widget)
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._on_scroll)
        splitter.addWidget(self.scroll_area)
        splitter.setStretchFactor(1, 1)

        self._build_report(compounds)
        self._on_scroll(0)

    # -- navigation ---------------------------------------------------
    def _on_toc_item_clicked(self, item, _column):
        try:
            widget = item.data(0, Qt.ItemDataRole.UserRole)
            if widget is not None:
                # Scroll so the widget's top edge lands exactly at the top of the
                # viewport. ensureWidgetVisible() only scrolls the minimum amount
                # needed to bring the widget into view, which can leave its
                # header above the visible area, so position the scrollbar
                # directly instead.
                target_y = widget.y() - self.content_layout.spacing()
                self.scroll_area.verticalScrollBar().setValue(max(target_y, 0))
        except Exception:
            # Never let an exception escape a Qt signal callback.
            traceback.print_exc()

    def _on_scroll(self, value):
        try:
            if not self._page_records:
                return
            current = self._page_records[0]
            for label, widget, toc_item in self._page_records:
                if widget.y() <= value + 10:
                    current = (label, widget, toc_item)
                else:
                    break
            label, _widget, toc_item = current
            self.floating_header.setText(label)
            if toc_item is not None:
                self.toc_tree.setCurrentItem(toc_item)
        except Exception:
            # Never let an exception escape a Qt signal callback: PyQt6 can
            # abort the whole process (silently, before anything reaches the
            # console) if a slot invoked from C++ raises unexpectedly.
            traceback.print_exc()

    # -- report construction -------------------------------------------
    def _build_report(self, compounds: list):
        files_df = self.file_manager.get_files_data()
        selected_groups = self.options["selected_groups"]
        selected_samples = self.options["selected_samples"]

        included_files = files_df[files_df["group"].isin(selected_groups) & files_df["Filepath"].isin(selected_samples)]
        if included_files.empty:
            self.content_layout.addWidget(QLabel("No samples selected / available for this report."))
            return

        group_colors = self.file_manager.group_colors

        # Global sample ordering: group (natural), then acquisition time.
        sortable = included_files.copy()
        if ACQUISITION_DATETIME_COLUMN in sortable.columns:
            sortable["_acq_sort"] = sortable[ACQUISITION_DATETIME_COLUMN].fillna("")
        else:
            sortable["_acq_sort"] = ""
        group_order = {g: i for i, g in enumerate(natsorted(sortable["group"].unique()))}
        sortable["_group_sort"] = sortable["group"].map(group_order)
        sortable = sortable.sort_values(["_group_sort", "_acq_sort", "filename"])
        ordered_samples = list(sortable.itertuples(index=False))
        sample_index = {row.Filepath: i for i, row in enumerate(ordered_samples)}

        # Build the list of (compound_dict, [(adduct, mz, polarity), ...]) to process.
        plan = []
        for compound in compounds:
            compound_name = compound.get("Name")
            adducts = []
            seen = set()
            if self.options["include_common_adducts"]:
                for adduct in self.compound_manager.get_compound_adducts(compound_name):
                    if adduct not in seen:
                        adducts.append(adduct)
                        seen.add(adduct)
            for adduct in self.options["fallback_adducts"]:
                if adduct not in seen:
                    adducts.append(adduct)
                    seen.add(adduct)

            resolved = []
            for adduct in adducts:
                precalc = self.compound_manager.get_precalculated_data(compound_name, adduct)
                mz_value = precalc.get("mz") if precalc else None
                polarity = precalc.get("polarity") if precalc else None
                if mz_value is None:
                    mz_value = self.compound_manager.calculate_compound_mz(compound_name, adduct)
                if polarity is None:
                    polarity = self.compound_manager._determine_polarity(adduct)
                if mz_value is not None:
                    resolved.append((adduct, mz_value, polarity))
            if resolved:
                plan.append((compound, resolved))

        total_pages = sum(len(r) for _, r in plan)
        progress = QProgressDialog("Generating compound report...", "Cancel", 0, max(total_pages, 1), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        ppm = float(self.eic_defaults.get("mz_tolerance_ppm", 5.0))
        calc_method = self.eic_defaults.get("eic_method", "Sum of all signals")

        # Collected data for the overview heatmaps.
        overview_rows = []  # labels
        overview_area = []
        overview_rt_dev = []
        overview_mz_dev = []

        done = 0
        for compound, resolved in plan:
            compound_name = compound.get("Name")
            compound_toc_item = QTreeWidgetItem([compound_name])
            self.toc_tree.addTopLevelItem(compound_toc_item)

            for adduct, mz_value, polarity in resolved:
                if progress.wasCanceled():
                    break
                progress.setLabelText(f"{compound_name} — {adduct}")
                progress.setValue(done)
                print(f"[compound report] building page: {compound_name} — {adduct}", flush=True)

                try:
                    mz_tolerance_da = mz_value * ppm / 1e6
                    rt_start = float(compound.get("RT_start_min", 0.0))
                    rt_end = float(compound.get("RT_end_min", 100.0))
                    rt_center = float(compound.get("RT_min", (rt_start + rt_end) / 2.0))
                    peak_width = max(rt_end - rt_start, 0.0)
                    extension = peak_width * self.options["extension_factor"]
                    ext_start = rt_start - extension
                    ext_end = rt_end + extension
                    shift_width = max(ext_end - ext_start, 0.01)

                    sample_data = {}
                    for row in ordered_samples:
                        rt_arr, int_arr, mz_arr = _extract_eic_with_mz(
                            self.file_manager,
                            row.Filepath,
                            mz_value,
                            mz_tolerance_da,
                            rt_start=ext_start,
                            rt_end=ext_end,
                            calculation_method=calc_method,
                            polarity=polarity,
                        )
                        peak = _pick_peak(rt_arr, int_arr, mz_arr, rt_start, rt_end)
                        sample_data[row.Filepath] = {
                            "rt": rt_arr,
                            "intensity": int_arr,
                            "peak": peak,
                            "group": row.group,
                            "filename": row.filename,
                        }

                    page_widget = self._build_page(compound, adduct, mz_value, polarity, rt_center, rt_start, rt_end, ordered_samples, sample_data, group_colors, shift_width)
                except Exception:
                    print(f"[compound report] FAILED building page: {compound_name} — {adduct}", flush=True)
                    traceback.print_exc()
                    done += 1
                    progress.setValue(done)
                    continue

                self.content_layout.addWidget(page_widget)

                adduct_toc_item = QTreeWidgetItem([adduct])
                adduct_toc_item.setData(0, Qt.ItemDataRole.UserRole, page_widget)
                compound_toc_item.addChild(adduct_toc_item)
                self._page_records.append((f"{compound_name} — {adduct}", page_widget, adduct_toc_item))

                # Record data for the overview heatmaps.
                overview_rows.append(f"{compound_name} | {adduct}")
                area_row, rt_dev_row, mz_dev_row = [], [], []
                for row in ordered_samples:
                    peak = sample_data[row.Filepath]["peak"]
                    if peak is None:
                        area_row.append(np.nan)
                        rt_dev_row.append(np.nan)
                        mz_dev_row.append(np.nan)
                    else:
                        area_row.append(peak["area"])
                        rt_dev_row.append(peak["apex_rt"] - rt_center)
                        if peak["apex_mz"] is not None:
                            mz_dev_row.append((peak["apex_mz"] - mz_value) / mz_value * 1e6)
                        else:
                            mz_dev_row.append(np.nan)
                overview_area.append(area_row)
                overview_rt_dev.append(rt_dev_row)
                overview_mz_dev.append(mz_dev_row)

                done += 1
                progress.setValue(done)

            if progress.wasCanceled():
                break

        progress.setValue(max(total_pages, 1))

        # Insert the overview page at the very top, once all data is known.
        print("[compound report] building overview page", flush=True)
        try:
            overview_widget = self._build_overview_page(overview_rows, overview_area, overview_rt_dev, overview_mz_dev, [row.filename for row in ordered_samples])
        except Exception:
            print("[compound report] FAILED building overview page", flush=True)
            traceback.print_exc()
            overview_widget = QLabel("Failed to build the overview page (see console/crash log for details).")
        self.content_layout.insertWidget(0, overview_widget)
        overview_toc_item = QTreeWidgetItem(["Overview"])
        overview_toc_item.setData(0, Qt.ItemDataRole.UserRole, overview_widget)
        self.toc_tree.insertTopLevelItem(0, overview_toc_item)
        self._page_records.insert(0, ("Overview", overview_widget, overview_toc_item))

        self.content_layout.addStretch(1)
        print("[compound report] report construction finished", flush=True)

    # -- overview page --------------------------------------------------
    def _build_overview_page(self, row_labels, area_rows, rt_dev_rows, mz_dev_rows, col_labels) -> QWidget:
        container = QFrame()
        container.setFrameShape(QFrame.Shape.NoFrame)
        container.setStyleSheet("background-color: #ffffff;")
        layout = QVBoxLayout(container)
        layout.addWidget(QLabel("<h2>Overview — All Processed Compounds &amp; Adducts</h2>"))

        if not row_labels:
            layout.addWidget(QLabel("No compound/adduct combinations were processed."))
            return container

        area_matrix = np.array(area_rows, dtype=float)
        # Scale each row (compound-adduct) to its own maximum -> [0, 1].
        row_max = np.nanmax(area_matrix, axis=1, keepdims=True)
        row_max[row_max == 0] = np.nan
        scaled_area = np.divide(area_matrix, row_max, out=np.full_like(area_matrix, np.nan), where=~np.isnan(row_max))
        layout.addWidget(QLabel("<b>Peak areas, scaled to the row (compound \u00d7 adduct) maximum</b>"))
        layout.addWidget(_heatmap_canvas(scaled_area, row_labels, col_labels, "Scaled Peak Area", cmap="viridis", vmin=0, vmax=1, cbar_label="Area / row max"))

        rt_dev_matrix = np.array(rt_dev_rows, dtype=float)
        layout.addWidget(QLabel("<b>Peak apex RT, relative to the compound's RT_min (minutes)</b>"))
        layout.addWidget(_heatmap_canvas(rt_dev_matrix, row_labels, col_labels, "RT Apex Deviation from RT_min", cmap="coolwarm", cbar_label="\u0394RT (min)"))

        mz_dev_matrix = np.array(mz_dev_rows, dtype=float)
        layout.addWidget(QLabel("<b>Peak apex m/z deviation from the theoretical m/z (ppm)</b>"))
        layout.addWidget(_heatmap_canvas(mz_dev_matrix, row_labels, col_labels, "m/z Deviation", cmap="coolwarm", cbar_label="\u0394m/z (ppm)"))

        return container

    # -- per compound/adduct page ---------------------------------------
    def _build_page(self, compound, adduct, mz_value, polarity, rt_center, rt_start, rt_end, ordered_samples, sample_data, group_colors, shift_width) -> QWidget:
        container = QFrame()
        container.setFrameShape(QFrame.Shape.NoFrame)
        container.setStyleSheet("background-color: #ffffff;")
        layout = QVBoxLayout(container)

        # -- header ---------------------------------------------------
        header_layout = QHBoxLayout()
        smiles = compound.get("SMILES") or compound.get("smiles")
        pixmap = _render_smiles_pixmap(smiles) if smiles else None
        image_label = QLabel()
        image_label.setFixedSize(220, 220)
        if pixmap is not None and not pixmap.isNull():
            image_label.setPixmap(pixmap)
        else:
            image_label.setText("No structure\navailable")
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setStyleSheet("color: #888888; border: 1px dashed #cccccc;")

        formula_or_mass = compound.get("ChemicalFormula") or (f"Mass: {compound.get('Mass')}" if compound.get("Mass") else "")
        comment = compound.get("Comment", "") or ""
        info_html = (
            f"<h2>{compound.get('Name')} — {adduct}</h2>"
            f"<b>Formula/Mass:</b> {formula_or_mass}<br>"
            f"<b>m/z ({polarity}):</b> {format_mz(mz_value)}<br>"
            f"<b>RT window:</b> {format_retention_time(rt_start)} \u2013 {format_retention_time(rt_end)} min "
            f"(RT_min: {format_retention_time(rt_center)} min)<br>"
            f"<b>Comment:</b> {comment}"
        )
        info_label = QLabel(info_html)
        info_label.setTextFormat(Qt.TextFormat.RichText)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        # Information on the left, structure image on the right.
        header_layout.addWidget(info_label, stretch=1)
        header_layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_layout)

        def color_for(filepath):
            group = sample_data[filepath]["group"]
            return QColor(group_colors.get(group, "#888888"))

        # -- raw + scaled overlay --------------------------------------
        row1 = QHBoxLayout()
        raw_traces = [
            (sample_data[row.Filepath]["filename"], sample_data[row.Filepath]["rt"], sample_data[row.Filepath]["intensity"], color_for(row.Filepath)) for row in ordered_samples
        ]
        row1.addWidget(build_eic_overlay_widget(raw_traces, scale_to_apex=False, peak_bounds=(rt_start, rt_end)))
        row1.addWidget(build_eic_overlay_widget(raw_traces, scale_to_apex=True, peak_bounds=(rt_start, rt_end)))
        layout.addLayout(row1)

        # -- separated by experimental group (RT-offset waterfall) -------
        layout.addWidget(QLabel("<b>EIC separated by experimental group</b>"))
        groups_present = natsorted({sample_data[row.Filepath]["group"] for row in ordered_samples})
        group_rank = {group: i for i, group in enumerate(groups_present)}
        group_entries = [
            (
                sample_data[row.Filepath]["filename"],
                sample_data[row.Filepath]["rt"],
                sample_data[row.Filepath]["intensity"],
                color_for(row.Filepath),
                group_rank[sample_data[row.Filepath]["group"]],
            )
            for row in ordered_samples
        ]
        layout.addWidget(build_eic_offset_widget(group_entries, shift_width))

        # -- ordered by group & measurement time (RT-offset waterfall) ---
        layout.addWidget(QLabel("<b>EIC ordered by experimental group &amp; measurement time</b>"))
        time_entries = [
            (sample_data[row.Filepath]["filename"], sample_data[row.Filepath]["rt"], sample_data[row.Filepath]["intensity"], color_for(row.Filepath), i)
            for i, row in enumerate(ordered_samples)
        ]
        layout.addWidget(build_eic_offset_widget(time_entries, shift_width))

        # -- peak metrics: area / apex RT / apex m/z ----------------------
        area_by_group, rt_by_group, mz_by_group = {}, {}, {}
        area_points, rt_points, mz_points = [], [], []
        for i, row in enumerate(ordered_samples):
            peak = sample_data[row.Filepath]["peak"]
            group = sample_data[row.Filepath]["group"]
            filename = sample_data[row.Filepath]["filename"]
            area = peak["area"] if peak else np.nan
            apex_rt = peak["apex_rt"] if peak else np.nan
            apex_mz = peak["apex_mz"] if peak and peak["apex_mz"] is not None else np.nan
            area_by_group.setdefault(group, []).append((area, filename))
            rt_by_group.setdefault(group, []).append((apex_rt, filename))
            mz_by_group.setdefault(group, []).append((apex_mz, filename))
            if not np.isnan(area):
                area_points.append((i, area, group, filename))
            if not np.isnan(apex_rt):
                rt_points.append((i, apex_rt, group, filename))
            if not np.isnan(apex_mz):
                mz_points.append((i, apex_mz, group, filename))

        mz_bands = [(mz_value * ppm / 1e6, color) for ppm, color in zip((1, 3, 5), _MZ_BAND_COLORS)]
        rt_bands = [(minutes, color) for minutes, color in zip((0.02, 0.05, 0.1, 0.2), _RT_BAND_COLORS)]

        layout.addWidget(QLabel("<b>Peak Area</b>"))
        row3 = QHBoxLayout()
        row3.addWidget(build_box_jitter_widget(area_by_group, group_colors, "Peak Area (log10)", log_y=True), stretch=1)
        row3.addWidget(build_scatter_by_group_widget(area_points, group_colors, "Peak Area (log10)", log_y=True), stretch=3)
        layout.addLayout(row3)

        layout.addWidget(QLabel("<b>Peak Apex RT</b>"))
        row4 = QHBoxLayout()
        row4.addWidget(build_box_jitter_widget(rt_by_group, group_colors, "Apex RT (min)", reference_value=rt_center, bands=rt_bands), stretch=1)
        row4.addWidget(build_scatter_by_group_widget(rt_points, group_colors, "Apex RT (min)", reference_value=rt_center, bands=rt_bands), stretch=3)
        layout.addLayout(row4)

        layout.addWidget(QLabel("<b>Peak Apex m/z</b>"))
        row5 = QHBoxLayout()
        row5.addWidget(build_box_jitter_widget(mz_by_group, group_colors, "Apex m/z", reference_value=mz_value, bands=mz_bands, tick_format="%.4f"), stretch=1)
        row5.addWidget(build_scatter_by_group_widget(mz_points, group_colors, "Apex m/z", reference_value=mz_value, bands=mz_bands, tick_format="%.4f"), stretch=3)
        layout.addLayout(row5)

        return container
