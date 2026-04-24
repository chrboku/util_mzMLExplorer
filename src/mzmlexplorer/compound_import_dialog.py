"""
Compound Import Dialog for CSV/TSV files
"""

import pandas as pd
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QHeaderView,
    QSplitter,
    QTextEdit,
)
from PyQt6.QtCore import Qt, QTimer
from typing import Optional, Dict, Any
from .FormulaTools import formulaTools
from .window_shared import NoScrollComboBox, NoScrollSpinBox, NoScrollDoubleSpinBox

import traceback

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def validate_formula_smiles_agreement(
    df: pd.DataFrame,
    formula_col: str,
    smiles_col: str,
    name_col: str = "",
) -> list:
    """
    Check whether the sum formula and the SMILES agree on element composition for
    every row in *df* that has both values populated.

    Uses formulaTools to parse the written chemical formula and RDKit to derive
    the molecular formula from the SMILES.  Both are normalised to base elements
    (isotope-specific keys are stripped) before comparison.

    Returns a list of human-readable labels for every row with a discrepancy or
    a parse error.
    """

    ft = formulaTools()
    problematic = []

    for idx, row in df.iterrows():
        formula_val = row[formula_col] if formula_col in df.columns else None
        smiles_val = row[smiles_col] if smiles_col in df.columns else None

        if formula_val is None or pd.isna(formula_val):
            continue
        if smiles_val is None or pd.isna(smiles_val):
            continue

        formula_str = str(formula_val).strip()
        smiles_str = str(smiles_val).strip()
        if not formula_str or not smiles_str:
            continue

        # Human-readable row label
        compound_name = f"Row {idx + 1}"
        if name_col and name_col in df.columns:
            raw_name = row[name_col]
            if raw_name is not None and not pd.isna(raw_name):
                compound_name = str(raw_name)

        # Parse the written formula
        try:
            formula_elems = ft.parseFormula(formula_str)
            formula_base = {k: v for k, v in formula_elems.items() if k[0].isalpha()}
        except Exception:
            problematic.append(f"{compound_name} (invalid formula: {formula_str})")
            continue

        # Derive formula from SMILES via RDKit
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                problematic.append(f"{compound_name} (invalid SMILES)")
                continue
            rdkit_formula_str = rdMolDescriptors.CalcMolFormula(mol)
            rdkit_elems = ft.parseFormula(rdkit_formula_str)
            rdkit_base = {k: v for k, v in rdkit_elems.items() if k[0].isalpha()}
        except Exception:
            problematic.append(f"{compound_name} (SMILES parsing error)")
            continue

        if formula_base != rdkit_base:
            problematic.append(compound_name)

    return problematic


class CompoundImportDialog(QDialog):
    """Dialog for importing compounds from CSV/TSV files"""

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.df = None
        self.preview_df = None
        self._initializing = True  # Flag to prevent premature updates

        # Set up timer for debounced updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_preview)

        self.setWindowTitle("Import Compounds")
        self.setModal(True)
        self.resize(800, 600)

        # Initialize UI
        self.init_ui()

        # Load initial data
        self.load_file_with_delimiter(";")

        # Initialization complete
        self._initializing = False

        # Now do the first update (directly, not scheduled, since we're done initializing)
        self.update_preview()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)

        # File path display
        file_info_label = QLabel(f"Importing from: {self.file_path}")
        file_info_label.setStyleSheet("QLabel { font-weight: bold; padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc; }")
        file_info_label.setWordWrap(True)
        layout.addWidget(file_info_label)

        # Create splitter for parameters and preview
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Parameters section
        params_widget = self.create_parameters_section()
        splitter.addWidget(params_widget)

        # Preview section
        preview_widget = self.create_preview_section()
        splitter.addWidget(preview_widget)

        # Set splitter proportions
        splitter.setSizes([300, 300])
        layout.addWidget(splitter)

        # Buttons
        button_layout = QHBoxLayout()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        self.import_btn = QPushButton("Import")
        self.import_btn.clicked.connect(self.on_import_clicked)
        self.import_btn.setDefault(True)

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.import_btn)

        layout.addLayout(button_layout)

    def create_parameters_section(self):
        """Create the parameters configuration section"""
        group = QGroupBox("Import Parameters")
        layout = QFormLayout(group)

        # Delimiter selection
        self.delimiter_combo = NoScrollComboBox()
        self.delimiter_combo.addItems([";", ",", "\\t (Tab)", "|", ":"])
        self.delimiter_combo.currentTextChanged.connect(self.on_delimiter_changed)
        layout.addRow("Delimiter:", self.delimiter_combo)

        # Compound name prefix
        self.name_prefix_edit = QLineEdit("")
        self.name_prefix_edit.setPlaceholderText("Optional prefix for compound names")
        self.name_prefix_edit.textChanged.connect(self.schedule_update)
        layout.addRow("Name Prefix:", self.name_prefix_edit)

        # Column selections
        self.name_column_combo = NoScrollComboBox()
        self.name_column_combo.currentTextChanged.connect(self.schedule_update)
        layout.addRow("Name Column:", self.name_column_combo)

        self.mz_column_combo = NoScrollComboBox()
        self.mz_column_combo.currentTextChanged.connect(self.schedule_update)
        layout.addRow("m/z Column:", self.mz_column_combo)

        self.rt_column_combo = NoScrollComboBox()
        self.rt_column_combo.currentTextChanged.connect(self.schedule_update)
        layout.addRow("RT Column (optional):", self.rt_column_combo)

        # Formula column (optional)
        self.formula_column_combo = NoScrollComboBox()
        self.formula_column_combo.currentTextChanged.connect(self.schedule_update)
        layout.addRow("Formula Column (optional):", self.formula_column_combo)

        # SMILES column (optional)
        self.smiles_column_combo = NoScrollComboBox()
        self.smiles_column_combo.currentTextChanged.connect(self.schedule_update)
        layout.addRow("SMILES Column (optional):", self.smiles_column_combo)

        # Polarity selection
        self.polarity_combo = NoScrollComboBox()
        self.polarity_combo.addItems(["Use Global Polarity", "Use Column"])
        self.polarity_combo.currentTextChanged.connect(self.on_polarity_method_changed)
        layout.addRow("Polarity Method:", self.polarity_combo)

        # Global polarity selection
        self.global_polarity_combo = NoScrollComboBox()
        self.global_polarity_combo.addItems(["+", "-"])
        self.global_polarity_combo.currentTextChanged.connect(self.on_global_polarity_changed)
        self.global_polarity_label = QLabel("Global Polarity:")
        layout.addRow(self.global_polarity_label, self.global_polarity_combo)

        # Polarity column selection
        self.polarity_column_combo = NoScrollComboBox()
        self.polarity_column_combo.currentTextChanged.connect(self.schedule_update)
        self.polarity_column_label = QLabel("Polarity Column:")
        layout.addRow(self.polarity_column_label, self.polarity_column_combo)

        # Initially hide polarity column controls
        self.polarity_column_label.setVisible(False)
        self.polarity_column_combo.setVisible(False)

        # RT window
        self.rt_window_spinbox = NoScrollDoubleSpinBox()
        self.rt_window_spinbox.setRange(0.1, 10.0)
        self.rt_window_spinbox.setValue(0.5)
        self.rt_window_spinbox.setSuffix(" min")
        self.rt_window_spinbox.setDecimals(2)
        self.rt_window_spinbox.valueChanged.connect(self.schedule_update)
        layout.addRow("RT Window (±):", self.rt_window_spinbox)

        return group

    def create_preview_section(self):
        """Create the preview section"""
        group = QGroupBox("Preview")
        layout = QVBoxLayout(group)

        # Info text
        self.info_label = QLabel("Preview of compounds to be imported:")
        layout.addWidget(self.info_label)

        # Preview table
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        layout.addWidget(self.preview_table)

        return group

    def on_delimiter_changed(self, text):
        """Handle delimiter change"""
        delimiter = text
        if delimiter == "\\t (Tab)":
            delimiter = "\t"

        # Block signals during file reloading to prevent cascading updates
        self._block_all_signals(True)
        try:
            self.load_file_with_delimiter(delimiter)
        finally:
            self._block_all_signals(False)

        # Only update preview if not initializing
        if not getattr(self, "_initializing", False):
            self.schedule_update()

    def _block_all_signals(self, block):
        """Block or unblock signals for all input controls"""
        controls = [
            self.delimiter_combo,
            self.name_prefix_edit,
            self.name_column_combo,
            self.mz_column_combo,
            self.rt_column_combo,
            self.formula_column_combo,
            self.smiles_column_combo,
            self.polarity_combo,
            self.polarity_column_combo,
            self.global_polarity_combo,
            self.rt_window_spinbox,
        ]
        for control in controls:
            control.blockSignals(block)

    def _update_polarity_ui_visibility(self, use_column):
        """Update polarity UI visibility without triggering signals"""
        self.polarity_column_label.setVisible(use_column)
        self.polarity_column_combo.setVisible(use_column)
        self.global_polarity_label.setVisible(not use_column)
        self.global_polarity_combo.setVisible(not use_column)

    def on_polarity_method_changed(self, text):
        """Handle polarity method change"""
        use_column = text == "Use Column"

        # Show/hide appropriate controls
        self._update_polarity_ui_visibility(use_column)

        # Only update preview if not initializing
        if not getattr(self, "_initializing", False):
            self.schedule_update()

    def on_global_polarity_changed(self, text):
        """Handle global polarity change"""
        # Only update preview if not initializing
        if not getattr(self, "_initializing", False):
            self.schedule_update()

    def schedule_update(self):
        """Schedule an update with a small delay to debounce rapid changes"""
        if not getattr(self, "_initializing", False) and hasattr(self, "update_timer"):
            # Stop any existing timer and start a new one
            self.update_timer.stop()
            self.update_timer.start(200)  # 200ms delay for more stability

    def load_file_with_delimiter(self, delimiter: str):
        """Load the file with the specified delimiter"""
        try:
            # Try to read the file
            if self.file_path.endswith(".xlsx"):
                self.df = pd.read_excel(self.file_path)
            else:
                self.df = pd.read_csv(self.file_path, sep=delimiter)

            # Update column combo boxes
            self.update_column_combos()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
            self.df = None

    def update_column_combos(self):
        """Update the column selection combo boxes"""
        if self.df is None:
            return

        columns = [""] + list(self.df.columns)

        # Block signals to prevent recursive calls during population
        combos = [
            self.name_column_combo,
            self.mz_column_combo,
            self.rt_column_combo,
            self.formula_column_combo,
            self.smiles_column_combo,
            self.polarity_column_combo,
        ]
        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            combo.blockSignals(False)

        # Try to auto-detect columns based on common names
        self.auto_detect_columns()

    def auto_detect_columns(self):
        """Auto-detect likely column mappings"""
        if self.df is None:
            return

        columns = list(self.df.columns)

        # Block signals during auto-detection to prevent recursive calls
        combos = [
            self.name_column_combo,
            self.mz_column_combo,
            self.rt_column_combo,
            self.formula_column_combo,
            self.smiles_column_combo,
            self.polarity_column_combo,
            self.polarity_combo,
        ]
        for combo in combos:
            combo.blockSignals(True)

        try:
            # Auto-detect name column
            name_candidates = [
                "name",
                "compound",
                "compound_name",
                "substance",
                "chemical",
            ]
            for col in columns:
                if any(candidate in col.lower() for candidate in name_candidates):
                    idx = self.name_column_combo.findText(col)
                    if idx >= 0:
                        self.name_column_combo.setCurrentIndex(idx)
                    break

            # Auto-detect m/z column
            mz_candidates = ["mz", "m/z", "mass", "molecular_mass", "exact_mass"]
            for col in columns:
                if any(candidate in col.lower() for candidate in mz_candidates):
                    idx = self.mz_column_combo.findText(col)
                    if idx >= 0:
                        self.mz_column_combo.setCurrentIndex(idx)
                    break

            # Auto-detect RT column
            rt_candidates = ["rt", "retention_time", "time", "retention", "rt_min"]
            for col in columns:
                if any(candidate in col.lower() for candidate in rt_candidates):
                    idx = self.rt_column_combo.findText(col)
                    if idx >= 0:
                        self.rt_column_combo.setCurrentIndex(idx)
                    break

            # Auto-detect formula column
            formula_candidates = [
                "formula",
                "chemicalformula",
                "sum_formula",
                "molformula",
            ]
            for col in columns:
                if any(candidate in col.lower() for candidate in formula_candidates):
                    idx = self.formula_column_combo.findText(col)
                    if idx >= 0:
                        self.formula_column_combo.setCurrentIndex(idx)
                    break

            # Auto-detect SMILES column
            smiles_candidates = ["smiles", "smi"]
            for col in columns:
                if any(candidate in col.lower() for candidate in smiles_candidates):
                    idx = self.smiles_column_combo.findText(col)
                    if idx >= 0:
                        self.smiles_column_combo.setCurrentIndex(idx)
                    break

            # Auto-detect polarity column
            polarity_candidates = ["polarity", "pol", "charge", "mode", "ion_mode"]
            for col in columns:
                if any(candidate in col.lower() for candidate in polarity_candidates):
                    idx = self.polarity_column_combo.findText(col)
                    if idx >= 0:
                        self.polarity_column_combo.setCurrentIndex(idx)
                        # Switch to column mode if polarity column detected
                        self.polarity_combo.setCurrentText("Use Column")
                        # Manually trigger the UI changes without signals
                        self._update_polarity_ui_visibility(True)
                    break

        finally:
            # Re-enable signals
            for combo in combos:
                combo.blockSignals(False)

    def update_preview(self):
        """Update the preview table"""
        if self.df is None or getattr(self, "_initializing", False):
            return

        try:
            # Get current selections
            name_col = self.name_column_combo.currentText()
            mz_col = self.mz_column_combo.currentText()
            rt_col = self.rt_column_combo.currentText()
            name_prefix = self.name_prefix_edit.text()
            rt_window = self.rt_window_spinbox.value()

            # Polarity settings
            use_polarity_column = self.polarity_combo.currentText() == "Use Column"
            polarity_col = self.polarity_column_combo.currentText() if use_polarity_column else ""
            global_polarity = self.global_polarity_combo.currentText()

        except Exception as e:
            # If there's an error getting the selections (e.g., during initialization), just return
            print(f"Error getting selections: {e}")
            return

        # Check if required columns are selected
        required_missing = []
        if not name_col:
            required_missing.append("Name")
        if not mz_col:
            required_missing.append("m/z")
        # RT column is now optional
        if use_polarity_column and not polarity_col:
            required_missing.append("Polarity")

        if required_missing:
            self.preview_table.setRowCount(0)
            self.info_label.setText(f"Please select all required columns: {', '.join(required_missing)}")
            self.import_btn.setEnabled(False)
            return

        try:
            # Create preview dataframe
            preview_data = []

            for idx, row in self.df.iterrows():
                # Check required fields
                if pd.isna(row[name_col]) or pd.isna(row[mz_col]):
                    continue

                # RT is now optional - if not provided, defaults will be used (0-100 min)
                has_rt = rt_col and not pd.isna(row[rt_col])

                # Check polarity field if using column
                if use_polarity_column and pd.isna(row[polarity_col]):
                    continue

                compound_name = name_prefix + str(row[name_col])
                mz_value = float(row[mz_col])

                # Handle RT value - use provided value or defaults
                if has_rt:
                    rt_value = float(row[rt_col])
                    rt_start = rt_value - rt_window
                    rt_end = rt_value + rt_window
                else:
                    # Default RT values when not specified
                    rt_value = 50.0  # Center of default range
                    rt_start = 0.0  # Start of default range
                    rt_end = 100.0  # End of default range

                # Determine polarity
                if use_polarity_column:
                    polarity_value = str(row[polarity_col]).strip().lower()
                    # Map common polarity representations
                    if polarity_value in ["pos", "positive", "+", "1"]:
                        polarity = "+"
                    elif polarity_value in ["neg", "negative", "-", "0"]:
                        polarity = "-"
                    else:
                        # Default to positive if unclear
                        polarity = "+"
                else:
                    polarity = global_polarity

                # Create m/z adduct format
                adduct = f"[{mz_value:.4f}]{polarity}"

                preview_data.append(
                    {
                        "Name": compound_name,
                        "RT_min": rt_value,
                        "RT_start_min": rt_start,
                        "RT_end_min": rt_end,
                        "Common_adducts": adduct,
                    }
                )

            self.preview_df = pd.DataFrame(preview_data)

            # Update preview table
            self.populate_preview_table()

            # Update info label
            self.info_label.setText(f"Preview: {len(self.preview_df)} compounds will be imported")

            # Enable import button if we have data
            self.import_btn.setEnabled(len(self.preview_df) > 0)

        except Exception as e:
            print(f"Exception in update_preview: {str(e)}")

            traceback.print_exc()
            self.preview_table.setRowCount(0)
            self.info_label.setText(f"Error creating preview: {str(e)}")
            self.import_btn.setEnabled(False)

    def populate_preview_table(self, max_rows=5):
        """Populate the preview table with data"""
        if self.preview_df is None or self.preview_df.empty:
            self.preview_table.setRowCount(0)
            return

        preview_df = self.preview_df.head(max_rows)

        # Set up table
        self.preview_table.setRowCount(len(preview_df))
        self.preview_table.setColumnCount(len(preview_df.columns))
        self.preview_table.setHorizontalHeaderLabels(list(preview_df.columns))

        # Populate data
        for row_idx, (_, row) in enumerate(preview_df.iterrows()):
            for col_idx, value in enumerate(row):
                display_text = "" if (value is None or (isinstance(value, float) and pd.isna(value))) else str(value)
                item = QTableWidgetItem(display_text)
                self.preview_table.setItem(row_idx, col_idx, item)

        # Resize columns to content
        header = self.preview_table.horizontalHeader()
        for i in range(len(preview_df.columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

    def on_import_clicked(self):
        """Handle Import button click: validate formula/SMILES agreement before accepting."""
        formula_col = self.formula_column_combo.currentText()
        smiles_col = self.smiles_column_combo.currentText()

        if formula_col and smiles_col and self.df is not None:
            problematic = self._validate_formula_smiles(formula_col, smiles_col)
            if problematic:
                names_text = "\n".join(f"  \u2022 {name}" for name in problematic)
                msg_text = f"{len(problematic)} compound(s) have a mismatch between sum formula and SMILES:\n\n{names_text}\n\nDo you want to continue the import anyway?"
                result = QMessageBox.question(
                    self,
                    "Formula/SMILES Mismatch",
                    msg_text,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if result != QMessageBox.StandardButton.Yes:
                    return  # Keep dialog open so the user can review/cancel

        self.accept()

    def _validate_formula_smiles(self, formula_col: str, smiles_col: str) -> list:
        """Delegate to the module-level validation function."""
        return validate_formula_smiles_agreement(
            self.df,
            formula_col,
            smiles_col,
            name_col=self.name_column_combo.currentText(),
        )

    def get_import_data(self) -> Optional[pd.DataFrame]:
        """Get the data to be imported"""
        return self.preview_df

    def get_import_parameters(self) -> Dict[str, Any]:
        """Get the import parameters"""
        return {
            "delimiter": self.delimiter_combo.currentText(),
            "name_prefix": self.name_prefix_edit.text(),
            "name_column": self.name_column_combo.currentText(),
            "mz_column": self.mz_column_combo.currentText(),
            "rt_column": self.rt_column_combo.currentText(),
            "rt_window": self.rt_window_spinbox.value(),
            "polarity_method": self.polarity_combo.currentText(),
            "polarity_column": self.polarity_column_combo.currentText(),
            "global_polarity": self.global_polarity_combo.currentText(),
        }
