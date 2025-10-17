import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QLabel,
    QDoubleSpinBox,
    QLineEdit,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QHeaderView,
    QMenuBar,
    QMenu,
    QDialog,
    QFormLayout,
    QProgressDialog,
    QFrame,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QTimer, QMimeData, QUrl, QSettings
from PyQt6.QtGui import QFont, QAction, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QFont
import pandas as pd
import pymzml
import numpy as np
from .compound_manager import CompoundManager
from .file_manager import FileManager
from .eic_window import EICWindow
from .compound_import_dialog import CompoundImportDialog
from .multi_adduct_window import MultiAdductWindow
from natsort import natsorted, index_natsorted

import toml


class MzMLExplorerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mzML Explorer")
        self.setGeometry(100, 100, 1200, 800)

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Initialize settings
        self.settings = QSettings("mzMLExplorer", "mzMLExplorer")

        # Data storage (initialize before loading settings that depend on it)
        self.file_manager = FileManager()
        self.compound_manager = CompoundManager()
        self.eic_windows = []

        # Peak integration data storage
        # Key: (compound_name, ion_name) -> dict with integration data
        self.peak_integration_data = {}

        # Load settings after data storage is initialized
        self.load_eic_defaults()

        self.init_ui()
        self.load_stylesheet()

    def init_ui(self):
        """Initialize the user interface"""
        # Create menu bar
        self.create_menu_bar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Files table
        left_panel = QGroupBox("Loaded Files")
        left_panel.setAcceptDrops(True)
        left_layout = QVBoxLayout(left_panel)

        self.files_table = QTableWidget()
        self.files_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.files_table.setAcceptDrops(True)
        self.files_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.files_table.customContextMenuRequested.connect(
            self.show_files_context_menu
        )
        left_layout.addWidget(self.files_table)

        # Right panel: Compounds tree
        right_panel = QGroupBox("Compounds")
        right_panel.setAcceptDrops(True)
        right_layout = QVBoxLayout(right_panel)

        # Add filter line
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.compound_filter = QLineEdit()
        self.compound_filter.setPlaceholderText(
            "mz 100-200, rt 5-10, or compound name pattern"
        )
        self.compound_filter.textChanged.connect(self.filter_compounds)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.compound_filter)
        right_layout.addLayout(filter_layout)

        self.compounds_table = QTableWidget()
        self.compounds_table.setColumnCount(3)
        self.compounds_table.setHorizontalHeaderLabels(
            ["Name", "Retention Time", "Type"]
        )
        self.compounds_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.compounds_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.compounds_table.customContextMenuRequested.connect(
            self.show_compound_context_menu
        )
        self.compounds_table.setAcceptDrops(True)
        self.compounds_table.setSortingEnabled(True)

        # Configure table headers
        header = self.compounds_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        right_layout.addWidget(self.compounds_table)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])

        main_layout.addWidget(splitter)

        # Create memory usage label for bottom left corner
        self.memory_label = QLabel("Memory: -- MB")
        self.memory_label.setStyleSheet("QLabel { font-size: 9px; color: #666; }")

        # Add memory label to status bar
        self.statusBar().addPermanentWidget(self.memory_label)
        self.statusBar().showMessage("Ready")

        # Timer for updating memory usage
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_label)
        self.memory_timer.start(2000)  # Update every 2 seconds

    def load_files(self):
        """Load mzML files from a TSV or Excel file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load File List",
            "",
            "Excel files (*.xlsx);;TSV files (*.tsv);;CSV files (*.csv)",
        )

        if file_path:
            try:
                # Load the file list
                if file_path.endswith(".xlsx"):
                    df = pd.read_excel(file_path)
                elif file_path.endswith(".tsv"):
                    df = pd.read_csv(file_path, sep="\t")
                else:
                    df = pd.read_csv(file_path)

                # Validate required columns
                if "Filepath" not in df.columns:
                    QMessageBox.warning(
                        self, "Error", "The file must contain a 'Filepath' column!"
                    )
                    return

                # Load files using file manager
                self.file_manager.load_files(df)
                self.update_files_table()

                # If memory mode is enabled, load the new files into memory with progress
                if self.file_manager.keep_in_memory:
                    self.load_files_to_memory_with_progress()

                total_files = len(self.file_manager.get_files_data())
                self.statusBar().showMessage(
                    f"Files loaded. Total: {total_files} files"
                )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load files: {str(e)}")

    def load_compounds(self):
        """Load compounds from a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Compounds",
            "",
            "All supported (*.xlsx *.csv *.tsv);;Excel files (*.xlsx);;CSV files (*.csv);;TSV files (*.tsv)",
        )

        if file_path:
            self.load_compounds_from_file(file_path)

    def clear_compounds(self):
        """Clear all loaded compounds"""
        if self.compound_manager.get_compounds_data().empty:
            QMessageBox.information(
                self, "Information", "No compounds loaded to clear."
            )
            return

        reply = QMessageBox.question(
            self,
            "Clear Compounds",
            "Are you sure you want to clear all loaded compounds?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Clear compounds data
            self.compound_manager.compounds_data = pd.DataFrame()
            self.compound_manager.compound_adduct_data = {}

            # Update the UI
            self.update_compounds_table()

            # Update status
            self.statusBar().showMessage("All compounds cleared.")

    def clear_files(self):
        """Clear all loaded files"""
        if self.file_manager.get_files_data().empty:
            QMessageBox.information(self, "Information", "No files loaded to clear.")
            return

        reply = QMessageBox.question(
            self,
            "Clear Files",
            "Are you sure you want to clear all loaded files?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Clear files data
            self.file_manager.files_data = pd.DataFrame()

            # Clear any cached file data in memory
            if hasattr(self.file_manager, "files_in_memory"):
                self.file_manager.files_in_memory = {}

            # Update the UI
            self.update_files_table()

            # Update status
            self.statusBar().showMessage("All files cleared.")

    def generate_templates(self):
        """Generate template Excel files for file list and compounds"""
        try:
            # Ask user where to save templates
            save_dir = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Templates", ""
            )

            if not save_dir:
                return

            # Generate file list template
            files_template_data = {
                "Filepath": [
                    "C:\\path\\to\\your\\file1.mzML",
                    "C:\\path\\to\\your\\file2.mzML",
                    "C:\\path\\to\\your\\file3.mzML",
                    "C:\\path\\to\\your\\file4.mzML",
                ],
                "group": ["Control", "Control", "Treatment", "Treatment"],
                "color": ["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"],
                "batch": ["Batch1", "Batch1", "Batch1", "Batch2"],
                "injection_volume": [5.0, 5.0, 5.0, 5.0],
                "sample_id": ["CTL_001", "CTL_002", "TRT_001", "TRT_002"],
            }

            files_template_df = pd.DataFrame(files_template_data)
            files_template_path = os.path.join(save_dir, "file_list_template.xlsx")
            files_template_df.to_excel(files_template_path, index=False)

            # Generate compounds template
            compounds_template_data = {
                "Name": [
                    "Caffeine",
                    "Theophylline",
                    "Unknown_Compound_1",
                    "Unknown_Compound_2",
                ],
                "ChemicalFormula": [
                    "C8H10N4O2",
                    "C7H8N4O2",
                    "",  # Empty for mass-based compound
                    "",  # Empty for mass-based compound
                ],
                "Mass": [
                    "",  # Empty for formula-based compound
                    "",  # Empty for formula-based compound
                    194.0579,  # Mass-based compound
                    256.1234,  # Mass-based compound
                ],
                "RT_min": [5.2, 4.8, 3.1, 7.5],
                "RT_start_min": [4.8, 4.4, 2.7, 7.0],
                "RT_end_min": [5.6, 5.2, 3.5, 8.0],
                "Common_adducts": [
                    "[M+H]+, [M+Na]+, [M+K]+",
                    "[M+H]+, [M+Na]+, [M-H]-",
                    "[M+H]+, [195.0652]+",  # Mix of standard and custom m/z
                    "[257.1307]+, [255.1151]-",  # Custom m/z values only
                ],
                "compound_class": ["Alkaloid", "Alkaloid", "Unknown", "Unknown"],
                "cas_number": ["58-08-2", "58-55-9", "", ""],
            }

            compounds_template_df = pd.DataFrame(compounds_template_data)

            # Create adducts template
            adducts_template_data = {
                "Adduct": [
                    "[M+3H]+++",
                    "[M+2H+Na]+++",
                    "[M+H+2Na]+++",
                    "[M+3Na]+++",
                    "[M+2H]++",
                    "[M+H+NH4]++",
                    "[M+H+Na]++",
                    "[M+H+K]++",
                    "[M+ACN+2H]++",
                    "[M+2Na]++",
                    "[M+2ACN+2H]++",
                    "[M+3ACN+2H]++",
                    "[M+H]+",
                    "[M+NH4]+",
                    "[M+Na]+",
                    "[M+CH3OH+H]+",
                    "[M+K]+",
                    "[M+ACN+H]+",
                    "[M+2Na-H]+",
                    "[M+IsoProp+H]+",
                    "[M+ACN+Na]+",
                    "[M+2K-H]+",
                    "[M+DMSO+H]+",
                    "[M+2ACN+H]+",
                    "[M+IsoProp+Na+H]+",
                    "[2M+H]+",
                    "[2M+NH4]+",
                    "[2M+Na]+",
                    "[2M+K]+",
                    "[2M+ACN+H]+",
                    "[2M+ACN+Na]+",
                    "[M-3H]---",
                    "[M-2H]--",
                    "[M-H2O-H]-",
                    "[M-H]-",
                    "[M+Na-2H]-",
                    "[M+Cl]-",
                    "[M+K-2H]-",
                    "[M+FA-H]-",
                    "[M+Hac-H]-",
                    "[M+Br]-",
                    "[M+TFA-H]-",
                    "[2M-H]-",
                    "[2M+FA-H]-",
                    "[2M+Hac-H]-",
                    "[3M-H]-",
                ],
                "Mass_change": [
                    1.007276,
                    8.33459,
                    15.76619,
                    22.989218,
                    1.007276,
                    9.52055,
                    11.998247,
                    19.985217,
                    21.52055,
                    22.989218,
                    42.033823,
                    62.547097,
                    1.007276,
                    18.033823,
                    22.989218,
                    33.033489,
                    38.963158,
                    42.033823,
                    44.97116,
                    61.06534,
                    64.015765,
                    76.91904,
                    79.02122,
                    83.06037,
                    84.05511,
                    1.007276,
                    18.033823,
                    22.989218,
                    38.963158,
                    42.033823,
                    64.015765,
                    -1.007276,
                    -1.007276,
                    -19.01839,
                    -1.007276,
                    20.974666,
                    34.969402,
                    36.948606,
                    44.998201,
                    59.013851,
                    78.918885,
                    112.985586,
                    -1.007276,
                    44.998201,
                    59.013851,
                    1.007276,
                ],
                "Charge": [
                    3,
                    3,
                    3,
                    3,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    -3,
                    -2,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                "Multiplier": [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                ],
            }

            adducts_template_df = pd.DataFrame(adducts_template_data)

            # Save to Excel with multiple sheets
            compounds_template_path = os.path.join(save_dir, "compounds_template.xlsx")
            with pd.ExcelWriter(compounds_template_path, engine="openpyxl") as writer:
                compounds_template_df.to_excel(
                    writer, sheet_name="Compounds", index=False
                )
                adducts_template_df.to_excel(writer, sheet_name="Adducts", index=False)

            # Show success message
            QMessageBox.information(
                self,
                "Templates Generated",
                f"Template files have been generated successfully:\n\n"
                f"• File List Template: {files_template_path}\n"
                f"• Compounds Template: {compounds_template_path}\n\n"
                f"Instructions:\n"
                f"• In the file list template:\n"
                f"  - 'Filepath' column: Full path to your mzML files (required)\n"
                f"  - 'group' column: Group names for organizing samples\n"
                f"  - 'color' column: Hex colors for visualization (optional)\n"
                f"  - Other columns: Additional metadata as needed\n\n"
                f"• In the compounds template:\n"
                f"  - Use either 'ChemicalFormula' OR 'Mass' column\n"
                f"  - 'Common_adducts': Standard adducts (e.g., [M+H]+) or\n"
                f"    custom m/z values (e.g., [197.1234]+, [255.0987]-)\n"
                f"  - 'Adducts' sheet: Defines standard adduct properties\n\n"
                f"• Files will be sorted by group first, then by filename\n"
                f"• The table will show filename first, then metadata, then full path\n\n"
                f"Please edit these files with your data before loading.",
            )

            self.statusBar().showMessage("Template files generated successfully")

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to generate templates: {str(e)}"
            )

    def update_files_table(self):
        """Update the files table with loaded data"""
        files_display_data = self.file_manager.get_files_display_data()

        if files_display_data.empty:
            self.files_table.setRowCount(0)
            self.files_table.setColumnCount(0)
            return

        # Set up table
        self.files_table.setRowCount(len(files_display_data))
        self.files_table.setColumnCount(len(files_display_data.columns))
        self.files_table.setHorizontalHeaderLabels(files_display_data.columns.tolist())

        # Get original files data for group information
        files_data = self.file_manager.get_files_data()

        # Populate table
        for i, (index, row) in enumerate(files_display_data.iterrows()):
            for j, (col_name, value) in enumerate(row.items()):
                item = QTableWidgetItem(str(value))

                # Special handling for color column - only show background color
                if col_name == "color":
                    # Don't show the color text, just the background
                    item.setText("")
                    if pd.notna(value) and value:
                        item.setBackground(QColor(str(value)))
                else:
                    # For other columns, set background color based on group
                    if "group" in row.index and pd.notna(row["group"]):
                        group_color = self.file_manager.get_group_color(row["group"])
                        if group_color and col_name != "color":
                            # Apply a lighter version of the group color for non-color columns
                            color = QColor(group_color)
                            color.setAlpha(50)  # Make it semi-transparent
                            item.setBackground(color)

                self.files_table.setItem(i, j, item)

        # Adjust column widths
        self.files_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

    def update_compounds_table(self):
        """Update the compounds table with loaded data"""
        self.compounds_table.setRowCount(0)
        compounds_data = self.compound_manager.get_compounds_data()

        if compounds_data.empty:
            return

        # Set row count
        self.compounds_table.setRowCount(len(compounds_data))

        # Populate table
        for row_idx, (_, compound) in enumerate(compounds_data.iterrows()):
            compound_name = compound["Name"]

            # Compound name
            name_item = QTableWidgetItem(compound_name)
            name_item.setData(Qt.ItemDataRole.UserRole, compound.to_dict())
            self.compounds_table.setItem(row_idx, 0, name_item)

            # Retention time info
            rt_text = ""
            if "RT_minutes" in compound and pd.notna(compound["RT_minutes"]):
                avg_rt = compound["RT_minutes"]
                rt_text = f"{avg_rt:.1f} min"
            elif compound.get("RT_start_min") and compound.get("RT_end_min"):
                rt_start = compound["RT_start_min"]
                rt_end = compound["RT_end_min"]
                rt_text = f"{rt_start:.1f}-{rt_end:.1f} min"
            else:
                rt_text = "not set"

            rt_item = QTableWidgetItem(rt_text)
            self.compounds_table.setItem(row_idx, 1, rt_item)

            # Compound type
            compound_type = compound.get("compound_type", "formula")
            type_display = {
                "formula": "Formula",
                "mass": "Mass",
                "mz_only": "Adduct",
            }.get(compound_type, compound_type)

            type_item = QTableWidgetItem(type_display)
            self.compounds_table.setItem(row_idx, 2, type_item)

    def show_files_context_menu(self, position):
        """Show context menu for file operations"""
        item = self.files_table.itemAt(position)
        if item is None:
            return

        # Get the selected row
        row = item.row()

        # Create context menu
        menu = QMenu(self)

        # Get file information for display
        file_name = "Unknown"
        if self.files_table.columnCount() > 0:
            name_item = self.files_table.item(row, 0)
            if name_item:
                file_name = name_item.text()

        # Add remove file action
        remove_action = QAction(f"Remove File: {file_name}", self)
        remove_action.triggered.connect(
            lambda checked, r=row: self.remove_file_at_row(r)
        )
        menu.addAction(remove_action)

        # Show menu at cursor position
        menu.exec(self.files_table.mapToGlobal(position))

    def remove_file_at_row(self, row):
        """Remove a file at the specified row"""
        if row < 0 or row >= self.files_table.rowCount():
            return

        # Get file information for confirmation
        file_name = "Unknown"
        if self.files_table.columnCount() > 0:
            name_item = self.files_table.item(row, 0)
            if name_item:
                file_name = name_item.text()

        # Confirm removal
        reply = QMessageBox.question(
            self,
            "Remove File",
            f"Are you sure you want to remove the file '{file_name}' from the list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Get the current files data
                files_data = self.file_manager.get_files_data()

                if not files_data.empty and row < len(files_data):
                    # Remove the file from the file manager
                    files_data_copy = files_data.copy()
                    files_data_copy = files_data_copy.drop(
                        files_data_copy.index[row]
                    ).reset_index(drop=True)

                    # Update the file manager with the new data
                    self.file_manager.files_data = files_data_copy

                    # Update the UI
                    self.update_files_table()

                    # Update status
                    remaining_files = len(self.file_manager.get_files_data())
                    self.statusBar().showMessage(
                        f"File removed. Remaining: {remaining_files} files"
                    )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to remove file: {str(e)}")

    def show_compound_context_menu(self, position):
        """Show context menu for compound adducts"""
        item = self.compounds_table.itemAt(position)
        if item is None:
            return

        # Get the compound data from the first column (name column)
        row = item.row()
        name_item = self.compounds_table.item(row, 0)
        compound_data = name_item.data(Qt.ItemDataRole.UserRole)

        if not compound_data:
            return

        compound_name = compound_data["Name"]

        # Create context menu
        menu = QMenu(self)
        menu.setTitle(f"Adducts for {compound_name}")

        # Get categorized adducts
        adducts_info = self.compound_manager.get_compound_adducts_categorized(
            compound_name
        )
        can_calculate = self.compound_manager.can_calculate_adducts_from_formula(
            compound_name
        )

        # Add specified adducts
        specified_adducts = adducts_info["specified"]
        if specified_adducts:
            if can_calculate:
                # Show specified adducts at top for compounds with formula/mass
                for adduct in specified_adducts:
                    self._add_adduct_action(menu, compound_data, adduct, specified=True)

                # Add separator
                if adducts_info["remaining"]:
                    menu.addSeparator()

                # Add remaining adducts
                for adduct in adducts_info["remaining"]:
                    self._add_adduct_action(
                        menu, compound_data, adduct, specified=False
                    )
            else:
                # For m/z only compounds, show only the specified adducts
                for adduct in specified_adducts:
                    self._add_adduct_action(menu, compound_data, adduct, specified=True)
        elif can_calculate:
            # No adducts specified but can calculate from formula - show all possible
            all_adducts = self.compound_manager.get_all_available_adducts()
            for adduct in all_adducts:
                self._add_adduct_action(menu, compound_data, adduct, specified=False)
        else:
            # No adducts and can't calculate - show message
            no_adducts_action = QAction("No adducts available", self)
            no_adducts_action.setEnabled(False)
            menu.addAction(no_adducts_action)

        # Add separator before multi-adduct options
        if menu.actions():
            menu.addSeparator()

        # Add multi-adduct options
        self._add_multi_adduct_actions(menu, compound_data, adducts_info, can_calculate)

        # Show menu at cursor position
        if menu.actions():
            menu.exec(self.compounds_table.mapToGlobal(position))

    def _add_adduct_action(self, menu, compound_data, adduct, specified=True):
        """Add an adduct action to the context menu"""
        # Get pre-calculated data for display
        compound_name = compound_data["Name"]
        precalc_data = self.compound_manager.get_precalculated_data(
            compound_name, adduct
        )

        if precalc_data:
            display_name = precalc_data["display_name"]
            mz_value = precalc_data["mz"]
            polarity = precalc_data["polarity"]

            if mz_value is not None:
                action_text = f"{display_name} (m/z: {mz_value:.4f})"
            else:
                action_text = f"{display_name} (m/z: calculation failed)"
        else:
            # Try to calculate on the fly
            display_name = self.compound_manager.get_adduct_display_name(
                compound_name, adduct
            )
            mz_value = self.compound_manager.calculate_compound_mz(
                compound_name, adduct
            )
            polarity = self.compound_manager._determine_polarity(adduct)

            if mz_value is not None:
                action_text = f"{display_name} (m/z: {mz_value:.4f})"
            else:
                action_text = f"{display_name} (m/z: not calculated)"

        # Create action
        action = QAction(action_text, self)

        # Make specified adducts bold
        if specified:
            font = action.font()
            font.setBold(True)
            action.setFont(font)

        # Connect to EIC window function
        action.triggered.connect(
            lambda checked,
            c=compound_data,
            a=adduct,
            m=mz_value,
            p=polarity: self.show_eic_window(c, a, m, p)
        )

        menu.addAction(action)

    def _add_multi_adduct_actions(
        self, menu, compound_data, adducts_info, can_calculate
    ):
        """Add multi-adduct window actions to the context menu"""
        compound_name = compound_data["Name"]
        compound_type = compound_data.get("compound_type", "formula")

        # Check if multi-adduct options should be enabled
        # Enable for compounds with formula or mass, disable for m/z only compounds
        enable_multi_adduct = compound_type in ["formula", "mass"]

        # Option 1: Show predefined adducts in multi-EIC window
        specified_adducts = adducts_info["specified"]
        if specified_adducts:
            predefined_action = QAction("📊 Show Predefined Adducts (Multi-EIC)", self)
            predefined_action.setEnabled(enable_multi_adduct)
            if enable_multi_adduct:
                predefined_action.triggered.connect(
                    lambda checked, c=compound_data: self.show_multi_adduct_window(
                        c, show_predefined_only=True
                    )
                )
            else:
                # Add tooltip explaining why it's disabled
                predefined_action.setToolTip(
                    "Multi-adduct view not available for specific m/z adducts. "
                    "Please provide chemical formula or mass to calculate adducts."
                )
            menu.addAction(predefined_action)

        # Option 2: Show all possible adducts in multi-EIC window
        # This option is only shown if we can calculate adducts (formula or mass available)
        if can_calculate and enable_multi_adduct:
            all_adducts_action = QAction("📊 Show All Adducts (Multi-EIC)", self)
            all_adducts_action.triggered.connect(
                lambda checked, c=compound_data: self.show_multi_adduct_window(
                    c, show_predefined_only=False
                )
            )
            menu.addAction(all_adducts_action)

        # If no multi-adduct options are available for m/z only compounds, show informational message
        if not enable_multi_adduct and not specified_adducts:
            info_action = QAction("ℹ️ Multi-adduct view requires formula or mass", self)
            info_action.setEnabled(False)
            menu.addAction(info_action)
            menu.addAction(all_adducts_action)

    def show_multi_adduct_window(self, compound, show_predefined_only=True):
        """Show multi-adduct EIC window"""
        if self.file_manager.get_files_data().empty:
            QMessageBox.warning(self, "Warning", "No files loaded!")
            return

        try:
            compound_name = compound["Name"]

            # Prepare adducts data
            adducts_data = []

            if show_predefined_only:
                # Get only predefined adducts
                specified_adducts = self.compound_manager.get_compound_adducts(
                    compound_name
                )
                for adduct in specified_adducts:
                    mz_value = self.compound_manager.calculate_compound_mz(
                        compound_name, adduct
                    )
                    polarity = self.compound_manager._determine_polarity(adduct)
                    adducts_data.append((adduct, mz_value, polarity))
            else:
                # Get all possible adducts
                if self.compound_manager.can_calculate_adducts_from_formula(
                    compound_name
                ):
                    all_adducts = self.compound_manager.get_all_available_adducts()
                    for adduct in all_adducts:
                        mz_value = self.compound_manager.calculate_compound_mz(
                            compound_name, adduct
                        )
                        polarity = self.compound_manager._determine_polarity(adduct)
                        if mz_value is not None:  # Only add adducts with valid m/z
                            adducts_data.append((adduct, mz_value, polarity))
                else:
                    # For m/z only compounds, fall back to predefined
                    specified_adducts = self.compound_manager.get_compound_adducts(
                        compound_name
                    )
                    for adduct in specified_adducts:
                        mz_value = self.compound_manager.calculate_compound_mz(
                            compound_name, adduct
                        )
                        polarity = self.compound_manager._determine_polarity(adduct)
                        adducts_data.append((adduct, mz_value, polarity))

            if not adducts_data:
                QMessageBox.information(
                    self, "Information", "No adducts available for this compound."
                )
                return

            # Create multi-adduct window
            multi_window = MultiAdductWindow(
                compound,
                adducts_data,
                self.file_manager,
                defaults=self.eic_defaults,
                show_predefined_only=show_predefined_only,
                parent=None,
            )

            # Show the window
            multi_window.show()
            multi_window.raise_()
            multi_window.activateWindow()

            # Keep reference to prevent garbage collection
            self.eic_windows.append(multi_window)

            # Remove from list when window is closed
            def on_window_closed():
                if multi_window in self.eic_windows:
                    self.eic_windows.remove(multi_window)

            multi_window.destroyed.connect(on_window_closed)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to create multi-adduct window: {str(e)}"
            )
            import traceback

            traceback.print_exc()

    def filter_compounds(self):
        """Filter compounds based on the filter text"""
        filter_text = self.compound_filter.text().strip()

        if not filter_text:
            # Show all compounds if filter is empty
            for i in range(self.compounds_table.rowCount()):
                self.compounds_table.setRowHidden(i, False)
            return

        # Parse filter text
        filter_type, filter_params = self._parse_filter_text(filter_text)

        for i in range(self.compounds_table.rowCount()):
            name_item = self.compounds_table.item(i, 0)
            compound_data = name_item.data(Qt.ItemDataRole.UserRole)

            if not compound_data:
                self.compounds_table.setRowHidden(i, True)
                continue

            compound_name = compound_data["Name"]
            show_compound = False

            if filter_type == "mz":
                # Check if any adducts have m/z in range
                min_mz, max_mz = filter_params
                adducts = compound_data.get("Common_adducts", "")

                if isinstance(adducts, str) and adducts.strip():
                    adduct_list = [a.strip() for a in adducts.split(",") if a.strip()]
                    for adduct in adduct_list:
                        try:
                            # Use pre-calculated m/z value if available
                            precalc_data = self.compound_manager.get_precalculated_data(
                                compound_name, adduct
                            )

                            if precalc_data and precalc_data["mz"] is not None:
                                mz_value = precalc_data["mz"]
                            else:
                                # Fallback to calculation if pre-calculated data not available
                                mz_value = self.compound_manager.calculate_compound_mz(
                                    compound_name, adduct
                                )

                            if mz_value is not None and min_mz <= mz_value <= max_mz:
                                show_compound = True
                                break
                        except:
                            continue

            elif filter_type == "rt":
                # Check if RT is in range
                min_rt, max_rt = filter_params
                if "RT_minutes" in compound_data and pd.notna(
                    compound_data["RT_minutes"]
                ):
                    rt_value = float(compound_data["RT_minutes"])
                    show_compound = min_rt <= rt_value <= max_rt
                elif compound_data.get("RT_start_min") and compound_data.get(
                    "RT_end_min"
                ):
                    # Use average of RT window if RT_minutes not available
                    avg_rt = (
                        float(compound_data["RT_start_min"])
                        + float(compound_data["RT_end_min"])
                    ) / 2
                    show_compound = min_rt <= avg_rt <= max_rt

            elif filter_type == "name":
                # Regex search on compound name
                import re

                try:
                    pattern = re.compile(filter_params, re.IGNORECASE)
                    show_compound = bool(pattern.search(compound_name))
                except re.error:
                    # If regex is invalid, fall back to simple string search
                    show_compound = filter_params.lower() in compound_name.lower()

            # Show/hide compound based on filter result
            self.compounds_table.setRowHidden(i, not show_compound)

    def _parse_filter_text(self, filter_text):
        """
        Parse filter text to determine filter type and parameters.

        Returns:
            tuple: (filter_type, parameters)
                - 'mz': parameters = (min_mz, max_mz)
                - 'rt': parameters = (min_rt, max_rt)
                - 'name': parameters = regex_pattern
        """
        import re

        # Check for mz filter: "mz 100-200" or "mz 100 - 200"
        mz_match = re.match(
            r"mz\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", filter_text, re.IGNORECASE
        )
        if mz_match:
            min_mz = float(mz_match.group(1))
            max_mz = float(mz_match.group(2))
            return "mz", (min_mz, max_mz)

        # Check for rt filter: "rt 5-10" or "rt 5 - 10"
        rt_match = re.match(
            r"rt\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", filter_text, re.IGNORECASE
        )
        if rt_match:
            min_rt = float(rt_match.group(1))
            max_rt = float(rt_match.group(2))
            return "rt", (min_rt, max_rt)

        # Otherwise, treat as name regex pattern
        return "name", filter_text

    def show_eic_window(self, compound, adduct, mz_value=None, polarity=None):
        """Show EIC window for the selected compound and adduct"""
        if self.file_manager.get_files_data().empty:
            QMessageBox.warning(self, "Warning", "No files loaded!")
            return

        try:
            # Create EIC window as independent window (no parent)
            eic_window = EICWindow(
                compound,
                adduct,
                self.file_manager,
                mz_value=mz_value,
                polarity=polarity,
                defaults=self.eic_defaults,  # Pass the defaults
                parent=None,  # Make it independent
                integration_callback=self.record_peak_integration,  # Pass integration callback
            )

            # Show the window
            eic_window.show()
            eic_window.raise_()  # Bring to front
            eic_window.activateWindow()  # Make it the active window

            # Keep reference to prevent garbage collection
            self.eic_windows.append(eic_window)

            # Remove from list when window is closed
            def on_window_closed():
                if eic_window in self.eic_windows:
                    self.eic_windows.remove(eic_window)

            eic_window.destroyed.connect(on_window_closed)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to create EIC window: {str(e)}"
            )
            import traceback

            traceback.print_exc()

    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Load Files action
        load_files_action = QAction("Load mzML Files", self)
        load_files_action.triggered.connect(self.load_files)
        file_menu.addAction(load_files_action)

        # Load Compounds action
        load_compounds_action = QAction("Load Compounds", self)
        load_compounds_action.triggered.connect(self.load_compounds)
        file_menu.addAction(load_compounds_action)

        file_menu.addSeparator()

        # Clear Files action
        clear_files_action = QAction("Clear mzML Files", self)
        clear_files_action.triggered.connect(self.clear_files)
        file_menu.addAction(clear_files_action)

        # Clear Compounds action
        clear_compounds_action = QAction("Clear Compounds", self)
        clear_compounds_action.triggered.connect(self.clear_compounds)
        file_menu.addAction(clear_compounds_action)

        file_menu.addSeparator()

        # Generate Templates action
        generate_templates_action = QAction("Generate Templates", self)
        generate_templates_action.triggered.connect(self.generate_templates)
        file_menu.addAction(generate_templates_action)

        file_menu.addSeparator()

        # Export Peak Integration Data action
        export_integration_action = QAction("Export Peak Integration Data", self)
        export_integration_action.triggered.connect(self.export_peak_integration_excel)
        file_menu.addAction(export_integration_action)

        # Generate R Code action
        generate_r_code_action = QAction("Generate R Code for Peak Data", self)
        generate_r_code_action.triggered.connect(self.generate_r_code)
        file_menu.addAction(generate_r_code_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Options menu
        options_menu = menubar.addMenu("Options")

        # Unified Options action
        options_action = QAction("Options", self)
        options_action.setShortcut("Ctrl+P")
        options_action.triggered.connect(self.show_options_dialog)
        options_menu.addAction(options_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_about_dialog(self):
        """Show the about dialog"""
        # Extract version from the project's pyproject.toml file
        try:
            toml_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "pyproject.toml"
            )
            with open(toml_path, "r") as f:
                project_data = toml.load(f)
                version = project_data.get("project", {}).get("version", "Unknown")
        except Exception as ex:
            version = "Unknown"

        QMessageBox.about(
            self,
            "About mzML Explorer",
            f"mzML Explorer v{version}\n\n"
            "A tool for visualizing LC-HRMS data from mzML files.\n\n"
            "Features:\n"
            "• Load mzML files via Excel templates\n"
            "• Extract ion chromatograms (EICs)\n"
            "• Interactive plotting with zoom and pan\n"
            "• Group-based color coding\n\n"
            "Built with PyQt6 and pymzml.\n\n"
            "(c) 2025 Plant-Microbe Metabolomics, BOKU University",
        )

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            # Check if any of the files are Excel files
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.endswith((".xlsx", ".csv", ".tsv")):
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Handle drop events"""
        files = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.endswith((".xlsx", ".csv", ".tsv")):
                    files.append(file_path)

        if files:
            # Determine drop location
            widget_under_mouse = self.childAt(event.position().toPoint())

            # Try to find which panel the drop occurred on
            files_panel_drop = False
            compounds_panel_drop = False

            # Walk up the widget hierarchy to find the target
            current_widget = widget_under_mouse
            while current_widget:
                if current_widget == self.files_table:
                    files_panel_drop = True
                    break
                elif current_widget == self.compounds_table:
                    compounds_panel_drop = True
                    break
                current_widget = current_widget.parent()

            # Process the first file
            file_path = files[0]

            try:
                if compounds_panel_drop:
                    # Load as compounds file
                    self.load_compounds_from_file(file_path)
                else:
                    # Default to loading as files list
                    self.load_files_from_file(file_path)

                event.acceptProposedAction()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

        event.ignore()

    def load_files_from_file(self, file_path):
        """Load files from a dropped file"""
        try:
            # Load the file list
            if file_path.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            elif file_path.endswith(".tsv"):
                df = pd.read_csv(file_path, sep="\t")
            else:
                df = pd.read_csv(file_path)

            # Validate required columns
            if "Filepath" not in df.columns:
                QMessageBox.warning(
                    self, "Error", "The file must contain a 'Filepath' column!"
                )
                return

            # Load files using file manager
            self.file_manager.load_files(df)
            self.update_files_table()

            # If memory mode is enabled, load the new files into memory with progress
            if self.file_manager.keep_in_memory:
                self.load_files_to_memory_with_progress()

            total_files = len(self.file_manager.get_files_data())
            self.statusBar().showMessage(
                f"Files loaded via drag & drop. Total: {total_files} files"
            )

        except Exception as e:
            raise Exception(f"Failed to load files: {str(e)}")

    def load_compounds_from_file(self, file_path):
        """Load compounds from a dropped file"""
        try:
            file_ext = file_path.lower().split(".")[-1]

            if file_ext in ["csv", "tsv"]:
                # Use import dialog for CSV/TSV files
                self.load_compounds_from_csv_tsv(file_path)
            elif file_ext == "xlsx":
                # Load Excel file directly (existing functionality)
                self.load_compounds_from_excel(file_path)
            else:
                QMessageBox.warning(
                    self,
                    "Unsupported Format",
                    f"Unsupported file format: {file_ext}. "
                    "Supported formats: xlsx, csv, tsv",
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load compounds: {str(e)}")

    def load_compounds_from_csv_tsv(self, file_path):
        """Load compounds from CSV/TSV file using import dialog"""
        dialog = CompoundImportDialog(file_path, self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            import_data = dialog.get_import_data()
            if import_data is not None and not import_data.empty:
                # Load compounds
                self.compound_manager.load_compounds(import_data)
                self.update_compounds_table()

                compound_count = len(self.compound_manager.get_compounds_data())
                self.statusBar().showMessage(
                    f"Compounds imported from {file_path}. Total: {compound_count} compounds"
                )

    def load_compounds_from_excel(self, file_path):
        """Load compounds from Excel file (existing functionality)"""
        # Load all sheets from Excel file
        excel_data = pd.read_excel(file_path, sheet_name=None)

        # Determine compounds sheet and validate
        if len(excel_data) == 1:
            # Single sheet file
            compounds_sheet = list(excel_data.values())[0]
            compounds_input = compounds_sheet
        else:
            # Multi-sheet file
            if "Compounds" in excel_data:
                compounds_input = excel_data
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Multi-sheet Excel file found, but no 'Compounds' sheet detected. "
                    "Using the first sheet for compounds.",
                )
                compounds_input = list(excel_data.values())[0]

        # Load compounds
        self.compound_manager.load_compounds(compounds_input)
        self.update_compounds_table()

        compound_count = len(self.compound_manager.get_compounds_data())
        self.statusBar().showMessage(
            f"Compounds loaded via drag & drop. Total: {compound_count} compounds"
        )

    def load_stylesheet(self):
        """Load the CSS stylesheet"""
        stylesheet_path = os.path.join(os.path.dirname(__file__), "style.css")
        if os.path.exists(stylesheet_path):
            with open(stylesheet_path, "r") as f:
                self.setStyleSheet(f.read())

    def closeEvent(self, event):
        """Clean up when closing the application"""
        # Close all EIC windows
        for window in self.eic_windows:
            window.close()
        event.accept()

    def load_eic_defaults(self):
        """Load EIC window default settings"""
        self.eic_defaults = {
            "mz_tolerance_ppm": float(self.settings.value("eic/mz_tolerance_ppm", 5.0)),
            "separate_groups": self.settings.value(
                "eic/separate_groups", True, type=bool
            ),
            "rt_shift_min": float(self.settings.value("eic/rt_shift_min", 1.0)),
            "crop_rt_window": self.settings.value(
                "eic/crop_rt_window", False, type=bool
            ),
            "normalize_samples": self.settings.value(
                "eic/normalize_samples", False, type=bool
            ),
        }

        # Load memory settings
        self.memory_settings = {
            "keep_in_memory": self.settings.value(
                "memory/keep_in_memory", False, type=bool
            )
        }

        # Apply memory settings to file manager
        self.file_manager.set_memory_mode(
            self.memory_settings["keep_in_memory"], auto_load=True
        )

    def save_eic_defaults(self):
        """Save EIC window default settings"""
        self.settings.setValue(
            "eic/mz_tolerance_ppm", self.eic_defaults["mz_tolerance_ppm"]
        )
        self.settings.setValue(
            "eic/separate_groups", self.eic_defaults["separate_groups"]
        )
        self.settings.setValue("eic/rt_shift_min", self.eic_defaults["rt_shift_min"])
        self.settings.setValue(
            "eic/crop_rt_window", self.eic_defaults["crop_rt_window"]
        )
        self.settings.setValue(
            "eic/normalize_samples", self.eic_defaults["normalize_samples"]
        )

    def save_memory_settings(self):
        """Save memory settings"""
        self.settings.setValue(
            "memory/keep_in_memory", self.memory_settings["keep_in_memory"]
        )

    def update_memory_label(self):
        """Update the memory usage label"""
        try:
            memory_info = self.file_manager.get_memory_usage()

            if "error" in memory_info:
                self.memory_label.setText("Memory: Error")
            else:
                rss_mb = memory_info["rss_mb"]
                cached_files = memory_info["cached_files"]

                if memory_info["keep_in_memory"] and cached_files > 0:
                    self.memory_label.setText(
                        f"Memory: {rss_mb:.1f} MB ({cached_files} files cached)"
                    )
                else:
                    self.memory_label.setText(f"Memory: {rss_mb:.1f} MB")
        except Exception as e:
            self.memory_label.setText("Memory: Error")
        self.settings.sync()

    def show_options_dialog(self):
        """Show the unified options configuration dialog"""
        dialog = UnifiedOptionsDialog(
            self.eic_defaults, self.memory_settings, self.file_manager, self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.eic_defaults, self.memory_settings = dialog.get_values()
            self.save_eic_defaults()
            self.save_memory_settings()

            # Apply memory settings to file manager
            old_memory_mode = self.file_manager.keep_in_memory
            new_memory_mode = self.memory_settings["keep_in_memory"]

            if old_memory_mode != new_memory_mode:
                if new_memory_mode:
                    # Show loading dialog when enabling memory mode
                    self.file_manager.set_memory_mode(
                        True, auto_load=False
                    )  # Enable mode without auto-loading
                    self.load_files_to_memory_with_progress()
                else:
                    # Just disable memory mode
                    self.file_manager.set_memory_mode(False)

            QMessageBox.information(
                self,
                "Settings Saved",
                "Options have been saved and applied successfully!",
            )

    def load_files_to_memory_with_progress(self):
        """Load files to memory with a progress dialog"""
        if self.file_manager.files_data.empty:
            return

        num_files = len(self.file_manager.files_data)

        # Create progress dialog
        progress = QProgressDialog(
            "Loading mzML files into memory...", "Cancel", 0, num_files, self
        )
        progress.setWindowTitle("Loading Files")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        # Clear any existing cache first
        self.file_manager._clear_memory_cache()

        # Load files one by one with progress updates
        for i, (_, row) in enumerate(self.file_manager.files_data.iterrows()):
            if progress.wasCanceled():
                self.file_manager.set_memory_mode(False)
                QMessageBox.information(
                    self, "Cancelled", "Memory loading was cancelled."
                )
                return

            filepath = row["Filepath"]
            filename = row["filename"]

            progress.setLabelText(f"Loading {filename}...")
            progress.setValue(i)
            QApplication.processEvents()  # Update UI

            try:
                # Load single file
                reader = self.file_manager.get_mzml_reader(filepath)
                ms1_spectra_data = []
                ms2_spectra_data = []

                for spectrum in reader:
                    if spectrum.ms_level == 1:  # MS1 spectra
                        spectrum_data = {
                            "scan_time": spectrum.scan_time_in_minutes(),
                            "mz": spectrum.mz,
                            "intensity": spectrum.i,
                            "polarity": self.file_manager._get_spectrum_polarity(
                                spectrum
                            ),
                        }
                        ms1_spectra_data.append(spectrum_data)
                    elif spectrum.ms_level == 2:  # MS2/MSMS spectra
                        try:
                            # Get precursor information
                            precursor_mz = None
                            if (
                                hasattr(spectrum, "selected_precursors")
                                and spectrum.selected_precursors
                            ):
                                precursor_info = spectrum.selected_precursors[0]
                                if "mz" in precursor_info:
                                    precursor_mz = float(precursor_info["mz"])

                            # Alternative method to get precursor m/z
                            if precursor_mz is None and hasattr(spectrum, "element"):
                                for elem in spectrum.element.iter():
                                    if elem.tag.endswith("precursorList"):
                                        for precursor in elem:
                                            if precursor.tag.endswith("precursor"):
                                                for selected_ion in precursor:
                                                    if selected_ion.tag.endswith(
                                                        "selectedIonList"
                                                    ):
                                                        for ion in selected_ion:
                                                            for cv_param in ion:
                                                                if (
                                                                    cv_param.tag.endswith(
                                                                        "cvParam"
                                                                    )
                                                                    and cv_param.get(
                                                                        "accession"
                                                                    )
                                                                    == "MS:1000744"
                                                                ):  # selected ion m/z
                                                                    precursor_mz = float(
                                                                        cv_param.get(
                                                                            "value"
                                                                        )
                                                                    )
                                                                    break

                            spectrum_data = {
                                "scan_time": spectrum.scan_time_in_minutes(),
                                "mz": spectrum.mz,
                                "intensity": spectrum.i,
                                "polarity": self.file_manager._get_spectrum_polarity(
                                    spectrum
                                ),
                                "precursor_mz": precursor_mz,
                                "scan_id": spectrum.ID
                                if hasattr(spectrum, "ID")
                                else f"RT_{spectrum.scan_time_in_minutes():.2f}",
                            }
                            ms2_spectra_data.append(spectrum_data)
                        except Exception as e:
                            print(f"Error processing MS2 spectrum: {e}")
                            continue

                # Store both MS1 and MS2 data
                self.file_manager.cached_data[filepath] = {
                    "ms1": ms1_spectra_data,
                    "ms2": ms2_spectra_data,
                }
                reader.close()

            except Exception as e:
                print(f"Error loading {filepath} to memory: {str(e)}")

        progress.setValue(num_files)
        progress.close()

    def record_peak_integration(
        self,
        compound_name,
        ion_name,
        mz_value,
        rt_value,
        ion_mode,
        sample_name,
        group_name,
        peak_start_rt,
        peak_end_rt,
        peak_area,
    ):
        """
        Record peak integration data for a compound and ion.
        Only keeps the latest integration for each compound-ion combination.

        Args:
            compound_name: Name of the compound
            ion_name: Name/description of the ion (e.g., adduct type)
            mz_value: m/z value
            rt_value: retention time value (center)
            ion_mode: ionization mode (positive/negative)
            sample_name: name of the sample file
            group_name: group name for the sample
            peak_start_rt: start of peak boundary
            peak_end_rt: end of peak boundary
            peak_area: integrated peak area
        """
        key = (compound_name, ion_name)

        if key not in self.peak_integration_data:
            self.peak_integration_data[key] = {
                "compound_name": compound_name,
                "ion_name": ion_name,
                "mz_value": mz_value,
                "rt_value": rt_value,
                "ion_mode": ion_mode,
                "peak_start_rt": peak_start_rt,
                "peak_end_rt": peak_end_rt,
                "sample_data": [],  # List of (sample_name, group_name, peak_area) tuples
            }

        # Update the integration data - only keep latest boundaries and sample data
        integration_data = self.peak_integration_data[key]
        integration_data["peak_start_rt"] = peak_start_rt
        integration_data["peak_end_rt"] = peak_end_rt
        integration_data["sample_data"] = [(sample_name, group_name, peak_area)]

    def update_peak_integration_samples(
        self, compound_name, ion_name, sample_data_list
    ):
        """
        Update the sample data for a compound-ion integration.

        Args:
            compound_name: Name of the compound
            ion_name: Name/description of the ion
            sample_data_list: List of (sample_name, group_name, peak_area) tuples
        """
        key = (compound_name, ion_name)

        if key in self.peak_integration_data:
            self.peak_integration_data[key]["sample_data"] = sample_data_list

    def export_peak_integration_excel(self):
        """Export peak integration data to Excel file in long format"""
        if not self.peak_integration_data:
            QMessageBox.information(
                self, "No Data", "No peak integration data available to export."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Peak Integration Data",
            "peak_integration_data.xlsx",
            "Excel files (*.xlsx)",
        )

        if not file_path:
            return

        try:
            # Prepare data for long format table
            export_data = []

            for (
                compound_name,
                ion_name,
            ), integration_data in self.peak_integration_data.items():
                base_row = {
                    "compound_name": integration_data["compound_name"],
                    "ion_name": integration_data["ion_name"],
                    "mz_value": integration_data["mz_value"],
                    "rt_value": integration_data["rt_value"],
                    "ion_mode": integration_data["ion_mode"],
                    "peak_start_rt": integration_data["peak_start_rt"],
                    "peak_end_rt": integration_data["peak_end_rt"],
                }

                # Add a row for each sample
                for sample_name, group_name, peak_area in integration_data[
                    "sample_data"
                ]:
                    row = base_row.copy()
                    row.update(
                        {
                            "sample_name": sample_name,
                            "group_name": group_name,
                            "peak_area": peak_area,
                        }
                    )
                    export_data.append(row)

            # Create DataFrame and export
            df = pd.DataFrame(export_data)
            df.to_excel(file_path, index=False)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Peak integration data exported to:\n{file_path}",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Failed to export data:\n{str(e)}"
            )

    def generate_r_code(self):
        """Generate R code for loading the exported Excel file"""
        if not self.peak_integration_data:
            QMessageBox.information(
                self,
                "No Data",
                "No peak integration data available to generate R code for.",
            )
            return

        r_code = """# R code to load peak integration data
# Install readxl package if not already installed
# install.packages("readxl")

library(readxl)
library(dplyr)

# Load the peak integration data
peak_data <- read_excel("peak_integration_data.xlsx")

# Display structure of the data
str(peak_data)

# Summary of the data
summary(peak_data)

# View first few rows
head(peak_data)

# Example: Group by compound and ion to get summary statistics
peak_summary <- peak_data %>%
  group_by(compound_name, ion_name, group_name) %>%
  summarise(
    n_samples = n(),
    mean_area = mean(peak_area, na.rm = TRUE),
    sd_area = sd(peak_area, na.rm = TRUE),
    median_area = median(peak_area, na.rm = TRUE),
    .groups = 'drop'
  )

print(peak_summary)

# Example: Create boxplot by groups
library(ggplot2)

ggplot(peak_data, aes(x = group_name, y = peak_area)) +
  geom_boxplot() +
  geom_jitter() +
  facet_wrap(~ paste(compound_name, ion_name, sep = " - "), scales = "free_y") +
  theme_minimal() +
  labs(
    title = "Peak Areas by Group",
    x = "Group",
    y = "Peak Area",
    caption = "Data from mzML Explorer peak integration"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
"""

        # Show R code in a dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("R Code for Loading Peak Integration Data")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        # Text area with R code
        from PyQt6.QtWidgets import QTextEdit

        text_edit = QTextEdit()
        text_edit.setPlainText(r_code)
        text_edit.setFont(QFont("Courier", 10))
        layout.addWidget(text_edit)

        # Buttons
        button_layout = QHBoxLayout()

        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(r_code))
        button_layout.addWidget(copy_button)

        save_button = QPushButton("Save to File")
        save_button.clicked.connect(lambda: self._save_r_code(r_code))
        button_layout.addWidget(save_button)

        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        dialog.exec()

    def _save_r_code(self, r_code):
        """Save R code to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save R Code",
            "peak_integration_analysis.R",
            "R files (*.R);;Text files (*.txt);;All files (*)",
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(r_code)
                QMessageBox.information(
                    self, "Save Complete", f"R code saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", f"Failed to save R code:\n{str(e)}"
                )


class CollapsibleSection(QWidget):
    """A collapsible section widget with a title and content area"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.is_expanded = True
        self.init_ui()

    def init_ui(self):
        """Initialize the collapsible section UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button
        self.header_button = QPushButton(f"▼ {self.title}")
        self.header_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                border: 1px solid #ccc;
                background-color: #f0f0f0;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.header_button.clicked.connect(self.toggle_expansion)
        layout.addWidget(self.header_button)

        # Content frame
        self.content_frame = QFrame()
        self.content_frame.setFrameStyle(QFrame.Shape.Box)
        self.content_frame.setStyleSheet(
            "QFrame { border: 1px solid #ccc; border-top: none; }"
        )
        self.content_layout = QVBoxLayout(self.content_frame)
        layout.addWidget(self.content_frame)

    def toggle_expansion(self):
        """Toggle the expansion state of the section"""
        self.is_expanded = not self.is_expanded
        self.content_frame.setVisible(self.is_expanded)
        arrow = "▼" if self.is_expanded else "▶"
        self.header_button.setText(f"{arrow} {self.title}")

    def add_content(self, widget):
        """Add a widget to the content area"""
        self.content_layout.addWidget(widget)

    def set_expanded(self, expanded):
        """Set the expansion state"""
        if self.is_expanded != expanded:
            self.toggle_expansion()


class UnifiedOptionsDialog(QDialog):
    """Unified dialog for all application options with collapsible sections"""

    def __init__(self, eic_defaults, memory_settings, file_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.setModal(True)
        self.setMinimumSize(600, 500)

        self.eic_defaults = eic_defaults.copy()
        self.memory_settings = memory_settings.copy()
        self.file_manager = file_manager
        self.init_ui()

    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)

        # Create scroll area for the content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        # Memory Settings Section
        memory_section = CollapsibleSection("Memory Settings")
        memory_content = self.create_memory_settings_content()
        memory_section.add_content(memory_content)
        content_layout.addWidget(memory_section)

        # EIC Defaults Section
        eic_section = CollapsibleSection("EIC Window Defaults")
        eic_content = self.create_eic_defaults_content()
        eic_section.add_content(eic_content)
        content_layout.addWidget(eic_section)

        content_layout.addStretch()
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

        # Buttons
        button_layout = QHBoxLayout()

        self.reset_button = QPushButton("Reset All to Defaults")
        self.reset_button.clicked.connect(self.reset_all_to_defaults)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)

        layout.addLayout(button_layout)

    def create_memory_settings_content(self):
        """Create the memory settings content widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description
        desc_label = QLabel(
            "Configure how mzML data is handled in memory. Keeping data in memory "
            "provides faster access but uses more RAM."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("QLabel { color: #555; margin-bottom: 10px; }")
        layout.addWidget(desc_label)

        # Form layout
        form_layout = QFormLayout()

        # Keep in memory checkbox
        self.keep_in_memory_cb = QCheckBox()
        self.keep_in_memory_cb.setChecked(self.memory_settings["keep_in_memory"])
        self.keep_in_memory_cb.toggled.connect(self.on_memory_mode_changed)
        form_layout.addRow("Keep all mzML data in memory:", self.keep_in_memory_cb)

        layout.addLayout(form_layout)

        # Current memory usage
        memory_group = QGroupBox("Current Memory Usage")
        memory_layout = QVBoxLayout(memory_group)

        memory_info = self.file_manager.get_memory_usage()

        if "error" in memory_info:
            memory_text = f"Error getting memory info: {memory_info['error']}"
        else:
            rss_mb = memory_info["rss_mb"]
            cached_files = memory_info["cached_files"]
            memory_text = f"Current memory usage: {rss_mb:.1f} MB\n"
            memory_text += f"Cached files: {cached_files}\n"
            memory_text += f"Memory mode: {'Enabled' if memory_info['keep_in_memory'] else 'Disabled'}"

        self.memory_info_label = QLabel(memory_text)
        self.memory_info_label.setStyleSheet("QLabel { font-family: monospace; }")
        memory_layout.addWidget(self.memory_info_label)

        # Refresh button
        refresh_button = QPushButton("Refresh Memory Info")
        refresh_button.clicked.connect(self.refresh_memory_info)
        memory_layout.addWidget(refresh_button)

        layout.addWidget(memory_group)

        # Warning label
        self.memory_warning_label = QLabel()
        self.memory_warning_label.setWordWrap(True)
        self.memory_warning_label.setStyleSheet(
            "QLabel { color: #d66; margin: 10px 0; }"
        )
        self.update_memory_warning()
        layout.addWidget(self.memory_warning_label)

        return widget

    def create_eic_defaults_content(self):
        """Create the EIC defaults content widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description
        desc_label = QLabel(
            "Configure default settings for new EIC (Extracted Ion Chromatogram) windows."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("QLabel { color: #555; margin-bottom: 10px; }")
        layout.addWidget(desc_label)

        # Form layout
        form_layout = QFormLayout()

        # m/z Tolerance (ppm)
        self.mz_tolerance_spin = QDoubleSpinBox()
        self.mz_tolerance_spin.setRange(0.1, 10000.0)
        self.mz_tolerance_spin.setValue(self.eic_defaults["mz_tolerance_ppm"])
        self.mz_tolerance_spin.setSuffix(" ppm")
        self.mz_tolerance_spin.setDecimals(1)
        self.mz_tolerance_spin.setSingleStep(1.0)
        form_layout.addRow("m/z Tolerance:", self.mz_tolerance_spin)

        # Separate by groups
        self.separate_groups_cb = QCheckBox()
        self.separate_groups_cb.setChecked(self.eic_defaults["separate_groups"])
        form_layout.addRow("Separate by groups:", self.separate_groups_cb)

        # Group RT Shift
        self.rt_shift_spin = QDoubleSpinBox()
        self.rt_shift_spin.setRange(0.0, 60.0)
        self.rt_shift_spin.setValue(self.eic_defaults["rt_shift_min"])
        self.rt_shift_spin.setSuffix(" min")
        self.rt_shift_spin.setDecimals(1)
        form_layout.addRow("Group RT Shift:", self.rt_shift_spin)

        # Crop to RT Window
        self.crop_rt_cb = QCheckBox()
        self.crop_rt_cb.setChecked(self.eic_defaults["crop_rt_window"])
        form_layout.addRow("Crop to RT Window:", self.crop_rt_cb)

        # Normalize to Max per Sample
        self.normalize_cb = QCheckBox()
        self.normalize_cb.setChecked(self.eic_defaults["normalize_samples"])
        form_layout.addRow("Normalize to Max per Sample:", self.normalize_cb)

        layout.addLayout(form_layout)

        return widget

    def on_memory_mode_changed(self):
        """Handle memory mode checkbox change"""
        self.update_memory_warning()

    def update_memory_warning(self):
        """Update the memory warning label based on current settings"""
        if self.keep_in_memory_cb.isChecked():
            num_files = len(self.file_manager.get_files_data())
            if num_files > 0:
                self.memory_warning_label.setText(
                    f"⚠ Warning: Enabling memory mode will load all {num_files} mzML files "
                    "into RAM. This may use significant memory and take time to load."
                )
            else:
                self.memory_warning_label.setText(
                    "ℹ Note: Memory mode will load all mzML files into RAM when files are added."
                )
        else:
            if self.file_manager.keep_in_memory:
                self.memory_warning_label.setText(
                    "ℹ Disabling memory mode will clear the current cache and revert to "
                    "file-based reading."
                )
            else:
                self.memory_warning_label.setText("")

    def refresh_memory_info(self):
        """Refresh the memory information display"""
        memory_info = self.file_manager.get_memory_usage()

        if "error" in memory_info:
            memory_text = f"Error getting memory info: {memory_info['error']}"
        else:
            rss_mb = memory_info["rss_mb"]
            cached_files = memory_info["cached_files"]
            memory_text = f"Current memory usage: {rss_mb:.1f} MB\n"
            memory_text += f"Cached files: {cached_files}\n"
            memory_text += f"Memory mode: {'Enabled' if memory_info['keep_in_memory'] else 'Disabled'}"

        self.memory_info_label.setText(memory_text)

    def reset_all_to_defaults(self):
        """Reset all values to application defaults"""
        # Reset EIC defaults
        self.mz_tolerance_spin.setValue(5.0)
        self.separate_groups_cb.setChecked(True)
        self.rt_shift_spin.setValue(1.0)
        self.crop_rt_cb.setChecked(False)
        self.normalize_cb.setChecked(False)

        # Reset memory settings
        self.keep_in_memory_cb.setChecked(False)
        self.update_memory_warning()

    def get_values(self):
        """Get the current values from the dialog"""
        eic_values = {
            "mz_tolerance_ppm": self.mz_tolerance_spin.value(),
            "separate_groups": self.separate_groups_cb.isChecked(),
            "rt_shift_min": self.rt_shift_spin.value(),
            "crop_rt_window": self.crop_rt_cb.isChecked(),
            "normalize_samples": self.normalize_cb.isChecked(),
        }

        memory_values = {"keep_in_memory": self.keep_in_memory_cb.isChecked()}

        return eic_values, memory_values


class EICDefaultsDialog(QDialog):
    """Dialog for configuring EIC window defaults"""

    def __init__(self, current_defaults, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EIC Window Defaults")
        self.setModal(True)
        self.setFixedSize(400, 300)

        self.current_defaults = current_defaults.copy()
        self.init_ui()

    def init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)

        # Create form layout
        form_layout = QFormLayout()

        # m/z Tolerance (ppm)
        self.mz_tolerance_spin = QDoubleSpinBox()
        self.mz_tolerance_spin.setRange(0.1, 10000.0)
        self.mz_tolerance_spin.setValue(self.current_defaults["mz_tolerance_ppm"])
        self.mz_tolerance_spin.setSuffix(" ppm")
        self.mz_tolerance_spin.setDecimals(1)
        self.mz_tolerance_spin.setSingleStep(1.0)
        form_layout.addRow("m/z Tolerance:", self.mz_tolerance_spin)

        # Separate by groups
        self.separate_groups_cb = QCheckBox()
        self.separate_groups_cb.setChecked(self.current_defaults["separate_groups"])
        form_layout.addRow("Separate by groups:", self.separate_groups_cb)

        # Group RT Shift
        self.rt_shift_spin = QDoubleSpinBox()
        self.rt_shift_spin.setRange(0.0, 60.0)
        self.rt_shift_spin.setValue(self.current_defaults["rt_shift_min"])
        self.rt_shift_spin.setSuffix(" min")
        self.rt_shift_spin.setDecimals(1)
        form_layout.addRow("Group RT Shift:", self.rt_shift_spin)

        # Crop to RT Window
        self.crop_rt_cb = QCheckBox()
        self.crop_rt_cb.setChecked(self.current_defaults["crop_rt_window"])
        form_layout.addRow("Crop to RT Window:", self.crop_rt_cb)

        # Normalize to Max per Sample
        self.normalize_cb = QCheckBox()
        self.normalize_cb.setChecked(self.current_defaults["normalize_samples"])
        form_layout.addRow("Normalize to Max per Sample:", self.normalize_cb)

        layout.addLayout(form_layout)

        # Buttons
        button_layout = QHBoxLayout()

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)

        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)

        layout.addLayout(button_layout)

    def reset_to_defaults(self):
        """Reset all values to application defaults"""
        self.mz_tolerance_spin.setValue(5.0)
        self.separate_groups_cb.setChecked(True)
        self.rt_shift_spin.setValue(1.0)
        self.crop_rt_cb.setChecked(False)
        self.normalize_cb.setChecked(False)

    def get_values(self):
        """Get the current values from the dialog"""
        return {
            "mz_tolerance_ppm": self.mz_tolerance_spin.value(),
            "separate_groups": self.separate_groups_cb.isChecked(),
            "rt_shift_min": self.rt_shift_spin.value(),
            "crop_rt_window": self.crop_rt_cb.isChecked(),
            "normalize_samples": self.normalize_cb.isChecked(),
        }


def main():
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("mzML Explorer")
    app.setApplicationVersion("1.0.0")

    # Create and show main window
    window = MzMLExplorerMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
