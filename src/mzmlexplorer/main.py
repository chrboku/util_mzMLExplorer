import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QTableWidget, QTableWidgetItem, 
                             QTreeWidget, QTreeWidgetItem, QLabel, QDoubleSpinBox,
                             QLineEdit, QGroupBox, QCheckBox, QComboBox, QSpinBox,
                             QSplitter, QFileDialog, QMessageBox, QHeaderView,
                             QMenuBar, QMenu, QDialog, QFormLayout)
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
from .utils import calculate_mz_from_formula


class MzMLExplorerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mzML Explorer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # Initialize settings
        self.settings = QSettings('mzMLExplorer', 'mzMLExplorer')
        self.load_eic_defaults()
        
        # Data storage
        self.file_manager = FileManager()
        self.compound_manager = CompoundManager()
        self.eic_windows = []
        
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
        left_layout.addWidget(self.files_table)
        
        # Right panel: Compounds tree
        right_panel = QGroupBox("Compounds")
        right_panel.setAcceptDrops(True)
        right_layout = QVBoxLayout(right_panel)
        
        # Add filter line
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.compound_filter = QLineEdit()
        self.compound_filter.setPlaceholderText("mz 100-200, rt 5-10, or compound name pattern")
        self.compound_filter.textChanged.connect(self.filter_compounds)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.compound_filter)
        right_layout.addLayout(filter_layout)
        
        self.compounds_tree = QTreeWidget()
        self.compounds_tree.setHeaderLabel("Compounds and Adducts")
        self.compounds_tree.itemClicked.connect(self.on_compound_selected)
        self.compounds_tree.setAcceptDrops(True)
        right_layout.addWidget(self.compounds_tree)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def load_files(self):
        """Load mzML files from a TSV or Excel file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load File List", 
            "", 
            "Excel files (*.xlsx);;TSV files (*.tsv);;CSV files (*.csv)"
        )
        
        if file_path:
            try:
                # Load the file list
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                elif file_path.endswith('.tsv'):
                    df = pd.read_csv(file_path, sep='\t')
                else:
                    df = pd.read_csv(file_path)
                
                # Validate required columns
                if 'Filepath' not in df.columns:
                    QMessageBox.warning(self, "Error", "The file must contain a 'Filepath' column!")
                    return
                
                # Load files using file manager
                self.file_manager.load_files(df)
                self.update_files_table()
                
                total_files = len(self.file_manager.get_files_data())
                self.statusBar().showMessage(f"Files loaded. Total: {total_files} files")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load files: {str(e)}")
    
    def load_compounds(self):
        """Load compounds from an Excel file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Compounds", 
            "", 
            "Excel files (*.xlsx)"
        )
        
        if file_path:
            try:
                # Load all sheets from Excel file
                excel_data = pd.read_excel(file_path, sheet_name=None)
                
                # Determine compounds sheet and validate
                if len(excel_data) == 1:
                    # Single sheet file
                    compounds_sheet = list(excel_data.values())[0]
                    compounds_input = compounds_sheet
                else:
                    # Multi-sheet file
                    if 'Compounds' in excel_data:
                        compounds_sheet = excel_data['Compounds']
                    else:
                        compounds_sheet = list(excel_data.values())[0]
                    compounds_input = excel_data
                
                # Validate required columns for compounds sheet
                required_cols = ['Name', 'RT_min', 'RT_start_min', 'RT_end_min', 'Common_adducts']
                missing_cols = [col for col in required_cols if col not in compounds_sheet.columns]
                
                # Check for either ChemicalFormula or Mass
                has_formula = 'ChemicalFormula' in compounds_sheet.columns
                has_mass = 'Mass' in compounds_sheet.columns
                
                if not has_formula and not has_mass:
                    missing_cols.append('ChemicalFormula OR Mass')
                
                if missing_cols:
                    QMessageBox.warning(
                        self, 
                        "Error", 
                        f"Missing required columns in compounds sheet: {', '.join(missing_cols)}"
                    )
                    return
                
                # Load compounds using compound manager
                self.compound_manager.load_compounds(compounds_input)
                self.update_compounds_tree()
                
                compounds_count = len(self.compound_manager.get_compounds_data())
                adducts_count = len(self.compound_manager.get_adducts_data())
                self.statusBar().showMessage(f"Compounds loaded. Total: {compounds_count} compounds with {adducts_count} adduct definitions")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load compounds: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def generate_templates(self):
        """Generate template Excel files for file list and compounds"""
        try:
            # Ask user where to save templates
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory to Save Templates",
                ""
            )
            
            if not save_dir:
                return
            
            # Generate file list template
            files_template_data = {
                'Filepath': [
                    'C:\\path\\to\\your\\file1.mzML',
                    'C:\\path\\to\\your\\file2.mzML',
                    'C:\\path\\to\\your\\file3.mzML',
                    'C:\\path\\to\\your\\file4.mzML'
                ],
                'group': [
                    'Control',
                    'Control',
                    'Treatment',
                    'Treatment'
                ],
                'color': [
                    '#1f77b4',
                    '#1f77b4',
                    '#ff7f0e',
                    '#ff7f0e'
                ],
                'batch': [
                    'Batch1',
                    'Batch1',
                    'Batch1',
                    'Batch2'
                ],
                'injection_volume': [
                    5.0,
                    5.0,
                    5.0,
                    5.0
                ],
                'sample_id': [
                    'CTL_001',
                    'CTL_002',
                    'TRT_001',
                    'TRT_002'
                ]
            }
            
            files_template_df = pd.DataFrame(files_template_data)
            files_template_path = os.path.join(save_dir, "file_list_template.xlsx")
            files_template_df.to_excel(files_template_path, index=False)
            
            # Generate compounds template
            compounds_template_data = {
                'Name': [
                    'Caffeine',
                    'Theophylline',
                    'Unknown_Compound_1',
                    'Unknown_Compound_2'
                ],
                'ChemicalFormula': [
                    'C8H10N4O2',
                    'C7H8N4O2',
                    '',  # Empty for mass-based compound
                    ''   # Empty for mass-based compound
                ],
                'Mass': [
                    '',   # Empty for formula-based compound
                    '',   # Empty for formula-based compound
                    194.0579,  # Mass-based compound
                    256.1234   # Mass-based compound
                ],
                'RT_min': [
                    5.2,
                    4.8,
                    3.1,
                    7.5
                ],
                'RT_start_min': [
                    4.8,
                    4.4,
                    2.7,
                    7.0
                ],
                'RT_end_min': [
                    5.6,
                    5.2,
                    3.5,
                    8.0
                ],
                'Common_adducts': [
                    '[M+H]+, [M+Na]+, [M+K]+',
                    '[M+H]+, [M+Na]+, [M-H]-',
                    '[M+H]+, [195.0652]+',  # Mix of standard and custom m/z
                    '[257.1307]+, [255.1151]-'  # Custom m/z values only
                ],
                'compound_class': [
                    'Alkaloid',
                    'Alkaloid',
                    'Unknown',
                    'Unknown'
                ],
                'cas_number': [
                    '58-08-2',
                    '58-55-9',
                    '',
                    ''
                ]
            }
            
            compounds_template_df = pd.DataFrame(compounds_template_data)
            
            # Create adducts template
            adducts_template_data = {
                'Adduct': [
                    '[M+H]+',
                    '[M+Na]+',
                    '[M+K]+',
                    '[M+NH4]+',
                    '[M+2H]2+',
                    '[M-H]-',
                    '[M+Cl]-',
                    '[M+HCOO]-',
                    '[M+CH3COO]-',
                    '[M-H2O-H]-',
                    '[M-H2O+H]+',
                    '[M+H-NH3]+',
                    '[2M+H]+',
                    '[2M+Na]+',
                    '[2M-H]-'
                ],
                'Mass_change': [
                    1.007276,
                    22.989218,
                    38.963158,
                    18.033823,
                    2.014552,
                    -1.007276,
                    34.969402,
                    44.998201,
                    59.013851,
                    -19.018393,
                    -17.003288,
                    -16.018724,
                    1.007276,
                    22.989218,
                    -1.007276
                ],
                'Charge': [
                    1, 1, 1, 1, 2, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1
                ],
                'Multiplier': [
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2
                ]
            }
            
            adducts_template_df = pd.DataFrame(adducts_template_data)
            
            # Save to Excel with multiple sheets
            compounds_template_path = os.path.join(save_dir, "compounds_template.xlsx")
            with pd.ExcelWriter(compounds_template_path, engine='openpyxl') as writer:
                compounds_template_df.to_excel(writer, sheet_name='Compounds', index=False)
                adducts_template_df.to_excel(writer, sheet_name='Adducts', index=False)
            
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
                f"Please edit these files with your data before loading."
            )
            
            self.statusBar().showMessage("Template files generated successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate templates: {str(e)}")
    
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
                if col_name == 'color':
                    # Don't show the color text, just the background
                    item.setText("")
                    if pd.notna(value) and value:
                        item.setBackground(QColor(str(value)))
                else:
                    # For other columns, set background color based on group
                    if 'group' in row.index and pd.notna(row['group']):
                        group_color = self.file_manager.get_group_color(row['group'])
                        if group_color and col_name != 'color':
                            # Apply a lighter version of the group color for non-color columns
                            color = QColor(group_color)
                            color.setAlpha(50)  # Make it semi-transparent
                            item.setBackground(color)
                
                self.files_table.setItem(i, j, item)
        
        # Adjust column widths
        self.files_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    
    def update_compounds_tree(self):
        """Update the compounds tree with loaded data"""
        self.compounds_tree.clear()
        compounds_data = self.compound_manager.get_compounds_data()
        
        for _, compound in compounds_data.iterrows():
            # Create compound item with retention time info
            compound_name = compound['Name']
            
            # Try to get average RT from RT_minutes column, fallback to RT window
            rt_text = ""
            if 'RT_minutes' in compound and pd.notna(compound['RT_minutes']):
                avg_rt = compound['RT_minutes']
                rt_text = f"RT: {avg_rt:.1f} min"
            elif compound.get('RT_start_min') and compound.get('RT_end_min'):
                rt_start = compound['RT_start_min']
                rt_end = compound['RT_end_min']
                rt_text = f"RT: {rt_start:.1f}-{rt_end:.1f} min"
            else:
                rt_text = "RT: not set"
            
            compound_text = f"{compound_name} ({rt_text})"
            compound_item = QTreeWidgetItem([compound_text])
            compound_item.setFont(0, QFont("Arial", 10, QFont.Weight.Bold))
            
            # Parse adducts and create child items
            adducts = compound['Common_adducts']
            if isinstance(adducts, str):
                adduct_list = [a.strip() for a in adducts.split(',') if a.strip()]
                
                for adduct in adduct_list:
                    # Get pre-calculated data for this compound-adduct combination
                    precalc_data = self.compound_manager.get_precalculated_data(
                        compound['Name'], adduct
                    )
                    
                    if precalc_data:
                        display_name = precalc_data['display_name']
                        mz_value = precalc_data['mz']
                        polarity = precalc_data['polarity']
                        
                        if mz_value is not None:
                            adduct_text = f"{display_name} (m/z: {mz_value:.4f})"
                        else:
                            adduct_text = f"{display_name} (m/z: calculation failed)"
                    else:
                        # Fallback to old method if pre-calculated data not available
                        display_name = self.compound_manager.get_adduct_display_name(
                            compound['Name'], adduct
                        )
                        adduct_text = f"{display_name} (m/z: not calculated)"
                        mz_value = None
                        polarity = None
                    
                    adduct_item = QTreeWidgetItem([adduct_text])
                    # Store compound data and pre-calculated values in the item
                    adduct_item.setData(0, Qt.ItemDataRole.UserRole, {
                        'compound': compound.to_dict(),
                        'adduct': adduct,
                        'display_name': display_name,
                        'mz': mz_value,
                        'polarity': polarity
                    })
                    compound_item.addChild(adduct_item)
            
            self.compounds_tree.addTopLevelItem(compound_item)
        
        # Expand all items
        self.compounds_tree.expandAll()
    
    def filter_compounds(self):
        """Filter compounds based on the filter text"""
        filter_text = self.compound_filter.text().strip()
        
        if not filter_text:
            # Show all compounds if filter is empty
            for i in range(self.compounds_tree.topLevelItemCount()):
                item = self.compounds_tree.topLevelItem(i)
                item.setHidden(False)
                # Show all children too
                for j in range(item.childCount()):
                    child = item.child(j)
                    child.setHidden(False)
            return
        
        # Parse filter text
        filter_type, filter_params = self._parse_filter_text(filter_text)
        
        compounds_data = self.compound_manager.get_compounds_data()
        
        for i in range(self.compounds_tree.topLevelItemCount()):
            item = self.compounds_tree.topLevelItem(i)
            compound_name = None
            
            # Extract compound name from tree item text (remove RT info)
            item_text = item.text(0)
            if '(' in item_text:
                compound_name = item_text.split('(')[0].strip()
            else:
                compound_name = item_text
            
            # Find corresponding compound data
            compound_row = compounds_data[compounds_data['Name'] == compound_name]
            if compound_row.empty:
                item.setHidden(True)
                continue
            
            compound = compound_row.iloc[0]
            show_compound = False
            
            if filter_type == 'mz':
                # Check which adducts have m/z in range and show only those
                min_mz, max_mz = filter_params
                adducts = compound['Common_adducts']
                matching_adducts = []
                
                if isinstance(adducts, str):
                    adduct_list = [a.strip() for a in adducts.split(',') if a.strip()]
                    for adduct in adduct_list:
                        try:
                            # Use pre-calculated m/z value if available
                            precalc_data = self.compound_manager.get_precalculated_data(
                                compound['Name'], adduct
                            )
                            
                            if precalc_data and precalc_data['mz'] is not None:
                                mz_value = precalc_data['mz']
                            else:
                                # Fallback to calculation if pre-calculated data not available
                                mz_value = self.compound_manager.calculate_compound_mz(
                                    compound['Name'], adduct
                                )
                            
                            if mz_value is not None and min_mz <= mz_value <= max_mz:
                                matching_adducts.append(adduct)
                        except:
                            continue
                
                if matching_adducts:
                    show_compound = True
                    # Show compound but hide non-matching adducts
                    item.setHidden(False)
                    for j in range(item.childCount()):
                        child = item.child(j)
                        child_data = child.data(0, Qt.ItemDataRole.UserRole)
                        if child_data:
                            child_adduct = child_data['adduct']
                            child.setHidden(child_adduct not in matching_adducts)
                else:
                    show_compound = False
            
            elif filter_type == 'rt':
                # Check if RT is in range
                min_rt, max_rt = filter_params
                if 'RT_minutes' in compound and pd.notna(compound['RT_minutes']):
                    rt_value = float(compound['RT_minutes'])
                    show_compound = min_rt <= rt_value <= max_rt
                elif compound.get('RT_start_min') and compound.get('RT_end_min'):
                    # Use average of RT window if RT_minutes not available
                    avg_rt = (float(compound['RT_start_min']) + float(compound['RT_end_min'])) / 2
                    show_compound = min_rt <= avg_rt <= max_rt
            
            elif filter_type == 'name':
                # Regex search on compound name
                import re
                try:
                    pattern = re.compile(filter_params, re.IGNORECASE)
                    show_compound = bool(pattern.search(compound['Name']))
                except re.error:
                    # If regex is invalid, fall back to simple string search
                    show_compound = filter_params.lower() in compound['Name'].lower()
            
            # Show/hide compound based on filter result
            if filter_type == 'mz':
                # For m/z filter, we already handled individual adduct visibility above
                item.setHidden(not show_compound)
            else:
                # For other filters, show/hide compound and all its adducts together
                item.setHidden(not show_compound)
                if show_compound:
                    # Show all adduct children when compound matches
                    for j in range(item.childCount()):
                        child = item.child(j)
                        child.setHidden(False)
    
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
        mz_match = re.match(r'mz\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', filter_text, re.IGNORECASE)
        if mz_match:
            min_mz = float(mz_match.group(1))
            max_mz = float(mz_match.group(2))
            return 'mz', (min_mz, max_mz)
        
        # Check for rt filter: "rt 5-10" or "rt 5 - 10"
        rt_match = re.match(r'rt\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', filter_text, re.IGNORECASE)
        if rt_match:
            min_rt = float(rt_match.group(1))
            max_rt = float(rt_match.group(2))
            return 'rt', (min_rt, max_rt)
        
        # Otherwise, treat as name regex pattern
        return 'name', filter_text
    
    def on_compound_selected(self, item, column):
        """Handle compound/adduct selection"""
        if item.parent() is not None:  # This is an adduct item
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                # Pass the pre-calculated m/z and polarity values
                self.show_eic_window(
                    data['compound'], 
                    data['adduct'], 
                    data.get('mz'), 
                    data.get('polarity')
                )
    
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
                parent=None  # Make it independent
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
            QMessageBox.critical(self, "Error", f"Failed to create EIC window: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Load Files action
        load_files_action = QAction('Load mzML Files...', self)
        load_files_action.setShortcut('Ctrl+O')
        load_files_action.triggered.connect(self.load_files)
        file_menu.addAction(load_files_action)
        
        # Load Compounds action
        load_compounds_action = QAction('Load Compounds...', self)
        load_compounds_action.setShortcut('Ctrl+C')
        load_compounds_action.triggered.connect(self.load_compounds)
        file_menu.addAction(load_compounds_action)
        
        file_menu.addSeparator()
        
        # Generate Templates action
        generate_templates_action = QAction('Generate Templates...', self)
        generate_templates_action.triggered.connect(self.generate_templates)
        file_menu.addAction(generate_templates_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Options menu
        options_menu = menubar.addMenu('Options')
        
        # EIC Defaults action
        eic_defaults_action = QAction('EIC Window Defaults...', self)
        eic_defaults_action.triggered.connect(self.show_eic_defaults_dialog)
        options_menu.addAction(eic_defaults_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        # About action
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def show_about_dialog(self):
        """Show the about dialog"""
        QMessageBox.about(self, "About mzML Explorer", 
                         "mzML Explorer v1.0\n\n"
                         "A tool for visualizing LC-HRMS data from mzML files.\n\n"
                         "Features:\n"
                         "• Load mzML files via Excel templates\n"
                         "• Extract ion chromatograms (EICs)\n"
                         "• Interactive plotting with zoom and pan\n"
                         "• Group-based color coding\n"
                         "• Drag and drop support\n\n"
                         "Built with PyQt6 and pymzml.")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            # Check if any of the files are Excel files
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.endswith(('.xlsx', '.csv', '.tsv')):
                        event.acceptProposedAction()
                        return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop events"""
        files = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.endswith(('.xlsx', '.csv', '.tsv')):
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
                elif current_widget == self.compounds_tree:
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
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
            else:
                df = pd.read_csv(file_path)
            
            # Validate required columns
            if 'Filepath' not in df.columns:
                QMessageBox.warning(self, "Error", "The file must contain a 'Filepath' column!")
                return
            
            # Load files using file manager
            self.file_manager.load_files(df)
            self.update_files_table()
            
            total_files = len(self.file_manager.get_files_data())
            self.statusBar().showMessage(f"Files loaded via drag & drop. Total: {total_files} files")
            
        except Exception as e:
            raise Exception(f"Failed to load files: {str(e)}")
    
    def load_compounds_from_file(self, file_path):
        """Load compounds from a dropped file"""
        try:
            # Load all sheets from Excel file
            excel_data = pd.read_excel(file_path, sheet_name=None)
            
            # Determine compounds sheet and validate
            if len(excel_data) == 1:
                # Single sheet file
                compounds_sheet = list(excel_data.values())[0]
                compounds_input = compounds_sheet
            else:
                # Multi-sheet file
                if 'Compounds' in excel_data:
                    compounds_input = excel_data
                else:
                    QMessageBox.warning(self, "Warning", 
                                      "Multi-sheet Excel file found, but no 'Compounds' sheet detected. "
                                      "Using the first sheet for compounds.")
                    compounds_input = list(excel_data.values())[0]
            
            # Load compounds
            self.compound_manager.load_compounds(compounds_input)
            self.update_compounds_tree()
            
            compound_count = len(self.compound_manager.get_compounds_data())
            self.statusBar().showMessage(f"Compounds loaded via drag & drop. Total: {compound_count} compounds")
            
        except Exception as e:
            raise Exception(f"Failed to load compounds: {str(e)}")
    
    def load_stylesheet(self):
        """Load the CSS stylesheet"""
        stylesheet_path = os.path.join(os.path.dirname(__file__), 'style.css')
        if os.path.exists(stylesheet_path):
            with open(stylesheet_path, 'r') as f:
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
            'mz_tolerance_ppm': float(self.settings.value('eic/mz_tolerance_ppm', 5.0)),
            'separate_groups': self.settings.value('eic/separate_groups', True, type=bool),
            'rt_shift_min': float(self.settings.value('eic/rt_shift_min', 1.0)),
            'crop_rt_window': self.settings.value('eic/crop_rt_window', False, type=bool),
            'normalize_samples': self.settings.value('eic/normalize_samples', False, type=bool)
        }
    
    def save_eic_defaults(self):
        """Save EIC window default settings"""
        self.settings.setValue('eic/mz_tolerance_ppm', self.eic_defaults['mz_tolerance_ppm'])
        self.settings.setValue('eic/separate_groups', self.eic_defaults['separate_groups'])
        self.settings.setValue('eic/rt_shift_min', self.eic_defaults['rt_shift_min'])
        self.settings.setValue('eic/crop_rt_window', self.eic_defaults['crop_rt_window'])
        self.settings.setValue('eic/normalize_samples', self.eic_defaults['normalize_samples'])
        self.settings.sync()
    
    def show_eic_defaults_dialog(self):
        """Show the EIC defaults configuration dialog"""
        dialog = EICDefaultsDialog(self.eic_defaults, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.eic_defaults = dialog.get_values()
            self.save_eic_defaults()
            QMessageBox.information(self, "Settings Saved", 
                                  "EIC window defaults have been saved successfully!")


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
        self.mz_tolerance_spin.setValue(self.current_defaults['mz_tolerance_ppm'])
        self.mz_tolerance_spin.setSuffix(" ppm")
        self.mz_tolerance_spin.setDecimals(1)
        self.mz_tolerance_spin.setSingleStep(1.0)
        form_layout.addRow("m/z Tolerance:", self.mz_tolerance_spin)
        
        # Separate by groups
        self.separate_groups_cb = QCheckBox()
        self.separate_groups_cb.setChecked(self.current_defaults['separate_groups'])
        form_layout.addRow("Separate by groups:", self.separate_groups_cb)
        
        # Group RT Shift
        self.rt_shift_spin = QDoubleSpinBox()
        self.rt_shift_spin.setRange(0.0, 60.0)
        self.rt_shift_spin.setValue(self.current_defaults['rt_shift_min'])
        self.rt_shift_spin.setSuffix(" min")
        self.rt_shift_spin.setDecimals(1)
        form_layout.addRow("Group RT Shift:", self.rt_shift_spin)
        
        # Crop to RT Window
        self.crop_rt_cb = QCheckBox()
        self.crop_rt_cb.setChecked(self.current_defaults['crop_rt_window'])
        form_layout.addRow("Crop to RT Window:", self.crop_rt_cb)
        
        # Normalize to Max per Sample
        self.normalize_cb = QCheckBox()
        self.normalize_cb.setChecked(self.current_defaults['normalize_samples'])
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
            'mz_tolerance_ppm': self.mz_tolerance_spin.value(),
            'separate_groups': self.separate_groups_cb.isChecked(),
            'rt_shift_min': self.rt_shift_spin.value(),
            'crop_rt_window': self.crop_rt_cb.isChecked(),
            'normalize_samples': self.normalize_cb.isChecked()
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
