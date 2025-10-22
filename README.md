# mzML Explorer

A comprehensive GUI application for visualization and analysis of LC-HRMS data from mzML files. Built with PyQt6 and designed for analytical chemistry workflows.

## Features

### 🔬 **Core Functionality**
- **mzML File Loading**: Import multiple mzML files via Excel templates with metadata
- **Compound Management**: Load compound lists with chemical formulas, masses, and retention time windows
- **EIC Extraction**: Extract ion chromatograms with customizable parameters
- **Interactive Plotting**: Professional-grade charts with zoom, pan, and legend management

### 📊 **Data Visualization**
- **Group-based Color Coding**: Automatic color assignment for sample groups
- **Transparent Overlays**: Semi-transparent EIC traces for better comparison
- **Normalized Views**: Optional normalization to maximum intensity per sample
- **RT Window Cropping**: Focus on specific retention time ranges
- **Scientific Notation**: Proper formatting for intensity axes

### 🎯 **Advanced Analysis**
- **Polarity-Aware Extraction**: Considers adduct polarity for MS1 filtering
- **Multiple EIC Methods**: Sum of signals or most intensive signal
- **m/z Tolerance**: Flexible tolerance in both ppm and Da units
- **Group Separation**: Optional RT shifting for group comparison with visual indicators
- **Reference Lines**: Dashed vertical lines show expected retention times from compound database
- **Performance Optimization**: Pre-calculated m/z values for faster EIC generation
- **User Preferences**: Configurable defaults for EIC window parameters

### 🖱️ **User Interface**
- **Drag & Drop Support**: Direct file loading by dropping onto panels
- **Professional Menu System**: File operations, preferences, and help via menu bar
- **Compound Filtering**: Advanced filtering by m/z range, RT range, or name patterns
- **Interactive Mouse Controls**: 
  - Left-click + drag: Pan the plot
  - Right-click + drag: Zoom in/out
- **Responsive Layout**: Narrow control panel with optimized space usage
- **Configurable Defaults**: Set default parameters for EIC windows via menu
- **Performance Indicators**: Pre-calculated m/z values with polarity detection

### 📁 **File Management**
- **Incremental Loading**: Add new files and compounds without replacing existing data
- **Template Generation**: Generate Excel templates for file lists and compounds
- **Multi-format Support**: Excel (.xlsx), TSV, and CSV file formats

## Installation
### Setup
1. Install the uv package manager, for details refer to [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
1. Clone or download the repository


## Usage

### Starting the Application
```bash
uv run mzmlexplorer
```

### Loading Data

#### 1. File Lists
Create an Excel file with columns:
- `Filepath`: Full path to mzML files
- `group`: Sample group name (optional)

Use **File → Load mzML Files...** or drag the Excel file onto the "Loaded Files" panel.

#### 2. Compounds
Create an Excel file with columns:
- `Group`: Group name for organizing compounds (optional, can contain multiple groups separated by ";")
- `Name`: Compound name
- `ChemicalFormula` OR `Mass`: Chemical formula or molecular mass
- `RT_minutes`: Average retention time (optional)
- `RT_start_min`: RT window start (optional, defaults to 0 min if not specified)
- `RT_end_min`: RT window end (optional, defaults to 100 min if not specified)
- `Common_adducts`: Comma-separated list of adducts

**Note:** 
- If RT columns are not provided or left empty, the compound will use the full retention time range (0-100 minutes) for extraction.
- The `Group` column allows you to organize compounds hierarchically in the table. Group names appear as bold headers, with compounds listed underneath. Multiple groups can be specified separated by semicolons (e.g., "Alkaloids;Stimulants") to add a compound to multiple groups. Compounds without a group will appear under an empty group header at the bottom.

Use **File → Load Compounds...** or drag the Excel file onto the "Compounds" panel.

#### 3. Template Generation
Use **File → Generate Templates...** to create template Excel files with example data.

### Extracting EICs

1. Load your mzML files and compounds
2. **Set Default Parameters** (optional): Use **Preferences → EIC Defaults...** to configure:
   - Default m/z tolerance (ppm)
   - Group separation preference
   - RT shift amount for group separation
   - RT cropping and normalization defaults
3. Click on any adduct in the compounds tree
4. A new window opens with:
   - **Compound Information**: Name, formula, m/z, RT window
   - **Extraction Parameters**: Tolerance, method, normalization options (using your defaults)
   - **Interactive Plot**: EIC traces with mouse controls and reference lines

### Filtering Compounds

Use the filter box above the compounds tree:

- **m/z range**: `mz 100-500` (shows only adducts in this m/z range)
- **RT range**: `rt 5-10` (shows compounds with RT in this range)
- **Name search**: Any other text (regex pattern matching compound names)

### Plot Controls

#### Mouse Interactions
- **Pan**: Left-click and drag to move the view
- **Zoom**: Right-click and drag to zoom in/out
- **Reset**: Double-click to reset zoom

#### Options Panel
- **EIC Method**: Sum of signals vs. most intensive signal
- **m/z Tolerance**: Adjustable in ppm or Da (linked)
- **Group Separation**: Offset groups by RT for comparison
  - When enabled, legend shows shift amounts (e.g., "Group A (+ 0.0 min)", "Group B (+ 1.0 min)")
- **RT Cropping**: Limit view to compound's RT window
- **Normalization**: Scale each sample to its maximum intensity
- **Reference Lines**: 
  - Horizontal baseline at intensity 0
  - Vertical dashed lines at expected retention times from compound database
  - Reference lines adjust with group separation shifts

## File Formats

### File List Template
```excel
Filepath                             | group
C:\data\sample1.mzML                 | Control
C:\data\sample2.mzML                 | 
C:\data\sample3.mzML                 | Treatment
C:\data\sample4.mzML                 | 
```

**Important Rules for File Lists:**
- The **first row** must be completely filled (no empty cells)
- The **Filepath column** must be completely filled in all rows
- Other columns (like `group`) can have empty cells
- Empty cells automatically inherit the value from the previous row in that column
- In the example above, sample2 inherits "Control" and sample4 inherits "Treatment"

### Compounds Template
```excel
Name     | ChemicalFormula | RT_minutes | RT_start_min | RT_end_min | Common_adducts
Caffeine | C8H10N4O2      | 8.5        | 7.5          | 9.5        | [M+H]+,[M+Na]+
Glucose  |                 | 2.1        | 1.5          | 3.0        | [M+H]+,[M+NH4]+
Unknown  | C10H15N5O      |            |              |            | [M+H]+
```

*Note: Either `ChemicalFormula` OR `Mass` column is required. RT columns are optional - if not specified, the full range (0-100 min) will be used.*

### Adducts Sheet (Optional)
```excel
Adduct     | Mass_change | Charge
[M+H]+     | 1.007276   | 1
[M+Na]+    | 22.989218  | 1
[M-H]-     | -1.007276  | -1
```

## Features in Detail

### Polarity-Aware Analysis
The application automatically detects ion polarity from adduct specifications:
- Positive adducts (containing `+`): Only positive MS1 scans used
- Negative adducts (containing `-`): Only negative MS1 scans used
- Neutral adducts: All MS1 scans used

### Group Management
- Automatic color assignment for different sample groups
- Natural sorting (Group1, Group2, Group10 sorts correctly)
- Consistent colors across all plots
- Legend shows one entry per group
- **Group Separation Display**: When enabled, legend includes RT shift information
- **Visual Reference Lines**: Dashed vertical lines indicate expected retention times for each group

### Data Processing
- **RT Window Filtering**: Focus analysis on biologically relevant time ranges
- **Peak Integration**: Choose between sum or maximum intensity methods
- **Normalization**: Compare relative intensities across samples
- **Missing Data Handling**: Graceful handling of files without target ions
- **Performance Optimization**: m/z values pre-calculated during compound import for faster EIC extraction
- **Global Adduct Management**: Adducts from compound lists automatically added to global dictionary

## Troubleshooting

### Common Issues

1. **Files not loading**: Ensure mzML files exist at specified paths
2. **No EIC displayed**: Check m/z tolerance and RT window settings
3. **Slow performance**: Reduce number of files or use narrower RT windows
4. **Formula errors**: Validate chemical formulas using standard notation

### Debug Information
The application prints debug information to the console, including:
- Polarity filtering statistics
- File loading progress
- Compound validation results

## Development

### Project Structure
```
src/mzmlexplorer/
├── main.py              # Main application window
├── file_manager.py      # mzML file handling
├── compound_manager.py  # Compound data management
├── eic_window.py       # EIC plotting window
├── utils.py            # Utility functions
└── style.css           # Application styling
```

### Running in Development
```bash
uv run python run.py
```

## License

This project is provided as-is for research and educational purposes.

## Version History

### v1.1.0
- **Performance Optimization**: Pre-calculated m/z values during compound import for faster EIC extraction
- **User Preferences**: Configurable defaults for EIC window parameters via Preferences menu
- **Visual Enhancements**: Reference lines showing expected retention times and baseline
- **Improved Group Separation**: Legend displays RT shift amounts when groups are separated
- **Enhanced Data Management**: Automatic adduct integration from compound lists

### v1.0.0
- Initial release with full LC-HRMS visualization capabilities
- Interactive plotting with mouse controls
- Polarity-aware EIC extraction
- Professional UI with drag & drop support
- Advanced compound filtering
- Normalization and RT cropping options