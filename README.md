# mzML Explorer

A comprehensive GUI application for visualization and analysis of LC-HRMS data from mzML files. Built with PyQt6 and designed for analytical chemistry workflows.

## Features
- **mzML File Loading**: Import multiple mzML files via Excel templates with metadata
- **Compound Management**: Load compound lists with chemical formulas, masses, and retention time windows
- **EIC Extraction**: Extract ion chromatograms with customizable parameters
- **Interactive Plotting**: Professional-grade charts with zoom, pan, and legend management
- **Manual Peak Picking**: The user can manually define a chromatographic peak (start/end only, no baseline correction)
- **MS1 Spectra Visualization**: MS1 spectra of all loaded files are illustrated for particular retention times
- **MS/MS Spectra Visualizationa and Comparison**: MS/MS spectra around the selected retention time are extracted, compared, and results are tabularly illustrated
- **Quantification**: Using the established peak boundaries, the compounds are quantified using a set of reference samples



## Installation
### Setup
1. Install the uv package manager, for details refer to [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
2. Clone (`git clone https://github.com/chrboku/util_mzmlexplorer`) or download the repository from [https://github.com/chrboku/util_mzmlexplorer](https://github.com/chrboku/util_mzmlexplorer).



## Usage
### Starting the Application
```bash
uv run mzmlexplorer
```



## License
This project is licensed under the MIT license (see LICENSE.txt for a full copy of the licensing conditions). 
