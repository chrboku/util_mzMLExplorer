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
1. Install the uv package manager if it is not available, for details refer to [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv). 
2. Install git if it is not available, see [https://git-scm.com/install/](https://git-scm.com/install/) for further details. 
3. Obtain a copy of the mzmlexplorer repository with either: 
   - with git available: execute the command `git clone https://github.com/chrboku/util_mzmlexplorer` from a command line and the folder where you want the application to be saved. 



## Usage

### Start mzmlexplorer
To start mzmlexplorer, open the folder where it has been cloned or downloaded to and double-click the file `run.bat` (Windows) or `run.sh` (Linux, Mac). Note: On Linux and Mac one might have to allow the `run.sh` file to be an executable. 

### Starting the Application from the console
```bash
uv run mzmlexplorer
```



## Keyboard shortcuts, zooming, etc.
The single-compound EIC window provides mouse- and keyboard-driven controls for navigating and annotating chromatograms. Left-click-hold and drag anywhere on the plot to pan the view, or right-click-hold and drag the plot area to zoom in/out interactively. Holding **Alt** unlocks two additional zoom modes: **Alt + mouse wheel** steps the x-axis (retention time) range in/out by a fixed increment, and **Alt + left-click-drag** draws a rubber-band selection that zooms directly to the enclosed retention-time range. A plain right-click (without dragging) opens a context menu with further view, trace, and lookup options, while **Ctrl + left-click** are used to set peak-boundary/baseline points and **Ctrl + right-click** is used to set the RT_min marker, respectively. Double-clicking the plot resets the zoom to show the full chromatogram. The table below summarizes these interactions together with the available keyboard shortcuts.

| Action | Interaction |
| --- | --- |
| Pan the plot | Left-click-hold and drag |
| Zoom in/out (free) | Right-click-hold and drag outside the plot area |
| Zoom the x-axis (retention time) | `Alt` + mouse wheel |
| Zoom to a selected retention-time range | `Alt` + left-click-drag (rubber-band selection) |
| Add a peak boundary / baseline point | `Ctrl` + left-click inside the plot |
| Set the RT_min marker | `Ctrl` + right-click inside the plot |
| Open the context menu (view, traces, lookups) | Right-click (no drag) inside the plot |
| Reset the zoom to the full chromatogram | Double-click the plot |
| Save peak boundaries to the compound list | `Ctrl` + S |
| Clear integration boundaries and baseline | `Ctrl`+ D |
| Zoom out to show the entire EIC | `Ctrl`+ E |
| Set the intensity value under the mouse cursor as the compound intensity | `Ctrl`+ I |

## License
This project is licensed under the MIT license (see LICENSE.txt for a full copy of the licensing conditions). 
