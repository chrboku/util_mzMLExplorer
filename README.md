# mzML Explorer

A comprehensive GUI application for visualization and analysis of LC-HRMS data from mzML files. Available in both Python (original) and C++ (rewrite) implementations.

## Features
- **mzML File Loading**: Import multiple mzML files via TSV/CSV templates with metadata
- **Compound Management**: Load compound lists with chemical formulas, masses, and retention time windows
- **EIC Extraction**: Extract ion chromatograms with customizable parameters
- **Interactive Plotting**: Professional-grade charts with zoom, pan, and legend management
- **Manual Peak Picking**: The user can manually define a chromatographic peak (start/end only, no baseline correction)
- **MS1 Spectra Visualization**: MS1 spectra of all loaded files are illustrated for particular retention times
- **MS/MS Spectra Visualization and Comparison**: MS/MS spectra around the selected retention time are extracted, compared, and results are tabularly illustrated
- **Quantification**: Using the established peak boundaries, the compounds are quantified using a set of reference samples

---

## C++ Implementation

The `cpp/` directory contains a complete C++ rewrite of mzML Explorer using Qt6 and standard C++ libraries.

### C++ Dependencies

| Library | Purpose |
|---------|---------|
| Qt 6 (Core, Gui, Widgets, Charts, Concurrent) | GUI and plotting |
| zlib | Compressed mzML (.mzML.gz) support |
| OpenSSL | SHA-256 for disk cache validation |
| [pugixml](https://pugixml.org/) | XML/mzML parsing (bundled in `cpp/third_party/`) |
| CMake ≥ 3.16 | Build system |

### Building the C++ Version

#### Linux / macOS

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install qt6-base-dev qt6-charts-dev libssl-dev zlib1g-dev cmake

# Build
cd cpp
./build.sh
```

#### Windows (MSVC or MinGW)

```bat
cd cpp
build.bat
```

> **Note:** Qt6 must be installed and `CMAKE_PREFIX_PATH` must point to its installation
> (e.g. `cmake .. -DCMAKE_PREFIX_PATH=C:/Qt/6.x.x/msvc2022_64`).

#### Manual cmake build

```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

### Running the C++ Application

```bash
./cpp/build/mzmlexplorer
```

The C++ binary looks for `style.css` in the same directory.
The CSS stylesheet is automatically copied to the build directory during the CMake build.

### C++ Implementation Notes

- **mzML parsing**: Uses pugixml (bundled) for XML parsing and zlib for gzip-compressed files
- **Disk cache**: Parsed spectra are cached as binary files alongside the mzML files (`.cached.bin`); the cache is invalidated when the mzML file changes (SHA-256)
- **Adduct calculations**: All adduct m/z values are pre-calculated at load time using the same formulas as the Python version
- **Interactive charts**: Qt Charts with pan (left-click drag) and zoom (mouse wheel)
- **Thread-safe EIC extraction**: EIC extraction runs in a background thread (QThread) with progress reporting

### C++ Source Structure

```
cpp/
├── CMakeLists.txt             # Build configuration
├── build.sh                   # Linux/macOS build script
├── build.bat                  # Windows build script
├── src/
│   ├── main.cpp               # Application entry point
│   ├── FormulaTools.h/.cpp    # Chemical formula parser
│   ├── Utils.h/.cpp           # Utility functions (m/z, cosine similarity, colours)
│   ├── MzMLReader.h/.cpp      # mzML file parser (XML + base64 + zlib)
│   ├── FileManager.h/.cpp     # File list management with disk cache
│   ├── CompoundManager.h/.cpp # Compound and adduct management
│   ├── SharedWidgets.h/.cpp   # Reusable Qt widgets
│   ├── MainWindow.h/.cpp      # Main application window
│   ├── EICWindow.h/.cpp       # EIC extraction and plotting window
│   ├── MS1Window.h/.cpp       # MS1 spectrum viewer
│   ├── MSMSWindow.h/.cpp      # MS/MS spectrum viewer and mirror plot
│   ├── MultiAdductWindow.h/.cpp # Multi-adduct analysis window
│   ├── FileExplorerWindow.h/.cpp # Single-file mzML explorer
│   └── CompoundImportDialog.h/.cpp # CSV/TSV import dialog
└── third_party/
    └── pugixml/               # Bundled pugixml XML parser
```

---

## Python Implementation (Original)

### Installation
1. Install the uv package manager, for details refer to [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
2. Download the repository from [https://github.com/chrboku/util_mzMLExplorer/archive/refs/heads/master.zip](https://github.com/chrboku/util_mzMLExplorer/archive/refs/heads/master.zip) if you do not have git installed. Unpack the zip archive to a folder of your choice. Alternatively, the repository can be cloned with the command `git clone https://github.com/chrboku/util_mzmlexplorer`

### Usage

#### Start mzmlexplorer
To start mzmlexplorer, open the folder where it has been cloned or downloaded to and double-click the file `run.bat` (Windows) or `run.sh` (Linux, Mac). Note: On Linux and Mac one might have to allow the `run.sh` file to be an executable.

#### Starting the Application from the console
```bash
uv run mzmlexplorer
```

## License
This project is licensed under the MIT license (see LICENSE.txt for a full copy of the licensing conditions). 
