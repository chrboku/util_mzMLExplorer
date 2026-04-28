"""
mzML Explorer - A GUI application for visualization of mzML LC-HRMS files
"""

__version__ = "1.0.0"
__author__ = "mzML Explorer Team"
__description__ = "GUI application for mzML LC-HRMS file visualization"

from .compound_manager import CompoundManager
from .file_manager import FileManager
from .main import MzMLExplorerMainWindow, main
from .utils import calculate_molecular_mass, calculate_mz_from_formula
from .windows import EICWindow

__all__ = [
    "main",
    "MzMLExplorerMainWindow",
    "FileManager",
    "CompoundManager",
    "EICWindow",
    "calculate_mz_from_formula",
    "calculate_molecular_mass",
]
