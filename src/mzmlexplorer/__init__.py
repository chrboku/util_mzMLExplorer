"""
mzML Explorer - A GUI application for visualization of mzML LC-HRMS files
"""

__version__ = "1.0.0"
__author__ = "mzML Explorer Team"
__description__ = "GUI application for mzML LC-HRMS file visualization"

from .main import main, MzMLExplorerMainWindow
from .file_manager import FileManager
from .compound_manager import CompoundManager
from .eic_window import EICWindow
from .utils import calculate_mz_from_formula, calculate_molecular_mass

__all__ = [
    'main',
    'MzMLExplorerMainWindow',
    'FileManager', 
    'CompoundManager',
    'EICWindow',
    'calculate_mz_from_formula',
    'calculate_molecular_mass'
]
