"""
Utility functions for mzML Explorer
"""

from typing import Dict, Optional

from .FormulaTools import formulaTools

import numpy as np
from matchms import Spectrum as MatchmsSpectrum
from matchms.similarity import CosineGreedy, CosineHungarian
import colorsys

# Atomic masses (monoisotopic)
ATOMIC_MASSES = {
    "H": 1.007825032,
    "C": 12.0,
    "N": 14.003074004,
    "O": 15.994914620,
    "P": 30.973761998,
    "S": 31.972071174,
    "Cl": 34.96885268,
}

# Heavy isotope labels and their mass differences relative to the common isotope
ISOTOPE_DATA = {
    "C": {"label": "13C", "mass_delta": 1.003354835},
    "H": {"label": "2H", "mass_delta": 1.006276745},
    "N": {"label": "15N", "mass_delta": 0.997034893},
    "O": {"label": "18O", "mass_delta": 2.004245205},
    "S": {"label": "34S", "mass_delta": 1.9957959},
    "Cl": {"label": "37Cl", "mass_delta": 1.997048},
}


# Reuse a single parser/mass calculator so formula handling is consistent app-wide.
_FORMULA_TOOLS = formulaTools()


def parse_molecular_formula(formula: str) -> Dict[str, int]:
    """
    Parse a molecular formula string into a dictionary of element counts.

    Args:
        formula: Molecular formula string (e.g., "C6H12O6")

    Returns:
        Dictionary mapping element symbols to their counts
    """
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("Formula must be a non-empty string")

    return _FORMULA_TOOLS.parseFormula(formula)


def calculate_molecular_mass(formula: str) -> float:
    """
    Calculate the monoisotopic molecular mass of a compound.

    Args:
        formula: Molecular formula string

    Returns:
        Monoisotopic molecular mass in Da
    """
    composition = parse_molecular_formula(formula)
    return _FORMULA_TOOLS.calcMolWeight(composition)


def calculate_mz_from_formula(formula: str, adduct: str, adducts_data) -> float:
    """
    Calculate the m/z value for a given molecular formula and adduct.

    Args:
        formula: Molecular formula string
        adduct: Adduct string (e.g., "[M+H]+")

    Returns:
        m/z value
    """
    molecular_mass = calculate_molecular_mass(formula)

    adduct_row = adducts_data[adducts_data["Adduct"] == adduct]

    if adduct_row.empty:
        raise ValueError(f"Unknown adduct: {adduct}")

    adduct_info = adduct_row.iloc[0]
    mass_change = adduct_info["Mass_change"]
    charge = adduct_info["Charge"]
    multiplier = adduct_info.get("Multiplier", 1)

    # Calculate m/z
    total_mass = (molecular_mass * multiplier) + mass_change
    mz = total_mass / abs(charge)

    return mz


def get_mass_tolerance_window(
    mz: float, tolerance_ppm: float = 5.0
) -> tuple[float, float]:
    """
    Calculate the mass tolerance window for a given m/z value.

    Args:
        mz: Target m/z value
        tolerance_ppm: Mass tolerance in ppm

    Returns:
        Tuple of (min_mz, max_mz)
    """
    delta = mz * tolerance_ppm / 1e6
    return (mz - delta, mz + delta)


def generate_color_palette(n_colors: int) -> list[str]:
    """
    Generate a list of distinct colors for grouping.

    Args:
        n_colors: Number of colors needed

    Returns:
        List of hex color strings
    """
    # Predefined color palette
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d3",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]

    # If we need more colors than available, generate more using a simple algorithm
    if n_colors > len(colors):
        for i in range(len(colors), n_colors):
            hue = (i * 0.618033988749895) % 1  # Golden ratio conjugate
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colors.append(hex_color)

    return colors[:n_colors]


def format_retention_time(rt_minutes: float) -> str:
    """
    Format retention time for display.

    Args:
        rt_minutes: Retention time in minutes

    Returns:
        Formatted string
    """
    return f"{rt_minutes:.2f} min"


def format_mz(mz: float, decimals: int = 4) -> str:
    """
    Format m/z value for display.

    Args:
        mz: m/z value
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    return f"{mz:.{decimals}f}"


def calculate_cosine_similarity(
    spectrum1: Dict,
    spectrum2: Dict,
    mz_tolerance: float = 0.1,
    method: str = "CosineHungarian",
) -> float:
    """
    Calculate the cosine similarity between two MS/MS spectra using matchms.

    Args:
        spectrum1: First spectrum data with 'mz' and 'intensity' arrays
        spectrum2: Second spectrum data with 'mz' and 'intensity' arrays
        mz_tolerance: m/z tolerance for peak matching in Da
        method: Scoring method — 'CosineHungarian' (default) or 'CosineGreedy'

    Returns:
        Cosine similarity score (0-1)
    """

    mz1 = np.array(spectrum1["mz"], dtype=float)
    int1 = np.array(spectrum1["intensity"], dtype=float)
    mz2 = np.array(spectrum2["mz"], dtype=float)
    int2 = np.array(spectrum2["intensity"], dtype=float)

    if len(mz1) == 0 or len(mz2) == 0:
        return 0.0

    spec_a = MatchmsSpectrum(
        mz=mz1,
        intensities=int1,
        metadata={"precursor_mz": spectrum1.get("precursor_mz", 0.0)},
    )
    spec_b = MatchmsSpectrum(
        mz=mz2,
        intensities=int2,
        metadata={"precursor_mz": spectrum2.get("precursor_mz", 0.0)},
    )

    if method == "CosineGreedy":
        scorer = CosineGreedy(tolerance=mz_tolerance)
    else:
        scorer = CosineHungarian(tolerance=mz_tolerance)

    result = scorer.pair(spec_a, spec_b)
    return float(result["score"])


def calculate_similarity_statistics(similarities: list) -> Dict[str, float]:
    """
    Calculate statistical measures for a list of similarity scores.

    Args:
        similarities: List of similarity scores

    Returns:
        Dictionary with statistical measures
    """
    if not similarities:
        return {
            "min": 0.0,
            "percentile_10": 0.0,
            "median": 0.0,
            "percentile_90": 0.0,
            "max": 0.0,
        }

    similarities = np.array(similarities)

    return {
        "min": float(np.min(similarities)),
        "percentile_10": float(np.percentile(similarities, 10)),
        "median": float(np.median(similarities)),
        "percentile_90": float(np.percentile(similarities, 90)),
        "max": float(np.max(similarities)),
    }
