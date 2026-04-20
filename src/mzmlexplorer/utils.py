"""
Utility functions for mzML Explorer
"""

from typing import Dict, Optional

from .FormulaTools import formulaTools

import numpy as np
from matchms import Spectrum as MatchmsSpectrum
from matchms.similarity import CosineGreedy, CosineHungarian
import colorsys
import pandas as pd

# Monoisotopic electron mass in Da (used for adduct m/z calculations)
ELECTRON_MASS = 0.000549  # Da

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


def adduct_mass_change(adduct_info) -> tuple:
    """Return (mass_change, charge, multiplier) from an adduct info row or dict.

    Prefers *ElementsAdded* / *ElementsLost* columns when they are present and
    non-empty; falls back to the pre-computed *Mass_change* value otherwise.

    The relationship is::

        ion_mass = multiplier * neutral_mass + mass_change
        m/z      = ion_mass / abs(charge)

    For ElementsAdded/ElementsLost the ion mass is derived as::

        ion_mass = multiplier * neutral_mass
                   + mass(ElementsAdded)
                   - mass(ElementsLost)
                   - charge * ELECTRON_MASS
    """
    charge = int(adduct_info["Charge"])

    try:
        multiplier = int(adduct_info["Multiplier"])
        if multiplier < 1:
            multiplier = 1
    except (TypeError, ValueError, KeyError):
        multiplier = 1

    def _str_or_none(val):
        """Return val if it is a non-blank string, otherwise None."""
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        return val.strip() if isinstance(val, str) and val.strip() else None

    # Support both dict and pandas Series
    ea = _str_or_none(adduct_info.get("ElementsAdded") if hasattr(adduct_info, "get") else adduct_info["ElementsAdded"] if "ElementsAdded" in adduct_info else None)
    el = _str_or_none(adduct_info.get("ElementsLost") if hasattr(adduct_info, "get") else adduct_info["ElementsLost"] if "ElementsLost" in adduct_info else None)

    if ea is not None or el is not None:
        added_mass = calculate_molecular_mass(ea) if ea else 0.0
        lost_mass = calculate_molecular_mass(el) if el else 0.0
        mass_change = added_mass - lost_mass - charge * ELECTRON_MASS
    else:
        mass_change = float(adduct_info["Mass_change"])

    return mass_change, charge, multiplier


def calculate_mz_from_formula(formula: str, adduct: str, adducts_data) -> float:
    """
    Calculate the m/z value for a given molecular formula and adduct.

    Uses *ElementsAdded* / *ElementsLost* adduct columns when present;
    falls back to *Mass_change* for legacy adduct tables.

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

    mass_change, charge, multiplier = adduct_mass_change(adduct_row.iloc[0])
    return (multiplier * molecular_mass + mass_change) / abs(charge)


def get_mass_tolerance_window(mz: float, tolerance_ppm: float = 5.0) -> tuple[float, float]:
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
            hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
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


def make_usi(spectrum_data: dict, filename: str) -> str:
    """Generate a Unique Spectrum Identifier (USI) for a spectrum.

    Format: filename:RT_min:polarity:(MS1|MS2-precursor_mz):scan_id

    Short fallback if data is sparse: filename:scan_id

    Args:
        spectrum_data: Spectrum dict with keys such as ``rt``, ``polarity``,
            ``precursor_mz``, ``scan_id``, ``ms_level``.
        filename: Display filename (basename, not full path).

    Returns:
        A human-readable USI string.
    """
    if not spectrum_data or not filename:
        return str(filename or "unknown")

    parts = [str(filename)]

    rt = spectrum_data.get("rt") or spectrum_data.get("scan_time")
    if rt is not None:
        try:
            parts.append(f"RT{float(rt):.3f}min")
        except (TypeError, ValueError):
            pass

    polarity = spectrum_data.get("polarity", "")
    if isinstance(polarity, str) and polarity.strip():
        p = polarity.strip().lower()
        if p in {"+", "positive", "pos", "pos."}:
            parts.append("pos")
        elif p in {"-", "negative", "neg", "neg."}:
            parts.append("neg")
        else:
            parts.append(polarity.strip())

    ms_level = spectrum_data.get("ms_level")
    precursor_mz = spectrum_data.get("precursor_mz")
    if precursor_mz is not None:
        try:
            parts.append(f"MS2-{float(precursor_mz):.4f}")
        except (TypeError, ValueError):
            parts.append("MS2")
    elif ms_level is not None:
        parts.append(f"MS{int(ms_level)}")
    else:
        parts.append("MS1")

    scan_id = spectrum_data.get("scan_id") or spectrum_data.get("id") or spectrum_data.get("index")
    if scan_id is not None:
        sid_str = str(scan_id)
        # Avoid double "scan=" prefix when the stored id already contains it
        if sid_str.lower().startswith("scan="):
            parts.append(sid_str)
        else:
            parts.append(f"scan={sid_str}")

    return ":".join(parts)
