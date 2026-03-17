"""
Utility functions for mzML Explorer
"""

from typing import Dict, Optional

from .FormulaTools import formulaTools


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
        import colorsys

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
    spectrum1: Dict, spectrum2: Dict, mz_tolerance: float = 0.01
) -> float:
    """
    Calculate the cosine similarity between two MS/MS spectra.

    Args:
        spectrum1: First spectrum data with 'mz' and 'intensity' arrays
        spectrum2: Second spectrum data with 'mz' and 'intensity' arrays
        mz_tolerance: m/z tolerance for peak matching in Da

    Returns:
        Cosine similarity score (0-1)
    """
    import numpy as np

    # Get spectrum data
    mz1 = np.array(spectrum1["mz"])
    intensity1 = np.array(spectrum1["intensity"])
    mz2 = np.array(spectrum2["mz"])
    intensity2 = np.array(spectrum2["intensity"])

    # Normalize intensities
    if len(intensity1) > 0 and np.max(intensity1) > 0:
        intensity1 = intensity1 / np.max(intensity1)
    if len(intensity2) > 0 and np.max(intensity2) > 0:
        intensity2 = intensity2 / np.max(intensity2)

    # If either spectrum is empty, return 0
    if len(mz1) == 0 or len(mz2) == 0:
        return 0.0

    # Create intensity vectors for matching peaks
    matched_intensity1 = []
    matched_intensity2 = []

    # For each peak in spectrum1, find matching peak in spectrum2
    for i, mz in enumerate(mz1):
        intensity = intensity1[i]

        # Find matching peak in spectrum2
        matches = np.where(np.abs(mz2 - mz) <= mz_tolerance)[0]
        if len(matches) > 0:
            # Use the closest match
            closest_idx = matches[np.argmin(np.abs(mz2[matches] - mz))]
            matched_intensity1.append(intensity)
            matched_intensity2.append(intensity2[closest_idx])

        else:
            matched_intensity1.append(intensity)
            matched_intensity2.append(0.0)

    # For each peak in spectrum2 that doesn't have a match in spectrum1
    for i, mz in enumerate(mz2):
        intensity = intensity2[i]

        # Check if this peak was already matched
        if not np.any(np.abs(mz1 - mz) <= mz_tolerance):
            matched_intensity1.append(0.0)
            matched_intensity2.append(intensity)

    if len(matched_intensity1) == 0:
        return 0.0

    matched_intensity1 = np.array(matched_intensity1)
    matched_intensity2 = np.array(matched_intensity2)

    # Calculate cosine similarity
    dot_product = np.dot(matched_intensity1, matched_intensity2)
    norm1 = np.linalg.norm(matched_intensity1)
    norm2 = np.linalg.norm(matched_intensity2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def calculate_similarity_statistics(similarities: list) -> Dict[str, float]:
    """
    Calculate statistical measures for a list of similarity scores.

    Args:
        similarities: List of similarity scores

    Returns:
        Dictionary with statistical measures
    """
    import numpy as np

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
