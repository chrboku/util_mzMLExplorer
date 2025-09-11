"""
Utility functions for mzML Explorer
"""
import re
from typing import Dict, Optional


# Atomic masses (monoisotopic)
ATOMIC_MASSES = {
    'H': 1.007825032,
    'C': 12.0,
    'N': 14.003074004,
    'O': 15.994914620,
    'P': 30.973761998,
    'S': 31.972071174,
}

def parse_molecular_formula(formula: str) -> Dict[str, int]:
    """
    Parse a molecular formula string into a dictionary of element counts.
    
    Args:
        formula: Molecular formula string (e.g., "C6H12O6")
        
    Returns:
        Dictionary mapping element symbols to their counts
    """
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    composition = {}
    for element, count in matches:
        count = int(count) if count else 1
        composition[element] = composition.get(element, 0) + count
    
    return composition


def calculate_molecular_mass(formula: str) -> float:
    """
    Calculate the monoisotopic molecular mass of a compound.
    
    Args:
        formula: Molecular formula string
        
    Returns:
        Monoisotopic molecular mass in Da
    """
    composition = parse_molecular_formula(formula)
    mass = 0.0
    
    for element, count in composition.items():
        if element in ATOMIC_MASSES:
            mass += ATOMIC_MASSES[element] * count
        else:
            raise ValueError(f"Unknown element: {element}")
    
    return mass


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

    adduct_row = adducts_data[adducts_data['Adduct'] == adduct]
        
    if adduct_row.empty:
        raise ValueError(f"Unknown adduct: {adduct}")
    
    adduct_info = adduct_row.iloc[0]
    mass_change = adduct_info['Mass_change']
    charge = adduct_info['Charge']
    multiplier = adduct_info.get('Multiplier', 1)
    
    # Calculate m/z
    total_mass = (molecular_mass * multiplier) + mass_change
    mz = total_mass / abs(charge)
    
    return mz


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
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    # If we need more colors than available, generate more using a simple algorithm
    if n_colors > len(colors):
        import colorsys
        for i in range(len(colors), n_colors):
            hue = (i * 0.618033988749895) % 1  # Golden ratio conjugate
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
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
