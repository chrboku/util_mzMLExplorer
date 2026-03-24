"""
Compound manager for handling compound information and adducts
"""

import pandas as pd
import re
from typing import List, Dict, Optional, Tuple
from .utils import (
    calculate_mz_from_formula,
    parse_molecular_formula,
    calculate_molecular_mass,
    ISOTOPE_DATA,
)


class CompoundManager:
    """Manages compound information and adduct calculations"""

    def __init__(self):
        self.compounds_data = pd.DataFrame()
        self.adducts_data = pd.DataFrame()
        self.compound_adduct_data = {}  # Dictionary to store pre-calculated m/z values

    def load_compounds(self, compounds_input):
        """
        Load compounds data from DataFrame or dict of DataFrames. New compounds are added to existing ones.

        Args:
            compounds_input: Either a DataFrame or dict of DataFrames from Excel sheets
        """
        # Canonical column name mappings (lowercase key -> canonical name)
        _COMPOUNDS_COLUMNS = {
            "name": "Name",
            "common_adducts": "Common_adducts",
            "chemicalformula": "ChemicalFormula",
            "mass": "Mass",
            "rt_min": "RT_min",
            "rt_start_min": "RT_start_min",
            "rt_end_min": "RT_end_min",
            "group": "Group",
            "smiles": "SMILES",
            "isotopologs": "Isotopologs",
        }
        _ADDUCTS_COLUMNS = {
            "adduct": "Adduct",
            "mass_change": "Mass_change",
            "charge": "Charge",
            "multiplier": "Multiplier",
        }

        # Handle different input types
        if isinstance(compounds_input, dict):
            # Multi-sheet Excel file
            if "Compounds" in compounds_input:
                compounds_data = compounds_input["Compounds"]
            else:
                # Use first sheet as compounds data
                compounds_data = list(compounds_input.values())[0]

            if "Adducts" in compounds_input:
                new_adducts_data = compounds_input["Adducts"]
                # Normalize adducts column names (case-insensitive)
                new_adducts_data = new_adducts_data.rename(columns={col: _ADDUCTS_COLUMNS.get(col.lower(), col) for col in new_adducts_data.columns})
                # Merge with existing adducts
                if self.adducts_data.empty:
                    self.adducts_data = new_adducts_data
                else:
                    # Combine adducts, avoiding duplicates based on 'Adduct' column
                    existing_adducts = set(self.adducts_data["Adduct"].tolist())
                    new_adducts = new_adducts_data[~new_adducts_data["Adduct"].isin(existing_adducts)]
                    if not new_adducts.empty:
                        self.adducts_data = pd.concat([self.adducts_data, new_adducts], ignore_index=True)
            elif self.adducts_data.empty:
                self.adducts_data = self._get_default_adducts()
        else:
            # Single DataFrame
            compounds_data = compounds_input
            if self.adducts_data.empty:
                self.adducts_data = self._get_default_adducts()

        # Normalize compounds column names (case-insensitive)
        compounds_data = compounds_data.rename(columns={col: _COMPOUNDS_COLUMNS.get(col.lower(), col) for col in compounds_data.columns})

        # Validate required columns for compounds
        # RT columns are now optional and will default to 0-100 min if not provided
        required_columns = [
            "Name",
            "Common_adducts",
        ]
        missing_columns = [col for col in required_columns if col not in compounds_data.columns]

        # Check for either ChemicalFormula or Mass column (or just adducts with m/z values)
        has_formula = "ChemicalFormula" in compounds_data.columns
        has_mass = "Mass" in compounds_data.columns

        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Add RT columns if they don't exist (will be filled with defaults)
        if "RT_min" not in compounds_data.columns:
            compounds_data["RT_min"] = None
        if "RT_start_min" not in compounds_data.columns:
            compounds_data["RT_start_min"] = None
        if "RT_end_min" not in compounds_data.columns:
            compounds_data["RT_end_min"] = None

        # Add Group column if it doesn't exist (optional column)
        if "Group" not in compounds_data.columns:
            compounds_data["Group"] = None

        # Get existing compound names to avoid duplicates
        existing_compounds = set(self.compounds_data["Name"].tolist()) if not self.compounds_data.empty else set()

        # Validate and process compounds
        valid_compounds = []
        for idx, row in compounds_data.iterrows():
            try:
                compound_name = row["Name"]

                # Skip if compound already exists
                if compound_name in existing_compounds:
                    print(f"Info: Compound already exists, skipping: {compound_name}")
                    continue

                compound_dict = row.to_dict()

                # Set default RT values if not provided (0-100 minutes)
                if pd.isna(compound_dict.get("RT_min")) or compound_dict.get("RT_min") is None:
                    compound_dict["RT_min"] = 50.0  # Default center
                if pd.isna(compound_dict.get("RT_start_min")) or compound_dict.get("RT_start_min") is None:
                    compound_dict["RT_start_min"] = 0.0  # Default start
                if pd.isna(compound_dict.get("RT_end_min")) or compound_dict.get("RT_end_min") is None:
                    compound_dict["RT_end_min"] = 100.0  # Default end

                # Determine compound type based on available data
                if has_formula and pd.notna(row.get("ChemicalFormula")) and str(row.get("ChemicalFormula")).strip():
                    # Test if the formula can be parsed
                    parse_molecular_formula(str(row["ChemicalFormula"]))
                    compound_dict["compound_type"] = "formula"
                elif has_mass and pd.notna(row.get("Mass")) and str(row.get("Mass")).strip():
                    # Validate mass value
                    mass = float(row["Mass"])
                    if mass <= 0:
                        raise ValueError(f"Mass must be positive: {mass}")
                    compound_dict["compound_type"] = "mass"
                else:
                    # Check if compound has adducts with m/z values (e.g., [197.23234]+)
                    adducts_str = row.get("Common_adducts", "")
                    if isinstance(adducts_str, str) and adducts_str.strip():
                        adduct_list = [a.strip() for a in adducts_str.split(",") if a.strip()]
                        has_mz_adducts = any(self._is_mz_adduct(adduct) for adduct in adduct_list)
                        if has_mz_adducts:
                            compound_dict["compound_type"] = "mz_only"
                        else:
                            raise ValueError("Either ChemicalFormula, Mass, or m/z adducts must be provided")
                    else:
                        raise ValueError("Either ChemicalFormula, Mass, or m/z adducts must be provided")

                valid_compounds.append(compound_dict)

            except Exception as e:
                print(f"Warning: Invalid compound {row.get('Name', 'Unknown')}: {str(e)}")

        if not valid_compounds:
            if not self.compounds_data.empty:
                print("No new valid compounds to add.")
                return
            else:
                raise ValueError("No valid compounds found!")

        # Create DataFrame for new compounds
        new_compounds_df = pd.DataFrame(valid_compounds)

        # Combine with existing data
        if self.compounds_data.empty:
            self.compounds_data = new_compounds_df
        else:
            self.compounds_data = pd.concat([self.compounds_data, new_compounds_df], ignore_index=True)

        print(f"Added {len(valid_compounds)} new compounds. Total compounds: {len(self.compounds_data)}")

        # Pre-calculate m/z values for all compound-adduct combinations
        self._precalculate_mz_values()

    def _precalculate_mz_values(self):
        """Pre-calculate m/z values and polarity for all compound-adduct combinations"""
        self.compound_adduct_data = {}  # Dictionary to store pre-calculated data

        for _, compound in self.compounds_data.iterrows():
            compound_name = compound["Name"]
            adducts = self.get_compound_adducts(compound_name)

            self.compound_adduct_data[compound_name] = {}

            for adduct in adducts:
                try:
                    mz_value = self.calculate_compound_mz(compound_name, adduct)
                    polarity = self._determine_polarity(adduct)

                    self.compound_adduct_data[compound_name][adduct] = {
                        "mz": mz_value,
                        "polarity": polarity,
                        "display_name": self.get_adduct_display_name(compound_name, adduct),
                    }
                except Exception as e:
                    print(f"Warning: Could not calculate m/z for {compound_name} with {adduct}: {str(e)}")
                    self.compound_adduct_data[compound_name][adduct] = {
                        "mz": None,
                        "polarity": None,
                        "display_name": self.get_adduct_display_name(compound_name, adduct),
                    }

        print(f"Pre-calculated m/z values for {len(self.compound_adduct_data)} compounds")

    def _determine_polarity(self, adduct: str) -> Optional[str]:
        """Determine polarity from adduct string"""
        polarity = None
        if adduct.endswith("+") or adduct.endswith("]+"):
            polarity = "positive"
        elif adduct.endswith("-") or adduct.endswith("]-"):
            polarity = "negative"
        else:
            # Check in adducts data table
            adduct_row = self.adducts_data[self.adducts_data["Adduct"] == adduct]
            if not adduct_row.empty:
                charge = adduct_row.iloc[0]["Charge"]
                polarity = "positive" if charge > 0 else "negative"
        return polarity

    def get_precalculated_data(self, compound_name: str, adduct: str) -> Optional[Dict]:
        """Get pre-calculated m/z, polarity and display name for compound-adduct combination"""
        if hasattr(self, "compound_adduct_data"):
            return self.compound_adduct_data.get(compound_name, {}).get(adduct)
        return None

    def _get_default_adducts(self) -> pd.DataFrame:
        """Get default adduct table"""
        default_adducts = [
            {"Adduct": "[M+H]+", "Mass_change": 1.007276, "Charge": 1},
            {"Adduct": "[M-H]-", "Mass_change": -1.007276, "Charge": -1},
        ]
        return pd.DataFrame(default_adducts)

    def _is_mz_adduct(self, adduct_str: str) -> bool:
        """
        Check if adduct string is a direct m/z specification like [197.23234]+ or [197.23234]-

        Returns:
            True if it's an m/z adduct, False otherwise
        """
        pattern = r"\[(\d+\.?\d*)\]([+-])"
        return bool(re.match(pattern, adduct_str.strip()))

    def _parse_mz_adduct(self, adduct_str: str) -> Tuple[float, str]:
        """
        Parse m/z value from adduct string like [197.23234]+ or [197.23234]-

        Returns:
            Tuple of (mz_value, polarity)
        """
        pattern = r"\[(\d+\.?\d*)\]([+-])"
        match = re.match(pattern, adduct_str.strip())

        if match:
            mz_value = float(match.group(1))
            polarity = match.group(2)
            return mz_value, polarity
        else:
            raise ValueError(f"Invalid m/z adduct format: {adduct_str}")

    def get_compounds_data(self) -> pd.DataFrame:
        """Get the loaded compounds data"""
        return self.compounds_data.copy()

    def get_adducts_data(self) -> pd.DataFrame:
        """Get the loaded adducts data"""
        return self.adducts_data.copy()

    def get_compound_by_name(self, name: str) -> Optional[pd.Series]:
        """Get compound data by name"""
        compound_rows = self.compounds_data[self.compounds_data["Name"] == name]
        if not compound_rows.empty:
            return compound_rows.iloc[0]
        return None

    def get_compound_adducts(self, compound_name: str) -> List[str]:
        """
        Get list of adducts for a compound.

        Args:
            compound_name: Name of the compound

        Returns:
            List of adduct strings
        """
        compound = self.get_compound_by_name(compound_name)
        if compound is not None:
            adducts_str = compound["Common_adducts"]
            if isinstance(adducts_str, str):
                return [adduct.strip() for adduct in adducts_str.split(",") if adduct.strip()]
        return []

    def calculate_compound_mz(self, compound_name: str, adduct: str) -> Optional[float]:
        """
        Calculate m/z for a compound with a specific adduct.

        Args:
            compound_name: Name of the compound
            adduct: Adduct string or m/z specification

        Returns:
            m/z value or None if calculation fails
        """
        compound = self.get_compound_by_name(compound_name)
        if compound is None:
            return None

        try:
            # Check if adduct is a direct m/z specification
            if adduct.startswith("[") and "]" in adduct and not any(letter in adduct for letter in "MHCNO"):
                mz_value, polarity = self._parse_mz_adduct(adduct)
                return mz_value

            # Handle standard adducts
            compound_type = compound.get("compound_type", "formula")

            if compound_type == "formula":
                mz_value = calculate_mz_from_formula(compound["ChemicalFormula"], adduct, self.adducts_data)
                return mz_value
            elif compound_type == "mass":
                mz_value = self._calculate_mz_from_mass(compound["Mass"], adduct)
                return mz_value
            elif compound_type == "mz_only":
                # For mz_only compounds, the adduct should be an m/z specification
                mz_value, polarity = self._parse_mz_adduct(adduct)
                return mz_value
            else:
                raise ValueError(f"Unknown compound type: {compound_type}")

        except Exception as e:
            print(f"Error calculating m/z for {compound_name} with {adduct}: {str(e)}")
            return None

    def _get_compound_composition(self, compound_name: str) -> Dict[str, int]:
        """Return parsed elemental composition for the compound if available."""
        compound = self.get_compound_by_name(compound_name)
        if compound is None:
            return {}

        formula = compound.get("ChemicalFormula")
        if isinstance(formula, str) and formula.strip():
            try:
                return parse_molecular_formula(formula.strip())
            except Exception:
                return {}
        return {}

    def _get_adduct_charge(self, adduct: str) -> Optional[float]:
        """Look up the charge for an adduct if defined in the adduct table."""
        if self.adducts_data.empty:
            return None

        adduct_row = self.adducts_data[self.adducts_data["Adduct"] == adduct]
        if not adduct_row.empty:
            try:
                return float(adduct_row.iloc[0]["Charge"])
            except Exception:
                return None
        return None

    def get_isotope_label(self, element: str) -> Optional[str]:
        """Return the heavy isotope label for a given element."""
        isotope_info = ISOTOPE_DATA.get(element)
        if isotope_info:
            return isotope_info["label"]
        return None

    def get_isotopolog_elements(self, compound_name: str) -> Dict[str, Optional[int]]:
        """Return elements eligible for isotopolog calculations with their counts."""
        composition = self._get_compound_composition(compound_name)

        elements: Dict[str, Optional[int]] = {}
        for element, count in composition.items():
            if element in ISOTOPE_DATA:
                elements[element] = int(count)

        if not elements and "C" in ISOTOPE_DATA:
            # Default to carbon if no composition is available
            elements["C"] = composition.get("C")

        return elements

    def get_isotopolog_counts(self, element: str, total_count: Optional[int]) -> List[int]:
        """Return the list of labeled atom counts to show for a given element."""
        if element not in ISOTOPE_DATA:
            return []

        counts = set()

        max_base = 3
        if total_count is not None:
            try:
                total_int = int(total_count)
            except (TypeError, ValueError):
                total_int = None
        else:
            total_int = None

        if total_int is not None and total_int > 0:
            max_base = min(3, total_int)

        for num in range(1, max_base + 1):
            counts.add(num)

        if total_int is not None and total_int > 0:
            for num in range(total_int, total_int - 3, -1):
                if num >= 1:
                    counts.add(num)

        return sorted(counts)

    def calculate_isotopolog_mz(self, compound_name: str, adduct: str, element: str, labeled_count: int) -> Optional[float]:
        """Calculate m/z for a specific isotopolog and adduct selection."""
        if labeled_count <= 0:
            return None

        isotope_info = ISOTOPE_DATA.get(element)
        if not isotope_info:
            return None

        base_mz = self.calculate_compound_mz(compound_name, adduct)
        if base_mz is None:
            return None

        mass_delta = isotope_info.get("mass_delta")
        if mass_delta is None:
            return None

        charge = self._get_adduct_charge(adduct)
        shift = mass_delta * labeled_count
        if charge and charge != 0:
            shift /= abs(charge)

        return base_mz + shift

    def calculate_custom_isotopolog_mz(self, compound_name: str, adduct: str, isotopolog_formula: str) -> Optional[float]:
        """Calculate m/z shift for a custom isotopolog formula.

        Expected format is one or multiple terms like ``[13C]15`` or
        ``[13C]9[15N]2``. The shift is the sum of the isotope mass deltas
        multiplied by their counts and adjusted by adduct charge magnitude.
        """
        parsed = self._parse_custom_isotopolog_formula(isotopolog_formula)
        if not parsed:
            return None

        base_mz = self.calculate_compound_mz(compound_name, adduct)
        if base_mz is None:
            return None

        shift = 0.0
        for isotope_label, count in parsed:
            mass_delta = self._get_mass_delta_for_isotope_label(isotope_label)
            if mass_delta is None:
                return None
            shift += mass_delta * count

        charge = self._get_adduct_charge(adduct)
        if charge and charge != 0:
            shift /= abs(charge)

        return base_mz + shift

    def _get_mass_delta_for_isotope_label(self, isotope_label: str) -> Optional[float]:
        """Return the mass delta for an isotope label (for example ``13C``)."""
        for isotope_info in ISOTOPE_DATA.values():
            if isotope_info.get("label") == isotope_label:
                return isotope_info.get("mass_delta")
        return None

    def _parse_custom_isotopolog_formula(self, isotopolog_formula: str) -> Optional[List[Tuple[str, int]]]:
        """Parse a custom isotopolog formula into ``(isotope_label, count)`` terms."""
        if not isinstance(isotopolog_formula, str):
            return None

        formula = isotopolog_formula.strip()
        if not formula:
            return None

        pattern = re.compile(r"\[([^\]]+)\](\d+)")
        terms: List[Tuple[str, int]] = []
        pos = 0

        for match in pattern.finditer(formula):
            # Only allow optional whitespace between terms.
            between = formula[pos : match.start()]
            if between.strip():
                return None

            isotope_label = match.group(1).strip()
            try:
                count = int(match.group(2))
            except ValueError:
                return None

            if not isotope_label or count <= 0:
                return None

            terms.append((isotope_label, count))
            pos = match.end()

        if not terms:
            return None

        # Disallow trailing non-whitespace characters.
        if formula[pos:].strip():
            return None

        return terms

    def _calculate_mz_from_mass(self, molecular_mass: float, adduct: str) -> float:
        """
        Calculate m/z from molecular mass and adduct.

        Args:
            molecular_mass: Molecular mass in Da
            adduct: Adduct string

        Returns:
            m/z value
        """
        # Look up adduct in adducts table
        adduct_row = self.adducts_data[self.adducts_data["Adduct"] == adduct]

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

    def get_all_available_adducts(self) -> List[str]:
        """
        Get all available adducts from the adducts data table.

        Returns:
            List of all available adduct strings
        """
        if not self.adducts_data.empty:
            return self.adducts_data["Adduct"].tolist()
        return []

    def get_compound_adducts_categorized(self, compound_name: str) -> Dict[str, List[str]]:
        """
        Get adducts for a compound categorized by specified/remaining.

        Args:
            compound_name: Name of the compound

        Returns:
            Dictionary with 'specified' and 'remaining' adduct lists
        """
        compound = self.get_compound_by_name(compound_name)
        if compound is None:
            return {"specified": [], "remaining": []}

        compound_type = compound.get("compound_type", "formula")

        # Get specified adducts for this compound
        specified_adducts = self.get_compound_adducts(compound_name)

        if compound_type == "mz_only":
            # For m/z only compounds, only show the specified adducts
            return {"specified": specified_adducts, "remaining": []}

        # For compounds with formula or mass, show all possible adducts
        all_adducts = self.get_all_available_adducts()
        remaining_adducts = [a for a in all_adducts if a not in specified_adducts]

        return {"specified": specified_adducts, "remaining": remaining_adducts}

    def can_calculate_adducts_from_formula(self, compound_name: str) -> bool:
        """
        Check if compound can calculate adducts from sum formula.

        Args:
            compound_name: Name of the compound

        Returns:
            True if compound has formula or mass (not m/z only)
        """
        compound = self.get_compound_by_name(compound_name)
        if compound is None:
            return False

        compound_type = compound.get("compound_type", "formula")
        return compound_type in ["formula", "mass"]

    def get_adduct_display_name(self, compound_name: str, adduct: str) -> str:
        """
        Get display name for adduct in the tree view.

        Args:
            compound_name: Name of the compound
            adduct: Adduct string

        Returns:
            Display name for the adduct
        """
        try:
            # Check if adduct is a direct m/z specification
            if adduct.startswith("[") and "]" in adduct and not any(letter in adduct for letter in "MHCNO"):
                mz_value, polarity = self._parse_mz_adduct(adduct)
                return f"m/z {mz_value:.4f} ({polarity})"
            else:
                # Standard adduct, just return the adduct name
                return adduct
        except:
            return adduct

    def get_compound_rt_window(self, compound_name: str) -> Optional[tuple[float, float, float]]:
        """
        Get retention time window for a compound.

        Args:
            compound_name: Name of the compound

        Returns:
            Tuple of (rt_center, rt_start, rt_end) or None
        """
        compound = self.get_compound_by_name(compound_name)
        if compound is not None:
            return (
                float(compound["RT_min"]),
                float(compound["RT_start_min"]),
                float(compound["RT_end_min"]),
            )
        return None

    def search_compounds_by_formula(self, formula: str) -> List[str]:
        """
        Search for compounds with a specific chemical formula.

        Args:
            formula: Chemical formula to search for

        Returns:
            List of compound names
        """
        if "ChemicalFormula" in self.compounds_data.columns:
            matching_compounds = self.compounds_data[self.compounds_data["ChemicalFormula"] == formula]
            return matching_compounds["Name"].tolist()
        return []

    def search_compounds_by_mass(self, mass: float, tolerance: float = 0.01) -> List[str]:
        """
        Search for compounds with a specific mass.

        Args:
            mass: Target mass in Da
            tolerance: Mass tolerance in Da

        Returns:
            List of compound names
        """
        matching_compounds = []

        for _, compound in self.compounds_data.iterrows():
            compound_mass = None

            if compound.get("compound_type") == "mass" and "Mass" in compound:
                compound_mass = float(compound["Mass"])
            elif compound.get("compound_type") == "formula" and "ChemicalFormula" in compound:
                try:
                    compound_mass = calculate_molecular_mass(compound["ChemicalFormula"])
                except:
                    continue

            if compound_mass and abs(compound_mass - mass) <= tolerance:
                matching_compounds.append(compound["Name"])

        return matching_compounds

    def search_compounds_by_rt(self, rt: float, tolerance: float = 0.5) -> List[str]:
        """
        Search for compounds eluting within a retention time window.

        Args:
            rt: Target retention time in minutes
            tolerance: RT tolerance in minutes

        Returns:
            List of compound names
        """
        matching_compounds = self.compounds_data[(self.compounds_data["RT_start_min"] <= rt + tolerance) & (self.compounds_data["RT_end_min"] >= rt - tolerance)]
        return matching_compounds["Name"].tolist()

    def get_compounds_summary(self) -> Dict:
        """
        Get summary statistics about loaded compounds.

        Returns:
            Dictionary with summary information
        """
        if self.compounds_data.empty:
            return {
                "total_compounds": 0,
                "formula_compounds": 0,
                "mass_compounds": 0,
                "unique_formulas": 0,
                "rt_range": (0, 0),
                "total_adducts": 0,
            }

        # Count total adducts
        total_adducts = 0
        for _, row in self.compounds_data.iterrows():
            adducts_str = row["Common_adducts"]
            if isinstance(adducts_str, str):
                adducts_count = len([a.strip() for a in adducts_str.split(",") if a.strip()])
                total_adducts += adducts_count

        # Count compound types
        formula_count = len(self.compounds_data[self.compounds_data.get("compound_type") == "formula"])
        mass_count = len(self.compounds_data[self.compounds_data.get("compound_type") == "mass"])

        unique_formulas = 0
        if "ChemicalFormula" in self.compounds_data.columns:
            unique_formulas = self.compounds_data["ChemicalFormula"].nunique()

        return {
            "total_compounds": len(self.compounds_data),
            "formula_compounds": formula_count,
            "mass_compounds": mass_count,
            "unique_formulas": unique_formulas,
            "rt_range": (
                self.compounds_data["RT_start_min"].min(),
                self.compounds_data["RT_end_min"].max(),
            ),
            "total_adducts": total_adducts,
        }

    def export_compounds_with_mz(self, output_path: str):
        """
        Export compounds data with calculated m/z values for all adducts.

        Args:
            output_path: Path to save the Excel file
        """
        if self.compounds_data.empty:
            raise ValueError("No compounds data to export!")

        export_data = []

        for _, compound in self.compounds_data.iterrows():
            adducts = self.get_compound_adducts(compound["Name"])

            for adduct in adducts:
                mz = self.calculate_compound_mz(compound["Name"], adduct)

                export_row = compound.to_dict()
                export_row["Adduct"] = adduct
                export_row["MZ"] = mz
                export_data.append(export_row)

        export_df = pd.DataFrame(export_data)
        export_df.to_excel(output_path, index=False)

    def validate_compound_data(self) -> List[str]:
        """
        Validate the loaded compound data and return list of issues.

        Returns:
            List of validation error messages
        """
        issues = []

        if self.compounds_data.empty:
            issues.append("No compounds data loaded")
            return issues

        for idx, compound in self.compounds_data.iterrows():
            compound_name = compound["Name"]

            # Check RT values
            rt_center = compound["RT_min"]
            rt_start = compound["RT_start_min"]
            rt_end = compound["RT_end_min"]

            if rt_start >= rt_end:
                issues.append(f"{compound_name}: RT_start_min must be less than RT_end_min")

            if not (rt_start <= rt_center <= rt_end):
                issues.append(f"{compound_name}: RT_min should be between RT_start_min and RT_end_min")

            # Check adducts
            adducts = self.get_compound_adducts(compound_name)
            if not adducts:
                issues.append(f"{compound_name}: No valid adducts found")

            # Try to calculate m/z for each adduct
            for adduct in adducts:
                mz = self.calculate_compound_mz(compound_name, adduct)
                if mz is None:
                    issues.append(f"{compound_name}: Cannot calculate m/z for adduct {adduct}")

        return issues
