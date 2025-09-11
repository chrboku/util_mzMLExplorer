"""
File manager for handling mzML files and their metadata
"""
import os
import pandas as pd
import pymzml
from typing import Dict, List, Optional, Tuple
import numpy as np
from natsort import natsorted, index_natsorted
import psutil
import gc
from .utils import generate_color_palette


class FileManager:
    """Manages mzML files and their associated metadata"""
    
    def __init__(self):
        self.files_data = pd.DataFrame()
        self.group_colors = {}
        self.mzml_readers = {}  # Cache for mzML readers
        self.keep_in_memory = False  # Whether to keep all data in memory
        self.cached_data = {}  # Cache for raw mzML data when keep_in_memory is True
    
    def set_memory_mode(self, keep_in_memory: bool, auto_load: bool = True):
        """
        Set whether to keep all mzML data in memory.
        
        Args:
            keep_in_memory: If True, load and cache all mzML data in memory
            auto_load: If True, automatically load files to memory when enabled
        """
        if self.keep_in_memory == keep_in_memory:
            return  # No change needed
            
        self.keep_in_memory = keep_in_memory
        
        if keep_in_memory and auto_load:
            # Load all files into memory automatically (for initial settings load)
            self._load_all_files_to_memory()
        elif not keep_in_memory:
            # Clear memory cache and revert to file-based reading
            self._clear_memory_cache()
    
    def _load_all_files_to_memory(self):
        """Load all mzML files into memory cache (for use without progress dialog)"""
        if self.files_data.empty:
            return
            
        print("Loading all mzML files into memory...")
        self.cached_data = {}
        
        for _, row in self.files_data.iterrows():
            filepath = row['Filepath']
            try:
                print(f"Loading {os.path.basename(filepath)}...")
                # Read all spectra into memory
                reader = pymzml.run.Reader(filepath)
                ms1_spectra_data = []
                ms2_spectra_data = []
                
                for spectrum in reader:
                    if spectrum.ms_level == 1:  # MS1 spectra
                        spectrum_data = {
                            'scan_time': spectrum.scan_time_in_minutes(),
                            'mz': spectrum.mz,
                            'intensity': spectrum.i,
                            'polarity': self._get_spectrum_polarity(spectrum)
                        }
                        ms1_spectra_data.append(spectrum_data)
                    elif spectrum.ms_level == 2:  # MS2/MSMS spectra
                        try:
                            # Get precursor information
                            precursor_mz = None
                            precursor_intensity = 0
                            if hasattr(spectrum, 'selected_precursors') and spectrum.selected_precursors:
                                precursor_info = spectrum.selected_precursors[0]
                                if 'mz' in precursor_info:
                                    precursor_mz = float(precursor_info['mz'])
                                if 'intensity' in precursor_info:
                                    try:
                                        precursor_intensity = float(precursor_info['intensity'])
                                    except:
                                        precursor_intensity = 0
                            
                            # Alternative method to get precursor m/z
                            if precursor_mz is None and hasattr(spectrum, 'element'):
                                for elem in spectrum.element.iter():
                                    if elem.tag.endswith('precursorList'):
                                        for precursor in elem:
                                            if precursor.tag.endswith('precursor'):
                                                for selected_ion in precursor:
                                                    if selected_ion.tag.endswith('selectedIonList'):
                                                        for ion in selected_ion:
                                                            for cv_param in ion:
                                                                if (cv_param.tag.endswith('cvParam') and 
                                                                    cv_param.get('accession') == 'MS:1000744'):  # selected ion m/z
                                                                    precursor_mz = float(cv_param.get('value'))
                                                                elif (cv_param.tag.endswith('cvParam') and 
                                                                      cv_param.get('accession') == 'MS:1000042'):  # peak intensity
                                                                    try:
                                                                        precursor_intensity = float(cv_param.get('value'))
                                                                    except:
                                                                        precursor_intensity = 0
                            
                            spectrum_data = {
                                'scan_time': spectrum.scan_time_in_minutes(),
                                'mz': spectrum.mz,
                                'intensity': spectrum.i,
                                'polarity': self._get_spectrum_polarity(spectrum),
                                'precursor_mz': precursor_mz,
                                'precursor_intensity': precursor_intensity,
                                'scan_id': spectrum.ID if hasattr(spectrum, 'ID') else f"RT_{spectrum.scan_time_in_minutes():.2f}"
                            }
                            ms2_spectra_data.append(spectrum_data)
                        except Exception as e:
                            print(f"Error processing MS2 spectrum: {e}")
                            continue
                
                # Store both MS1 and MS2 data
                self.cached_data[filepath] = {
                    'ms1': ms1_spectra_data,
                    'ms2': ms2_spectra_data
                }
                reader.close()
                
            except Exception as e:
                print(f"Error loading {filepath} to memory: {str(e)}")
                
        print(f"Loaded {len(self.cached_data)} files into memory.")
    
    def _clear_memory_cache(self):
        """Clear the memory cache to free up RAM"""
        print("Clearing memory cache...")
        self.cached_data = {}
        gc.collect()  # Force garbage collection
    
    def get_memory_usage(self) -> dict:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
                'percent': memory_percent,
                'cached_files': len(self.cached_data),
                'keep_in_memory': self.keep_in_memory
            }
        except Exception as e:
            return {
                'rss_mb': 0,
                'vms_mb': 0,
                'percent': 0,
                'cached_files': len(self.cached_data),
                'keep_in_memory': self.keep_in_memory,
                'error': str(e)
            }

    def load_files(self, files_df: pd.DataFrame):
        """
        Load file list with metadata. New files are added to existing ones.
        
        Args:
            files_df: DataFrame with columns Filepath and optional group, color, etc.
        """
        # Validate file paths
        valid_files = []
        existing_paths = set(self.files_data['Filepath'].tolist()) if not self.files_data.empty else set()
        
        for idx, row in files_df.iterrows():
            filepath = row['Filepath']
            
            # Skip if file already exists in the list
            if filepath in existing_paths:
                print(f"Info: File already loaded, skipping: {os.path.basename(filepath)}")
                continue
                
            if os.path.exists(filepath) and filepath.lower().endswith('.mzml'):
                valid_files.append(row.to_dict())
            else:
                print(f"Warning: File not found or not mzML: {filepath}")
        
        if not valid_files:
            if not self.files_data.empty:
                print("No new valid files to add.")
                return
            else:
                raise ValueError("No valid mzML files found!")
        
        # Create DataFrame for new files
        new_files_df = pd.DataFrame(valid_files)
        
        # Add filename column
        new_files_df['filename'] = new_files_df['Filepath'].apply(
            lambda x: os.path.basename(x)
        )
        
        # Combine with existing data
        if self.files_data.empty:
            self.files_data = new_files_df
        else:
            self.files_data = pd.concat([self.files_data, new_files_df], ignore_index=True)
        
        # Assign colors to groups (this will update colors for new groups)
        self._assign_group_colors()
        
        # Sort the data
        self._sort_files_data()
        
        print(f"Added {len(valid_files)} new files. Total files: {len(self.files_data)}")
        
        # Note: Memory loading is now handled by the main window with progress dialog
        # The main window will call load_files_to_memory_with_progress() if needed
    
    def _sort_files_data(self):
        """Sort files data by group first, then by filename using natural sort"""
        if self.files_data.empty:
            return
        
        # Ensure group column exists
        if 'group' not in self.files_data.columns:
            self.files_data['group'] = 'default'
        
        # Create sorting keys
        groups = self.files_data['group'].tolist()
        filenames = self.files_data['filename'].tolist()
        
        # Natural sort by group first, then by filename
        sort_indices = sorted(
            range(len(self.files_data)),
            key=lambda i: (groups[i], filenames[i]),
            # Use natsort for the filename part
        )
        
        # Apply natural sort to filenames within each group
        grouped_indices = []
        current_group = None
        group_start = 0
        
        for i, idx in enumerate(sort_indices):
            if groups[idx] != current_group:
                if current_group is not None:
                    # Sort the previous group's filenames naturally
                    group_slice = sort_indices[group_start:i]
                    group_filenames = [filenames[idx] for idx in group_slice]
                    nat_sort_indices = index_natsorted(group_filenames)
                    sorted_group = [group_slice[j] for j in nat_sort_indices]
                    grouped_indices.extend(sorted_group)
                
                current_group = groups[idx]
                group_start = i
        
        # Handle the last group
        if current_group is not None:
            group_slice = sort_indices[group_start:]
            group_filenames = [filenames[idx] for idx in group_slice]
            nat_sort_indices = index_natsorted(group_filenames)
            sorted_group = [group_slice[j] for j in nat_sort_indices]
            grouped_indices.extend(sorted_group)
        
        # Reorder the dataframe
        self.files_data = self.files_data.iloc[grouped_indices].reset_index(drop=True)
    
    def _assign_group_colors(self):
        """Assign colors to groups based on the group column or color column"""
        self.group_colors = {}
        
        if 'group' not in self.files_data.columns:
            # If no group column, create a default group
            self.files_data['group'] = 'default'
        
        # Get unique groups
        unique_groups = self.files_data['group'].unique()
        
        # Check if color column exists
        if 'color' in self.files_data.columns:
            # Map existing colors to groups
            for group in unique_groups:
                group_files = self.files_data[self.files_data['group'] == group]
                group_colors = group_files['color'].dropna().unique()
                if len(group_colors) > 0:
                    self.group_colors[group] = group_colors[0]
        
        # Generate colors for groups that don't have assigned colors
        unassigned_groups = [g for g in unique_groups if g not in self.group_colors]
        if unassigned_groups:
            colors = generate_color_palette(len(unassigned_groups))
            for group, color in zip(unassigned_groups, colors):
                self.group_colors[group] = color
    
    def get_files_data(self) -> pd.DataFrame:
        """Get the loaded files data"""
        return self.files_data.copy()
    
    def get_files_display_data(self) -> pd.DataFrame:
        """Get files data formatted for display in the table"""
        if self.files_data.empty:
            return pd.DataFrame()
        
        display_data = self.files_data.copy()
        
        # Reorder columns: filename first, then other columns, filepath last
        columns = list(display_data.columns)
        
        # Remove filename and Filepath from their current positions
        if 'filename' in columns:
            columns.remove('filename')
        if 'Filepath' in columns:
            columns.remove('Filepath')
        
        # Create new column order
        new_columns = ['filename'] + [col for col in columns if col != 'filename'] + ['Filepath']
        
        # Filter to only include columns that exist
        existing_columns = [col for col in new_columns if col in display_data.columns]
        
        return display_data[existing_columns]
    
    def get_group_color(self, group: str) -> Optional[str]:
        """Get the color assigned to a group"""
        return self.group_colors.get(group)
    
    def get_groups(self) -> List[str]:
        """Get list of unique groups"""
        if 'group' in self.files_data.columns:
            return sorted(self.files_data['group'].unique())
        return []
    
    def get_files_by_group(self, group: str) -> pd.DataFrame:
        """Get files belonging to a specific group"""
        if 'group' in self.files_data.columns:
            return self.files_data[self.files_data['group'] == group]
        return pd.DataFrame()
    
    def get_mzml_reader(self, filepath: str) -> pymzml.run.Reader:
        """
        Get or create an mzML reader for the specified file.
        
        Args:
            filepath: Path to the mzML file
            
        Returns:
            pymzml Reader object
        """
        if filepath not in self.mzml_readers:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"mzML file not found: {filepath}")
            
            try:
                self.mzml_readers[filepath] = pymzml.run.Reader(filepath)
            except Exception as e:
                raise RuntimeError(f"Failed to open mzML file {filepath}: {str(e)}")
        
        return self.mzml_readers[filepath]
    
    def _get_spectrum_polarity(self, spectrum):
        """
        Get the polarity of a spectrum from pymzml.
        
        Args:
            spectrum: pymzml spectrum object
            
        Returns:
            str: '+' for positive, '-' for negative, None if unknown
        """
        try:
            # Method 1: Check spectrum attributes directly
            if hasattr(spectrum, 'polarity'):
                polarity = spectrum.polarity
                if polarity in ['+', 'positive', 'pos']:
                    return '+'
                elif polarity in ['-', 'negative', 'neg']:
                    return '-'
            
            # Method 2: Check CV parameters for polarity
            if hasattr(spectrum, 'get_element_by_path'):
                # Try to find polarity in CV params
                try:
                    scan_list = spectrum.get_element_by_path(['scanList'])
                    if scan_list:
                        for scan in scan_list:
                            if hasattr(scan, 'get_element_by_path'):
                                cv_params = scan.get_element_by_path(['cvParam'])
                                if cv_params:
                                    for cv_param in cv_params:
                                        accession = cv_param.get('accession', '')
                                        if accession == 'MS:1000130':  # positive scan
                                            return '+'
                                        elif accession == 'MS:1000129':  # negative scan
                                            return '-'
                except:
                    pass
            
            # Method 3: Check spectrum element directly for CV params
            if hasattr(spectrum, 'element'):
                try:
                    # Look for polarity CV params in the spectrum element
                    for elem in spectrum.element.iter():
                        if elem.tag.endswith('cvParam'):
                            accession = elem.get('accession')
                            if accession == 'MS:1000130':  # positive scan
                                return '+'
                            elif accession == 'MS:1000129':  # negative scan
                                return '-'
                except:
                    pass
            
            # Method 4: Try to access internal spectrum data
            if hasattr(spectrum, '_spectrum_dict'):
                spectrum_dict = spectrum._spectrum_dict
                if 'polarity' in spectrum_dict:
                    polarity = spectrum_dict['polarity']
                    if polarity in ['+', 'positive', 'pos']:
                        return '+'
                    elif polarity in ['-', 'negative', 'neg']:
                        return '-'
            
            return None  # Unknown polarity
            
        except Exception as e:
            # If any error occurs, return None (unknown polarity)
            return None
    
    def extract_eic(self, filepath: str, target_mz: float, mz_tolerance: float = 0.01, 
                   rt_start: Optional[float] = None, rt_end: Optional[float] = None,
                   calculation_method: str = "Sum of all signals", polarity: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract extracted ion chromatogram (EIC) from an mzML file.
        
        Args:
            filepath: Path to the mzML file
            target_mz: Target m/z value
            mz_tolerance: m/z tolerance in Da
            rt_start: Start retention time in minutes (optional)
            rt_end: End retention time in minutes (optional)
            calculation_method: "Sum of all signals" or "Most intensive signal"
            polarity: Ion polarity ('+', '-', or None for both)
            
        Returns:
            Tuple of (retention_times, intensities) arrays
        """
        try:
            rt_list = []
            intensity_list = []
            
            # Use cached data if available
            if self.keep_in_memory and filepath in self.cached_data:
                cached_file_data = self.cached_data[filepath]
                
                # Handle both old format (list) and new format (dict with ms1/ms2)
                if isinstance(cached_file_data, dict) and 'ms1' in cached_file_data:
                    spectra_data = cached_file_data['ms1']
                else:
                    # Old format - assume it's MS1 data
                    spectra_data = cached_file_data
                
                for spectrum_data in spectra_data:
                    # Check polarity if specified
                    if polarity is not None and spectrum_data['polarity'] is not None:
                        if polarity != spectrum_data['polarity']:
                            continue
                    
                    rt = spectrum_data['scan_time']
                    
                    # Filter by retention time if specified
                    if rt_start is not None and rt < rt_start:
                        continue
                    if rt_end is not None and rt > rt_end:
                        continue
                    
                    # Find peaks within m/z tolerance
                    mz_array = spectrum_data['mz']
                    intensity_array = spectrum_data['intensity']
                    
                    if len(mz_array) > 0:
                        # Find indices of peaks within tolerance
                        mz_mask = np.abs(mz_array - target_mz) <= mz_tolerance
                        
                        if np.any(mz_mask):
                            if calculation_method == "Sum of all signals":
                                # Sum intensities of all peaks within tolerance
                                total_intensity = np.sum(intensity_array[mz_mask])
                            else:  # "Most intensive signal"
                                # Take the maximum intensity within tolerance
                                total_intensity = np.max(intensity_array[mz_mask])
                            
                            rt_list.append(rt)
                            intensity_list.append(total_intensity)
                        else:
                            # No peaks found, add zero intensity
                            rt_list.append(rt)
                            intensity_list.append(0.0)
                
                return np.array(rt_list), np.array(intensity_list)
            
            else:
                # Use file-based reading (original method)
                reader = self.get_mzml_reader(filepath)
                
                # Debug: count spectra by polarity
                polarity_debug = {'total': 0, 'used': 0, 'filtered': 0}
                
                for spectrum in reader:
                    if spectrum.ms_level == 1:  # Only MS1 spectra
                        polarity_debug['total'] += 1
                        
                        # Check polarity if specified
                        spectrum_used = True
                        if polarity is not None:
                            spectrum_polarity = self._get_spectrum_polarity(spectrum)
                            
                            # Skip if polarity doesn't match
                            if spectrum_polarity is not None:
                                if polarity == '+' and spectrum_polarity != '+':
                                    spectrum_used = False
                                elif polarity == '-' and spectrum_polarity != '-':
                                    spectrum_used = False
                            # If spectrum_polarity is None, we include the spectrum
                            # (better to include uncertain spectra than miss data)
                        
                        if not spectrum_used:
                            polarity_debug['filtered'] += 1
                            continue
                        
                        polarity_debug['used'] += 1
                        
                        rt = spectrum.scan_time_in_minutes()
                        
                        # Filter by retention time if specified
                        if rt_start is not None and rt < rt_start:
                            continue
                        if rt_end is not None and rt > rt_end:
                            continue
                        
                        # Find peaks within m/z tolerance
                        mz_array = spectrum.mz
                        intensity_array = spectrum.i
                        
                        if len(mz_array) > 0:
                            # Find indices of peaks within tolerance
                            mz_mask = np.abs(mz_array - target_mz) <= mz_tolerance
                            
                            if np.any(mz_mask):
                                if calculation_method == "Sum of all signals":
                                    # Sum intensities of all peaks within tolerance
                                    total_intensity = np.sum(intensity_array[mz_mask])
                                else:  # "Most intensive signal"
                                    # Take the maximum intensity within tolerance
                                    total_intensity = np.max(intensity_array[mz_mask])
                                
                                rt_list.append(rt)
                                intensity_list.append(total_intensity)
                            else:
                                # No peaks found, add zero intensity
                                rt_list.append(rt)
                                intensity_list.append(0.0)
                
                return np.array(rt_list), np.array(intensity_list)
            
        except Exception as e:
            print(f"Error extracting EIC from {filepath}: {str(e)}")
            return np.array([]), np.array([])
    
    def extract_eic_all_files(self, target_mz: float, mz_tolerance: float = 0.01,
                            rt_start: Optional[float] = None, rt_end: Optional[float] = None,
                            calculation_method: str = "Sum of all signals", polarity: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract EIC from all loaded files.
        
        Args:
            target_mz: Target m/z value
            mz_tolerance: m/z tolerance in Da
            rt_start: Start retention time in minutes (optional)
            rt_end: End retention time in minutes (optional)
            calculation_method: "Sum of all signals" or "Most intensive signal"
            polarity: Ion polarity ('+', '-', or None for both)
            
        Returns:
            Dictionary mapping filepath to (retention_times, intensities) tuples
        """
        eic_data = {}
        
        for _, row in self.files_data.iterrows():
            filepath = row['Filepath']
            try:
                rt, intensity = self.extract_eic(filepath, target_mz, mz_tolerance, rt_start, rt_end, 
                                               calculation_method, polarity)
                eic_data[filepath] = (rt, intensity)
            except Exception as e:
                print(f"Failed to extract EIC from {filepath}: {str(e)}")
                eic_data[filepath] = (np.array([]), np.array([]))
        
        return eic_data
    
    def get_file_info(self, filepath: str) -> Dict:
        """Get information about a specific file"""
        file_row = self.files_data[self.files_data['Filepath'] == filepath]
        if not file_row.empty:
            return file_row.iloc[0].to_dict()
        return {}
    
    def close_readers(self):
        """Close all mzML readers to free memory"""
        for reader in self.mzml_readers.values():
            try:
                reader.close()
            except:
                pass
        self.mzml_readers.clear()
        
        # Also clear memory cache if not in memory mode
        if not self.keep_in_memory:
            self._clear_memory_cache()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close_readers()
