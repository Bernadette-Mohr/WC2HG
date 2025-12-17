import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Union


class CVDataManager:
    """
    Manages storage and retrieval of CV data with trajectory relationships.
    For different molecular systems:
    dna_manager = create_dna_cv_manager()
    protein_manager = create_protein_cv_manager()
    custom_manager = CVDataManager(['distance', 'angle', 'dihedral'])
    """



    def __init__(self, cv_names: Optional[list[str]] = None):
        """
        Initialize CVDataManager.

        Args:
            cv_names: List of CV names. Defaults to standard DNA CV names.
        """
        self.data = {}
        self.cv_names = cv_names or ['dHG', 'dWC', 'lambda', 'theta', 'phi']

    def add_trajectory_data(self, mcc: int, cv_results_list: list) -> None:
        """
        Add CV data for one trajectory.

        Args:
            mcc: MC cycle number (trajectory identifier)
            cv_results_list: List containing [cv_tuples_list, weight] where
                           cv_tuples_list is list of CV value tuples per frame
        """
        cv_arrays = {}
        weight, cv_tuples_list = None, None

        if cv_results_list:
            cv_tuples_list = cv_results_list[0]
            weight = cv_results_list[1]
            n_cvs = len(cv_results_list[0][0])

            for idx, cv_name in enumerate(self.cv_names[:n_cvs]):
                cv_arrays[cv_name] = [frame_cvs[idx] for frame_cvs in cv_tuples_list]

        self.data[mcc] = {
            'cv_arrays': cv_arrays,
            'weight': weight,
            'n_frames': len(cv_tuples_list) if cv_tuples_list else 0
        }

    def save_to_dataframe(self, filepath: Union[str, Path]) -> None:
        """Save CV data to parquet format."""
        rows = []
        for mcc, traj_data in self.data.items():
            row = {
                'MCC': mcc,
                'weight': traj_data['weight'],
                'n_frames': traj_data['n_frames']
            }
            # Add CV arrays as columns
            for cv_name, cv_array in traj_data['cv_arrays'].items():
                row[f'{cv_name}_array'] = np.array(cv_array)
            rows.append(row)

        if rows:  # Only create DataFrame if we have data
            df = pd.DataFrame(rows)
            df.to_parquet(filepath, compression='gzip', index=False)
        else:
            print("Warning: No data to save to DataFrame")

    def save_to_pickle(self, filepath: Union[str, Path]) -> None:
        """Save nested structure to pickle format."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(self, filepath: Union[str, Path]) -> None:
        """Load nested structure from pickle format."""
        with open(filepath, 'rb') as f:
            self.data = pickle.load(f)

    def load_from_parquet(self, filepath: Union[str, Path]) -> None:
        """Load CV data from parquet file and convert to nested structure."""
        df = pd.read_parquet(filepath)
        self._dataframe_to_nested_dict(df)

    def _dataframe_to_nested_dict(self, df: pd.DataFrame) -> None:
        """Convert DataFrame back to nested dictionary structure."""
        self.data = {}

        # Handle different DataFrame formats
        if 'MCC' in df.columns:
            # Format with MCC column (aggregated format)
            for _, row in df.iterrows():
                mcc = row['MCC']
                cv_arrays = {}

                # Extract CV arrays from columns ending with '_array'
                for col in df.columns:
                    if col.endswith('_array'):
                        cv_name = col.replace('_array', '')
                        if cv_name in self.cv_names:
                            cv_arrays[cv_name] = list(row[col])

                self.data[mcc] = {
                    'cv_arrays': cv_arrays,
                    'weight': row.get('weight', 1.0),
                    'n_frames': row.get('n_frames', len(cv_arrays.get(self.cv_names[0], [])))
                }
        else:
            # Format with per-frame rows
            for mcc, group in df.groupby('MCC' if 'MCC' in df.columns else df.index):
                weight = group['weight'].iloc[0] if 'weight' in group.columns else 1.0
                cv_arrays = {}

                for cv_name in self.cv_names:
                    if cv_name in group.columns:
                        cv_arrays[cv_name] = group[cv_name].tolist()

                self.data[mcc] = {
                    'cv_arrays': cv_arrays,
                    'weight': weight,
                    'n_frames': len(group)
                }

    def load_previous_cvs(self,
                          filepath: Optional[Union[str, Path]],
                          dir_path: Optional[Union[str, Path]] = None,
                          expected_mc_cycle: Optional[int] = None) -> tuple[Optional[int], bool]:
        """
        Load existing CV data from file for resuming calculations.

        Args:
            filepath: Path to existing CV file (pickle or parquet)
            dir_path: Directory to search if filepath is relative
            expected_mc_cycle: Expected last MC cycle number for validation

        Returns:
            Tuple of (old_cycle, resumed) where:
            - old_cycle: Last MC cycle number found (None if error)
            - resumed: Whether resuming from existing data
        """
        old_cycle = 0
        resumed = False

        if not filepath:
            print('No previous CVs file provided. Starting fresh calculation...')
            return old_cycle, resumed

        filepath = Path(filepath)

        # Try to find file in dir_path if not found directly
        if not filepath.is_file() and dir_path:
            filepath = Path(dir_path) / filepath
            if not filepath.is_file():
                print(f'CV file {filepath} does not exist. Starting fresh calculation...')
                return old_cycle, resumed

        if not filepath.is_file():
            print(f'CV file {filepath} does not exist. Starting fresh calculation...')
            return old_cycle, resumed

        print(f'Loading previous CVs from {filepath}. Resuming calculation...')

        try:
            # Determine file format and load accordingly
            if filepath.suffix == '.pkl':
                self.load_from_pickle(filepath)
            elif filepath.suffix == '.parquet' or '.parquet' in filepath.suffixes:
                self.load_from_parquet(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")

            # Process loaded data
            if self.data:
                old_cycle = max(self.data.keys())

                if expected_mc_cycle is not None and old_cycle != expected_mc_cycle:
                    print(f'MC cycle mismatch: expected {expected_mc_cycle}, found {old_cycle}')
                    return None, False

                resumed = True
                print(f'Loaded {len(self.data)} trajectories, resuming from cycle {old_cycle}')
            else:
                print('Warning: Loaded file contains no CV data.')

        except Exception as e:
            print(f'Error loading CV file {filepath}: {e}')
            return None, False

        return old_cycle, resumed

    def get_trajectory_cvs(self, mcc: int) -> dict[str, list]:
        """Get all CV arrays for a specific trajectory."""
        if mcc not in self.data:
            raise KeyError(f"Trajectory {mcc} not found")
        return self.data[mcc]['cv_arrays']

    def get_cv_for_all_trajectories(self, cv_name: str) -> dict[int, list]:
        """Get a specific CV array for all trajectories."""
        if cv_name not in self.cv_names:
            print(f"Warning: {cv_name} not in expected CV names: {self.cv_names}")

        return {mcc: traj_data['cv_arrays'].get(cv_name, [])
                for mcc, traj_data in self.data.items()}

    def get_summary(self) -> dict:
        """Get summary statistics of loaded data."""
        if not self.data:
            return {"total_trajectories": 0, "cv_names": self.cv_names}

        total_frames = sum(traj['n_frames'] for traj in self.data.values())
        mc_cycles = list(self.data.keys())

        # Find which CVs actually have data
        available_cvs = set()
        for traj_data in self.data.values():
            available_cvs.update(traj_data['cv_arrays'].keys())

        return {
            "total_trajectories": len(self.data),
            "total_frames": total_frames,
            "mc_cycle_range": (min(mc_cycles), max(mc_cycles)),
            "available_cvs": sorted(available_cvs),
            "expected_cvs": self.cv_names
        }

    def clear_data(self) -> None:
        """Clear all stored CV data."""
        self.data = {}

    def __len__(self) -> int:
        """Return number of stored trajectories."""
        return len(self.data)

    def __contains__(self, mcc: int) -> bool:
        """Check if trajectory with given MC cycle exists."""
        return mcc in self.data


# Factory function for common use cases
def create_dna_cv_manager() -> CVDataManager:
    """Create CVDataManager with standard DNA CV names."""
    return CVDataManager(['dHG', 'dWC', 'lambda', 'theta', 'phi'])

# TODO: Extend for common system specific CV sets if needed
# def create_protein_cv_manager() -> CVDataManager:
#     """Create CVDataManager with common protein CV names."""
#     return CVDataManager(['phi', 'psi', 'chi1', 'rmsd', 'rg'])


# Utility functions
def merge_cv_data(managers: list[CVDataManager]) -> CVDataManager:
    """Merge data from multiple CVDataManager instances."""
    if not managers:
        return CVDataManager()

    # Use CV names from first manager
    merged = CVDataManager(managers[0].cv_names)

    for manager in managers:
        # Check for MC cycle conflicts
        conflicts = set(merged.data.keys()) & set(manager.data.keys())
        if conflicts:
            print(f"Warning: MC cycle conflicts found: {conflicts}")

        merged.data.update(manager.data)

    return merged