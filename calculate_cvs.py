import argparse
from collections import deque
import mdtraj as md
import numpy as np
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage
import pandas as pd
from pathlib import Path
import pickle
import time
from tqdm import tqdm
from pathsampling_utilities import PathsamplingUtilities
from collective_variable import CollectiveVariable

paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True


class DataLoader:
    """
    Load and manage OPS trajectory data.
    """
    def __init__(self):
        self.trajectories = list()
        self.new_storage = None

    def load_data(self, root_path, dir_name, new_name) -> tuple:
        """
        Load OPS trajectories and weights from databases and dictionaries.

        Args:
            root_path: Path to the directory containing the TPS run folders.
            dir_name: Basename of the folders containing parts of a TPS run.
            new_name: Name of the .db file that will contain all accepted paths.

        Returns:
            Tuple of (trajectories, storage, cvs_list, engine)
        """

        storage_dirs = sorted(root_path.glob(f'{dir_name}*'))
        for dir_ in storage_dirs:
            self.trajectories.extend(sorted(dir_.glob('*.db')))

        db_name = f'{new_name}.db'
        db_path = root_path / db_name
        if not db_path.is_file():
            self.new_storage = Storage(filename=f'{root_path}/{db_name}', mode='w')
            first_storage = Storage(filename=str(self.trajectories[0]), mode='r')
            cvs_list = [first_storage.cvs['d_WC'], first_storage.cvs['d_HG']]
            engine = first_storage.engines[0]
            for cv in tqdm(first_storage.storable_functions, desc='Preloading cache'):
                cv.preload_cache()
            for obj in tqdm(first_storage.simulation_objects, desc='Copying simulation objects'):
                self.new_storage.save(obj)
            first_storage.close()
        else:
            print(f'Resuming: processing TPS trajectories')
            self.new_storage = Storage(filename=f'{root_path}/{db_name}', mode='a')
            cvs_list = [self.new_storage.cvs['d_WC'], self.new_storage.cvs['d_HG']]
            engine = self.new_storage.engines[0]

        return self.trajectories, self.new_storage, cvs_list, engine


class CVDataManager:
    """Manages storage and retrieval of CV data with trajectory relationships."""

    def __init__(self):
        self.data = {}
        self.cv_names = ['dHG', 'dWC', 'lambda', 'theta', 'phi']

    def add_trajectory_data(self, mcc, cv_results_list) -> None:
        """
        Add CV data for one trajectory.

        Args:
            :param mcc: MC cycle number (trajectory identifier)
            :param cv_results_list: List of list of tuples, each tuple contains CV values for one frame of the trajectory,
            and corresponding weight
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
            'n_frames': len(cv_tuples_list)
        }

    def save_to_dataframe(self, filepath) -> None:
        """Save to parquet format."""
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

        df = pd.DataFrame(rows)
        df.to_parquet(filepath, index=False)

    def save_to_pickle(self, filepath) -> None:
        """Save nested structure to pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(self, filepath) -> None:
        """Load nested structure from pickle."""
        with open(filepath, 'rb') as f:
            self.data = pickle.load(f)

    def _load_from_parquet(self, filepath) -> None:
        """Load CV data from parquet file and convert to nested structure."""
        df = pd.read_parquet(filepath)
        self._dataframe_to_nested_dict(df)

    def _dataframe_to_nested_dict(self, df) -> None:
        """
        Convert DataFrame back to nested dictionary structure.

        Expected DataFrame columns:
        - MCC: MC cycle number (trajectory identifier)
        - frame_idx or snapshot_idx: Frame number within trajectory
        - weight: Weight for the trajectory
        - CV columns: dHG, dWC, lambda, theta, phi, etc.
        """
        self.data = {}  # Reset the data dictionary

        # Group by MCC (each group = one trajectory)
        for mcc, group in df.groupby('MCC'):
            # Sort frames in correct order
            if 'frame_idx' in group.columns:
                group = group.sort_values('frame_idx')
            elif 'snapshot_idx' in group.columns:
                group = group.sort_values('snapshot_idx')

            weight = group['weight'].iloc[0]
            cv_arrays = {}

            for cv_name in self.cv_names:
                if cv_name in group.columns:
                    cv_arrays[cv_name] = group[cv_name].tolist()

            self.data[mcc] = {
                'cv_arrays': cv_arrays,  # Dict of CV_name -> list of values
                'weight': weight,  # Single weight for whole trajectory
                'n_frames': len(group)  # Number of frames in trajectory
            }

    def load_previous_cvs(self, filepath, dir_path, expected_mc_cycle=None) -> tuple:
        """Load existing CV data from file for resuming calculations.
        Args: filepath: Path to existing CV file (pickle or parquet).
              expected_mc_cycle: Expected last MC cycle number (for validation).
        Returns: Tuple of (old_cycle, resumed) where old_cycle is the last MC cycle
                 number found in the file, and resumed is a boolean indicating
                 whether we are resuming from existing data.
            """
        old_cycle = 0  # Default: start from cycle 0
        resumed = False  # Default: not resuming from existing data

        if not filepath:
            print('No previous CVs file provided. Starting fresh calculation...')
            return old_cycle, resumed

        filepath = Path(filepath)  # Convert to Path object for easier handling
        if not filepath.is_file() and dir_path:
            filepath = Path(dir_path) / filepath
            if not filepath.is_file():
                print(f'CV file {filepath} does not exist. Starting fresh calculation...')
                return old_cycle, resumed

        print(f'Loading previous CVs from {filepath}. Resuming calculation...')

        try:
            if str(filepath).endswith('.pkl'):
                self.load_from_pickle(filepath)
            elif str(filepath).endswith('.parquet'):
                self._load_from_parquet(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")

            # Process the loaded data
            if self.data:
                old_cycle = max(self.data.keys())

                if expected_mc_cycle is not None and old_cycle != expected_mc_cycle:
                    print(f'MC cycle mismatch: expected {expected_mc_cycle}, found {old_cycle}')
                    return None, None

                resumed = True

            else:
                print('Warning: Loaded file contains no CV data.')

        except Exception as e:
            print(f'Error loading CV file {filepath}: {e}')
            return None, None

        return old_cycle, resumed


class CVCalculator:
    def __init__(self, directory, base_name, rolling_residues, backbone_residues,
                 angle, identifier, cvs_name=None, mc_cycle=None, max_steps=None, wall_time=None):
        self.root_path = Path(directory)
        self.dir_name = base_name
        self.id_str = identifier
        self.rolling_residues = rolling_residues
        self.backbone_residues = backbone_residues
        self.angle = angle
        self.cvs_name = cvs_name
        self.mc_cycle = mc_cycle
        self.n_steps = max_steps
        self.wall_time = wall_time
        self.start_time = time.time()

        self.trajectory_loader = DataLoader()
        self.utils = PathsamplingUtilities()
        self.cv_manager = CVDataManager()
        self.cvs = list()

    def _save_results(self, output_dir) -> None:
        """Save results in multiple formats."""
        output_dir = Path(output_dir)

        pickle_path = output_dir / f'{self.id_str}_cv_data.pkl'
        self.cv_manager.save_to_pickle(pickle_path)

        dataframe_path = output_dir / f'{self.id_str}_cv_data.parquet'
        self.cv_manager.save_to_dataframe(dataframe_path)

        print(f"Saved CV data:")
        print(f"  - Nested structure: {pickle_path}")
        print(f"  - DataFrame: {dataframe_path}")

    def _check_time_and_step_limits(self, iteration_durations, old_cycle, fname) -> bool:
        """Check if we should stop due to time or step limits."""
        # Check step limits
        if self.n_steps and len(self.cv_manager.data) >= self.n_steps:
            print(f"Reached maximum steps limit: {self.n_steps}")
            print(f"Stopping after mcc {old_cycle} in {fname}.")
            return True

        # Check time limits
        if self.wall_time and iteration_durations:
            elapsed_time = time.time() - self.start_time
            remaining_time = self.wall_time - elapsed_time
            rolling_avg_duration = sum(iteration_durations) / len(iteration_durations)

            if rolling_avg_duration >= remaining_time:
                print(f"Approaching cutoff time. Exiting after mcc {old_cycle} in {fname}.")
                print(f'Estimated remaining time: {remaining_time:.2f} s')
                print(f'Average iteration duration: {rolling_avg_duration:.2f} s')
                return True

        return False

    def _calculate_and_store(self, trajectories, new_storage) -> None:
        """Main calculation loop with data storage."""
        # Copied from OpenPathSampling PathDensityHistogram(PathHistogram), l. 367
        def _add_ops_trajectory(trajectory, weight) -> list:
            cv_traj = [cv(trajectory) for cv in self.cvs]
            return [list(zip(*cv_traj)), weight]

        iteration_durations = deque(maxlen=10)  # Rolling window of last 10 iterations

        old_cycle, resumed = self.cv_manager.load_previous_cvs(self.cvs_name, self.root_path, self.mc_cycle)
        if old_cycle is None:  # Error occurred in loading previous CVs or generating empty data dictionary
            return

        # Process each trajectory file
        for file_idx, fname in enumerate(tqdm(trajectories,
                                              total=len(trajectories),
                                              desc='Processing trajectory files...')):
            print(f"Processing file: {fname}")
            iteration_start_time = time.time()

            try:
                storage = Storage(filename=str(fname), mode='r')
                start = 1 if not resumed and file_idx == 0 else 0
                len_db = len(storage.steps) if not resumed and file_idx == 0 else len(storage.steps) + 1
                steps = self.utils.wrapper(storage.steps, fname, start, len_db)

                for step in steps:
                    if resumed and step.mccycle <= old_cycle:
                        continue

                    if step.change.accepted:
                        new_cycle = step.mccycle
                        wt = new_cycle - old_cycle

                        # Calculate CVs and add to DataFrame
                        results_list = _add_ops_trajectory(step, wt)
                        self.cv_manager.add_trajectory_data(new_cycle, results_list)
                        new_storage.save(step)
                        old_cycle = new_cycle

                        # Periodically save progress
                        if len(self.cv_manager.data) % 100 == 0:
                            new_storage.sync_all()
                            pickle_path = self.root_path / f'{self.id_str}_cv_data.pkl'
                            self.cv_manager.save_to_pickle(pickle_path)

                    # Track iteration duration and update rolling average
                    iteration_duration = time.time() - iteration_start_time
                    iteration_durations.append(iteration_duration)

                    if self._check_time_and_step_limits(iteration_durations, old_cycle, fname):
                        new_storage.sync_all()
                        new_storage.close()
                        self._save_results(self.root_path)
                        return

                storage.close()

            except Exception as e:
                print(f"Error processing file {fname}: {e}")
                if 'storage' in locals():
                    storage.close()
                continue

        new_storage.sync_all()
        new_storage.close()
        self._save_results(self.root_path)


    def calculate_cvs(self) -> None:
        """Main method to calculate collective variables."""
        residA, residT = self.rolling_residues  # 6, 16

        try:
            (trajectories,
             storage,
             cvs_list,
             engine) = self.trajectory_loader.load_data(self.root_path, self.dir_name, f'{self.id_str}_accepted')
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        self.cvs.extend(cvs_list)

        if not self.angle:
            print('Using arccos for angle calculation (use -a flag for arctan2).')

        d_WC, d_HG = self.cvs[0], self.cvs[1]

        # Define atom selections
        o3_prime = '"O3\'"'
        o5_prime = '"O5\'"'
        angI_atoms = (
            f"(resid {residA - 1} and not type H) or (resid {residA + 1} and not type H) or (resid {residT - 1} "
            f"and not type H) or (resid {residT + 1} and not type H)")
        angII_atoms = f"(resid {residA - 1} and name {o3_prime}) or (resid {residA} and name P OP1 OP2 {o5_prime})"
        angIII_atoms = f"(resid {residA} and name  {o3_prime}) or (resid {residA + 1} and name P OP1 OP2 {o5_prime})"
        angIV_atoms = f"resid {residA} and not type H"

        # Create collective variables
        try:
            comI = paths.MDTrajFunctionCV("comI", md.compute_center_of_mass,
                                          topology=engine.topology, select=angI_atoms)
            comII = paths.MDTrajFunctionCV("comII", md.compute_center_of_mass,
                                           topology=engine.topology, select=angII_atoms)
            comIII = paths.MDTrajFunctionCV("comIII", md.compute_center_of_mass,
                                            topology=engine.topology, select=angIII_atoms)
            comIV = paths.MDTrajFunctionCV("comIV", md.compute_center_of_mass,
                                           topology=engine.topology, select=angIV_atoms)

            lambda_ = paths.CoordinateFunctionCV(
                name='lambda',
                f=CollectiveVariable.lambda_CV,
                d_WC_cv=d_WC,
                d_HG_cv=d_HG
            )

            theta_ = paths.CoordinateFunctionCV(
                name="theta",
                f=CollectiveVariable.base_opening_angle,
                comI_cv=comI,
                comII_cv=comII,
                comIII_cv=comIII,
                comIV_cv=comIV,
                angle_between_vectors_cv=CollectiveVariable.angle_between_vectors,
                angle=self.angle,
            )

            resid_bb_start = self.backbone_residues[0]  # 1
            resid_bb_end = self.backbone_residues[1]  # 13
            backbone_atoms = f'resid {resid_bb_start} {resid_bb_end} and name P'
            rollingbase_atoms = f'resid {residA} and name N7 N3 N1'

            phi_ = paths.CoordinateFunctionCV(
                name="phi",
                f=CollectiveVariable.base_rolling_angle,
                backbone_idx=engine.topology.mdtraj.select(backbone_atoms),
                rollingbase_idx=engine.topology.mdtraj.select(rollingbase_atoms),
                angle_between_vectors_cv=CollectiveVariable.angle_between_vectors,
                angle=self.angle,
            )

            self.cvs.extend([lambda_, theta_, phi_])

        except Exception as e:
            print(f"Error creating collective variables: {e}")
            return

        # Start calculation
        self._calculate_and_store(trajectories, storage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Calculate collective variables for OPS trajectories. Returns a parquet file with CV values and '
                    'weights for each trajectory.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output. Needs to contain OPS databases.')
    parser.add_argument('-bn', '--base_name', type=str, required=True,
                        help='Directory (base) name for directory/directories containing the raw TPS output files.')
    parser.add_argument('-rr', '--rolling_residues', type=int, nargs=2, required=True,
                        help='Residue indices of 1. the rolling base and 2. the other base of the pair.')
    parser.add_argument('-bb', '--backbone_residues', type=int, nargs=2, required=True,
                        help='Residue indices at 1. the 5\' end and 2. the 3\' end for backbone orientation.')
    parser.add_argument('-a', '--angle', type=bool, required=False,
                        help='Calculate angle as arccos (default) or arctan2 (if set).')
    parser.add_argument('-id', '--identifier', type=str, required=True,
                        help='Identifier for the output files.')
    parser.add_argument('-cvs', '--cvs_name', type=Path, required=False, default=None,
                        help='Existing CV file to continue from (parquet or pickle format).')
    parser.add_argument('-mcc', '--mc_cycle', type=int, required=False, default=None,
                        help='Last processed MC cycle number for resuming.')
    parser.add_argument('-max', '--max_steps', type=int, required=False, default=None,
                        help='Maximum number of steps to process.')
    parser.add_argument('-wt', '--wall_time', type=int, required=False, default=None,
                        help='Wall time limit in seconds.')

    args = parser.parse_args()
    args_dict = vars(args)

    try:
        cvs = CVCalculator(**args_dict)
        cvs.calculate_cvs()
    except Exception as e:
        print(f"Error during CV calculation: {e}")
        raise
