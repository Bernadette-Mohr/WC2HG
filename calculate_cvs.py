import argparse
from collections import deque
import mdtraj as md
import numpy as np
import openpathsampling as paths
import pandas as pd
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage
from pathlib import Path
from pathsampling_utilities import PathsamplingUtilities
from collective_variable import CollectiveVariable
import time
from tqdm import tqdm

paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True


class DataLoader:
    """
    Load and manage OPS trajectory data.
    """
    def __init__(self):
        self.trajectories = list()
        self.new_storage = None

    def load_data(self, dir_path, dir_name, new_name):
        """
        Load OPS trajectories and weights from databases and dictionaries.

        Args:
            dir_path: Path to the directory containing the TPS run folders.
            dir_name: Basename of the folders containing parts of a TPS run.
            new_name: Name of the .db file that will contain all accepted paths.

        Returns:
            Tuple of (trajectories, storage, cvs_list, engine)
        """

        storage_dirs = sorted(dir_path.glob(f'{dir_name}*'))
        for dir_idx, dir_ in enumerate(storage_dirs):
            self.trajectories.extend(sorted(dir_.glob('*.db')))

        db_name = f'{new_name}.db'
        db_path = dir_path / db_name

        if not db_path.is_file():
            self.new_storage = Storage(filename=f'{dir_path}/{db_name}', mode='w')
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
            self.new_storage = Storage(filename=f'{dir_path}/{db_name}', mode='a')
            cvs_list = [self.new_storage.cvs['d_WC'], self.new_storage.cvs['d_HG']]
            engine = self.new_storage.engines[0]

        return self.trajectories, self.new_storage, cvs_list, engine


class CVCalculator:
    def __init__(self,
                 directory,
                 identifier,
                 rolling_residues,
                 backbone_residues,
                 angle,
                 cvs_name,
                 mc_cycle,
                 max_steps,
                 wall_time):
        self.dir_path = Path(directory)
        self.id_str = identifier
        self.rolling_residues = rolling_residues
        self.backbone_residues = backbone_residues
        self.angle = angle
        self.cvs_name = cvs_name
        self.mc_cycle = mc_cycle
        self.n_steps = max_steps
        self.wall_time = wall_time
        self.start_time = time.time()
        self.loader = DataLoader()
        self.utils = PathsamplingUtilities()
        self.cvs_frame = None
        self.cvs = list()

    def _get_previous_cvs(self):
        """Load existing CV data or create new DataFrame."""
        old_cycle = 0
        resumed = False

        if not self.cvs_name:
            print('No previous CVs passed. Starting calculation...')
            cvs_frame = pd.DataFrame(columns=['MCC', 'dHG', 'dWC', 'lambda', 'theta', 'phi', 'weight'])
        else:
            print(f'Loading previous CVs from {self.cvs_name}. Resuming calculation...')

            # Handle both parquet and pickle files
            if str(self.cvs_name).endswith('.parquet'):
                cvs_frame = pd.read_parquet(self.cvs_name)
            elif str(self.cvs_name).endswith('.csv'):
                cvs_frame = pd.read_csv(self.cvs_name)
            else:
                raise ValueError(f"Unsupported file format: {self.cvs_name}")

            if self.mc_cycle and len(cvs_frame) > 0:
                last_mcc = cvs_frame['MCC'].iloc[-1]
                if last_mcc != self.mc_cycle:
                    print(
                        f'MC cycle number {self.mc_cycle} does not match the last entry {last_mcc} in the CVs file. Exiting...')
                    return None, None, None
                else:
                    old_cycle = last_mcc
                    resumed = True

        return cvs_frame, old_cycle, resumed

    def _save_progress(self, cvs_frame):
        """Save intermediate progress."""
        temp_path = self.dir_path / f'{self.id_str}_accepted_CVs_weights_temp.parquet'
        cvs_frame.to_parquet(temp_path)

    def _save_final_results(self, cvs_frame):
        """Save final results and clean up temporary files."""
        final_path = self.dir_path / f'{self.id_str}_accepted_CVs_weights.parquet'
        cvs_frame.to_parquet(final_path)

        # Clean up temporary file
        temp_path = self.dir_path / f'{self.id_str}_accepted_CVs_weights_temp.parquet'
        if temp_path.exists():
            temp_path.unlink()

        print(f"Saved {len(cvs_frame)} CV calculations to {final_path}")

    def _calculate_and_store(self, trajectories, new_storage):
        """Main calculation loop with data storage."""
        # Copied from OpenPathSampling PathDensityHistogram(PathHistogram), l. 367
        def _add_ops_trajectory(trajectory, weight):
            cv_traj = [cv(trajectory) for cv in self.cvs]
            # self.add_trajectory(list(zip(*cv_traj)), weight)
            return [list(zip(*cv_traj)), weight]

        iteration_durations = deque(maxlen=10)  # Rolling window of last 10 iterations

        result = self._get_previous_cvs()
        if result[0] is None:  # Error occurred in loading previous CVs
            return

        cvs_frame, old_cycle, resumed = result

        elapsed_time = time.time() - self.start_time
        remaining_time = self.wall_time - elapsed_time

        # Process each trajectory file
        for file_idx, fname in enumerate(tqdm(trajectories,
                                              total=len(trajectories),
                                              desc='Processing trajectory files...')):

            # Calculate rolling average duration of last n iterations
            rolling_avg_duration = (sum(iteration_durations) / len(iteration_durations)
                                    if iteration_durations else 0)

            iteration_start_time = time.time()

            try:
                storage = Storage(filename=str(fname), mode='r')

                start = 1 if not resumed and file_idx == 0 else 0
                len_db = len(storage.steps)

                steps = self.utils.wrapper(storage.steps, fname, start, len_db + 1)

                for step in steps:
                    if resumed and step.mccycle <= old_cycle:
                        continue

                    if step.change.accepted:
                        new_cycle = step.mccycle
                        wt = new_cycle - old_cycle
                        old_cycle = new_cycle

                        # Calculate CVs and add to DataFrame
                        results_list = _add_ops_trajectory(step, wt)
                        results_df = pd.DataFrame(
                            {'MCC': [new_cycle] * len(results_list[0]),
                             'dHG': results_list[0] if len(results_list) > 0 else np.nan,
                             'dWC': results_list[1] if len(results_list) > 1 else np.nan,
                             'lambda': results_list[2] if len(results_list) > 2 else np.nan,
                             'theta': results_list[3] if len(results_list) > 3 else np.nan,
                             'phi': results_list[4] if len(results_list) > 4 else np.nan,
                             'weight': [wt] * len(results_list[0])}

                        )
                        cvs_frame = pd.concat([cvs_frame, results_df], ignore_index=True)
                        # Periodically save progress
                        if len(cvs_frame) % 100 == 0:
                            self._save_progress(cvs_frame)
                        # new_storage.save(step)

                    # Track iteration duration and update rolling average
                    iteration_duration = time.time() - iteration_start_time
                    iteration_durations.append(iteration_duration)

                    # Check if estimated finishing time is still within the cutoff time
                    if (self.wall_time and rolling_avg_duration > 0 and
                            rolling_avg_duration >= remaining_time):
                        print(f"Approaching cutoff time. Exiting after mcc {old_cycle} in {fname}.")
                        print(f'Estimated remaining time: {remaining_time:.2f} s')
                        print(f'Average iteration duration: {rolling_avg_duration:.2f} s')
                        self._save_final_results(cvs_frame)
                        break

                    # Check step limits
                    if self.n_steps and len(cvs_frame) >= self.n_steps:
                        print(f"Reached maximum steps limit: {self.n_steps}")
                        print(f"Stopping after mcc {old_cycle} in {fname}.")
                        self._save_final_results(cvs_frame)
                        return

            except Exception as e:
                print(f"Error processing file {fname}: {e}")
                continue
            finally:
                if 'storage' in locals():
                    storage.close()

        new_storage.sync_all()

        self._save_final_results(cvs_frame)
        new_storage.close()


    def calculate_cvs(self):
        """Main method to calculate collective variables."""
        residA = self.rolling_residues[0]  # 6
        residT = self.rolling_residues[1]  # 16

        try:
            (trajectories,
             storage,
             cvs_list,
             engine) = self.loader.load_data(self.dir_path, self.id_str, f'{self.id_str}_accepted')
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        self.cvs.extend(cvs_list)

        if not self.angle:
            print('The angle between the rolling base and backbone is calculated as arctan2 in the literature. '
                  'Using arcos because \'-a\' flag not set to True.')

        d_WC = self.cvs[0]
        d_HG = self.cvs[1]

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
            comI = paths.MDTrajFunctionCV(
                "comI", md.compute_center_of_mass, topology=engine.topology, select=angI_atoms
            )
            comII = paths.MDTrajFunctionCV(
                "comII", md.compute_center_of_mass, topology=engine.topology, select=angII_atoms
            )
            comIII = paths.MDTrajFunctionCV(
                "comIII", md.compute_center_of_mass, topology=engine.topology, select=angIII_atoms
            )
            comIV = paths.MDTrajFunctionCV(
                "comIV", md.compute_center_of_mass, topology=engine.topology, select=angIV_atoms
            )

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
    parser = argparse.ArgumentParser('Calculate collective variables for OPS trajectories. Returns a parquet file with '
                                     'CV values and weights for each trajectory.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output. Needs to contain OPS databases.')
    parser.add_argument('-cv', '--collective_variable', type=str, required=True,
                        choices=['distances', 'theta', 'phi'],
                        help='Collective variables to be analyzed.')
    parser.add_argument('-rr', '--rolling_residues', type=int, nargs=2, required=True,
                        help='Residue indices of 1. the rolling base and 2. the other base of the pair.')
    parser.add_argument('-bb', '--backbone_residues', type=int, nargs=2, required=True,
                        help='Residue indices at 1. the 5\' end and 2. the 3\' end for backbone orientation.')
    parser.add_argument('-a', '--angle', action='store_true',  # Fixed boolean argument
                        help='Calculate angle as arccos (default) or arctan2 (if set).')
    parser.add_argument('-id', '--identifier', type=str, required=True,
                        help='Identifier for the output files.')
    parser.add_argument('-pkl', '--pkl_name', type=Path, required=False, default=None,
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
