import argparse
from collections import deque
import mdtraj as md
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
from openpathsampling.experimental.storage import Storage
from pathlib import Path
import time
from tqdm import tqdm
from collective_variable import CollectiveVariable
from cv_data_manager import create_dna_cv_manager
from pathsampling_utilities import PathsamplingUtilities
from typing import Any


class DataLoader:
    """
    Load and manage OPS trajectory data.
    """
    def __init__(self):
        self.raw_storages = list()
        self.new_storage = None

    def load_data(self, root_path: Path, dir_name: str, new_name: str | None) -> tuple:
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
            self.raw_storages.extend(sorted(dir_.glob('*.db')))

        db_name = f'{new_name}.db'
        db_path = root_path / db_name
        if not db_path.is_file():
            self.new_storage = Storage(filename=f'{db_path}', mode='w')
            first_storage = Storage(filename=str(self.raw_storages[0]), mode='r')
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


        return self.raw_storages, self.new_storage, cvs_list, engine


class CVCalculator:
    def __init__(self, directory: Path, base_name: str, rolling_residues: list[int], backbone_residues: list[int],
                 angle: bool, identifier: str, cvs_name: int =None, mc_cycle: int =None,
                 max_steps: int =None, wall_time: int =None,):
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
        self.cv_manager = create_dna_cv_manager()
        self.cvs = list()

    def _save_results(self, output_dir: str | None) -> None:
        """Save results in multiple formats."""
        output_dir = Path(output_dir)

        pickle_path = output_dir / f'{self.id_str}_cv_data.pkl'
        self.cv_manager.save_to_pickle(pickle_path)

        dataframe_path = output_dir / f'{self.id_str}_cv_data.parquet.gzip'
        self.cv_manager.save_to_dataframe(dataframe_path)

        print(f"Saved CV data:")
        print(f"  - Nested structure: {pickle_path}")
        print(f"  - DataFrame: {dataframe_path}")

    def _check_time_and_step_limits(self, iteration_durations: deque[Any] , old_cycle: int, fname: str) -> bool:
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

    def _calculate_and_store(self, raw_storages: list[Path], new_storage: Storage) -> None:
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
        for file_idx, fname in enumerate(tqdm(raw_storages,
                                              total=len(raw_storages),
                                              desc='Processing raw storage files')):
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
                        results_list = _add_ops_trajectory(step.active[0].trajectory, wt)
                        self.cv_manager.add_trajectory_data(new_cycle, results_list)
                        new_storage.save(step)
                        old_cycle = new_cycle

                        # Periodically save progress
                        if len(self.cv_manager.data) % 10 == 0:
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
                        storage.close()
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
            (raw_storages,
             new_storage,
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
                angle=False,  # self.angle,
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
                angle=True,  # self.angle,
            )

            self.cvs.extend([lambda_, theta_, phi_])

        except Exception as e:
            print(f"Error creating collective variables: {e}")
            return

        # Start calculation
        self._calculate_and_store(raw_storages, new_storage)


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
