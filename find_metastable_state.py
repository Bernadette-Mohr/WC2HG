import argparse
from collections import deque
import mdtraj as md
import numpy as np
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage
from pathlib import Path
from pathsampling_utilities import PathsamplingUtilities
from collective_variable import CollectiveVariable
import pandas as pd
import time

paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True


class DataLoader:
    """
    """
    def __init__(self):
        self.utils = PathsamplingUtilities()
        # self.frames = list()
        # self.mc_cycles = list()

    def select_frames(self, db_file, frames, mc_cycles):
        """
        """
        storage = Storage(filename=str(db_file), mode='r')
        n_steps = len(storage.steps)

        steps = self.utils.wrapper(storage.steps, db_file.name, len_db=n_steps)
        for step in steps:
            if not step.change.accepted:
                mover = step.change.canonical.mover
                details = step.change.canonical.details.load()
                rejection_reason = details.__dict__.get('rejection_reason', None)
                if rejection_reason == 'max_length' and mover.direction == 'forward':
                    frames.append(step.change.trials[0].trajectory[-1])
                    mc_cycles.append(step.mccycle)
                elif rejection_reason == 'max_length' and mover.direction == 'backward':
                    frames.append(step.change.trials[0].trajectory[0])
                    mc_cycles.append(step.mccycle)
                else:
                    # print('\nMetropolis acceptance:', details.__dict__.get('metropolis_acceptance', None))
                    pass
        return frames, mc_cycles

    def load_data(self, db_file):
        """
        """
        frames, mc_cycles = list(), list()
        frames, mc_cycles = self.select_frames(db_file, frames, mc_cycles)
        storage = Storage(filename=str(db_file), mode='r')
        engine = storage.engines[0]

        return frames, mc_cycles, storage, engine


class MetaStablesStateFinder(CollectiveVariable):
    def __init__(self, directory,
                 identifier,
                 rolling_residues,
                 backbone_residues,
                 df_name,
                 mc_cycle,
                 max_steps,
                 wall_time
                 ):
        super().__init__()
        # self.db_list = sorted(directory.glob('*.db'))
        self.db_list = sorted([_file for _file in directory.rglob("*.db")])
        self.directory = directory
        self.id_str = identifier
        self.rolling_residues = rolling_residues
        self.backbone_residues = backbone_residues
        self.df_name = df_name
        self.mc_cycle = mc_cycle
        self.n_steps = max_steps
        self.wall_time = wall_time
        self.start_time = time.time()
        self.loader = DataLoader()
        self.cvs = list()
        self.cvs_df = self._get_previous_cvs()

    def _get_previous_cvs(self):
        if not self.df_name:
            print('No previous CVs found. Starting calculation...')
            self.df_name = f'{self.id_str}_MetaStableStates.pickle'
            return pd.DataFrame(columns=['MCC', 'd_WC', 'd_HG', 'lambda', 'theta', 'phi'])
        else:
            print(f'Loading previous CVs from {self.df_name}. Resuming calculation...')
            cv_frame = pd.read_pickle(self.df_name)
            print(cv_frame)

            return cv_frame

    def _add_ops_trajectory(self, traj):
        cv_traj = [cv(traj) for cv in self.cvs]
        return cv_traj

    def calculate_cvs(self):
        residA = self.rolling_residues[0]  # 6
        residT = self.rolling_residues[1]  # 16
        resid_bb_start = self.backbone_residues[0]  # 1
        resid_bb_end = self.backbone_residues[1]  # 13
        backbone_atoms = f'resid {resid_bb_start} {resid_bb_end} and name P'
        rollingbase_atoms = f'resid {residA} and name N7 N3 N1'

        new_df = pd.DataFrame(columns=self.cvs_df.columns)
        iteration_durations = deque(maxlen=3)  # Rolling window of last 3 iterations
        for idx, database in enumerate(self.db_list):
            print(f'Processing database {database.name}...')
            elapsed_time = time.time() - self.start_time
            if self.wall_time:
                remaining_time = self.wall_time - elapsed_time
            else:
                remaining_time = None

            # Calculate rolling average duration of last n iterations
            if iteration_durations:
                rolling_avg_duration = sum(iteration_durations) / len(iteration_durations)
            else:
                rolling_avg_duration = 0

            iteration_start_time = time.time()
            frames, mc_cycles, storage, engine = self.loader.load_data(database)

            # self.mc_cycle = last_step
            d_WC = storage.cvs['d_WC']
            d_HG = storage.cvs['d_HG']

            o3_prime = '"O3\'"'
            o5_prime = '"O5\'"'
            angI_atoms = (
                f"(resid {residA - 1} and not type H) or (resid {residA + 1} and not type H) or (resid {residT - 1} "
                f"and not type H) or (resid {residT + 1} and not type H)")
            angII_atoms = f"(resid {residA - 1} and name {o3_prime}) or (resid {residA} and name P OP1 OP2 {o5_prime})"
            angIII_atoms = f"(resid {residA} and name  {o3_prime}) or (resid {residA + 1} and name P OP1 OP2 {o5_prime})"
            angIV_atoms = f"resid {residA} and not type H"

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
                angle=False,
            )

            phi_ = paths.CoordinateFunctionCV(
                name="phi",
                f=CollectiveVariable.base_rolling_angle,
                backbone_idx=engine.topology.mdtraj.select(backbone_atoms),
                rollingbase_idx=engine.topology.mdtraj.select(rollingbase_atoms),
                angle_between_vectors_cv=CollectiveVariable.angle_between_vectors,
                angle=True,
            )
            self.cvs = [d_WC, d_HG, lambda_, theta_, phi_]

            cv_values = self._add_ops_trajectory(frames)
            cv_values.insert(0, np.array(mc_cycles))
            df = pd.DataFrame({col: array for col, array in zip(self.cvs_df.columns, cv_values)})
            new_df = pd.concat([new_df, df], ignore_index=True)
            print(new_df)

            # Track iteration duration and update rolling average
            iteration_duration = time.time() - iteration_start_time
            iteration_durations.append(iteration_duration)

            # Check if estimated finishing time is still within the cutoff time
            if self.wall_time:
                if rolling_avg_duration > 0 and rolling_avg_duration >= remaining_time:
                    print(f"Approaching cutoff time. Exiting after iteration {idx}, file {database.name}.")
                    print(f'Estimated remaining time: {remaining_time:.2f} s')
                    print(f'Average iteration duration: {rolling_avg_duration:.2f} s')
                    break

        self.cvs_df = pd.concat([self.cvs_df, new_df], ignore_index=True)
        self.cvs_df.to_pickle(self.directory / self.df_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Calculate collective variables for OPS trajectories. Returns a pickle file with a '
        'list of two CV values and a weight for each trajectory.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output. Needs to contain OPS databases and '
                             'dictionaries with weights as values.')
    parser.add_argument('-rr', '--rolling_residues', type=int, nargs='+', required=True,
                        help='Residue indices of 1. the rolling base and 2. the other base of the pair on the '
                             'neighboring DNA strand.')
    parser.add_argument('-bb', '--backbone_residues', type=int, nargs='+', required=True,
                        help='Residue indices at 1. the 5\' end of the strand containing the rolling base and 2. at the'
                             ' 3\' end of the opposite strand to calculate the proxy for the backbone orientation. '
                             'Needed for calculating the base rolling angle.')
    parser.add_argument('-id', '--identifier', type=str, required=True,
                        help='Identifier for the output files, e.g. name of system.')
    parser.add_argument('-df', '--df_name', type=Path, required=False, default=None,
                        help='If the cv calculation is a continuation run, provide the name of the pandas dataframe with '
                             'existing CVs and MC cycle numbers. Example: \'SYSTEM_theta_CVs_weights.pkl\'.')
    parser.add_argument('-mcc', '--mc_cycle', type=int, required=False, default=None,
                        help='If a large storage database had to be split for processing, pass the number of the last '
                             'processed MC cycle (Stored as last entry in output pkl with collective variables as '
                             '{"mc_cyle": int}).')
    parser.add_argument('-max', '--max_steps', type=int, required=False, default=None,
                        help='If a large storage database has to be split for processing, pass the number of steps that'
                             'can be processed within memory on the available compute resources. When in doubt, try half.')
    parser.add_argument('-wt', '--wall_time', type=int, required=False, default=None,
                        help='If running on an HPC cluster, set the cluster wall time in seconds to prevent TIMEOUT and '
                             'data loss.')

    args = parser.parse_args()
    args_dict = vars(args)
    msf = MetaStablesStateFinder(**args_dict)
    msf.calculate_cvs()

# -dir $TMPDIR/ -cv distances -rr 6 16 -bb 1 13 -id DNAWC -a False -pkl DNAWC_distances_CVs_weights.pkl -mcc 346 -max 62 -wt 430200