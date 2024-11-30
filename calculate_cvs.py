import argparse
from collections import deque
from itertools import islice
import mdtraj as md
import numpy as np
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage
from pathlib import Path
from pathsampling_utilities import PathsamplingUtilities
import pickle
import time
from tqdm import tqdm

paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True

# import importlib.util
# import sys
# spec = importlib.util.spec_from_file_location(
#     "pathsampling_utilities",
#     "/home/bernadette/PycharmProjects/WC2HG/pathsampling_utilities.py",
# )
# psu = importlib.util.module_from_spec(spec)
# sys.modules['pathsampling_utilities'] = psu
# spec.loader.exec_module(psu)
# utils = psu.PathsamplingUtilities()


class DataLoader:
    """
    Load OPS trajectories and weights from databases and dictionaries
    """

    def __init__(self):
        self.utils = PathsamplingUtilities()
        self.max_steps = None
        self.n_steps = None
        self.remaining_steps = None
        self.mcc_idx = None
        self.last_mc_step = 0
        self.mc_steps = list()
        self.trajectories = list()
        self.weights = list()

    def append_trajectories(self, db_file, mc_cycle, weight=None):
        """
        Append OPS trajectories from storage file to the list of trajectories.
        If no weights are provided, append a default weight = 1.0 to the list of weights for each trajectory.
        :param mc_cycle: Last processed MC cycle number, if storage file was split.
        :param db_file: (Path) OPS storage file containing the trajectories.
        :param weight: (float) weight for the trajectories.
        :return: None
        """
        storage = Storage(filename=str(db_file), mode='r')
        n_steps = len(storage.steps)
        mc_steps = list()

        steps = self.utils.wrapper(storage.steps, db_file.name, start=0, len_db=n_steps)

        # Initialize or update the steps processed count
        if not hasattr(self, 'steps_processed'):
            self.steps_processed = 0

        # Calculate the remaining steps we can process
        if self.max_steps:
            remaining_steps = self.max_steps - self.steps_processed
            steps_to_process = min(remaining_steps, n_steps)
            self.n_steps = steps_to_process
        else:
            steps_to_process = n_steps


        # If we have already reached the maximum steps, exit early
        if steps_to_process <= 0:
            return

        # Determine the number of steps to process based on mc_cycle
        if not mc_cycle:
            for current_step in islice(steps, steps_to_process):
                mc_steps.append(current_step.mccycle)
                self.trajectories.append(current_step.active[0].trajectory)
                self.steps_processed += 1
        else:
            for step in steps:
                if step.mccycle <= mc_cycle:
                    continue
                if self.max_steps and self.steps_processed >= self.max_steps:
                    break
                mc_steps.append(step.mccycle)
                self.trajectories.append(step.active[0].trajectory)
                self.steps_processed += 1

        self.last_mc_step = mc_steps[-1] if mc_steps else None

        # Update remaining_steps to reflect how many more steps can be processed
        if self.max_steps:
            self.remaining_steps = self.max_steps - self.steps_processed

        if weight:
            self.weights = [weight] * len(mc_steps)
            self.mc_steps.extend(mc_steps)

    def append_weights(self, wt_file, mc_cycle):
        """
        Append weights from a dictionary to the list of weights.
        :param mc_cycle: Last processed MC cycle number, if storage file was split.
        :param wt_file: (Path) Pickled dictionary containing the weights.
        :return: None
        """
        # Initialize or update weights added count
        if not hasattr(self, 'weights_added'):
            self.weights_added = 0

        print(wt_file)
        with open(wt_file, 'rb') as f:
            weights = pickle.load(f)
        # Calculate how many more weights are needed to reach max_steps
        if self.max_steps:
            remaining_weights_to_add = self.max_steps - self.weights_added
        else:
            remaining_weights_to_add = len(weights)
        print(f'Remaining weights to add: {remaining_weights_to_add}')
        if remaining_weights_to_add <= 0:
            return

        # Start from beginning of file if no mc_cycle is provided
        if not mc_cycle:
            # Slice to get only the needed number of weights
            self.weights.extend(list(weights.values())[:remaining_weights_to_add])
            self.mc_steps.extend(list(weights.keys())[:remaining_weights_to_add])
            print('len self.weights', len(self.weights))
            self.weights_added = len(self.weights)
            print('len self.weights_added', self.weights_added)

        # Start from position of provided mc_cycle if continuation run
        else:
            if mc_cycle:
                self.mcc_idx = list(weights.keys()).index(mc_cycle) + 1
                if not self.max_steps:
                    weights_to_add = list(weights.values())[self.mcc_idx:]
                    mcc_to_add = list(weights.keys())[self.mcc_idx:]
                else:
                    weights_to_add = list(weights.values())[self.mcc_idx:self.mcc_idx + remaining_weights_to_add]
                    mcc_to_add = list(weights.keys())[self.mcc_idx:self.mcc_idx + remaining_weights_to_add]
            else:
                weights_to_add = list(weights.values())[:remaining_weights_to_add]
                mcc_to_add = list(weights.keys())[:remaining_weights_to_add]
            count = 0
            for cycle, wt in list(zip(mcc_to_add, weights_to_add)):
                if count > remaining_weights_to_add:
                    break
                self.weights.append(float(wt))
                self.mc_steps.append(cycle)
                count += 1
                self.weights_added += 1

    def load_data(self, wt_list, db_list, mc_cycle, max_steps):
        """
        Load OPS trajectories and weights from databases and dictionaries.
        :param max_steps:
        :param mc_cycle:
        :param db_list: List of OPS storage files containing the trajectories.
        :param wt_list: List of pickled dictionaries containing the weights.
        :return: List of OPS trajectories, list of weights, OPS storage object, OPS engine object.
        """

        self.max_steps = max_steps
        if not wt_list:
            for db in db_list:
                self.append_trajectories(db, mc_cycle, weight=1.0)
        else:
            for db, wt in zip(db_list, wt_list):
                self.append_trajectories(db, mc_cycle)
                self.append_weights(wt, mc_cycle)

        print(len(self.trajectories), len(self.weights), len(self.mc_steps))
        print(self.mc_steps[-1], self.last_mc_step)
        storage = Storage(filename=str(db_list[0]), mode='r')
        engine = storage.engines[0]

        return self.trajectories, self.weights, self.mc_steps, storage, engine, self.last_mc_step


class CollectiveVariable:
    def __init__(self):
        pass

    @classmethod
    def angle_between_vectors(cls, v1, v2, angle=False):
        # Chose whether to calculate the angle as arctan2 [-180°, 180°] or arccos [0°, 180°]
        if angle is True:
            normal = np.cross(v1, v2)
            # Use the sign of the z coordinate of the normal to determine if the angle is rotated (counter-)clockwise
            # and reflect the full angle range from -180° to 180° in the 3D case.
            angle = np.degrees(
                np.arctan2(np.linalg.norm(normal), np.dot(v1, v2))
            ) * np.sign(np.dot(normal, np.array([0.0, 0.0, 1.0])))
        else:
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            angle = np.degrees(
                np.arccos(np.clip(np.divide(dot_product, (norm_v1 * norm_v2)), -1.0, 1.0))
            )

        return angle

    @classmethod
    def base_opening_angle(
            cls, snapshot, comI_cv, comII_cv, comIII_cv, comIV_cv, angle_between_vectors_cv, angle
    ):
        """
        Parameters:
        :param snapshot:
        :param comI_cv:
        :param comII_cv:
        :param comIII_cv:
        :param comIV_cv:
        :param angle_between_vectors_cv:
        :param angle:
        :return:
        """
        comI = comI_cv(snapshot)
        comII = comII_cv(snapshot)
        comIII = comIII_cv(snapshot)
        comIV = comIV_cv(snapshot)

        vec_21 = np.subtract(comI, comII)
        vec_23 = np.subtract(comIII, comII)
        vec_24 = np.subtract(comIV, comII)
        norm1 = np.cross(vec_21, vec_23)
        norm2 = np.cross(vec_24, vec_23)

        return angle_between_vectors_cv(norm1, norm2,
                                        angle)  # hard-coded negative sign in the code to Vreede et al., 2019

    @classmethod
    def base_rolling_angle(
            cls, snapshot, backbone_idx, rollingbase_idx, angle_between_vectors_cv, angle
    ):
        """
        Parameters
        ----------
        :param angle: selects wether the angle between two vectors is calculated as atan2 (True) or arccos (False).
        :param snapshot: ops trajectory frame
        :param rollingbase_idx: list of the indices of the N1, N3 and N7 atoms defining the vectors of the rolling base
        :param backbone_idx: list of the P atom indices defining the backbone vector
        :param angle_between_vectors_cv: function to calculate the angle between two vectors.
        """

        def normalize(vector):
            norm = np.linalg.norm(vector)
            if norm == 0:
                return vector
            return vector / norm

        # Get the vectors connecting atoms N3 and N1 and N3 and N7 in the rolling base.
        bp_v1 = np.subtract(
            snapshot.xyz[rollingbase_idx[0]], snapshot.xyz[rollingbase_idx[1]]
        )
        bp_v2 = np.subtract(
            snapshot.xyz[rollingbase_idx[2]], snapshot.xyz[rollingbase_idx[1]]
        )

        # Calculate the normal of the rolling-base vectors
        bp_vector = normalize(np.cross(bp_v1, bp_v2))

        # Get the vector associated with the backbone
        bb_vector = np.subtract(
            snapshot.xyz[backbone_idx[1]], snapshot.xyz[backbone_idx[0]]
        )
        bb_vector = normalize(bb_vector)

        # calculate angle
        return angle_between_vectors_cv(bb_vector, bp_vector, angle)

    # lambda = arctan2(dHG, dWC)
    @classmethod
    def lambda_CV(cls, snapshot, d_WC_cv, d_HG_cv):
        """
        Parameters:
        :param snapshot:
        :param d_WC_cv:
        :param d_HG_cv:
        :return: Single CV combining the hydrogen bond lengths of the WC and the HG pairing.
        """
        d_wc = d_WC_cv(snapshot)
        d_hg = d_HG_cv(snapshot)

        return np.arctan2(d_wc, d_hg)


class CVCalculator:
    def __init__(self,
                 directory,
                 identifier,
                 collective_variable,
                 rolling_residues,
                 backbone_residues,
                 angle,
                 pkl_name,
                 mc_cycle,
                 max_steps,
                 wall_time):
        self.db_list = sorted(directory.glob('*.db'))
        self.wt_list = sorted(f for f in directory.glob('*.pkl') if f.name != pkl_name)
        self.id_str = identifier
        self.cv_name = collective_variable
        self.rolling_residues = rolling_residues
        self.backbone_residues = backbone_residues
        self.angle = angle
        self.pkl_name = pkl_name
        self.mc_cycle = mc_cycle
        self.n_steps = max_steps
        self.wall_time = wall_time
        self.start_time = time.time()
        self.loader = DataLoader()
        self.cvs = list()
        self.cvs_list = self._get_previous_cvs()

    def _get_previous_cvs(self):
        if not self.pkl_name:
            print('No previous CVs found. Starting calculation...')
            self.pkl_name = f'{self.id_str}_{self.cv_name}_CVs_weights.pkl'
            return list()
        else:
            print(f'Loading previous CVs from {self.pkl_name}. Resuming calculation...')
            with open(self.pkl_name, 'rb') as f:
                csv_list = pickle.load(f)
            if isinstance(csv_list[-1], dict):
                if self.mc_cycle:
                    if csv_list[-1]['mc_cycle'] != self.mc_cycle:
                        print('MC cycle number does not match the last entry in the pickle file. Exiting...')
                        return
                csv_list.pop()

            return csv_list

    def _calculate_and_store(self, trajectories, weights, mc_steps):

        # Copied from OpenPathSampling PathDensityHistogram(PathHistogram), l. 367
        def _add_ops_trajectory(trajectory, weight):
            cv_traj = [cv(trajectory) for cv in self.cvs]
            # self.add_trajectory(list(zip(*cv_traj)), weight)
            return [list(zip(*cv_traj)), weight]

        iteration_durations = deque(maxlen=10)  # Rolling window of last 10 iterations

        for idx, (traj, wt) in enumerate(tqdm(zip(trajectories, weights),
                                              total=len(weights),
                                              desc=f'Calculating {self.cv_name}')):
            elapsed_time = time.time() - self.start_time
            remaining_time = self.wall_time - elapsed_time

            # Calculate rolling average duration of last n iterations
            if iteration_durations:
                rolling_avg_duration = sum(iteration_durations) / len(iteration_durations)
            else:
                rolling_avg_duration = 0

            iteration_start_time = time.time()
            # Convert coordinates to CVs
            self.cvs_list.append(_add_ops_trajectory(traj, wt))

            # Track iteration duration and update rolling average
            iteration_duration = time.time() - iteration_start_time
            iteration_durations.append(iteration_duration)

            # Check if estimated finishing time is still within the cutoff time
            if rolling_avg_duration > 0 and rolling_avg_duration >= remaining_time:
                print(f"Approaching cutoff time. Exiting after iteration {idx}.")
                print(f'Estimated remaining time: {remaining_time:.2f} s')
                print(f'Average iteration duration: {rolling_avg_duration:.2f} s')
                self.mc_cycle = list(mc_steps)[idx]
                break
                
        if self.mc_cycle:
            self.cvs_list.append({'mc_cycle': self.mc_cycle})

        with open(self.pkl_name, 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(self.cvs_list, f, pickle.HIGHEST_PROTOCOL)

    def calculate_cvs(self):
        residA = self.rolling_residues[0]  # 6
        residT = self.rolling_residues[1]  # 16
        (trajectories,
         weights,
         mc_steps,
         storage,
         engine,
         last_step) = self.loader.load_data(self.wt_list, self.db_list, self.mc_cycle, self.n_steps)
        self.mc_cycle = last_step

        if self.cv_name == 'distances':
            self.cvs = [storage.cvs['d_WC'], storage.cvs['d_HG']]

            print(f'Calculating collective variable {self.cv_name}...')
            # Calculating collective variable: dWC - dHG
            self._calculate_and_store(trajectories, weights, mc_steps)

        elif self.cv_name == 'theta':
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
                f= CollectiveVariable.lambda_CV,
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

            self.cvs = [lambda_, theta_]

            print(f'Calculating collective variable {self.cv_name}...')
            # calculate CV: lambda-theta
            self._calculate_and_store(trajectories, weights, mc_steps)

        elif self.cv_name == 'phi':
            if not self.backbone_residues:
                print('Backbone residues are needed for calculating the base rolling angle. Exiting...')
                return
            if not self.angle:
                print('The angle between the rolling base and backbone is calculated as arctan2 in the literature. '
                      'Using arcos because \'-a\' flag not set to True.')

            resid_bb_start = self.backbone_residues[0]  # 1
            resid_bb_end = self.backbone_residues[1]  # 13
            backbone_atoms = f'resid {resid_bb_start} {resid_bb_end} and name P'
            rollingbase_atoms = f'resid {residA} and name N7 N3 N1'

            d_WC = storage.cvs['d_WC']
            d_HG = storage.cvs['d_HG']
            # d_BP = storage.cvs['d_BP']

            lambda_ = paths.CoordinateFunctionCV(
                name='lambda',
                f=CollectiveVariable.lambda_CV,
                d_WC_cv=d_WC,
                d_HG_cv=d_HG
            )

            phi_ = paths.CoordinateFunctionCV(
                name="phi",
                f=CollectiveVariable.base_rolling_angle,
                backbone_idx=engine.topology.mdtraj.select(backbone_atoms),
                rollingbase_idx=engine.topology.mdtraj.select(rollingbase_atoms),
                angle_between_vectors_cv=CollectiveVariable.angle_between_vectors,
                angle=self.angle,
            )

            self.cvs = [lambda_, phi_]

            print(f'Calculating collective variable {self.cv_name}...')
            # calculate CV: lambda-phi
            self._calculate_and_store(trajectories, weights, mc_steps)

        else:
            print(f'Collective variable {self.cv_name} not currently implemented or wrong/misspelled label. Exiting...')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Calculate collective variables for OPS trajectories. Returns a pickle file with a '
                                     'list of two CV values and a weight for each trajectory.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output. Needs to contain OPS databases and '
                             'dictionaries with weights as values.')
    parser.add_argument('-cv', '--collective_variable', type=str, required=True,
                        choices=['distances', 'theta', 'phi'],
                        help='Collective variables to be analyzed.')
    parser.add_argument('-rr', '--rolling_residues', type=int, nargs='+', required=True,
                        help='Residue indices of 1. the rolling base and 2. the other base of the pair on the '
                             'neighboring DNA strand.')
    parser.add_argument('-bb', '--backbone_residues', type=int, nargs='+', required=False,
                        help='Residue indices at 1. the 5\' end of the strand containing the rolling base and 2. at the'
                             ' 3\' end of the opposite strand to calculate the proxy for the backbone orientation. '
                             'Needed for calculating the base rolling angle.')
    parser.add_argument('-a', '--angle', type=bool, required=False, default=False,
                        help='Calculate the angle between two vectors as arccos (theta, default) '
                             'or arctan2 (phi, set to True).')
    parser.add_argument('-id', '--identifier', type=str, required=True,
                        help='Identifier for the output files, e.g. name of system.')
    parser.add_argument('-pkl', '--pkl_name', type=Path, required=False, default=None,
                        help='If the cv calculation is a continuation run, provide the name of the pickle file with '
                             'existing CVs and weights. Example: \'SYSTEM_theta_CVs_weights.pkl\'.')
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
    cvs = CVCalculator(**args_dict)
    cvs.calculate_cvs()
