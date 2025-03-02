import argparse
import mdtraj as md
import openpathsampling as paths
from openpathsampling.engines.openmm import tools
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage
from pathlib import Path
from pathsampling_utilities import PathsamplingUtilities
paths = monkey_patch_all(paths)

class TrajectoryExtractor:
    """
    Extracts trajectories from OPS storage file and saves them in XTC format.
    """
    def __init__(self, directory, storage, topology, mcc_list, file_name, traj_format, frames_to_extract):
        self.utils = PathsamplingUtilities()
        self.directory = directory
        self.storage = storage
        self.topology = md.load(topology).topology
        self.mcc_list = mcc_list
        self.file_name = file_name
        self.format = traj_format
        self.frames = frames_to_extract

    def extract_trajectory(self):
        """
        Loads OPS storage file and extracts trajectories from the specified MCC cycles. Converts trajectories to
        required format and writes to file.
        :return: None
        """
        try:
            if self.storage.is_file():
                storage = Storage(filename=str(self.storage), mode='r')
            else:
                storage = Storage(filename=str(self.directory / self.storage), mode='r')
        except FileNotFoundError:
            print('OPS storage file not found. Is it in the IO-directory or did you provide an absolute file path?')
            raise FileNotFoundError

        n_steps = len(storage.steps)

        steps = self.utils.wrapper(storage.steps, self.storage.name, len_db=n_steps)
        for step in steps:
            if not step.mccycle in self.mcc_list:
                continue
            else:
                traj = tools.trajectory_to_mdtraj(step.change.trials[0].trajectory, self.topology)
                if self.frames:
                    traj = traj.slice(self.frames)
                traj.save_xtc(self.directory / f'{self.file_name}_{str(step.mccycle).zfill(4)}.{self.format}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract trajectories from OPS storage file and convert to XTC format.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for reading OPS storage file and storing trajectories.')
    parser.add_argument('-storage', '--storage', type=Path, required=True,
                        help='Provide the name of the OPS storage file. Example: \'SYSTEM.db\'.')
    parser.add_argument('-top', '--topology', type=Path, required=True,
                        help='Provide the name of the topology file. Must be readable by MDTraj.load(). '
                             'Example: \'SYSTEM.pdb\'.')
    parser.add_argument('-mcc', '--mcc_list', type=int, nargs='+', required=True,
                        help='Provide a list of MCC cycles to extract trajectories from.')
    parser.add_argument('-name', '--file_name', type=str, required=True,
                        help='Provide the name of the output file. Example: \'SYSTEM_traj\'.')
    parser.add_argument('-format', '--traj_format', type=str, required=False, default='xtc',
                        help='Provide the format of the trajectory file. Default: \'xtc\'.')
    parser.add_argument('-f', '--frames_to_extract', type=int, nargs='+', required=False, default=None,
                        help='Provide the frames to extract from the trajectory. Default: None.')
    args = parser.parse_args()
    args_dict = vars(args)
    extractor = TrajectoryExtractor(**args_dict)
    extractor.extract_trajectory()