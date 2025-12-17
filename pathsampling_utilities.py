import configparser
from pathlib import Path
import h5py
import openmm as mm
from tqdm.auto import tqdm


class PathsamplingUtilities:

    def __init__(self):
        self.file_names = list()
        self.plumed_script = str()
        self.configs = None
        self.velocities = None
        self.sliced_trajectory = None

    def get_inputs(self, traj_file: str, cyc_no: int = None, input_path: Path = None,
                   config_file: str = None, pdb_file: str = None) -> tuple:
        """
        Load inputs for TPS simulation.

        Returns:
            If traj_file is .db with cyc_no: (trajectory, cvs_dict, network, engine, template, scheme)
            If traj_file is .h5: (traj_file_path, config_file_path, pdb_file_path)
        """
        # Resolve trajectory file path
        traj_path = Path(traj_file)
        if not traj_path.is_file() and input_path:
            traj_path = input_path / traj_file

        if not traj_path.exists():
            raise FileNotFoundError(f'Trajectory file not found: {traj_path}')

        # Case 1: Database file with cycle number - extract OPS objects
        if traj_path.suffix == '.db' and cyc_no is not None:
            from openpathsampling.experimental.storage import Storage

            storage = Storage(str(traj_path), 'r')
            trajectory = storage.steps[cyc_no].active[0].trajectory

            cvs = {cv.name: cv for cv in storage.storable_functions}
            network = storage.networks[0]
            engine = storage.engines[2]

            template = None
            scheme = None
            for obj in storage.simulation_objects:
                if obj.name == '[MDTrajTopology]':
                    template = obj
                elif obj.name == '[OneWayShootingMoveScheme]':
                    scheme = obj

            storage.close()
            return trajectory, cvs, network, engine, template, scheme

        # Case 2: HDF5 file - return file paths
        else:
            # Resolve config and pdb file paths
            resolved_files = [str(traj_path)]
            for file in [config_file, pdb_file]:
                if file:
                    file_path = Path(file)
                    if not file_path.is_file() and input_path:
                        file_path = input_path / file
                    if not file_path.exists():
                        raise FileNotFoundError(f'File not found: {file_path}')
                    resolved_files.append(str(file_path))
                else:
                    resolved_files.append(None)

            return tuple(resolved_files)

    def get_configs(self, config_file):
        self.configs = configparser.ConfigParser()
        self.configs.optionxform = str
        self.configs.read(config_file)

        return self.configs

    def get_plumed_settings(self, plumed_file):
        with open(Path(plumed_file), 'r') as file_:
            self.plumed_script = file_.read()
        return self.plumed_script

    def write_xml(self, filename, object_):
        with open(filename, mode='w') as self.file_:
            self.file_.write(mm.XmlSerializer.serialize(object_))

    def extract_velocities(self, filename, start=0, stop=None):
        with h5py.File(filename, "r") as file_:
            self.velocities = file_['velocities'][()]  # returns data as a numpy array
            if stop:
                self.velocities = self.velocities[start:stop]
            else:
                self.velocities = self.velocities[start:]

        return self.velocities

    def slice_trajectory(self, filename, new_name, start=0, stop=-1):
        try:
            self.sliced_trajectory = h5py.File(new_name, 'a')
            with h5py.File(filename, "r") as trajectory:

                for key in trajectory.keys():
                    if key == 'topology':
                        data = trajectory[key]
                        new_set = self.sliced_trajectory.create_dataset(key, data=data)
                        for attr in trajectory[key].attrs.keys():
                            new_set.attrs[attr] = trajectory[key].attrs[attr]
                    else:
                        data = trajectory[key][start:stop]
                        new_set = self.sliced_trajectory.create_dataset(key, data=data)
                        for attr in trajectory[key].attrs.keys():
                            new_set.attrs[attr] = trajectory[key].attrs[attr]
            self.sliced_trajectory.close()
        except ValueError:
            print('File exists, choose new name or delete old file!')

    @staticmethod
    def wrapper(gen, fname, start=None, len_db=None):
        if not len_db:
            len_db = len(list(gen))
        if not start:
            start = 0
        for idx in tqdm(range(start, len_db), desc=f'Reading steps'):
            try:
                yield gen[idx]
            except StopIteration:
                break
            except Exception as e:
                print(f'Unable to load step {idx} from {fname}: {e.__class__}: {e}')
