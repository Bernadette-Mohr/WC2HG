import configparser
from pathlib import Path
import h5py
import openmm as mm


class PathsamplingUtilities:

    def __init__(self):
        self.file_names = list()
        self.plumed_script = str()
        self.configs = None
        self.velocities = None
        self.sliced_trajectory = None

    def get_inputs(self, *args: str, cyc_no: int = None, input_path: Path = None) -> list:
        self.file_names = [name for name in args]
        for idx, file_name in enumerate(self.file_names):
            try:
                if Path(input_path / file_name).is_file():
                    self.file_names[idx] = str(input_path / file_name)
                elif Path(file_name).is_file():
                    pass
            except FileNotFoundError:
                print(f'File {self.file_names[idx]} not found!')
        if Path(self.file_names[0]).suffix == '.db' and cyc_no:
            from openpathsampling.experimental.storage import Storage
            storage = Storage(self.file_names[0], 'r')
            self.file_names[0] = storage.steps[cyc_no].active[0].trajectory
            cvs = dict()
            for cv in storage.storable_functions:
                cvs[cv.name] = cv
            network = storage.networks[0]
            template = None
            scheme = None
            for obj in storage.simulation_objects:
                if obj.name == '[MDTrajTopology]':
                    template = obj
                elif obj.name == '[OneWayShootingMoveScheme]':
                    scheme = obj
            # [cvs, network, engine, template, scheme]
            self.file_names.append(cvs)
            self.file_names.append(network)
            self.file_names.append(template)
            self.file_names.append(scheme)
            storage.close()

        return self.file_names

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
