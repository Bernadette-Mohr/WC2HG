import configparser
from pathlib import Path
import openmm as mm


class PathsamplingUtilities:

    def __init__(self):
        self.file_names = list()
        self.plumed_script = str()
        self.configs = None

    def get_inputs(self, *args: str, input_path: Path = None) -> list:
        self.file_names = [name for name in args]
        for idx, file_name in enumerate(self.file_names):
            try:
                if Path(input_path / file_name).is_file():
                    self.file_names[idx] = str(input_path / file_name)
                elif Path(file_name).is_file():
                    pass
            except FileNotFoundError:
                print(f'File {self.file_names[idx]} not found!')

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
