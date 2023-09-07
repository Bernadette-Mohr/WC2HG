import openmm as mm
# print(mm.__version__)
from openmm import app
from openmm import unit
from openmmtools.integrators import VVVRIntegrator
import openpathsampling.engines.openmm as ops_openmm
from pathsampling_utilities import PathsamplingUtilities


class TransitionPathSampling:

    def __init__(self, configs, forcefield_list, pdb_file, system_name, output_path):
        # User arguments provided via agrparse
        self.pdb = app.PDBFile(pdb_file)
        self.topology = self.pdb.topology
        self.ops_template = ops_openmm.snapshot_from_pdb(pdb_file=pdb_file)
        self.positions = self.pdb.positions
        self.forcefield_list = forcefield_list
        self.system_name = system_name
        self.output_path = output_path
        # Units extracted from the config file
        self.size_unit = getattr(unit, configs['UNITS']['size_unit'])
        self.press_unit = getattr(unit, configs['UNITS']['press_unit'])
        self.temp_unit = getattr(unit, configs['UNITS']['temp_unit'])
        self.time_unit = getattr(unit, configs['UNITS']['time_unit'])
        # OpenMM system: settings from config file
        self.nonbondedMethod = getattr(app, configs['SYSTEM']['nonbondedMethod'])
        self.nonbondedCutoff = configs['SYSTEM'].getfloat('nonbondedCutoff') * self.size_unit
        self.constraints = getattr(app, configs['SYSTEM']['constraints'])
        self.rigidWater = configs['SYSTEM'].getboolean('rigidWater')
        self.ewaldErrorTolerance = configs['SYSTEM'].getfloat('ewaldErrorTolerance')
        self.constraintTolerance = configs['SYSTEM'].getfloat('constraintTolerance')
        # OpenMM integrator: settings from config file
        self.pressure = configs['INTEGRATOR'].getfloat('pressure') * self.press_unit
        self.temperature = configs['INTEGRATOR'].getfloat('temperature') * self.temp_unit
        self.barostatInterval = configs['INTEGRATOR'].getint('barostatInterval')
        self.friction = configs['INTEGRATOR'].getfloat('friction') / self.time_unit
        self.dt = configs['INTEGRATOR'].getfloat('dt') * self.time_unit
        if configs['PLATFORM']['platform']:
            self.platform = mm.openmm.Platform.getPlatformByName(configs['PLATFORM']['platform'])
            properties = False
            for (key, val) in configs['PLATFORM_PROPERTIES'].items():
                if val:
                    properties = True
            if properties:
                self.platformProperties = {key: configs['PLATFORM_PROPERTIES'][key]
                                           for key in configs['PLATFORM_PROPERTIES']}
            else:
                self.platformProperties = None
        else:
            self.platform = None
            self.platformProperties = None
        self._options = {key: configs['OPS_OPTIONS'][key] for key in configs['OPS_OPTIONS']}
        self._forcefield = self.__get_forcefield()
        self.md_engine = self.setup_engine()

    def __get_forcefield(self):
        return app.ForceField(*self.forcefield_list)

    def __setup_system(self):
        system = self._forcefield.createSystem(self.topology, nonbondedMethod=self.nonbondedMethod,
                                               nonbondedCutoff=self.nonbondedCutoff, constraints=self.constraints,
                                               rigidWater=self.rigidWater,
                                               ewaldErrorTolerance=self.ewaldErrorTolerance)
        system.addForce(mm.MonteCarloBarostat(self.pressure, self.temperature, self.barostatInterval))
        utils = PathsamplingUtilities()
        utils.write_xml(filename=f'{self.output_path}/mtd_{self.system_name}_system.xml', object_=system)

        return system

    def __setup_integrator(self):
        integrator = VVVRIntegrator(self.temperature, self.friction, self.dt)
        integrator.setConstraintTolerance(self.constraintTolerance)
        utils = PathsamplingUtilities()
        utils.write_xml(filename=f'{self.output_path}/mtd_{self.system_name}_integrator.xml', object_=integrator)

        return integrator

    def setup_engine(self):
        if self.platform:
            self.md_engine = ops_openmm.Engine(topology=self.ops_template.topology, system=self.__setup_system(),
                                               integrator=self.__setup_integrator(),
                                               openmm_properties=self.platformProperties, options=self._options
                                               ).named('OMM_engine')
        else:
            self.md_engine = ops_openmm.Engine(topology=self.ops_template.topology, system=self.__setup_system(),
                                               integrator=self.__setup_integrator(), options=self._options
                                               ).named('OMM_engine')
        self.md_engine.initialize(self.platform)
        self.md_engine.current_snapshot = self.ops_template

        return self.md_engine
