import openmm as mm
print(mm.__version__)
from openmm import app
from openmm import unit
from pathsampling_utilities import PathsamplingUtilities


class MetadynamicsSimulation:

    def __init__(self, configs, forcefield_list, pdb_file, system_name, output_path):
        # User arguments provided via agrparse
        self.pdb = app.PDBFile(pdb_file)
        self.topology = self.pdb.topology
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
        self.platform = mm.Platform.getPlatformByName(configs['PLATFORM']['platform'])
        self.platformProperties = {key: configs['PLATFORM_PROPERTIES'][key]
                                   for key in configs['PLATFORM_PROPERTIES']}
        print(self.platformProperties)
        self.equilibrationSteps = configs['SIMULATION'].getint('equilibrationSteps')
        self.steps = configs['SIMULATION'].getint('steps')
        self.reportInterval = configs['SIMULATION'].getint('reportInterval')
        self.currentStep = configs['SIMULATION'].getint('currentStep')
        # OpenMM: initialize modules.
        self._forcefield = self.__get_forcefield()
        self.system = self.__setup_system()
        # self._integrator = self.__setup_integrator()
        self.simulation = self.setup_simulation()

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
        integrator = mm.LangevinMiddleIntegrator(self.temperature, self.friction, self.dt)
        integrator.setConstraintTolerance(self.constraintTolerance)
        utils = PathsamplingUtilities()
        utils.write_xml(filename=f'{self.output_path}/mtd_{self.system_name}_integrator.xml', object_=integrator)

        return integrator

    def setup_simulation(self):
        simulation = app.Simulation(self.topology, self.system, self.__setup_integrator(), self.platform,
                                    self.platformProperties)
        simulation.context.setPositions(self.positions)

        return simulation
