import sys
import argparse
from pathlib import Path
import configparser
from openmm import app
from openmmplumed import PlumedForce
import mdtraj as md
from pathsampling_utilities import PathsamplingUtilities
from metadynamics_setup import MetadynamicsSimulation

sys.path.append('')


def run_metadynamics_simulation(input_dir=None, config_file=None, plumed_file=None, pdb_file=None,
                                ff_list=None, system_name=None, output_dir=None):

    # Prepare the Simulation

    print('Building system...')
    utils: PathsamplingUtilities = PathsamplingUtilities()
    config_file, plumed_file, pdb_file = utils.get_inputs(config_file, plumed_file, pdb_file, input_path=input_dir)
    configs = configparser.ConfigParser()
    configs.optionxform = str
    configs.read(config_file)

    setup = MetadynamicsSimulation(configs=configs, forcefield_list=ff_list, pdb_file=pdb_file, system_name=system_name,
                                   output_path=output_dir)
    simulation = setup.setup_simulation()
    system = setup.system

    # Minimize and Equilibrate

    print('Performing energy minimization...')
    simulation.minimizeEnergy()
    print('Equilibrating...')
    simulation.context.setVelocitiesToTemperature(setup.temperature)
    simulation.step(setup.equilibrationSteps)

    # Simulate

    script = utils.get_plumed_settings(plumed_file)
    system.addForce(PlumedForce(script))
    simulation.context.reinitialize(preserveState=True)

    print('Running simulation...')
    dataReporter = app.StateDataReporter(f'{output_dir}/{system_name}_data.csv', setup.reportInterval,
                                         totalSteps=setup.steps, time=True, speed=True, progress=True,
                                         elapsedTime=True, remainingTime=True, potentialEnergy=True, kineticEnergy=True,
                                         totalEnergy=True, temperature=True, volume=True, density=True, separator=';')
    trajectoryReporter = md.reporters.HDF5Reporter(f'{output_dir}/{system_name}_trajectory.h5', setup.reportInterval,
                                                   coordinates=True, time=True, cell=True, potentialEnergy=True,
                                                   kineticEnergy=True, temperature=True, velocities=True,
                                                   enforcePeriodicBox=True)
    simulation.reporters.append(dataReporter)
    simulation.reporters.append(trajectoryReporter)
    simulation.currentStep = setup.currentStep
    simulation.step(setup.steps)

    trajectoryReporter.close()

    simulation.saveState(f'{output_dir}/{system_name}_final_state.xml')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Initialize and run a metadynamics simulation with OpenMM and Plumed. Requires a '
                                     'configparser file with units, number of simulation steps etc, a Plumed input '
                                     'file, the required force field and initial system coordinates in PDB format.')
    parser.add_argument('-in', '--input_directory', type=Path, required=False, default=Path.cwd(),
                        help='Path to a directory with the required input files, DEFAULT: current directory.')
    parser.add_argument('-cfg', '--config_file', type=str, required=True,
                        help='File with simulation settings in python configparser format.')
    parser.add_argument('-pmd', '--plumed_file', type=str, required=True,
                        help='File with Plumed settings. See https://www.plumed.org/doc for syntax and options.')
    parser.add_argument('-pdb', '--pdb_file', type=str, required=True,
                        help='File with initial coordinates of the simulation system in PDB format. '
                             'Useful resource: https://github.com/openmm/pdbfixer')
    parser.add_argument('-ff', '--forcefield', type=str, nargs='+', required=True,
                        help='All force field files required for simulation. '
                             'EXAMPLE: \'amber14-all.xml\' \'amber14/tip3p.xml\'.')
    parser.add_argument('-out', '--output_directory', type=Path, required=False, default=Path.cwd(),
                        help='Path to a directory for saving simulation results, DEFAULT: current directory.')
    parser.add_argument('-sys', '--system_name', type=str, required=False, default='TEST',
                        help='Name of the simulaiton system to identify output files.')

    args = parser.parse_args()
    input_ = args.input_directory
    config = args.config_file
    plumed = args.plumed_file
    coordinates = args.pdb_file
    forcefield = args.forcefield
    output = args.output_directory
    name = args.system_name

    run_metadynamics_simulation(input_dir=input_, config_file=config, plumed_file=plumed, pdb_file=coordinates,
                                ff_list=forcefield, output_dir=output, system_name=name)
