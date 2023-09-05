import sys
import os
import argparse
from pathlib import Path
from multiprocessing import Process
import time
from openmm import app
from openmmplumed import PlumedForce
import mdtraj as md
from pathsampling_utilities import PathsamplingUtilities
from metadynamics_setup import MetadynamicsSimulation

libdir = os.path.expanduser('/home/bmohr98/micromamba/envs/ops/lib/python3.9/site-packages/')
sys.path.append(libdir)


def run_metadynamics_simulation(input_path=None, config_file=None, plumed_file=None, pdb_file=None,
                                ff_list=None, system_name=None, output_dir=None, walltime=None):
    start_time = time.time()  # Time at the start of this process

    # Monitor elapsed time, close storage files if walltime is exceeded
    runtime = True
    while runtime:
        elapsed_time = time.time() - start_time
        if elapsed_time > walltime:
            trajectoryReporter.close()
            simulation.saveState(f'{output_dir}/{system_name}_final_state.xml')
            runtime = False

        # Prepare the Simulation
        print('Building system...')
        utils: PathsamplingUtilities = PathsamplingUtilities()
        config_file, plumed_file, pdb_file = utils.get_inputs(config_file, plumed_file, pdb_file,
                                                              input_path=input_path)
        configs = utils.get_configs(config_file)

        setup = MetadynamicsSimulation(configs=configs, forcefield_list=ff_list, pdb_file=pdb_file,
                                       system_name=system_name, output_path=output_dir)
        simulation = setup.setup_simulation()
        system = setup.system

        # Initializing reporters
        dataReporter = app.StateDataReporter(f'{output_dir}/{system_name}_data.csv', setup.reportInterval,
                                             totalSteps=setup.steps, time=True, speed=True, progress=True,
                                             elapsedTime=True, remainingTime=True, potentialEnergy=True,
                                             kineticEnergy=True,
                                             totalEnergy=True, temperature=True, volume=True, density=True,
                                             separator=';')
        statusReporter = app.StateDataReporter(sys.stdout, 1000, totalSteps=setup.steps, time=True, speed=True,
                                               progress=True, elapsedTime=True, remainingTime=True, separator=';')
        trajectoryReporter = md.reporters.HDF5Reporter(f'{output_dir}/{system_name}_trajectory.h5',
                                                       setup.reportInterval,
                                                       coordinates=True, cell=True, velocities=True)

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

        simulation.reporters.append(dataReporter)
        simulation.reporters.append(statusReporter)
        simulation.reporters.append(trajectoryReporter)
        simulation.currentStep = setup.currentStep
        simulation.step(setup.steps)

        trajectoryReporter.close()

        simulation.saveState(f'{output_dir}/{system_name}_final_state.xml')
        runtime = False


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
    parser.add_argument('-t', '--time', type=int, required=True, default=3600, help='Walltime for MTD run in seconds.')

    args = parser.parse_args()
    input_dir = args.input_directory
    config = args.config_file
    plumed = args.plumed_file
    coordinates = args.pdb_file
    forcefields = args.forcefield
    output = args.output_directory
    name = args.system_name
    WALLTIME = args.time

    process = Process(target=run_metadynamics_simulation,
                      args=(input_dir, config, plumed, coordinates, forcefields, name, output, WALLTIME))
    process.start()

    process.join(WALLTIME)

    if process.is_alive():
        print('MTD run timed out!')
        process.join(120)  # Wait 2 minutes for the process to finish closing storage files
        process.terminate()
