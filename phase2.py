import sys
import os
# libdir = os.path.expanduser('/home/bmohr98/micromamba/envs/ops/lib/python3.9/site-packages/')
# sys.path.append(libdir)
from pathlib import Path
import argparse
import h5py
# from datetime import datetime
import matplotlib.pyplot as plt
import logging.config
from multiprocessing import Process
import time

from openmm import app
import openmm as mm
import openmm.unit as unit
# import openmmtools
from openmmtools.integrators import VVVRIntegrator

import openpathsampling as paths
import openpathsampling.engines.openmm as ops_openmm
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage.collective_variables import MDTrajFunctionCV
from openpathsampling.experimental.storage import Storage


paths = monkey_patch_all(paths)
import mdtraj as md


def get_inputs(input_path, pdb_file, traj_file):
    file_names = [pdb_file, traj_file]
    for idx, file_name in enumerate(file_names):
        try:
            if Path(file_name).is_file():
                pass
            elif Path(input_path / file_name).is_file():
                file_names[idx] = str(input_path / file_name)
        except FileNotFoundError:
            print(f'File {file_names[idx]} not found!')

    return file_names


def extract_velocities(file_name, start=0, stop=None):
    with h5py.File(file_name, "r") as file_:
        velocities = file_['velocities'][()]  # returns data as a numpy array
        if stop:
            velocities = velocities[start:stop]
        else:
            velocities = velocities[start:]

    return velocities


def run_ops(input_path, file_name, pdb_file, traj_file, out_path, n_steps, run_id, walltime):
    start_time = time.time()  # Time at the start of this process

    # Storage
    paths.InterfaceSet.simstore = True
    fname = Path(out_path / f'{file_name}_{run_id}').with_suffix('.db')
    storage = Storage(str(fname), 'w')

    # Monitor elapsed time, close storage if walltime is exceeded
    runtime = True
    while runtime:
        elapsed_time = time.time() - start_time
        if elapsed_time > walltime:
            storage.close()
            runtime = False

        pdb_file, traj_file = get_inputs(input_path, pdb_file, traj_file)

        # cwd = Path().resolve()
        initial_traj = out_path / '000.h5'
        initial_traj.write_bytes(Path(traj_file).read_bytes())

        # OpenMM pdb file object
        pdb = app.PDBFile(pdb_file)

        # OpenMM Engine setup
        forcefield = app.ForceField('amber/protein.ff14SB.xml', 'amber/DNA.bsc1.xml', 'amber/tip3p_standard.xml')
        system = forcefield.createSystem(topology=pdb.topology,
                                         nonbondedMethod=app.PME,
                                         nonbondedCutoff=1.1 * unit.nanometer,
                                         constraints=app.HBonds,
                                         rigidWater=True,
                                         ewaldErrorTolerance=1e-05)
        system.addForce(mm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin, 25))

        integrator = VVVRIntegrator(
            300 * unit.kelvin,  # Temperature of heat bath
            5.0 / unit.picoseconds,  # Friction coefficient
            0.002 * unit.picoseconds  # Time step
        )
        integrator.setConstraintTolerance(0.00001)

        print('Loading trajectory with MDTraj')
        # load metadynamics trajectory from file
        wc = md.load_hdf5(initial_traj)
        topology = wc.topology
        # openmm_top = MDTrajTopology(topology)

        openmm_properties = {'Precision': 'mixed'}  # 'DeviceIndex': '0'
        ops_options = {
            'n_steps_per_frame': 2500,  # number of internal time steps per saved frame (snapshot)
            'n_frames_max': 4000  # maximum number of frames to store in the snapshot engine
        }

        # OPS snapshot file (Not equivalent to OpenMM PDBFile!?)
        ops_template = ops_openmm.snapshot_from_pdb(pdb_file=pdb_file)
        md_engine = ops_openmm.Engine(
            topology=ops_template.topology,
            system=system,
            integrator=integrator,
            openmm_properties=openmm_properties,
            options=ops_options
        ).named('OMM_engine')
        # md_engine._simulation = mm.app.Simulation(pdb.topology, system, integrator)
        # md_engine._simulation.context.setPositions(pdb.positions)
        md_engine.initialize(mm.openmm.Platform.getPlatformByName('CUDA'))
        md_engine.current_snapshot = ops_template

        bondlist = list()
        bondlist.append(topology.select('name N1 and resid 6 or name N3 and resid 16'))  # WC
        bondlist.append(topology.select('name N7 and resid 6 or name N3 and resid 16'))  # HG
        bondlist.append(topology.select('name N6 and resid 6 or name O4 and resid 16'))  # BP

        ha = topology.select('name "H3" and resid 16')[0]

        # Collective Variable
        d_WC = MDTrajFunctionCV(md.compute_distances, topology=ops_template.topology, atom_pairs=[bondlist[0]]).named(
            'd_WC')
        d_HG = MDTrajFunctionCV(md.compute_distances, topology=ops_template.topology, atom_pairs=[bondlist[1]]).named(
            'd_HG')
        d_BP = MDTrajFunctionCV(md.compute_distances, topology=ops_template.topology, atom_pairs=[bondlist[2]]).named(
            'd_BP')

        # a_hg = MDTrajFunctionCV("a_hg", md.compute_angles, template.topology,
        #                         angle_indices=[[ha] + bondlist[1]])
        # a_wc = MDTrajFunctionCV("a_wc", md.compute_angles, template.topology,
        #                         angle_indices=[[ha] + bondlist[1]])

        # Volumes
        distarr2 = [0, 0.35]  # Hoeken weer toevoegen!

        # Defining the stable states
        WC = (
                paths.CVDefinedVolume(d_WC, lambda_min=distarr2[0], lambda_max=distarr2[1]) &
                paths.CVDefinedVolume(d_BP, lambda_min=distarr2[0], lambda_max=distarr2[1])
        ).named("WC")

        HG = (
                paths.CVDefinedVolume(d_HG, lambda_min=distarr2[0], lambda_max=distarr2[1]) &
                paths.CVDefinedVolume(d_BP, lambda_min=distarr2[0], lambda_max=distarr2[1])
        ).named("noWC")

        # Trajectory
        ops_trj = paths.engines.openmm.tools.trajectory_from_mdtraj(wc, velocities=extract_velocities(traj_file))

        # Reaction network
        network = paths.TPSNetwork(initial_states=WC, final_states=HG).named('tps_network')

        print("Initial conformation")
        plt.plot(d_WC(ops_trj), d_HG(ops_trj), 'k.', label='Stable states')
        plt.legend()
        plt.xlabel("Hydrogen bond distance WC")
        plt.ylabel("Hydrogen bond distance HG")
        plt.title("Rotation")
        plt.savefig(out_path / f'{file_name}_{run_id}_h-bond_distances_initial.pdf', dpi=300)

        # Emsembles
        print('Subtrjajectories')
        subtrajectories = []
        for ens in network.analysis_ensembles:
            subtrajectories += ens.split(ops_trj)

        # for subtrajectory in subtrajectories[0]:
        plt.plot(d_WC(subtrajectories[0]), d_HG(subtrajectories[0]), color='r', label='State transitions')
        plt.legend()
        plt.xlabel("Hydrogen bond distance WC")
        plt.ylabel("Hydrogen bond distance HG")
        plt.title("Rotation")
        plt.savefig(out_path / f'{file_name}_{run_id}_h-bond_distances_subtrajectories.pdf', dpi=300)

        # Move scheme
        scheme = paths.OneWayShootingMoveScheme(network=network,
                                                selector=paths.UniformSelector(),
                                                engine=md_engine).named('move_scheme')

        # Initial conditions
        initial_conditions = scheme.initial_conditions_from_trajectories(subtrajectories)
        scheme.assert_initial_conditions(initial_conditions)

        print('Start TPS production run')

        # Storage
        storage.save(ops_template)
        storage.save(ops_trj)
        storage.save(initial_conditions)

        sampler = paths.PathSampling(storage=storage,
                                     move_scheme=scheme,
                                     sample_set=initial_conditions).named('TPS_sampler')

        logging.config.fileConfig(f'logging.conf', disable_existing_loggers=False)

        sampler.save_frequency = 10
        sampler.run(n_steps)

        print(storage.summary())
        storage.close()
        runtime = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='TODO')
    parser.add_argument('-fn', '--filename', type=str, required=True,
                        help='Name for output files, identifying the molecular system.')
    parser.add_argument('-pdb', '--coordinates', type=str, required=True,
                        help='Name of the coordinate file in pdb format.')
    parser.add_argument('-tr', '--trajectory', type=str, required=True,
                        help='Name of the trajectory file in MDTraj-HDF5 format.')
    parser.add_argument('-out', '--output_path', type=Path, required=True,
                        help='Name of the target directory for TPS run outputs.')
    parser.add_argument('-nr', '--n_steps', type=int, required=True, help='The number of desired TPS runs.')
    parser.add_argument('-id', '--run_id', type=str, required=False, default='test',
                        help='Id to correlate storage and trajectories of a specific run.')
    parser.add_argument('-t', '--time', type=int, required=True, default=3600, help='Walltime for TPS run in seconds.')
    args = parser.parse_args()

    in_path = args.directory  # -dir <input/directory/path>
    filename = args.filename  # -fn <filename>
    pdbfile = args.coordinates  # -pdb <initial_configuration.pdb>
    trajfile = args.trajectory  # -tr <mtd_trajectory.h5>
    outpath = args.output_path  # -out </path/to/output/directory>
    nsteps = args.n_steps  # -nr <number of runs>
    runid = args.run_id  # -id <which TPS run>
    WALLTIME = args.time  # -t <walltime in seconds>

    process = Process(target=run_ops, args=(in_path, filename, pdbfile, trajfile, outpath, nsteps, runid, WALLTIME))
    process.start()

    process.join(WALLTIME)

    if process.is_alive():
        print('TPS run timed out!')
        process.join(120)  # wait 2 minutes for the process to finish closing storage file
        process.terminate()
