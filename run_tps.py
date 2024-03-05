import sys
import os
# libdir = os.path.expanduser('/home/bmohr98/micromamba/envs/ops/lib/python3.9/site-packages/')
# sys.path.append(libdir)
from pathlib import Path
import argparse
import time
from memory_profiler import profile
import matplotlib.pyplot as plt
import seaborn as sns
import logging.config
from multiprocessing import Process
from tqdm.auto import tqdm

import mdtraj as md
from pathsampling_utilities import PathsamplingUtilities
from tps_setup import TransitionPathSampling

import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage.collective_variables import MDTrajFunctionCV
from openpathsampling.experimental.storage import Storage

paths = monkey_patch_all(paths)
sns.set(style='white', palette='muted', context='paper', color_codes=True, font_scale=1.8, rc={'text.usetex': False})
# logging.config.fileConfig(f'logging.conf', disable_existing_loggers=False)


# from openpathsampling.experimental.simstore import StorableObject, StorableNamedObject, StorableObjectStore

# decorator specifying the function to monitor the memory usage of (in this case, the function is run_ops)
@profile
def run_ops(input_path=None, file_name=None, pdb_file=None, traj_file=None, cyc_no=None, ff_list=None,
            config_file=None, out_path=None, n_steps=None, run_id=None, walltime=None):
    start_time = time.time()  # Time at the start of this process

    # Storage
    paths.InterfaceSet.simstore = True
    if cyc_no:
        fname = Path(out_path / f'{file_name}_{run_id}_{str(cyc_no).zfill(2)}').with_suffix('.db')
    else:
        fname = Path(out_path / f'{file_name}_{run_id}').with_suffix('.db')
    storage = Storage(str(fname), 'w')

    # Monitor elapsed time, close storage if walltime is exceeded
    runtime = True
    while runtime:
        elapsed_time = time.time() - start_time
        if elapsed_time > walltime:
            storage.close()
            runtime = False

        utils = PathsamplingUtilities()
        if config_file and not cyc_no:
            traj_file, config_file, pdb_file = utils.get_inputs(traj_file, config_file, pdb_file, input_path=input_path)
            configs = utils.get_configs(config_file)
            setup = TransitionPathSampling(configs=configs, forcefield_list=ff_list, pdb_file=pdb_file,
                                           system_name=run_id, output_path=out_path)
            md_engine = setup.md_engine
            ops_template = setup.ops_template

            print('Loading trajectory with MDTraj')
            # load initial trajectory from file
            wc = md.load_hdf5(traj_file)
            topology = wc.topology

            bondlist = list()
            bondlist.append(topology.select('name N1 and resid 6 or name N3 and resid 16'))  # WC
            bondlist.append(topology.select('name N7 and resid 6 or name N3 and resid 16'))  # HG
            bondlist.append(topology.select('name N6 and resid 6 or name O4 and resid 16'))  # BP

            # ha = topology.select('name "H3" and resid 16')[0]

            # Collective Variable
            d_WC = MDTrajFunctionCV(md.compute_distances, topology=ops_template.topology,
                                    atom_pairs=[bondlist[0]]).named('d_WC')
            d_HG = MDTrajFunctionCV(md.compute_distances, topology=ops_template.topology,
                                    atom_pairs=[bondlist[1]]).named('d_HG')
            d_BP = MDTrajFunctionCV(md.compute_distances, topology=ops_template.topology,
                                    atom_pairs=[bondlist[2]]).named('d_BP')

            # Volumes
            distarr = [0, 0.35]

            # Defining the stable states
            WC = (paths.CVDefinedVolume(d_WC, lambda_min=distarr[0], lambda_max=distarr[1]) &
                  paths.CVDefinedVolume(d_BP, lambda_min=distarr[0], lambda_max=distarr[1])).named("WC")

            HG = (paths.CVDefinedVolume(d_HG, lambda_min=distarr[0], lambda_max=distarr[1]) &
                  paths.CVDefinedVolume(d_BP, lambda_min=distarr[0], lambda_max=distarr[1])).named("noWC")

            # Trajectory
            ops_trj = paths.engines.openmm.tools.trajectory_from_mdtraj(wc,
                                                                        velocities=utils.extract_velocities(traj_file))

            # Reaction network
            network = paths.TPSNetwork(initial_states=WC, final_states=HG).named('tps_network')
            # Move scheme
            scheme = paths.OneWayShootingMoveScheme(network=network,
                                                    selector=paths.UniformSelector(),
                                                    engine=md_engine).named('move_scheme')
        else:
            traj, cvs, network, template, scheme = utils.get_inputs(traj_file, cyc_no=cyc_no, input_path=input_path)
            ops_template = template
            d_WC = cvs['d_WC']
            d_HG = cvs['d_HG']
            # d_BP = cvs['d_BP']
            ops_trj = traj
            network = network
            scheme = scheme

        print("Initial conformation")
        
        plt.plot(d_WC(ops_trj), d_HG(ops_trj), 'k.', label='Stable states')

        plt.xlabel("Hydrogen bond distance WC")
        plt.ylabel("Hydrogen bond distance HG")
        plt.title("Rotation")
        plt.tight_layout()
        if cyc_no:
            plt.savefig(out_path / f'{file_name}_{run_id}_{str(cyc_no).zfill(2)}_h-bond_distances_initial.pdf', dpi=300)
        else:
            plt.savefig(out_path / f'{file_name}_{run_id}_h-bond_distances_initial.pdf', dpi=300)

        # Ensembles
        print('Subtrajectories')
        subtrajectories = []
        n_ensembles = len(network.analysis_ensembles)
        for ens in tqdm(network.analysis_ensembles, total=n_ensembles):
            subtrajectories += ens.split(ops_trj)

        plt.plot(d_WC(subtrajectories[0]), d_HG(subtrajectories[0]), color='r', label='State transitions')

        plt.xlabel("Hydrogen bond distance WC")
        plt.ylabel("Hydrogen bond distance HG")
        plt.title("Rotation")
        plt.tight_layout()
        if cyc_no:
            plt.savefig(out_path / f'{file_name}_{run_id}_{str(cyc_no).zfill(2)}_h-bond_distances_subtrajectories.pdf',
                        dpi=300)
        else:
            plt.savefig(out_path / f'{file_name}_{run_id}_h-bond_distances_subtrajectories.pdf', dpi=300)

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

        sampler.save_frequency = 2
        sampler.run(n_steps)

        print(storage.summary())
        storage.close()
        runtime = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory containing initial trajectory and configuration files.')
    parser.add_argument('-fn', '--filename', type=str, required=True,
                        help='Name for output files, identifying the molecular system.')
    parser.add_argument('-pdb', '--coordinates', type=str, required=False,
                        help='Name of the coordinate file in pdb format.')
    parser.add_argument('-tr', '--trajectory', type=str, required=True,
                        help='Name of the trajectory file in MDTraj-HDF5 format.')
    parser.add_argument('-cyc', '--cycle_number', type=int, required=False,
                        help='In case of continuation run, provide number of cycle to start from in a TPS output file '
                             'in SQL *.db format.')
    parser.add_argument('-ff', '--forcefield', nargs='+', type=str, required=False,
                        help='List of force field files to be used for the simulation. '
                             'EXAMPLE: amber14-all.xml amber14/tip3p.xml.')
    parser.add_argument('-cfg', '--config_file', type=str, required=False,
                        help='File in python configparser format with simulation settings.')
    parser.add_argument('-out', '--output_path', type=Path, required=True,
                        help='Directory for storing TPS output files.')
    parser.add_argument('-nr', '--n_steps', type=int, required=True,
                        help='The number of desired TPS MC cycles.')
    parser.add_argument('-id', '--run_id', type=str, required=False, default='TEST',
                        help='Id to correlate storage and trajectories of a specific run.')
    parser.add_argument('-t', '--time', type=int, required=True, default=3600,
                        help='Walltime for entire TPS run in seconds, ensures the output database to be closed '
                             'correctly if run on a cluster with max. runtime for jobs.')
    args = parser.parse_args()

    in_path = args.directory  # -dir <input/directory/path>
    filename = args.filename  # -fn <filename>
    pdbfile = args.coordinates  # -pdb <initial_configuration.pdb>
    trajfile = args.trajectory  # -tr <mtd_trajectory.h5> or <TPS_output.db>
    cycno = args.cycle_number  # -cyc <number of cycle from previous TPS run to serve as initial trajectory>
    forcefieldlist = args.forcefield  # -ff List of force field files in OMM format to use for simulations
    configfile = args.config_file  # -cfg <config_file.config>
    outpath = args.output_path  # -out </path/to/output/directory>
    nsteps = args.n_steps  # -nr <number of runs>
    runid = args.run_id  # -id <which TPS run>
    WALLTIME = args.time  # -t <walltime in seconds>

    process = Process(target=run_ops, args=(in_path, filename, pdbfile, trajfile, cycno, forcefieldlist, configfile,
                                            outpath, nsteps, runid, WALLTIME))
    process.start()

    process.join(WALLTIME)

    if process.is_alive():
        print('TPS run timed out!')
        process.join(120)  # wait 2 minutes for the process to finish closing storage file
        process.terminate()
