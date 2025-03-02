from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging.config

import mdtraj as md
from pathsampling_utilities import PathsamplingUtilities
from tps_setup import TransitionPathSampling

import openpathsampling as paths
from openpathsampling.beta.hooks import GraciousKillHook, GraciousKillError
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage.collective_variables import MDTrajFunctionCV
from openpathsampling.experimental.storage import Storage

paths = monkey_patch_all(paths)
sns.set_theme(style='whitegrid', palette='deep')
sns.set_context(context='paper', font_scale=1.8)
logging.config.fileConfig(f'./logging.conf', disable_existing_loggers=False)


def run_ops(input_path=None, file_name=None, pdb_file=None, traj_file=None, cyc_no=None, ff_list=None,
            config_file=None, out_path=None, n_steps=None, run_id=None, walltime=None):

    # Storage
    paths.InterfaceSet.simstore = True
    if cyc_no:
        fname = Path(out_path / f'{file_name}_{run_id}_{str(cyc_no).zfill(2)}').with_suffix('.db')
    else:
        fname = Path(out_path / f'{file_name}_{run_id}').with_suffix('.db')
    storage = Storage(str(fname), 'w')

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

        anglelist = list()
        anglelist.append(np.hstack((
            topology.select('name N3 and resid 16'), topology.select('name H3 and resid 16'),
            topology.select('name N1 and resid 6'))))  # WC
        anglelist.append(np.hstack((
            topology.select('name N3 and resid 16'), topology.select('name H3 and resid 16'),
            topology.select('name N7 and resid 6'))))  # HG
        anglelist.append(np.vstack((np.hstack((topology.select('name N6 and resid 6'),
                                               topology.select('name H61 and resid 6'),
                                               topology.select('name O4 and resid 16'))),  # ON1
                                    np.hstack((topology.select('name N6 and resid 6'),
                                               topology.select('name H62 and resid 6'),
                                               topology.select('name O4 and resid 16'))))))  # ON2

        # Collective Variable
        d_WC = MDTrajFunctionCV(md.compute_distances, topology=ops_template.topology,
                                atom_pairs=[bondlist[0]]).named('d_WC')
        d_HG = MDTrajFunctionCV(md.compute_distances, topology=ops_template.topology,
                                atom_pairs=[bondlist[1]]).named('d_HG')
        d_BP = MDTrajFunctionCV(md.compute_distances, topology=ops_template.topology,
                                atom_pairs=[bondlist[2]]).named('d_BP')
        angle_WC = MDTrajFunctionCV(md.compute_angles, topology=ops_template.topology,
                                    angle_indices=[anglelist[0]]).named('angle_WC')
        angle_HG = MDTrajFunctionCV(md.compute_angles, topology=ops_template.topology,
                                    angle_indices=[anglelist[1]]).named('angle_HG')
        angles_ON = MDTrajFunctionCV(md.compute_angles, topology=ops_template.topology,
                                     angle_indices=anglelist[2]).named('angles_ON')

        # Additional CV: select the H atom of the amino group involved in the hydrogen bond DA7-N6 -- DT37-O4
        def get_angle_ON(snapshot, angles_ON_cv):
            import numpy as np
            return np.max(angles_ON_cv(snapshot))

        angle_ON = paths.FunctionCV("angle_ON", get_angle_ON, angles_ON_cv=angles_ON)

        # Volumes
        distarr = [0, 0.3]
        angarr = [155, 165, 180]
        deg = 180 / np.pi

        # Defining the stable states
        WC = (paths.CVDefinedVolume(d_WC, lambda_min=distarr[0], lambda_max=distarr[1]) &
              paths.CVDefinedVolume(d_BP, lambda_min=distarr[0], lambda_max=distarr[1]) &
              paths.CVDefinedVolume(angle_WC, lambda_min=angarr[1] / deg, lambda_max=angarr[2] / deg) &
              paths.CVDefinedVolume(angle_ON, lambda_min=angarr[1] / deg, lambda_max=angarr[2] / deg)).named("WC")

        HG = (paths.CVDefinedVolume(d_HG, lambda_min=distarr[0], lambda_max=distarr[1]) &
              paths.CVDefinedVolume(d_BP, lambda_min=distarr[0], lambda_max=distarr[1]) &
              paths.CVDefinedVolume(angle_HG, lambda_min=angarr[0] / deg, lambda_max=angarr[2] / deg) &
              paths.CVDefinedVolume(angle_ON, lambda_min=angarr[0] / deg, lambda_max=angarr[2] / deg)).named("HG")

        # Trajectory
        ops_trj = paths.engines.openmm.tools.trajectory_from_mdtraj(wc, velocities=utils.extract_velocities(traj_file))

        # Reaction network
        network = paths.TPSNetwork(initial_states=WC, final_states=HG).named('tps_network')
        # Move scheme
        scheme = paths.OneWayShootingMoveScheme(network=network,
                                                selector=paths.UniformSelector(),
                                                engine=md_engine).named('move_scheme')
    else:
        print('Continuing...')
        traj, cvs, network, _, template, scheme = utils.get_inputs(traj_file, cyc_no=cyc_no, input_path=input_path)
        ops_template = template
        d_WC = cvs['d_WC']
        d_HG = cvs['d_HG']
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
    print('Subtrjajectories')
    subtrajectories = []
    for ens in network.analysis_ensembles:
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

    # With long trajectories and slow IO, saving each step prevents data loss.
    sampler.save_frequency = 1

    # Monitor HPC cluster walltime, stop the simulation and close the storage before the job is killed.
    kill_hook = GraciousKillHook(walltime)
    sampler.attach_hook(kill_hook)
    try:
        sampler.run(n_steps)
    except GraciousKillError:
        print("TPS run timed out!")
        storage.close()

    print(storage.summary())
    storage.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory containing initial trajectory and configuration files.')
    parser.add_argument('-fn', '--filename', type=str, required=True,
                        help='Name for output files, identifying the molecular system.')
    parser.add_argument('-pdb', '--coordinates', type=str, required=False,
                        help='Name of the coordinate file in pdb format.')
    parser.add_argument('-tr', '--trajectory', type=str, required=True,
                        help='Name of the trajectory file in MDTraj-HDF5 format or of a TPS output file in SQL *.db '
                             'format.')
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
    parser.add_argument('-wt', '--walltime', type=str, required=True, default="4 days 12 hours",
                        help='Walltime for TPS run as str, e.g. \"23 hours 20 minutes\", ensures the output database to '
                             'be closed correctly if run on a cluster with max. runtime for jobs.\n '
                             'ATTENTION: Preprossessing steps are not included in the walltime.')
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
    WALLTIME = args.walltime  # -wt <walltime, str 'x days y hours z minutes'>

    run_ops(input_path=in_path, file_name=filename, pdb_file=pdbfile, traj_file=trajfile, cyc_no=cycno,
            ff_list=forcefieldlist, config_file=configfile, out_path=outpath, n_steps=nsteps, run_id=runid,
            walltime=WALLTIME)
