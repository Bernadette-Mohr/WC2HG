import sys
import os
libdir = os.path.expanduser('/home/jvanasselt/miniconda3/envs/tps/lib/python3.9/site-packages/')
sys.path.append(libdir)
from pathlib import Path
import argparse
from datetime import datetime
import mdtraj as md
import matplotlib.pyplot as plt
import logging.config

import openpathsampling as paths
from openpathsampling.engines import gromacs as ops_gmx
# from openpathsampling.engines.openmm.tools import ops_load_trajectory
# from openpathsampling.engines import MDTrajTopology


def get_inputs(input_path, grofile, mdpfile, topfile, trrfile):
    file_names = [grofile, mdpfile, topfile, trrfile]
    for idx, file_name in enumerate(file_names):
        if Path(file_name).is_file():
            pass
        else:
            file_names[idx] = str(input_path / file_name)

    return file_names


def run_ops(input_path, filename, grofile, mdpfile, topfile, trrfile, n_runs):

    grofile, mdpfile, topfile, trrfile = get_inputs(input_path, grofile, mdpfile, topfile, trrfile)

    cwd = Path().resolve()
    initial_traj = cwd / '000.trr'
    initial_traj.write_bytes(Path(trrfile).read_bytes())

    if Path(cwd / 'initial_frame.trr').is_file():
        Path(cwd / 'initial_frame.trr').unlink()

    # Engine
    options = {'gmx_executable': 'gmx -nobackup ',
               'snapshot_timestep': 0.5,
               'mdrun_args': ' -pin on -ntomp 32 -dlb yes -ntmpi 1 ',
               'base_dir': ".",
               'prefix': "md",
               'n_frames_max': 20000,
               'grompp_args': '-maxwarn 3'
               }

    md_engine = ops_gmx.Engine(gro=grofile,
                               mdp=mdpfile,
                               top=topfile,
                               options=options,
                               base_dir=".",
                               prefix="md").named("engine")

    wc = md.load(initial_traj, top=grofile)
    topology = wc.topology

    bondlist = list()
    bondlist.append(topology.select('name N1 and resid 6 or name N3 and resid 16'))  # WC
    bondlist.append(topology.select('name N7 and resid 6 or name N3 and resid 16'))  # HG
    bondlist.append(topology.select('name N6 and resid 6 or name O4 and resid 16'))  # BP

    ha = topology.select('name "H3" and resid 16')[0]

    # Collective Variable
    template = md_engine.current_snapshot
    d_WC = paths.MDTrajFunctionCV("d_WC", md.compute_distances, template.topology,
                                  atom_pairs=[bondlist[0]])
    d_HG = paths.MDTrajFunctionCV("d_HG", md.compute_distances, template.topology,
                                  atom_pairs=[bondlist[1]])
    d_BS = paths.MDTrajFunctionCV("d_BS", md.compute_distances, template.topology,
                                  atom_pairs=[bondlist[2]])

    a_hg = paths.MDTrajFunctionCV("a_hg", md.compute_angles, template.topology,
                                  angle_indices=[[ha] + bondlist[1]])
    a_wc = paths.MDTrajFunctionCV("a_wc", md.compute_angles, template.topology,
                                  angle_indices=[[ha] + bondlist[1]])

    # Volumes
    distarr2 = [0, 0.35]  # Hoeken weer toevoegen!

    # Defining the stable states
    WC = (
            paths.CVDefinedVolume(d_WC, lambda_min=distarr2[0], lambda_max=distarr2[1]) &
            paths.CVDefinedVolume(d_BS, lambda_min=distarr2[0], lambda_max=distarr2[1])
    ).named("WC")

    HG = (
            paths.CVDefinedVolume(d_HG, lambda_min=distarr2[0], lambda_max=distarr2[1]) &
            paths.CVDefinedVolume(d_BS, lambda_min=distarr2[0], lambda_max=distarr2[1])
    ).named("noWC")

    # initial trajectory (OPS/TPS can't handle pathlib path objects!)
    trajectory = paths.Trajectory([md_engine.read_frame_from_file(str(initial_traj), num)
                                   for num in range(len(wc.xyz))])

    # Reaction network
    network = paths.TPSNetwork(initial_states=WC, final_states=HG)

    # print(d_WC(trajectory))
    # print(d_HG(trajectory))
    print("Initial conformation")
    plt.plot(d_WC(trajectory), d_HG(trajectory))

    plt.xlabel("Hydrogen bond distance WC")
    plt.ylabel("Hydrogen bond distance HG")
    plt.title("Rotation")
    plt.savefig(cwd / f'{filename}_h-bond_distances_initial.png')

    # Ensembles
    subtrajectories = [network.analysis_ensembles[0].split(trajectory)]

    for subtrajectory in subtrajectories[0]:
        plt.plot(d_WC(subtrajectory), d_HG(subtrajectory),)

    plt.xlabel("Hydrogen bond distance WC")
    plt.ylabel("Hydrogen bond distance HG")
    plt.title("Rotation")
    plt.savefig(cwd / f'{filename}_h-bond_distances_subtrajectories.png')

    # Move scheme
    scheme = paths.OneWayShootingMoveScheme(network, selector=paths.UniformSelector(), engine=md_engine)

    # Initial conditions
    initial_conditions = scheme.initial_conditions_from_trajectories(subtrajectories[0][0])
    scheme.assert_initial_conditions(initial_conditions)

    # Storage
    current_datetime = datetime.now()
    suffix = 'GMX-test'
    fname = Path(cwd / f'{filename}_{suffix}').with_suffix('.nc')
    storage = paths.Storage(str(fname), "w", template)

    sampler = paths.PathSampling(storage, move_scheme=scheme, sample_set=initial_conditions)

    logging.config.fileConfig(f'/home/bmohr/Data/PycharmProjects/WC2HG/logging.conf', disable_existing_loggers=False)

    runs = n_runs
    sampler.run(runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='TODO')
    parser.add_argument('-fn', '--filename', type=str, required=True, help='Name for output files, should identify '
                                                                           'calculation.')
    parser.add_argument('-c', '--coordinates', type=str, required=True,
                        help='Name of the file with initial coordinates, e.g. *.gro, has to exist in the '
                             'input directory.')
    parser.add_argument('-mdp', '--parameters', type=str, required=True,
                        help='Name of the file with the run parameters, e.g. GROMACS *.mdp file')
    parser.add_argument('-top', '--topology', type=str, required=True, help='Name of the topology file, e.g. '
                                                                            'GROMACS *.top')
    parser.add_argument('-tr', '--trajectory', type=str, required=True, help='Name of the initial trajectory file to '
                                                                             'start the TPS calculations from.')
    parser.add_argument('-nr', '--n_steps', type=int, required=True, help='The number of desired TPS runs.')
    args = parser.parse_args()
    # -dir /media/bmohr/Backup/POSTDOC/WCHG/ZOE/DNAHG2MAT/inputs,
    # -c mdwater.gro,
    # -md md.mdp,
    # -top topol.top,
    # -tr combined.trr
    run_ops(args.directory, args.filename, args.coordinates, args.parameters, args.topology, args.trajectory,
            args.n_runs)
