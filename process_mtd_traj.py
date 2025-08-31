import argparse
from pathlib import Path
import mdtraj as md
import numpy as np
import h5py

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', palette='deep')
sns.set_context(context='paper', font_scale=2.2)


def get_pair_list(topology):
    # atoms involved in distances
    # DA7 N1 -- DT37 N3
    # DA7 N7 -- DT37 N3
    # DA7 N6 -- DT37 O4
    # WC to HG is A7 -- T37,
    # T6 A38
    # A8 T36
    atomsWC = topology.select('name N1 and resid 6 or name N3 and resid 16')
    atomsHG = topology.select('name N7 and resid 6 or name N3 and resid 16')
    atomsBP = topology.select('name N6 and resid 6 or name O4 and resid 16')
    pairs = [atomsWC, atomsHG, atomsBP]
    # print(pairs)
    for pair in pairs:
        # print(pair)
        atom1 = topology.atom(pair[0])
        atom2 = topology.atom(pair[1])
        # print('''%s in %s%s -- %s in %s%s''' % (atom1.name, atom1.residue.index, atom1.residue.name,
        #                                         atom2.name, atom2.residue.index, atom2.residue.name))
    return pairs


def get_bp_distances(trajectory, pairs):
    # calculate distances and order parameter for stable state runs
    # WC
    bpdistwc = md.compute_distances(trajectory, atom_pairs=pairs, periodic=True)
    # print(bpdistwc)
    opwc = np.zeros(trajectory.n_frames)
    for f in range(trajectory.n_frames):
        opwc[f] = np.arctan2(bpdistwc[f, 0], bpdistwc[f, 1])
    return opwc


def concatenate_slices(slices, dir_path, out_path, out_name):
    if out_path is None:
        out_path = dir_path
    try:
        outfile = h5py.File(f'{out_path}/{out_name}.h5', 'a')
        for idx in range(len(slices)):
            with (h5py.File(slices[idx], "r") as trajectory):
                if idx == 0:
                    for key in trajectory.keys():
                        if key == 'topology':
                            data = trajectory[key]
                            new_set = outfile.create_dataset(key, data=data)
                            for attr in trajectory[key].attrs.keys():
                                new_set.attrs[attr] = trajectory[key].attrs[attr]
                        elif (key == 'cell_angles' or key == 'cell_lengths' or key == 'coordinates'
                              or key == 'velocities'):
                            data = trajectory[key]
                            maxshape = (None,) + data.shape[1:]
                            new_set = outfile.create_dataset(key, data=data, chunks=True, maxshape=maxshape)
                            for attr in trajectory[key].attrs.keys():
                                new_set.attrs[attr] = trajectory[key].attrs[attr]
                        else:
                            pass
                else:
                    for key in trajectory.keys():
                        if key != 'topology':
                            data = trajectory[key]
                            outfile[key].resize((outfile[key].shape[0] + data.shape[0]), axis=0)
                            outfile[key][-data.shape[0]:] = data

        outfile.close()
    except ValueError:
        print('File exists, choose new name or delete old file!')


def plot_traj(traj_file, dir_path, out_name, new_path, sys_name):
    if isinstance(traj_file, str) and dir_path:
        traj = md.load_hdf5(Path(dir_path / traj_file))
    else:
        traj = traj_file

    if not new_path:
        new_path = dir_path

    pairs = get_pair_list(traj.topology)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=300)
    if sys_name:
        ax.set(xlabel='Frames', ylabel=r'$\mathrm{distance\ (\AA)}$', title=sys_name)
    else:
        ax.set(xlabel='Frames', ylabel=r'$\mathrm{distance\ (\AA)}$')
    ax.plot(get_bp_distances(traj, pairs), color='b')
    ax.set_ylim((0.38, 1.20))
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig(new_path / f'{out_name}.png')


def slice_trajectory(filename, new_name, dir_path, new_path, start, stop, count, sys_name):
    if new_path is None:
        new_path = dir_path

    def slice_h5traj(whole_traj, sliced_traj, first_frame, last_frame) -> None:
        for key in whole_traj.keys():
            if key == 'topology':
                data = whole_traj[key]
                new_set = sliced_traj.create_dataset(key, data=data)
                for attr in whole_traj[key].attrs.keys():
                    new_set.attrs[attr] = whole_traj[key].attrs[attr]
            else:
                data = whole_traj[key][first_frame:last_frame]
                new_set = sliced_traj.create_dataset(key, data=data)
                for attr in whole_traj[key].attrs.keys():
                    new_set.attrs[attr] = whole_traj[key].attrs[attr]

    with h5py.File(Path(dir_path / f'{filename}.h5'), "r") as trajectory:
        if start is not None and stop is not None and not count:
            try:
                sliced_trajectory = h5py.File(Path(new_path / f'{new_name}.h5'), 'a')
                slice_h5traj(trajectory, sliced_trajectory, start, stop)
                plot_traj(sliced_trajectory, None, new_name, new_path, sys_name)
                sliced_trajectory.close()
            except ValueError:
                print('File exists, choose new name or delete old file!')
        elif count and start is None and stop is None:
            print(f'Slicing trajectory...')
            for cnt in range(0, count):
                start_frame = cnt * 10000
                stop_frame = (cnt + 1) * 10000
                print(f'Processing chunk #{cnt}...')
                try:
                    sliced_trajectory = h5py.File(Path(new_path / f'{new_name}_{str(cnt).zfill(2)}.h5'), 'a')
                    slice_h5traj(trajectory, sliced_trajectory, start_frame, stop_frame)
                    plot_traj(sliced_trajectory, None, f'{new_name}_{str(cnt).zfill(2)}',
                              new_path, sys_name)
                    sliced_trajectory.close()
                except ValueError:
                    print('File exists, choose new name or delete old file!')
        else:
            print("Error: Please provide either (start AND end) OR count, not both or neither.")


if __name__ == '__main__':
    # Main parser with shared arguments
    parser = argparse.ArgumentParser(
        description="Concatenate or slice trajectories in HDF5 format, or plot basepair distances.")

    # Shared arguments that all commands need
    parser.add_argument('-ip', '--in_path', type=Path, required=True,
                        help='Directory containing trajectory file(s).')
    parser.add_argument('-on', '--out_name', type=str, required=True,
                        help='Output file name (without extension).')
    parser.add_argument('-op', '--out_path', type=Path, required=False, default=None,
                        help='Directory to store output file(s). Default: same as input.')
    parser.add_argument('-sys', '--system_name', type=str, required=False, default=None,
                        help='System name for plot title. Default: empty.')

    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # Concatenate command
    concatenate_parser = subparsers.add_parser('concatenate',
                                               help='Concatenate multiple trajectory files.')
    concatenate_parser.add_argument('-fl', '--file_list', type=str, nargs='+', required=True,
                                    help='List of HDF5 trajectory files to concatenate.')

    # Slice command
    slice_parser = subparsers.add_parser('slice',
                                         help='Slice a trajectory file. Provide either start/end frames OR count.')
    slice_parser.add_argument('trajectory_file', type=str,
                              help='Name of the MDTraj HDF5 trajectory file to slice (without .h5 extension).')

    # Create mutually exclusive group for slice options
    slice_group = slice_parser.add_mutually_exclusive_group(required=True)
    slice_group.add_argument('--frames', nargs=2, type=int, metavar=('START', 'END'),
                             help='Start and end frame numbers for single slice.')
    slice_group.add_argument('-c', '--count', type=int,
                             help='Number of 10000-frame chunks to create.')

    # Plot command
    plot_parser = subparsers.add_parser('plot',
                                        help='Plot basepair distances of trajectory.')
    plot_parser.add_argument('trajectory_file', type=str,
                             help='Name of the MDTraj HDF5 trajectory file to plot (without .h5 extension).')

    args = parser.parse_args()

    # Execute based on command
    if args.command == 'concatenate':
        # Convert file list to full paths
        full_file_list = [str(args.in_path / f) if not f.endswith('.h5') else str(args.in_path / f)
                          for f in args.file_list]
        concatenate_slices(full_file_list, args.in_path, args.out_path, args.out_name)

    elif args.command == 'slice':
        if hasattr(args, 'frames') and args.frames:
            start, end = args.frames
            slice_trajectory(args.trajectory_file, args.out_name, args.in_path, args.out_path,
                             start, end, None, args.system_name)
        else:
            slice_trajectory(args.trajectory_file, args.out_name, args.in_path, args.out_path,
                             None, None, args.count, args.system_name)

    elif args.command == 'plot':
        plot_traj(args.trajectory_file + '.h5', args.in_path, args.out_name, args.out_path, args.system_name)

# Example usage:
# python script.py -ip /path/to/data -on output_name concatenate -fl file1.h5 file2.h5 file3.h5
# python script.py -ip /path/to/data -on sliced_traj slice trajectory_name --frames 1000 5000
# python script.py -ip /path/to/data -on sliced_traj slice trajectory_name -c 5
# python script.py -ip /path/to/data -on plot_output plot trajectory_name -sys "My System"