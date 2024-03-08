from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
import mdtraj as md
import numpy as np
import h5py
from openpathsampling.engines.openmm.tools import trajectory_from_mdtraj

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', palette='deep')
sns.set_context(context='paper', font_scale=2.2)


def get_pair_list(topology):
    #atoms involved in distances
    #DA7 N1 -- DT37 N3
    #DA7 N7 -- DT37 N3
    #DA7 N6 -- DT37 O4
    #WC to HG is A7 -- T37,
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
    #calculate distances and order parameter for stable state runs
    #WC
    bpdistwc = md.compute_distances(trajectory, atom_pairs=pairs, periodic=True)
    # print(bpdistwc)
    opwc=np.zeros(trajectory.n_frames)
    for f in range(trajectory.n_frames):
        opwc[f] = np.arctan2(bpdistwc[f, 0], bpdistwc[f, 1])
    return opwc


def concatenate_slices(slices, path, out_name):
    try:
        outfile = h5py.File(f'{path}/{out_name}.h5', 'a')
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


def slice_trajectory(filename, new_name, start=None, stop=None):
    try:
        sliced_trajectory = h5py.File(new_name, 'a')
        with h5py.File(filename, "r") as trajectory:

            for key in trajectory.keys():
                if key == 'topology':
                    data = trajectory[key]
                    new_set = sliced_trajectory.create_dataset(key, data=data)
                    for attr in trajectory[key].attrs.keys():
                        new_set.attrs[attr] = trajectory[key].attrs[attr]
                else:
                    data = trajectory[key][start:stop]
                    new_set = sliced_trajectory.create_dataset(key, data=data)
                    for attr in trajectory[key].attrs.keys():
                        new_set.attrs[attr] = trajectory[key].attrs[attr]
        sliced_trajectory.close()
    except ValueError:
        print('File exists, choose new name or delete old file!')


def plot_traj(path, out_name):
    traj = md.load_hdf5(f'{path}/{out_name}.h5')
    pairs = get_pair_list(traj.topology)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=300)
    ax.set(xlabel='Frames', ylabel=r'$\mathrm{distance\ (\AA)}$', title=f'{path.parts[0]}')
    ax.plot(get_bp_distances(traj, pairs), color='b')
    ax.set_ylim([0.38, 1.20])
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig(dir_path / f'{out_name}.png')


if __name__ == '__main__':
    dir_path = Path('DNAWC2MAT/result-101893')
    # slice_list = [Path(dir_path / 'DNAWC2MAT_trajectory_15.h5'), Path(dir_path / 'DNAWC2MAT_trajectory_16.h5'),
    #               Path(dir_path / 'DNAWC2MAT_trajectory_17.h5')]
    # concatenate_slices(slice_list, dir_path, f'DNAWC2MAT_transition_100fs')
    slice_trajectory(Path(dir_path / 'DNAWC2MAT_transition_100fs.h5'),
                     Path(dir_path / 'DNAWC2MAT_transition_100fs_short.h5'), start=4000, stop=15000)
    plot_traj(dir_path, 'DNAWC2MAT_transition_100fs_short')
