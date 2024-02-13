import argparse
from pathlib import Path
import pandas as pd
import MDAnalysis
from MDAnalysis import transformations as trans
from MDAnalysis.analysis import distances
import MDAnalysis.analysis.rms as rms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mda = MDAnalysis
sns.set(style='whitegrid', palette='deep', context='paper', font_scale=1.8)


def load_universe(dir_path, traj_file, coord_file, clean):
    u = mda.Universe(str(dir_path / coord_file), str(dir_path / traj_file), dt=10.0)
    if clean:
        complex_ = u.select_atoms('protein or nucleic')
        solvent = u.select_atoms('resname SOL or resname NA or resname CL')
        workflow = [trans.unwrap(u.atoms),  # unwrap all fragments
                    trans.center_in_box(complex_,  # move atoms so complex
                                        center='geometry'),  # is centered
                    trans.wrap(solvent,  # wrap solvent back into box
                               compound='residues'),  # keep each solvent molecule whole
                    trans.fit_rot_trans(complex_,  # align complex to first frame
                                        complex_,
                                        weights='mass'),
                    ]
        u.trajectory.add_transformations(*workflow)

    return u


def plot_distances(dir_path, universe):
    protein = universe.select_atoms('resid 132-189')
    dna = universe.select_atoms('resid 1-42')
    prot_com = protein.center_of_mass(compound='residues')
    dna_com = dna.center_of_mass(compound='residues')

    res_dist = distances.distance_array(dna_com,  # reference
                                        prot_com,  # configuration
                                        box=universe.dimensions)
    print(res_dist.shape)

    fig2, ax2 = plt.subplots()
    im = ax2.imshow(res_dist, origin='upper')

    # add residue ID labels to axes
    tick_interval = 5
    ax2.set_yticks(np.arange(len(dna_com))[::tick_interval])
    ax2.set_xticks(np.arange(len(prot_com))[::tick_interval])
    ax2.set_yticklabels(dna.residues.resids[::tick_interval])
    ax2.set_xticklabels(protein.residues.resids[::tick_interval])

    # add figure labels and titles
    plt.ylabel('DNA')
    plt.xlabel(r'$\mathrm{MAT}\alpha\mathrm{2}$')
    plt.title('Distance residue COM')

    # colorbar
    cbar2 = fig2.colorbar(im)
    cbar2.ax.set_ylabel('Distance (Angstrom)')
    plt.tight_layout()
    plt.savefig(dir_path / 'DNAHG2MAT_residue_com_distance.png', dpi=300)


def plot_rmsd(dir_path, universe):
    R = rms.RMSD(universe, universe,
                 select='backbone or nucleicbackbone',  # superimpose on whole backbone of the whole complex
                 groupselections=['nucleicbackbone and resid 1-42',  # DNA
                                  'backbone and index 699-1728',  # B chain
                                  'backbone and index 1729-2739'],  # D chain
                 ref_frame=0)  # frame index of the reference
    R.run()
    df = pd.DataFrame(R.results.rmsd,
                      columns=['Frame', 'Time (ps)', 'Complex', 'DNA', 'B-Chain', 'D-Chain'])
    fig, ax = plt.subplots(figsize=(16, 9))
    df.plot(x='Time (ps)', y=['Complex', 'DNA', 'B-Chain', 'D-Chain'], kind='line', alpha=0.5, ax=ax)
    ax.set_ylabel(r'$\mathrm{RMSD}\ (\AA)$')
    plt.tight_layout()
    # plt.show()
    plt.savefig(dir_path / 'DNAHG2MAT_rmsd.png', dpi=300)


def analyze_trajectory(dir_path, traj_file, coord_file, clean):
    universe = load_universe(dir_path, traj_file, coord_file, clean)
    # plot_distances(dir_path, universe)
    plot_rmsd(dir_path, universe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analyze MD trajectories of a complex_-DNA complex.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory containing the GROMACS trajectory.')
    parser.add_argument('-t', '--trajectory', type=str, required=True,
                        help='Filename of the GROMACS trajectory.')
    parser.add_argument('-c', '--coordinates', type=str, required=True,
                        help='Filename of the GROMACS coordinate file.')
    parser.add_argument('-clean', type=bool, required=False, default=False,
                        help='Center system around molecule of interest, remove periodic boundary conditions.')

    args = parser.parse_args()
    dir_ = args.directory
    traj = args.trajectory
    coord = args.coordinates
    cl_sys = args.clean

    analyze_trajectory(dir_, traj, coord, cl_sys)
