import argparse
from pathlib import Path
import MDAnalysis as mda
import numpy as np


def translate_protein_positions(dirpath, filename):
    u = mda.Universe(dirpath / filename)
    # select COM of the DNA molecule as reference point
    dna = u.select_atoms('nucleic')
    dna_com = dna.center_of_mass()

    # select everything 'protein' in the coordinate file and get the subunit IDs
    chains = set(u.select_atoms('protein').segids)

    for chain in chains:
        # get the COM of the protein subunit
        chain_atoms = u.select_atoms(f'protein and segid {chain}')
        chain_com = chain_atoms.center_of_mass()
        # calculate the vector from the DNA COM to the protein COM
        vector = np.subtract(chain_com, dna_com)
        # normalize to unit vector
        # (norm of vector is never going to be zero, two MD system components can not occupy the same space)
        unit = vector / np.linalg.norm(vector)
        # translate the protein subunit 2 nm (20 Angstrom) away from the DNA molecule
        chain_atoms.positions = chain_atoms.positions + unit * 20

    u.atoms.write(dirpath / f'{filename.split("2")[0]}_{filename.split("2")[1]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TODO')
    parser.add_argument('-dir', '--directory', type=Path, required=True)
    parser.add_argument('-f', '--filename', type=str, required=True)

    args = parser.parse_args()

    translate_protein_positions(args.directory, args.filename)
