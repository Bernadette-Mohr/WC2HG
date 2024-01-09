import sys
from pathlib import Path
import argparse
from tqdm.auto import tqdm
import mdtraj as md
import numpy as np
import pickle
from contact_counts import ContactCount
from pathsampling_utilities import PathsamplingUtilities


# Load your trajectories
def load_traj(dir_path, dir_name):
    for traj_dir in sorted(dir_path.glob(f'{dir_name}*')):
        try:
            traj = md.load(str(traj_dir / f'short.xtc'), top=traj_dir / 'dry.gro').remove_solvent()
            yield traj
        except FileNotFoundError:
            print(f'No trajectory found for {traj_dir}')
            continue


def get_residues(traj, atom_idxs):
    residues = dict()
    for atom_idx in atom_idxs:
        residue = traj.topology.atom(atom_idx).residue
        residues[f'{residue.name}{residue.resSeq}'] = residue

    return residues


def get_atoms_per_residue(traj, protein=False, chain=None, pairing_idx=None, single_residue=None, major_acceptors=None):
    if protein:
        residues = get_residues(traj, traj.topology.select('protein'))
        if single_residue:
            residues = {key: value for (key, value) in residues.items() if key == single_residue}
        protein_dict = dict()
        protein_idx = list()
        for resname, residue in residues.items():
            atoms = list()
            for atom in residue.atoms:
                if atom.element.symbol == 'H' or atom.element.symbol == 'N':
                    atoms.append(atom.name)
                    # HG: 1698, WC: 1730; B-chain <, D-chain >=
                if chain == 'B':
                    if (atom.element.symbol == 'H' or atom.element.symbol == 'N') and atom.index < pairing_idx:
                        protein_idx.append(atom.index)
                else:
                    if (atom.element.symbol == 'H' or atom.element.symbol == 'N') and atom.index >= pairing_idx:
                        protein_idx.append(atom.index)
            protein_dict[resname] = atoms

        return protein_dict, protein_idx

    else:
        residues = get_residues(traj, traj.topology.select('not protein'))
        if major_acceptors:
            residues = {key: value for (key, value) in residues.items() if key in major_acceptors}
        nucleic_dict = dict()
        for resname, residue in residues.items():
            atoms = list()
            for atom in residue.atoms:
                if atom.element.symbol == 'O' or atom.element.symbol == 'N':
                    atoms.append(atom.name)
            nucleic_dict[resname] = atoms

        return nucleic_dict


def generate_contact_map(dir_path, dir_name, out_dir, out_name, configs):
    utils = PathsamplingUtilities()
    configs = utils.get_configs(configs)
    single_residue = configs['SELECTION'].get('residue_key')
    major_acceptors = configs['SELECTION'].get('major_acceptor_keys')
    if major_acceptors:
        major_acceptors = major_acceptors.split(',')
    atom_idx = configs['SELECTION'].getint('atom')
    chain = configs['SELECTION'].get('chain')
    d0 = configs['PARAMETERS'].getfloat('d0')
    r0 = configs['PARAMETERS'].getfloat('r0')
    nn = configs['PARAMETERS'].getint('nn')
    mm = configs['PARAMETERS'].getint('mm')
    trajs = load_traj(dir_path, dir_name)
    distance_matrix = None
    contact_matrix = None
    total_contacts = None
    nucleic_acids = list()
    amino_acids = list()
    n_trajs = len(sorted(dir_path.glob(f'{dir_name}*')))
    for idx, traj in enumerate(tqdm(trajs, total=n_trajs, desc='Processing trajectories')):
        # n_trajs += 1
        # Expects the system to be centered, PBC-corrected, and free of water and ions.
        protein_dict, protein_idx = get_atoms_per_residue(traj, protein=True, pairing_idx=atom_idx, chain=chain,
                                                          single_residue=single_residue)
        nucleic_dict = get_atoms_per_residue(traj, major_acceptors=major_acceptors)
        contacts = ContactCount(traj, protein_idx, protein_dict, nucleic_dict, d0=d0, r0=r0, nn=nn, mm=mm)
        if idx == 0:
            total_contacts = contacts.get_total_contacts()
            distance_matrix = contacts.get_distance_matrix()
            contact_matrix = contacts.get_contact_matrix()
            nucleic_acids = list(nucleic_dict.keys())
            amino_acids = list(protein_dict.keys())
        else:
            total_contacts += contacts.get_total_contacts()
            distance_matrix += contacts.get_distance_matrix()
            contact_matrix += contacts.get_contact_matrix()

    # print(type(total_contacts), type(contact_matrix), type(distance_matrix), type(np.array(n_trajs)))
    total_contacts /= np.array(n_trajs)
    contact_matrix /= np.array(n_trajs)
    distance_matrix /= np.array(n_trajs)
    # contact_matrix = np.mean(contact_matrix, axis=0)
    # distance_matrix = np.mean(distance_matrix, axis=0)

    with open(out_dir / f'{out_name}.pkl', 'wb') as file_:
        pickle.dump({'amino_acids': amino_acids, 'nucleic_acids': nucleic_acids, 'total_contacts': total_contacts,
                     'contact_matrix': contact_matrix, 'distance_matrix': distance_matrix}, file_)


# # Set the default text font size
# plt.rc('font', size=17)
# # Set the axes title font size
# plt.rc('axes', titlesize=18)
# # Set the axes labels font size
# plt.rc('axes', labelsize=18)
# # Set the font size for x tick labels
# plt.rc('xtick', labelsize=16)
# # Set the font size for y tick labels
# plt.rc('ytick', labelsize=16)
# # Set the legend font size
# plt.rc('legend', fontsize=14)
# # Set the font size of the figure title
# plt.rc('figure', titlesize=20)

# # plotting total contacts
# contacts = ContactCount(traj, protein_idx, protein_dict, nucleic_dict, d0=0.0, r0=0.3, nn=6, mm=12)
# c = contacts.get_total_contacts()
# plt.plot(c)
# plt.xlabel('Frame')
# plt.ylabel('$C_{Protein-DNA}$')
# plt.ylim(0, 120)
# plt.savefig('total_contacts.png')
# plt.clf()

# # contacts per base
# c_res = contacts.get_contacts_per_base()
# plt.xlabel('Frame')
# plt.ylabel('$C_{Protein-DNA}$')
# plt.ylim(0, 12)

# for r in c_res:
#     if np.mean(r.T) > 2:
#         plt.plot(r.T)

# plt.legend(labels=list(nucleic_dict.keys()))
# plt.savefig('total_per_base')
# plt.clf()


# # contacts for A7-T37
# contacts_bp = ContactCount(traj, protein_idx, protein_dict, major_acceptors, d0=0.0, r0=0.3, nn=6, mm=12)
# c_bp = contacts_bp.get_contacts_per_residue()

# plt.xlabel('Frame')
# plt.ylabel('$C_{Protein-DNA}$')
# plt.ylim(0, 12)

# for r in c_bp:
#     if np.mean(r.T) > 0.5:
#         plt.plot(r.T)

# plt.legend(labels=list(protein_dict.keys()))

# plt.savefig('A7D37.png')
# plt.clf()


# # solely arg132
# contacts_arg = ContactCount(traj, arg_idx, arg132, major_acceptors, d0=0.0, r0=0.3, nn=6, mm=12)
# c_arg = contacts_arg.get_total_contacts()
# plt.plot(c_arg)
# plt.xlabel('Frame')
# plt.ylabel('$C_{Protein-DNA}$')
# plt.ylim(0, 10)
# plt.savefig('argcontacts.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_path', type=Path, required=True,
                        help='Base path to trajectory directories')
    parser.add_argument('-t', '--traj_dir_name', type=str, required=True,
                        help='Name of trajectory directories')
    parser.add_argument('-od', '--output_dir', type=str, required=True,
                        help='Path to output directory.')
    parser.add_argument('-of', '--output_file', type=str, required=True,
                        help='Name of output file.')
    parser.add_argument('-cfg', '--config_file', type=str, required=True,
                        help='File in python configparser format with simulation settings.')
    args = parser.parse_args()

    base_path = Path(args.base_path)
    traj_dir_name = args.traj_dir_name
    output_dir = Path(args.output_dir)
    output_file = args.output_file
    config_file = args.config_file

    generate_contact_map(base_path, traj_dir_name, output_dir, output_file, config_file)
