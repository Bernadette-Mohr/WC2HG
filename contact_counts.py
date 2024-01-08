import numpy as np
import mdtraj as md
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# This script calculates the contacts between the protein placed at distance and the DNA and has several
# methods. It looks at the contacts per residue, per base, total contacts and solely ARG132.


class ContactCount:

    def __init__(self, traj, protein_ind, protein_queue, dna_haystack, d0=0.0, r0=0.3, nn=6, mm=12):
        # store trajectory and topology information
        self.traj = traj
        self.top = traj.topology
        self.atom_names = [at for at in map(str, self.top.atoms)]
        self.protein_queue = protein_queue
        self.dna_haystack = dna_haystack

        # store parameters
        self.d0 = d0
        self.r0 = r0
        self.nn = nn
        self.mm = mm

        # get indices of protein and dna atoms
        self.protein_indices = protein_ind
        self.dna_indices = self.get_dna_indices()

        # compute distances and contacts
        self.pairs = np.array([[p_idx, d_idx] for p_idx in self.protein_indices for d_idx in self.dna_indices])
        self.distances = md.geometry.compute_distances(self.traj, self.pairs)
        self.contacts = self.compute_contacts()
        self.contact_matrix = self.get_contact_matrix()

        # collect sections of protein  (for plotting)
        self.sections = self.collect_sections()

    def get_protein_indices(self):
        # Find protein indices corresponding to queue (Restype-Atomtype)
        return [self.atom_names.index(res + '-' + i) for res, at in self.protein_queue.items() for i in at]

    def get_dna_indices(self):
        return [self.atom_names.index(res + '-' + i) for res, at in self.dna_haystack.items() for i in at]

    def smooth_contact(self, r):
        # Compute contact based on distance smoothing function
        return (1 - ((r - self.d0) / self.r0) ** self.nn) / (1 - ((r - self.d0) / self.r0) ** self.mm)

    def compute_contacts(self):
        # Check where first condition holds
        ones = np.where(self.distances - self.d0 <= 0)
        # Apply second condition
        contacts = np.where(self.distances - self.d0 >= 0, self.smooth_contact(self.distances), self.distances)
        # Apply second condition (...)
        contacts[ones] = np.ones(ones[0].shape)
        return contacts

    def get_total_contacts(self):
        return np.sum(self.contacts, axis=1)

    def get_protein_names(self):
        return [self.atom_names[idx] for idx in self.protein_indices]

    def get_dna_names(self):
        return [self.atom_names[idx] for idx in sorted(self.dna_indices)]

    def get_distance_matrix(self):
        # Reshape pair distances to n x n matrix
        s = self.distances.shape
        return self.distances.reshape(s[0], len(self.protein_indices), len(self.dna_indices))

    def get_contact_matrix(self):
        # Reshape pair distances to n x n matrix
        s = self.contacts.shape
        return self.contacts.reshape(s[0], len(self.protein_indices), len(self.dna_indices))

    def collect_sections(self):
        section_ends = []
        count = 0
        for residue in self.protein_queue.keys():
            count += len(self.protein_queue[residue])
            section_ends.append(count)
        return section_ends[:-1]

    def split_data(self):
        return np.split(self.contact_matrix, self.sections, axis=1)

    def get_contacts_per_residue(self):
        return np.array([np.sum(d, axis=(1, 2)) for d in self.split_data()])

    def get_contacts_per_residue_per_base(self):
        return np.array([np.sum(d, axis=1) for d in self.split_data()])

    def get_contacts_per_base(self):
        contacts_per_residue_per_base = self.get_contacts_per_residue_per_base()
        return np.sum(contacts_per_residue_per_base, axis=0).T

    def get_contacts_per_bp(self):
        contacts_per_base = self.get_contacts_per_base()
        n_bases = len(contacts_per_base)
        return np.array(
            [a + b for a, b in zip(contacts_per_base[:n_bases // 2], contacts_per_base[n_bases // 2:][::-1])])

    def get_contacts_per_residue_per_bp(self):
        contacts_per_residue_per_base = self.get_contacts_per_residue_per_base()
        n_bases = len(contacts_per_residue_per_base.T)
        return np.array([a + b for a, b in zip(contacts_per_residue_per_base.T[:n_bases // 2],
                                               contacts_per_residue_per_base.T[n_bases // 2][::-1])]).T

    @staticmethod
    def check_axis(ax):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        return fig, ax

    def plot_contact_map(self, ax=None, frame=-1):
        fig, ax = self.check_axis(ax)
        contact_matrices = self.get_contact_matrix()
        if frame == -1:
            im = ax.imshow(np.mean(contact_matrices, axis=0), vmin=np.min(self.contacts), vmax=np.max(self.contacts),
                           aspect='auto')
        else:
            im = ax.imshow(contact_matrices[frame], vmin=np.min(self.contacts), vmax=np.max(self.contacts),
                           aspect='auto')

        protein_labels = self.get_protein_names()
        dna_labels = self.get_dna_names()

        ax.set_yticks(range(0, len(protein_labels)))
        ax.set_yticklabels(protein_labels)

        ax.set_xticks(range(0, len(dna_labels)))
        ax.set_xticklabels(dna_labels)
        ax.tick_params(axis="x", rotation=80)
        ax.set_title(f'Contact map of frame {frame}')
        plt.colorbar(im, ax=ax, label="$C_{Protein}$")

    def plot_contact_distribution(self, ax=None, c='Red'):
        fig, ax = self.check_axis(ax)
        total_contacts = self.get_total_contacts()
        df = pd.DataFrame(total_contacts)

        data = pd.DataFrame({
            "idx": np.tile(df.columns, len(df.index)),
            "$C_{Protein-DNA}$": df.values.ravel()})

        sns.kdeplot(
            data=data, y="$C_{Protein-DNA}$", legend=False, color=c,  # hue="idx",
            fill=True, common_norm=False, palette="Reds",
            alpha=.5, linewidth=1, ax=ax)

    @staticmethod
    def ns_to_steps(ns=1):
        # assume a timestep of 2 fs
        return int((ns * 1000) / 0.002)

    def make_plumed_file(self, filename='plumed.dat', write=False):

        output = f"""UNITS LENGTH=nm TIME=ps ENERGY=kj/mol\n"""
        output += f"""MOLINFO MOLTYPE=protein STRUCTURE=system.pdb
WHOLEMOLECULES ENTITY0={self.top._atoms[0].index + 1}-{self.top._atoms[-1].index + 1}\n
cmap: CONTACTMAP ...\n"""

        for idx, p in enumerate(self.pairs):
            output += f'\tATOMS{idx + 1}={p[0] + 1},{p[1] + 1}\n'

        output += f'\tSWITCH={{RATIONAL R_0={self.r0} D_0={self.d0} NN={self.nn} MM={self.mm}}}\n'
        output += '\n\t\tSUM \n\t...\n\n'

        output += f"""metad: METAD ARG=cmap PACE=500 HEIGHT=1.2 SIGMA=1 FILE=HILLS
PRINT ARG=* FILE=COLVAR STRIDE=50"""

        if write:
            with open(filename, "w") as f:
                f.write(output)

        print(output)
