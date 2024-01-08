import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', palette='deep', context='paper', font_scale=1.8)


def plot_contact_map(pickle_file, xname, yname, file_name, output_path):
    with open(output_path / pickle_file, 'rb') as file_:
        data = pickle.load(file_)
        amino_acids = data['amino_acids']
        nucleic_acids = data['nucleic_acids']
        total_contacts = data['total_contacts']
        contact_matrix = data['contact_matrix']
        distance_matrix = data['distance_matrix']

    print(contact_matrix.shape)
    print(distance_matrix.shape)
    print(amino_acids, len(amino_acids))
    print(nucleic_acids, len(nucleic_acids))

    # Plotting
    palette = sns.color_palette("mako_r", as_cmap=True)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)
    sns.scatterplot(total_contacts, ax=ax)
    ax.set_xlabel('Frames', fontsize=22)
    ax.set_ylabel('Total Contacts', fontsize=22)
    plt.tight_layout()
    plt.savefig(output_path / f'{file_name}_total_contacts_bound.png', bbox_inches='tight', dpi=300)
    # plt.clf()

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    dm = ax.imshow(distance_matrix, interpolation='nearest', cmap=palette,
                   vmin=distance_matrix.min(), vmax=distance_matrix.max(), aspect='auto')
    plt.colorbar(dm, ax=ax, label='Distance (nm)')
    ax.set_xlabel(xname, fontsize=22)
    ax.set_ylabel(yname, fontsize=22)
    plt.tight_layout()
    plt.savefig(output_path / f'{file_name}_distances_bound.png', bbox_inches='tight', dpi=300)
    # plt.clf()

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    dm = ax.imshow(contact_matrix, interpolation='nearest', cmap=palette,
                   vmin=contact_matrix.min(), vmax=contact_matrix.max(), aspect='auto')
    plt.colorbar(dm, ax=ax, label='Contacts')
    ax.set_xlabel(xname, fontsize=22)
    ax.set_ylabel(yname, fontsize=22)
    plt.tight_layout()
    plt.savefig(output_path / f'{file_name}_contacts_bound.png', bbox_inches='tight', dpi=300)
    # plt.clf()
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pickle_file', type=str, required=True,
                        help='Pickle file containing the contact matrices.')
    parser.add_argument('-x', '--xname', type=str, required=True,
                        help='Name of the x-axis.')
    parser.add_argument('-y', '--yname', type=str, required=True,
                        help='Name of the y-axis.')
    parser.add_argument('-fn', '--filename', type=str, required=True,
                        help='Name of output file.')
    parser.add_argument('-o', '--output_path', type=Path, required=True,
                        help='Output path for the plot.')
    args = parser.parse_args()

    plot_contact_map(args.pickle_file, args.xname, args.yname, args.filename, args.output_path)
