import argparse
from pathlib import Path
import pickle
from tqdm.auto import tqdm
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage
import openpathsampling.visualize as ops_vis
import matplotlib.pyplot as plt
import seaborn as sns

paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
sns.set(style='whitegrid', palette='deep', context='paper', font_scale=1.8)


def wrapper(gen, fname, start, len_db):
    for idx in tqdm(range(start, len_db), desc=f'Reading steps'):
        try:
            yield gen[idx]
        except StopIteration:
            break
        except Exception as e:
            print(f'Unable to load step {idx} from {fname}: {e.__class__}: {e}')


def plot_path_lengths(steps, output_path, fname):
    path_lengths = [len(step.active[0].trajectory) for step in steps]
    plt.hist(path_lengths, bins=40, alpha=0.5)
    plt.ylabel("Count")
    plt.xlabel("Path length (Frames)")
    plt.savefig(f'{output_path}/{fname.stem}.png', dpi=300)


def get_decorrelated_paths(steps):
    tree = ops_vis.PathTree(
        steps,
        ops_vis.ReplicaEvolution(replica=0, accepted=True)
    )
    print('Decorrelated paths', len(tree.generator.decorrelated))
    cycles = *map(tree.steps.get_mccycle, tree.generator.decorrelated),
    print('Cycles with decorrelated paths', cycles)
    print(f'Average of {1.0 * (cycles[-1] - cycles[0]) / (len(cycles) - 1)} cycles per decorrelated sample')


def read_database(dir_path, db_names, weights_names=None):

    if weights_names:
        for weights_name in weights_names:
            weights_file = Path(f'{dir_path}/{weights_name}')
            with open(weights_file, 'rb') as infile:
                weights_dict = pickle.load(infile)
            print(weights_dict)

    for db_name in db_names:
        fname = dir_path / db_name
        storage = Storage(filename=str(fname), mode='r')
        scheme = storage.schemes[0]
        scheme.move_summary(storage.steps)

        steps = wrapper(storage.steps, fname, 0, len(storage.steps))
        plot_path_lengths(steps, dir_path, fname)

        steps = wrapper(storage.steps, fname, 0, len(storage.steps))
        max_len = 0
        for step in steps:
            if len(step.active[0].trajectory) > max_len:
                max_len = len(step.active[0].trajectory)
            print(step.mccycle, len(step.active[0].trajectory))

        print(max_len)

        steps = wrapper(storage.steps, fname, 0, len(storage.steps))
        get_decorrelated_paths(list(steps))
        storage.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Read out contents from TPS database file, perform some statistics.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output.')
    parser.add_argument('-db', '--database', type=str, nargs='+', required=True,
                        help='Name of the *.db storage file for reading.')
    parser.add_argument('-w', '--weights', type=str, nargs='+', required=False, default=None,
                        help='Name of the weights file for reading, if present.')

    args = parser.parse_args()
    read_database(args.directory, args.database, args.weights)
