import argparse
from pathlib import Path
import pickle
from tqdm.auto import tqdm
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
from openpathsampling.experimental.storage import Storage


def wrapper(gen, fname, start, len_db):
    for idx in tqdm(range(start, len_db), desc=f'Reading steps'):
        try:
            yield gen[idx]
        except StopIteration:
            break
        except Exception as e:
            print(f'Unable to load step {idx} from {fname}: {e.__class__}: {e}')


def filter_trials(dir_path, dir_name, new_name):
    """Needs testing for correctness.

       Parameters:
       -----------
       dir_path: The base path to the folders containing the TPS run, storage locaton for files with collected paths
                 and weights.
       dir_name: The folders containing parts of a TPS run.
       new_name: the name of the .db file that will contain all accpted paths."""

    storage_dirs = sorted(dir_path.glob(f'{dir_name}*'))
    resumed = False
    for dir_idx, dir_ in enumerate(storage_dirs):
        storage_files = sorted(dir_.glob('*.db'))
        db_name = f'{new_name}.db'
        if not Path(dir_path / db_name).is_file():
            new_storage = Storage(filename=f'{dir_path}/{db_name}', mode='w')
            first_storage = Storage(filename=f'{sorted(dir_.glob("*.db"))[0]}', mode='r')
            for cv in tqdm(first_storage.storable_functions, desc='Preloading cache'):
                cv.preload_cache()
            for obj in tqdm(first_storage.simulation_objects, desc='Copying simulation objects'):
                new_storage.save(obj)
            first_storage.close()
        else:
            print('Resuming...')
            resumed = True
            new_storage = Storage(filename=f'{dir_path}/{db_name}', mode='a')
        
        old_cycle = 0
        for file_idx, fname in enumerate(storage_files):
            storage = Storage(filename=f'{fname}', mode='r')
            weights_file = Path(f'{dir_path}/{new_name}_{dir_idx}_weights.pkl')
            if weights_file.is_file():
                with open(weights_file, 'rb') as infile:
                    weights_dict = pickle.load(infile)
                    old_cycle = list(sorted(weights_dict.keys()))[-1]
            else:
                print('First file in for loop?')
                weights_dict = dict()

            print("File: {0} for {1} steps, {2} snapshots".format(
                fname.name,
                len(storage.steps),
                len(storage.snapshots),
            ))

            start = 0
            if not resumed and file_idx == 0:
                start = 1
                len_db = len(storage.steps)
            else:
                len_db = len(storage.steps) + 1

            steps = wrapper(storage.steps, fname, start, len_db)
            for step in steps:
                if resumed and step.mccycle <= old_cycle:
                    continue
                else:
                    if step.change.accepted:
                        new_cycle = step.mccycle
                        weights_dict[new_cycle] = new_cycle - old_cycle
                        old_cycle = new_cycle
                        new_storage.save(step)
            
            # Save weights dict
            with open(weights_file, 'wb') as f:
                pickle.dump(weights_dict, f, pickle.HIGHEST_PROTOCOL)

            new_storage.sync_all()
            new_storage.close()
            storage.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Concatenate accepted MC trials of multiple OPS databases into one.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output.')
    parser.add_argument('-dns', '--dir_names', type=str, required=True,
                        help='Subdirectory name snippet for globbing to iterate over output directories '
                             'on a cluster.')
    parser.add_argument('-nf', '--new_file_name', type=str, required=True,
                        help='Name of the new OPS database file with all accepted trials.')

    args = parser.parse_args()
    filter_trials(args.directory, args.dir_names, args.new_file_name)
