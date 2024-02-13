import sys
import argparse
from pathlib import Path

from tqdm.auto import tqdm
import logging.config
import openpathsampling as paths
from openpathsampling.experimental.storage import Storage
from openpathsampling.experimental.storage import monkey_patch_all

paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)


def wrapper(gen, fname, start, len_db):
    for idx in tqdm(range(start, len_db), desc=f'Reading steps'):
        try:
            yield gen[idx]
        except StopIteration:
            break
        except Exception as e:
            print(f'Unable to load step {idx} from {fname}: {e.__class__}: {e}')


def recover_database(dir_path, file_names, new_name):
    storage_files = sorted(dir_path.glob(f'{file_names}*'))
    resumed = False
    for file_idx, fname in enumerate(storage_files):
        db_name = f'{new_name}.db'
        if not Path(dir_path / db_name).is_file():
            print(f'Resuming: {resumed}')
            new_storage = Storage(filename=f'{dir_path}/{db_name}', mode='w')
            first_storage = Storage(filename=f'{storage_files[0]}', mode='r')
            for cv in tqdm(first_storage.storable_functions, desc='Preloading cache'):
                cv.preload_cache()
            for obj in tqdm(first_storage.simulation_objects, desc='Copying simulation objects'):
                new_storage.save(obj)
            first_storage.close()
        else:
            resumed = True
            print(f'Resuming: {resumed}')
            new_storage = Storage(filename=f'{dir_path}/{db_name}', mode='a')

        storage = Storage(filename=f'{fname}', mode='r')
        print("File: {0} for {1} steps, {2} snapshots".format(
            fname.name,
            len(storage.steps),
            len(storage.snapshots),
        ))

        start = 0
        len_db = len(storage.steps) + 1

        steps = wrapper(storage.steps, fname, start, len_db)
        for step in steps:
            new_storage.save(step)

        new_storage.sync_all()
        new_storage.close()
        storage.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Concatenate accepted MC trials of multiple OPS databases into one.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output.')
    parser.add_argument('-fns', '--file_names', type=str, required=True,
                        help='File name snippet of one or multiple database files.')
    parser.add_argument('-nf', '--new_file_name', type=str, required=True,
                        help='Name of the new OPS database file with all accepted trials.')

    args = parser.parse_args()
    recover_database(args.directory, args.file_names, args.new_file_name)
