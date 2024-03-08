import argparse
import sys
from pathlib import Path
from tqdm.auto import tqdm
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
from openpathsampling.experimental.storage import Storage


def concatenate_runs(dir_path, db_list, new_file_name):
    if Path(new_file_name).suffix == '.db':
        new_file_name = Path(new_file_name).stem

    for db in db_list:
        try:
            if Path(dir_path / db).is_file():
                storage = Storage(str(dir_path / db), 'r')
            else:
                storage = Storage(str(db), 'r')
        except FileNotFoundError:
            print('OPS storage file not found. Is it in the IO-directory or did you provide an absolute file path?')

        if not big_db:
            big_db_name = f'{dir_path}/{new_file_name}.db'
            big_db = Storage(str(), 'w')
            for cv in tqdm(storage.storable_functions, desc='Preloading cache'):
                cv.preload_cache()
            for obj in tqdm(storage.simulation_objects, desc='Copying simulation objects'):
                big_db.save(obj)

        try:
            for idx, step in enumerate(tqdm(storage.steps, desc='Copying steps')):
                big_db.save(step)
        except Exception as e:
            print(f'Unable to load step {idx} from storage: {e.__class__}: {str(e)}')
            pass

    big_db.sync_all()
    big_db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Concatenate multiple OPS databases into one.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output.')
    parser.add_argument('-db', '--database', nargs='+', required=True,
                        help='List of OPS databases to be concatenated.')
    parser.add_argument('-nf', '--new_file_name', type=str, required=True,
                        help='Name of the new OPS database file.')

    args = parser.parse_args()
    concatenate_runs(args.directory, args.database, args.new_file_name)
