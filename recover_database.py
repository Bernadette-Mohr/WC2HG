import sys
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import openpathsampling as paths
from mueller_brown_potential import MuellerBrown
from openpathsampling.experimental.storage import monkey_patch_all
paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
from openpathsampling.experimental.storage import Storage


def recover_database(old_storage_name, new_storage_name):
    old_storage = Storage(f'{old_storage_name}', 'r')

    if not Path(new_storage_name).is_file():
        new_storage = Storage(f'{new_storage_name}', 'w')

        for cv in tqdm(old_storage.storable_functions, desc='Preloading cache'):
            cv.preload_cache()

        for obj in tqdm(old_storage.simulation_objects, desc='Copying simulation objects'):
            new_storage.save(obj)
    else:
        new_storage = Storage(f'{new_storage_name}', 'a')

    try:
        if len(new_storage.steps) == 0:
            for idx, step in enumerate(tqdm(old_storage.steps)):
                if step.change.accepted:
                    new_storage.save(step)
                # print(idx, len(old_storage.steps[idx].active[0].trajectory))
        else:
            start = len(new_storage.steps) + 1
            for idx in tqdm(range(start, len(old_storage.steps))):
                if old_storage.steps[idx].change.accepted:
                    new_storage.save(old_storage.steps[idx])

    except Exception as e:
        print(f'Unable to load step {idx} from storage: {e.__class__}: {str(e)}')
        pass

    new_storage.sync_all()
    new_storage.close()
    old_storage.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Concatenate multiple OPS databases into one.')
    parser.add_argument('-old', '--old_storage', type=str, required=True, help='Filename/path of the old OPS database.')
    parser.add_argument('-new', '--new_storage', type=str, required=True, help='Filename/path of the new OPS database.')

    args = parser.parse_args()
    recover_database(args.old_storage, args.new_storage)
