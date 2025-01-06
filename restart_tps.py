import argparse
from pathlib import Path
import logging.config
from tqdm.auto import tqdm

import openpathsampling as paths
from openpathsampling.beta.hooks import GraciousKillHook, GraciousKillError
from openpathsampling.experimental.storage import monkey_patch_all
paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
from openpathsampling.experimental.storage import Storage

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)


def restart_tps(directory, filename, n_runs, walltime):
    try:
        if filename.is_file():
            storage = Storage(str(filename), 'r')
        else:
            filename = directory / filename
            storage = Storage(str(filename), 'r')
    except FileNotFoundError:
        print('OPS storage file not found. Is it in the IO-directory or did you provide an absolute file path?')
        raise FileNotFoundError

    sampler = storage.pathsimulators[0]
    # sampler.storage = storage

    step = -1
    while True:
        try:
            sampler.restart_at_step(storage.steps[step])
        except KeyError:
            print(f'KeyError at step {step} in {filename}. Trying previous step.')
            step -= 1
            continue
        else:
            break

    n_steps = storage.steps[step].mccycle

    new_storage_name = f'{directory}/{"_".join(filename.stem.split("_")[:-1])}_{n_steps + 1}{filename.suffix}'
    print(new_storage_name)
    new_storage = Storage(new_storage_name, 'w')

    for cv in tqdm(storage.storable_functions, desc='Preloading cache'):
        cv.preload_cache()

    for obj in tqdm(storage.simulation_objects, desc='Copying simulation objects'):
        new_storage.save(obj)
    storage.close()

    sampler.storage = new_storage

    # Monitor HPC cluster walltime and stop the simulation before the job is killed.
    kill_hook = GraciousKillHook(walltime)
    sampler.attach_hook(kill_hook)
    try:
        sampler.run(n_runs - n_steps)
    except GraciousKillError:
        print("TPS run timed out!")
        storage.close()

    print(storage.summary())
    storage.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Restart an unfinished TPS calculation from the last step in a *.db file.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory for storing TPS input and '
                                                                              'output.')
    parser.add_argument('-f', '--filename', type=Path, required=True, help='Filename of the *.db storage file for '
                                                                           'restarting.')
    parser.add_argument('-nr', '--n_runs', type=int, required=True, help='The original number of required TPS runs.')
    parser.add_argument('-wt', '--walltime', type=str, required=True, default="4 days 12 hours",
                        help='Walltime for TPS run as str, e.g. \"23 hours 20 minutes\", ensures the output database to '
                             'be closed correctly if run on a cluster with max. runtime for jobs.\n '
                             'ATTENTION: Preprossessing steps are not included the walltime!')

    args = parser.parse_args()

    restart_tps(directory=args.directory, filename=args.filename, n_runs=args.n_runs, walltime=args.walltime)
