import argparse
from pathlib import Path
import logging.config
from multiprocessing import Process
import time

import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
from openpathsampling.experimental.storage import Storage

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

# interface = 4
# logging.config.fileConfig('logging.conf',
#                           defaults={'initfilename': f'tis_init_{interface}.log',
#                                     'logfilename': f'tis_output_{interface}.log'},
#                           disable_existing_loggers=False)


def restart_tps(directory, filename, n_runs, walltime):
    start_time = time.time()  # Time at the start of this process
    try:
        if filename.is_file():
            storage = Storage(str(filename), 'a')
        else:
            filename = directory / filename
            storage = Storage(str(filename), 'a')
    except FileNotFoundError:
        print('OPS storage file not found. Is it in the IO-directory or did you provide an absolute file path?')

    runtime = True
    while runtime:
        elapsed_time = time.time() - start_time
        if elapsed_time > walltime:
            storage.close()
            runtime = False

        sampler = storage.pathsimulators[0]
        sampler.storage = storage

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
        sampler.run(n_runs - n_steps)
        print(storage.summary())
        storage.close()
        runtime = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Restart an unfinished TPS calculation from the last step in a *.db file.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory for storing TPS input and '
                                                                              'output.')
    parser.add_argument('-f', '--filename', type=Path, required=True, help='Filename of the *.db storage file for '
                                                                           'restarting.')
    parser.add_argument('-nr', '--n_runs', type=int, required=True, help='The original number of required TPS runs.')
    parser.add_argument('-wt', '--walltime', type=int, required=True,
                        help='WALLTIME of the cluster minus 30 mins to allow closing of the storage object.')

    args = parser.parse_args()

    process = Process(target=restart_tps, args=(args.directory, args.filename, args.n_runs, args.walltime))
    process.start()

    process.join(args.walltime)

    if process.is_alive():
        print('TPS run timed out!')
        process.join(120)  # wait 2 minutes for the process to finish closing storage file
        process.terminate()
