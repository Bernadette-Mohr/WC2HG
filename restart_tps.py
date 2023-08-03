import os
import sys
# libdir = os.path.expanduser('/home/jvanasselt/miniconda3/envs/tps/lib/python3.9/site-packages/')
# sys.path.append(libdir)

import argparse
from pathlib import Path
import numpy as np

import mdtraj as md
import openpathsampling as paths
from openpathsampling.engines import gromacs as ops_gmx


def restart_tps(directory, filename, n_runs):
    try:
        if filename.is_file():
            storage = paths.Storage(str(filename), 'a')
        else:
            filename = directory / filename
            storage = paths.Storage(str(filename), 'a')
    except FileNotFoundError:
        print('OPS storage file not found. Is it in the IO-directory or did you provide an absolute file path?')

    sampler = storage.pathsimulators[0]
    sampler.storage = storage
    engine = storage.engines[0]

    n_steps = len(storage.steps) - 1

    engine.filename_setter.count = n_steps
    sampler.restart_at_step(storage.steps[-1])
    sampler.run(n_runs - n_steps)
    storage.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Restart an unfinished TPS calculation from the last *.nc file.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory for storing TPS input and '
                                                                              'output.')
    parser.add_argument('-f', '--filename', type=Path, required=True, help='Filename of the *.nc storage file for '
                                                                           'restarting.')
    parser.add_argument('-nr', '--n_steps', type=int, required=True, help='The original number of TPS runs.')

    args = parser.parse_args()
    restart_tps(args.directory, args.filename, args.n_runs)
