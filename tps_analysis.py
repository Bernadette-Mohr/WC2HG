# general python functionalities
import os
import numpy as np
# OPS functionalities
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
from openpathsampling.experimental.storage.collective_variables import CoordinateFunctionCV, CollectiveVariable
from openpathsampling.experimental.simstore import SQLStorageBackend
from openpathsampling.experimental.storage import Storage
from openpathsampling.experimental.simstore import Processor, StorableFunctionConfig
import openpathsampling.visualize as ops_vis
from openpathsampling.numerics import HistogramPlotter2D
# plotting
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'figure.figsize': (8.8, 6.6)})
import cairosvg


def plot_path_tree(storage):
    tree = ops_vis.PathTree(
        # storage.steps,
        storage,
        ops_vis.ReplicaEvolution(replica=0, accepted=True)
    )
    print('Decorrelated paths', len(tree.generator.decorrelated))

    tree.options.css['scale_x'] = 0.2
    cairosvg.svg2pdf(bytestring=tree.svg(), output_width=None, output_height=None,
                     write_to='/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC/DNAWC_TEST_pathtree.pdf')


def plot_path_lengths(storage):
    path_lengths = [len(step.active[0].trajectory) for step in storage]  # storage.steps
    plt.hist(path_lengths, bins=40, alpha=0.5)
    plt.ylabel("Count")
    plt.xlabel("Path length (Frames)")
    plt.savefig('/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC/DNAWC_TEST_path-lengths.pdf', dpi=300)


def analyze_tps_runs():
    # storage = paths.AnalysisStorage('/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC/DNAWC_TEST.nc')
    storage = Storage(f'/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC/new.db', 'r')
    scheme = storage.schemes[0]
    steps = list()
    try:
        for idx, step in enumerate(storage.steps):
            steps.append(step)
    except Exception as e:
        print(f'Unable to load step {idx} from storage: {e.__class__}: {str(e)}')

    scheme.move_summary(steps=steps)  # just returns some statistics
    # plot_path_lengths(storage=steps)
    plot_path_tree(storage=steps)


if __name__ == '__main__':
    analyze_tps_runs()
