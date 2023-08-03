# general python functionalities
import os
import numpy as np
# OPS functionalities
import openpathsampling as paths
from openpathsampling.numerics import HistogramPlotter2D
import openpathsampling.visualize as ops_vis
# plotting
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'figure.figsize': (8.8, 6.6)})
import cairosvg


def plot_path_tree(storage):
    tree = ops_vis.PathTree(
        storage.steps,
        ops_vis.ReplicaEvolution(replica=0, accepted=True)
    )
    print('Decorrelated paths', len(tree.generator.decorrelated))

    tree.options.css['scale_x'] = 0.2
    cairosvg.svg2pdf(bytestring=tree.svg(), output_width=None, output_height=None,
                     write_to='/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC/DNAWC_TEST_pathtree.pdf')


def plot_path_lengths(storage):
    path_lengths = [len(step.active[0].trajectory) for step in storage.steps]
    plt.hist(path_lengths, bins=40, alpha=0.5)
    plt.ylabel("Count")
    plt.xlabel("Path length (Frames)")
    plt.savefig('/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC/DNAWC_TEST_path-lengths.pdf', dpi=300)


def analyze_tps_runs():
    storage = paths.AnalysisStorage('/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC/DNAWC_TEST.nc')
    scheme = storage.schemes[0]
    scheme.move_summary(storage.steps)  # just returns some statistics

    # plot_path_lengths(storage)
    plot_path_tree(storage)


if __name__ == '__main__':
    analyze_tps_runs()
