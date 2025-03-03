# general python functionalities
import os
import numpy as np
# OPS functionalities
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage
from openpathsampling.numerics import HistogramPlotter2D
import openpathsampling.visualize as ops_vis
from tqdm.auto import tqdm
# plotting
import matplotlib
import matplotlib.pyplot as plt
import cairosvg

paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'figure.figsize': (8.8, 6.6)})


def plot_path_tree(storage):
    tree = ops_vis.PathTree(
        storage.steps,
        # steps,
        ops_vis.ReplicaEvolution(replica=0, accepted=True),
    )
    print('Decorrelated paths', len(tree.generator.decorrelated))
    cycles = *map(tree.steps.get_mccycle, tree.generator.decorrelated),
    print('Cycles with decorrelated paths', cycles)
    # lengths = {cycle: len(storage.steps[cycle].active[0].trajectory) for cycle in cycles}
    lengths = {cycle: len(storage.steps[cycle].active[0].trajectory) for cycle in range(len(cycles))}
    print('Lengths of decorrelated paths', lengths)
    print(f'Average of {1.0 * (cycles[-1] - cycles[0]) / (len(cycles) - 1)} cycles per decorrelated sample')

    # tree.options.css['scale_x'] = 0.2
    # tree.__annotations__
    # # tree.options.ui['legends'] = ['step', 'active']
    # cairosvg.svg2pdf(bytestring=tree.svg(), output_width=None, output_height=None,
    #                  write_to='/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC2MAT/DNAWC2MAT_continued_10_pathtree.pdf',
    #                  dpi=300)


def plot_path_lengths(storage):
    path_lengths = [len(step.active[0].trajectory) for step in storage.steps]
    plt.hist(path_lengths, bins=40, alpha=0.5)
    plt.ylabel("Count")
    plt.xlabel("Path length (Frames)")
    plt.savefig('/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC2MAT/DNAWC2MAT_continued_10_path-lengths.pdf', dpi=300)


def analyze_tps_runs():
    # storage = paths.AnalysisStorage('/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC/DNAWC_TEST.nc')
    storage = Storage('/media/bmohr/Backup/POSTDOC/WCHG/TPS/DNAWC2MAT/', mode='r')
    scheme = storage.schemes[0]
    # steps = list()
    # for idx in tqdm(range(200)):
    #     steps.append(storage.steps[idx])
    # print(storage.schemes[1].move_summary(steps))  # why are there two schemes stored, but the second is empty?

    scheme.move_summary(storage.steps)  # just returns some statistics
    # plot_path_lengths(storage)
    plot_path_tree(storage)


if __name__ == '__main__':
    analyze_tps_runs()
