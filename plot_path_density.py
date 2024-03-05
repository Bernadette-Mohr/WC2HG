import argparse
from pathlib import Path
import pickle
import mdtraj as md
import openpathsampling as paths
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage import Storage
import importlib.util
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openpathsampling.numerics import HistogramPlotter2D

paths = monkey_patch_all(paths)
paths.InterfaceSet.simstore = True

spec = importlib.util.spec_from_file_location('pathsampling_utilities',
                                              '/home/bmohr/WCHG/TPS/scripts/pathsampling_utilities.py')
psu = importlib.util.module_from_spec(spec)
sys.modules['pathsampling_utilities'] = psu
spec.loader.exec_module(psu)

sns.set_theme(style='white', palette='muted', context='paper', color_codes=True, font_scale=1.5)
# micromamba/envs/ops/lib/python3.9/site-packages/

utils = psu.PathsamplingUtilities()


def append_trajectories(traj_list, db_file):
    storage = Storage(filename=str(db_file), mode='r')
    steps = utils.wrapper(storage.steps, db_file.name, 0, len(storage.steps))
    for step in steps:
        traj_list.append(step.active[0].trajectory)

    return traj_list


def append_weights(weight_list, wt_file):
    with open(wt_file, 'rb') as f:
        weights = pickle.load(f)
    weight_list.extend([float(wt) for wt in weights.values()])

    return weight_list


def load_data(wt_list, db_list):
    trajectories = list()
    weights = list()
    for db, wt in zip(db_list, wt_list):
        trajectories = append_trajectories(trajectories, db)
        weights = append_weights(weights, wt)

    return trajectories, weights


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.divide(dot_product, (norm_v1 * norm_v2)), -1.0, 1.0))

    return np.degrees(angle)


def base_opening_angle(snapshot, comI_cv, comII_cv, comIII_cv, angle_between_vectors_cv):
    comI = comI_cv(snapshot)
    comII = comII_cv(snapshot)
    comIII = comIII_cv(snapshot)

    vec1 = np.subtract(comI, comII)
    vec2 = np.subtract(comIII, comII)

    return angle_between_vectors_cv(vec1, vec2)


# Lambda: atan2(dHG, dWC)
def lambda_angle(snapshot, d_WC_cv, d_HG_cv):
    d_wc = d_WC_cv(snapshot)
    d_hg = d_HG_cv(snapshot)
    return np.arctan2(d_wc, d_hg)


def plot_path_density(directory):
    db_list = sorted(directory.glob('*.db'))
    wt_list = sorted(directory.glob('*.pkl'))
    trajectories, weights = load_data(wt_list, db_list)

    storage = Storage(filename=str(db_list[0]), mode='r')
    d_WC = storage.cvs['d_WC']
    d_HG = storage.cvs['d_HG']
    d_BP = storage.cvs['d_BP']
    
    engine = storage.engines[0]
    residA = 6
    residT = 16
    angI_atoms = f'resid {residA} and not type H'
    angII_atoms = f'(resid {residA} and name P OP1 OP2) or (resid {residA + 1} and name P OP1 OP2)'
    angIII_atoms = f'(resid {residA - 1 } and not type H) or (resid {residA + 1} and not type H) or (resid {residT - 1} and not type H) or (resid {residT + 1} and not type H)'
    
    comI = paths.MDTrajFunctionCV('center_of_mass', md.compute_center_of_mass, topology=engine.topology, select=angI_atoms)
    comII = paths.MDTrajFunctionCV('center_of_mass', md.compute_center_of_mass, topology=engine.topology, select=angII_atoms)
    comIII = paths.MDTrajFunctionCV('center_of_mass', md.compute_center_of_mass, topology=engine.topology, select=angIII_atoms)

    theta = paths.CoordinateFunctionCV(
        name='theta',
        f=base_opening_angle,
        comI_cv=comI,
        comII_cv=comII,
        comIII_cv=comIII,
        angle_between_vectors_cv=angle_between_vectors
    )

    lambda_ = paths.CoordinateFunctionCV(
        name='lambda', 
        f=lambda_angle,
        d_WC_cv=d_WC,
        d_HG_cv=d_HG
    )
    
    # # path density histogram for lambda-theta
    # path_density = paths.PathDensityHistogram(cvs=[lambda_, theta],
    #                                           left_bin_edges=(0.0, 0.0),
    #                                           bin_widths=(0.01,0.01))
    
    # # path density histogram for dWC-dHG
    path_density = paths.PathDensityHistogram(cvs=[d_WC, d_HG],
                                              left_bin_edges=(0.0, 0.0),
                                              bin_widths=(0.01, 0.01))

    path_dens_counter = path_density.histogram(trajectories, weights=weights)

    color_list = ['#FBFBFB00', '#787878', '#650015', '#B22F0B', '#FF5E00']
    gradient = sns.blend_palette(colors=color_list, n_colors=5, as_cmap=True, input='rgb')

    fig, ax = plt.subplots()
    
    # # ylim for lambda-theta -> comment out xticklabels in plotter!
    # tick_labels = np.linspace(0.0, 180.0, endpoint=True, num=7, dtype=int)
    # ax_xdim, ax_ydim = (0.0, 1.2), (0.0, 180.0)
    # # xlim and ylim for dWC-dHG -> set both xticklabels and yticklabels in plotter!
    tick_labels = np.linspace(0.0, 2.0, endpoint=False, num=4, dtype=float)
    ax_xdim, ax_ydim = (0.0, 2.0), (0.0, 2.0)

    plotter = HistogramPlotter2D(path_density,
                                 xticklabels=tick_labels,
                                 yticklabels=tick_labels,
                                 xlim=ax_xdim,
                                 ylim=ax_ydim,
                                 label_format="{:4.2f}")
    
    # # save histogram for local plotting
    hist_fcn = plotter.histogram.normalized(raw_probability=False)
    df = hist_fcn.df_2d(x_range=plotter.xrange_, y_range=plotter.yrange_)
    np.savetxt(directory / 'DNA_dWC_dHG_histogram.csv', df.fillna(0.0).transpose(), fmt='%1.3f', delimiter=',')
    
    # # plot histogram for sanity check
    ax = plotter.plot(cmap=gradient)
    # # labels for lambda-theta
    # plt.xlabel("lambda [rad]")
    # plt.ylabel("theta [deg]")
    # # labels for dWC-dHG
    plt.xlabel("distance WC [nm]")
    plt.ylabel("distance HG [nm]")
    
    plt.tight_layout()
    plt.savefig(directory / 'DNA_dWC_dHG.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot TPS path density.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output. Needs to contain OPS databases and '
                             'dictionaries with weights as values.')

    args = parser.parse_args()
    dir_path = args.directory

    plot_path_density(dir_path)
