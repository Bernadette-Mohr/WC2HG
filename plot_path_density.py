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

spec = importlib.util.spec_from_file_location(
    "pathsampling_utilities",
    "/home/bernadette/PycharmProjects/WC2HG/pathsampling_utilities.py",
)
psu = importlib.util.module_from_spec(spec)
sys.modules['pathsampling_utilities'] = psu
spec.loader.exec_module(psu)
utils = psu.PathsamplingUtilities()

sns.set_theme(style='white', palette='muted', context='paper', color_codes=True, font_scale=1.5)


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
    storage = Storage(filename=str(db_list[0]), mode='r')
    engine = storage.engines[0]

    return trajectories, weights, storage, engine


def angle_between_vectors(v1, v2, angle=False):
    # Chose whether to calculate the angle as arctan2 [-180°, 180°] or arccos [0°, 180°]
    if angle is True:
        normal = np.cross(v1, v2)
        # Use the sign of the z coordinate of the normal to determine if the angle is rotated (counter-)clockwise
        # and reflect the full angle range from -180° to 180° in the 3D case.
        angle = np.degrees(
            np.arctan2(np.linalg.norm(normal), np.dot(v1, v2))
        ) * np.sign(np.dot(normal, np.array([0.0, 0.0, 1.0])))
    else:
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        angle = np.degrees(
            np.arccos(np.clip(np.divide(dot_product, (norm_v1 * norm_v2)), -1.0, 1.0))
        )

    return angle


def base_opening_angle(
        snapshot, comI_cv, comII_cv, comIII_cv, comIV_cv, angle_between_vectors_cv, angle
):
    """
    Parameters:
    :param snapshot:
    :param comI_cv:
    :param comII_cv:
    :param comIII_cv:
    :param comIV_cv:
    :param angle_between_vectors_cv:
    :param angle:
    :return:
    """
    comI = comI_cv(snapshot)
    comII = comII_cv(snapshot)
    comIII = comIII_cv(snapshot)
    comIV = comIV_cv(snapshot)

    vec_21 = np.subtract(comI, comII)
    vec_23 = np.subtract(comIII, comII)
    vec_24 = np.subtract(comIV, comII)
    norm1 = np.cross(vec_21, vec_23)
    norm2 = np.cross(vec_24, vec_23)

    return angle_between_vectors_cv(norm1, norm2, angle)  # hard-coded negative sign in the code to Vreede et al., 2019


def base_rolling_angle(
        snapshot, backbone_idx, rollingbase_idx, angle_between_vectors_cv, angle
):
    """
    Parameters
    ----------
    :param angle: selects wether the angle between two vectors is calculated as atan2 (True) or arccos (False).
    :param snapshot: ops trajectory frame
    :param rollingbase_idx: list of the indices of the N1, N3 and N7 atoms defining the vectors of the rolling base
    :param backbone_idx: list of the P atom indices defining the backbone vector
    :param angle_between_vectors_cv: function to calculate the angle between two vectors.
    """

    def normalize(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    # Get the vectors connecting atoms N3 and N1 and N3 and N7 in the rolling base.
    bp_v1 = np.subtract(
        snapshot.xyz[rollingbase_idx[0]], snapshot.xyz[rollingbase_idx[1]]
    )
    bp_v2 = np.subtract(
        snapshot.xyz[rollingbase_idx[2]], snapshot.xyz[rollingbase_idx[1]]
    )

    # Calculate the normal of the rolling-base vectors
    bp_vector = normalize(np.cross(bp_v1, bp_v2))

    # Get the vector associated with the backbone
    bb_vector = np.subtract(
        snapshot.xyz[backbone_idx[1]], snapshot.xyz[backbone_idx[0]]
    )
    bb_vector = normalize(bb_vector)

    # calculate angle
    return angle_between_vectors_cv(bb_vector, bp_vector, angle)


# lambda = arctan2(dHG, dWC)
def lambda_CV(snapshot, d_WC_cv, d_HG_cv):
    """
    Parameters:
    :param snapshot:
    :param d_WC_cv:
    :param d_HG_cv:
    :return: Single CV combining the hydrogen bond lengths of the WC and the HG pairing.
    """
    d_wc = d_WC_cv(snapshot)
    d_hg = d_HG_cv(snapshot)
    return np.arctan2(d_wc, d_hg)


def plot_path_density(dir_path, id_str, collective_variable, rolling_residues, backbone_residues, angle, color_list):
    residA = rolling_residues[0]  # 6
    residT = rolling_residues[1]  # 16
    db_list = sorted(dir_path.glob('*.db'))
    wt_list = sorted(dir_path.glob('*.pkl'))

    if collective_variable == 'distances':
        trajectories, weights, storage, engine = load_data(wt_list, db_list)
        d_WC = storage.cvs['d_WC']
        d_HG = storage.cvs['d_HG']

        # path density histogram for dWC-dHG
        path_density = paths.PathDensityHistogram(cvs=[d_WC, d_HG],
                                                  left_bin_edges=(0.0, 0.0),
                                                  bin_widths=(0.01, 0.01))

        xtick_labels = np.linspace(0.5, 1.5, endpoint=True, num=3, dtype=float)
        ytick_labels = xtick_labels
        ax_xdim, ax_ydim = (0.0, 2.0), (0.0, 2.0)

        csv_name = f'{id_str}_dWC_dHG_histogram.csv'
        plot_name = f'{id_str}_dWC_dHG_plot.png'
        xax_label = "distance WC [nm]"
        yax_label = "distance HG [nm]"

    elif collective_variable == 'theta':
        trajectories, weights, storage, engine = load_data(wt_list, db_list)
        d_WC = storage.cvs['d_WC']
        d_HG = storage.cvs['d_HG']

        o3_prime = '"O3\'"'
        o5_prime = '"O5\'"'
        angI_atoms = (
            f"(resid {residA - 1} and not type H) or (resid {residA + 1} and not type H) or (resid {residT - 1} "
            f"and not type H) or (resid {residT + 1} and not type H)")
        angII_atoms = f"(resid {residA - 1} and name {o3_prime}) or (resid {residA} and name P OP1 OP2 {o5_prime})"
        angIII_atoms = f"(resid {residA} and name  {o3_prime}) or (resid {residA + 1} and name P OP1 OP2 {o5_prime})"
        angIV_atoms = f"resid {residA} and not type H"

        comI = paths.MDTrajFunctionCV(
            "comI", md.compute_center_of_mass, topology=engine.topology, select=angI_atoms
        )
        comII = paths.MDTrajFunctionCV(
            "comII", md.compute_center_of_mass, topology=engine.topology, select=angII_atoms
        )
        comIII = paths.MDTrajFunctionCV(
            "comIII", md.compute_center_of_mass, topology=engine.topology, select=angIII_atoms
        )
        comIV = paths.MDTrajFunctionCV(
            "comIV", md.compute_center_of_mass, topology=engine.topology, select=angIV_atoms
        )

        lambda_ = paths.CoordinateFunctionCV(
            name='lambda',
            f=lambda_CV,
            d_WC_cv=d_WC,
            d_HG_cv=d_HG
        )

        theta_ = paths.CoordinateFunctionCV(
            name="theta",
            f=base_opening_angle,
            comI_cv=comI,
            comII_cv=comII,
            comIII_cv=comIII,
            comIV_cv=comIV,
            angle_between_vectors_cv=angle_between_vectors,
            angle=angle,
        )

        # path density histogram for lambda-theta
        path_density = paths.PathDensityHistogram(
            cvs=[lambda_, theta_], left_bin_edges=(0.2, -180.0), bin_widths=(0.01, 0.01)
        )

        xtick_labels = np.linspace(0.4, 1.0, endpoint=True, num=4, dtype=float)
        ytick_labels = np.linspace(-180, 0, endpoint=True, num=7, dtype=int)
        ax_xdim, ax_ydim = (0.2, 1.2), (-180, 0)

        csv_name = f'{id_str}_lambda_{collective_variable}_histogram.csv'
        plot_name = f'{id_str}_lambda_{collective_variable}_plot.png'
        xax_label = "lambda [rad]"
        yax_label = "theta [deg]"

    elif collective_variable == 'phi':
        if not backbone_residues:
            print('Backbone residues are needed for calculating the base rolling angle. Exiting...')
            return
        if not angle:
            print('The angle between the rolling base and backbone is calculated as arctan2 in the literature. '
                  'Using arcos because \'-a\' flag not set to True.')

        resid_bb_start = backbone_residues[0]  # 1
        resid_bb_end = backbone_residues[1]  # 13
        backbone_atoms = f'resid {resid_bb_start} {resid_bb_end} and name P'
        rollingbase_atoms = f'resid {residA} and name N7 N3 N1'

        trajectories, weights, storage, engine = load_data(wt_list, db_list)
        d_WC = storage.cvs['d_WC']
        d_HG = storage.cvs['d_HG']
        # d_BP = storage.cvs['d_BP']

        lambda_ = paths.CoordinateFunctionCV(
            name='lambda',
            f=lambda_CV,
            d_WC_cv=d_WC,
            d_HG_cv=d_HG
        )

        phi_ = paths.CoordinateFunctionCV(
            name="phi",
            f=base_rolling_angle,
            backbone_idx=engine.topology.mdtraj.select(backbone_atoms),
            rollingbase_idx=engine.topology.mdtraj.select(rollingbase_atoms),
            angle_between_vectors_cv=angle_between_vectors,
            angle=angle,
        )

        # path density histogram for lambda-phi
        path_density = paths.PathDensityHistogram(
            cvs=[lambda_, phi_], left_bin_edges=(0.2, -180), bin_widths=(0.01, 0.01)
        )

        xtick_labels = np.linspace(0.4, 1.0, endpoint=True, num=4, dtype=float)
        ytick_labels = np.linspace(-180, 180, endpoint=True, num=9, dtype=int)
        ax_xdim, ax_ydim = (0.2, 1.2), (-180, 180)

        csv_name = f'{id_str}_lambda_{collective_variable}_histogram.csv'
        plot_name = f'{id_str}_lambda_{collective_variable}_plot.png'
        xax_label = "lambda [rad]"
        yax_label = "phi [deg]"

    else:
        print('Collective variable is not implemented or wrong label. Exiting...')
        return

    print(f'Calculating path density histogram for {collective_variable}...')
    path_dens_counter = path_density.histogram(trajectories[:100], weights=weights[:100])

    gradient = sns.blend_palette(colors=color_list, n_colors=5, as_cmap=True, input='rgb')

    plotter = HistogramPlotter2D(path_density,
                                 xticklabels=xtick_labels,
                                 yticklabels=ytick_labels,
                                 xlim=ax_xdim,
                                 ylim=ax_ydim,
                                 label_format="{:4.2f}")

    # # save histogram for local plotting
    hist_fcn = plotter.histogram.normalized(raw_probability=False)
    df = hist_fcn.df_2d(x_range=plotter.xrange_, y_range=plotter.yrange_)
    np.savetxt(dir_path / csv_name, df.fillna(0.0).transpose(), fmt='%1.3f', delimiter=',')

    # # plot histogram for sanity check
    ax = plotter.plot(cmap=gradient)
    plt.xlabel(xax_label)
    plt.ylabel(yax_label)

    plt.tight_layout()
    plt.savefig(dir_path / plot_name, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot TPS path densities of base-pairing transitions in DNA. Implemented.\n'
                                     'Collective vaiables are: hydrogen bond lengths (\'distances\'), the base opening '
                                     'angle (\'theta\') vs. the arctan2 between the hydrogen bond lengths (\'lambda\'),'
                                     ' and the base rolling angle (\'phi\'), again vs. lambda.\n'
                                     'Output: Probability density histograms (2D numpy array) in csv format. '
                                     'Final plotting can be performed locally.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output. Needs to contain OPS databases and '
                             'dictionaries with weights as values.')
    parser.add_argument('-cv', '--collective_variable', type=str, required=True,
                        choices=['distances', 'theta', 'phi'],
                        help='Collective variables to be analyzed.')
    parser.add_argument('-rr', '--rolling_residues', type=int, nargs='+', required=True,
                        help='Residue indices of 1. the rolling base and 2. the other base of the pair on the '
                             'neighboring DNA strand.')
    parser.add_argument('-bb', '--backbone_residues', type=int, nargs='+', required=False,
                        help='Residue indices at 1. the 5\' end of the strand containing the rolling base and 2. at the'
                             ' 3\' end of the opposite strand to calculate the proxy for the backbone orientation. '
                             'Needed for calculating the base rolling angle.')
    parser.add_argument('-a', '--angle', type=bool, required=False, default=False,
                        help='Calculate the angle between two vectors as arccos (theta, default) '
                             'or arctan2 (phi, set to True).')
    parser.add_argument('-id', '--identifier', type=str, required=True,
                        help='Identifier for the output files, e.g. name of system.')
    parser.add_argument('-c', '--colors', type=str, nargs=5, required=False,
                        default=['#FBFBFB00', '#787878', '#650015', '#B22F0B', '#FF5E00'],
                        help='List of 5 colors for the gradient colormap. Default is white-gray-red-orange.')

    args = parser.parse_args()
    directory = args.directory
    cv = args.collective_variable
    rb_res = args.rolling_residues
    bb_res = args.backbone_residues
    ang = args.angle
    colors = args.colors
    file_id = args.identifier

    plot_path_density(directory, file_id, cv, rb_res, bb_res, ang, colors)
