import argparse
import configparser
import numpy as np
import matplotlib.pyplot as plt
from openpathsampling.analysis import PathHistogram
from openpathsampling.numerics import HistogramPlotter2D
from pathlib import Path
import pickle
import seaborn as sns
import sys


sns.set_theme(style='white', palette='muted', context='paper', color_codes=True, font_scale=1.5)

class HistogramPlotter(PathHistogram):
    def __init__(self, directory,
                 cvs_pkl,
                 config_file,
                 left_bin_edges,
                 bin_widths,
                 identifier,
                 collective_variable,
                 colors):
        super(HistogramPlotter, self).__init__(left_bin_edges=tuple(left_bin_edges), bin_widths=tuple(bin_widths))
        self.directory = directory
        self.cv_trajs, self.weights = self.load_cvs(cvs_pkl)
        self.config_file = self.get_config_file(config_file)
        self.identifier = identifier
        self.collective_variable = collective_variable
        self.xtick_labels, self.ytick_labels = None, None
        self.ax_xdim, self.ax_ydim = None, None
        self.xax_label, self.yax_label = None, None
        self.label_format = None
        self.gradient = sns.blend_palette(colors=colors, n_colors=len(colors), as_cmap=True, input='rgb')

    def get_config_file(self, config_file):
        try:
            if config_file.absolute().is_file():
                return config_file
            elif Path(self.directory / config_file).is_file():
                return self.directory / config_file
        except FileNotFoundError:
            print('Config file not found. Exiting.')
            sys.exit(1)

    def load_cvs(self, cvs_pkl):
        try:
            if cvs_pkl.is_file():
                cvs_file = cvs_pkl
            else:
                cvs_file = self.directory / cvs_pkl
            with open(cvs_file, 'rb') as f:
                cvs = pickle.load(f)
            trajs, weights = zip(*cvs)
            return list(trajs), list(weights)
        except FileNotFoundError:
            print('Pickled CVs and weights not found. Exiting.')
            sys.exit(1)

    def set_plotting_options(self):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(self.config_file)
        cv_config = config[self.collective_variable.upper()]
        self.xtick_labels = np.linspace(cv_config.getfloat('x_start'), cv_config.getfloat('x_stop'),
                                        endpoint=True, num=cv_config.getint('x_num'), dtype=float)
        self.ytick_labels = np.linspace(cv_config.getfloat('y_start'), cv_config.getfloat('y_stop'),
                                        endpoint=True, num=cv_config.getint('y_num'), dtype=float)
        self.ax_xdim = tuple(float(val) for val in cv_config['ax_xdim'].split(','))
        self.ax_ydim = tuple(float(val) for val in cv_config['ax_ydim'].split(','))
        self.xax_label = cv_config['xax_label']
        self.yax_label = cv_config['yax_label']
        self.label_format = cv_config['label_format']

    def generate_histogram(self):

        self.set_plotting_options()
        _ = self.histogram(self.cv_trajs, self.weights)

        csv_name = f'{self.identifier}_{self.collective_variable}_histogram.csv'
        plot_name = f'{self.identifier}_{self.collective_variable}_plot.png'
        plotter = HistogramPlotter2D(self,
                                     xticklabels=self.xtick_labels,
                                     yticklabels=self.ytick_labels,
                                     xlim=self.ax_xdim,
                                     ylim=self.ax_ydim,
                                     label_format=self.label_format,
                                     )

        # save histogram for local plotting
        hist_fcn = plotter.histogram.normalized(raw_probability=False)
        df = hist_fcn.df_2d(x_range=plotter.xrange_, y_range=plotter.yrange_)
        np.savetxt(self.directory / csv_name, df.fillna(0.0).transpose(), fmt='%1.3f', delimiter=',')

        # Plot the histogram for sanity check
        plotter.plot(cmap=self.gradient)
        plt.xlabel(self.xax_label)
        plt.ylabel(self.yax_label)

        plt.tight_layout()
        plt.savefig(self.directory / plot_name, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Calculate collective variables for OPS trajectories. Returns a pickle file with a '
                                     'list of two CV values and a weight for each trajectory.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Directory for storing TPS input and output. Needs to contain OPS databases and '
                             'dictionaries with weights as values.')
    parser.add_argument('-pkl', '--cvs_pkl', type=Path, required=True,
                        help='Provide the name of the pickle file with '
                             'existing CVs and weights. Example: \'SYSTEM_theta_CVs_weights.pkl\'.')
    parser.add_argument('-cfg', '--config_file', type=Path, required=True,
                        help='File in python configparser format with simulation settings.')
    parser.add_argument('-lbe', '--left_bin_edges', type=float, nargs='+', required=True,
                        help='List of left bin edges for the histogram.')
    parser.add_argument('-bw', '--bin_widths', type=float, nargs='+', required=True,
                        help='List of bin widths for the histogram.')
    parser.add_argument('-id', '--identifier', type=str, required=True,
                        help='Identifier for the output files, e.g. name of system.')
    parser.add_argument('-cv', '--collective_variable', type=str, required=True,
                        choices=['distances', 'theta', 'phi'],
                        help='Collective variables to be plotted.')
    parser.add_argument('-c', '--colors', type=str, nargs=5, required=False,
                        default=['#FBFBFB00', '#787878', '#650015', '#B22F0B', '#FF5E00'],
                        help='List of 5 colors for the gradient colormap. Default is white-gray-red-orange.')

    args = parser.parse_args()
    args_dict = vars(args)
    histogrammer = HistogramPlotter(**args_dict)
    histogrammer.generate_histogram()