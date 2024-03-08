# Jocelynes color gradient for plotting: ['#FBFBFB00', '#787878', '#650015', '#B22F0B', '#FF5E00']
from matplotlib.colors import LinearSegmentedColormap


class CustomGradient:
    def __init__(self, color_list, nsteps, gradient_name):
        self.colors = color_list
        self.name = gradient_name  # name for the gradient
        self.n_bins = nsteps  # Discretizes the interpolation into bins

    def generate_gradient(self):
        """
        Generate gradient given a list of colors.
        :return: matplotlib.colors.LinearSegmentedColormap from the list of colors
        """
        return LinearSegmentedColormap.from_list(self.name, self.colors, N=self.n_bins)
