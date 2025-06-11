from __future__ import division, print_function
from past.utils import old_div
import numpy as np
from itertools import cycle

try:
    import matplotlib.pyplot as plt

    # more stuff for plotting histogram and comparing to prediction
    #######################SET LATEX OPTIONS###################
    plt.rc("text", usetex=True)
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    plt.rcParams.update({"font.size": 20})
    plt.rcParams["xtick.major.pad"] = 8
    plt.rcParams["ytick.major.pad"] = 8

    ##########################################################
    ####SET COLOUR MAP######
    cm = plt.get_cmap("Dark2")
    ########################
    #####################LINE STYLE CYCLER####################
    lines = ["-", "--", "-."]
    linecycler = cycle(lines)
    color_cycle = [cm(old_div(1.0 * i, 6)) for i in range(6)]
    ##########################################################
    
    MATPLOTLIB_AVAILABLE = True
except ImportError as err:
    print(err)
    MATPLOTLIB_AVAILABLE = False


def analytical_d2(x, k, N, boxdim=2):
    """
    Analytical function for displacement distribution
    """
    f = float(k * x) / 2
    g = float(boxdim * N - boxdim) / 2 - 1
    return np.exp(-f) * np.power(f, g)


# Vectorized version of analytical_d2
vec_analytical_d2 = np.vectorize(analytical_d2)


def setup_matplotlib_for_hypercube():
    """
    Set up matplotlib with settings appropriate for hypercube plotting
    (no LaTeX to avoid cluster bugs)
    """
    if MATPLOTLIB_AVAILABLE:
        plt.rc("text", usetex=False)  # True = bugs on the cluster!
        plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
        plt.rcParams.update({"font.size": 20})
        plt.rcParams["xtick.major.pad"] = 8
        plt.rcParams["ytick.major.pad"] = 8


def get_color_cycle():
    """
    Get the standard color cycle for plots
    """
    if MATPLOTLIB_AVAILABLE:
        return color_cycle
    else:
        return []


def get_line_cycler():
    """
    Get the standard line style cycler for plots
    """
    if MATPLOTLIB_AVAILABLE:
        return linecycler
    else:
        return iter([])


class PlottingMixin:
    """
    Mixin class providing common plotting functionality for MCRunners
    """
    
    def show_histogram(self):
        """
        Show histogram plot if matplotlib is available and histogram is recorded
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, cannot show histogram")
            return
            
        if not hasattr(self, 'histogram'):
            print("No histogram recorded")
            return
            
        hist = self.histogram.get_histogram()
        val = np.array([i * getattr(self, 'binsize', self.hbinsize) for i in range(len(hist))]) + 0.5 * getattr(self, 'binsize', self.hbinsize)
        plt.hist(val, weights=hist, bins=len(hist))
        plt.show()
    
    def show_histogram_analytical(self, output_directory=""):
        """
        Show histogram against analytical curve (for testing)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, cannot show histogram")
            return
            
        if not hasattr(self, 'get_timeseries'):
            print("No timeseries available")
            return
            
        plt.clf()
        timeseries = self.get_timeseries()
        n, bins, patch = plt.hist(
            timeseries,
            bins=500,
            range=(np.amin(timeseries), np.amax(timeseries)),
            density=True,
            stacked=True,
            alpha=0.4,
            edgecolor=get_color_cycle()[0] if get_color_cycle() else 'blue',
            color=get_color_cycle()[0] if get_color_cycle() else 'blue',
        )
        plt.show() 