from __future__ import division

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
from itertools import cycle
from cycler import cycler
from disk_square_deterministic_exp_stochastic_random_walk_r2 import *
from basinvolume.monte_carlo import CheckBarrierCrossing
from mcpele.monte_carlo import RandomCoordsDisplacement, RecordCoordsTimeseries
#######################SET LATEX OPTIONS###################
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
#rc('text.latex',preamble=r'\usepackage{times}')
glob_fontsize=24
plt.rcParams.update({'font.size': glob_fontsize})
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams.update({'figure.autolayout': True})
##########################################################

class FixedOrderFormatter(ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of
    magnitude"""
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset,
                                 useMathText=useMathText)
    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag

####SET COLOUR MAP######
def get_color_cycle(ncolors=20, reverse=True):
    cm = plt.get_cmap('Paired')
    if reverse:
        color_cycle=cycle([cm(1. * (i+0.5) / float(ncolors)) for i in xrange(ncolors)][::-1])
    else:
        color_cycle = cycle([cm(1. * (i - 0.5) / float(ncolors)) for i in xrange(ncolors)])
    return color_cycle
def get_marker_cycle():
    markers = ["s","o","v","^","<",">","*","D","h","8","p"]
    markercycle = cycle(markers)
    return markercycle
def get_line_cycle():
    lines = ["--","-"]
    linecycle = cycle(lines)
    return linecycle
def get_cycler(ncolors=20, reverse=True):
    cm = plt.get_cmap('Paired')
    if reverse:
        color_cycler = cycler('color', [cm(1. * (i+0.5) / float(ncolors)) for i in xrange(ncolors)][::-1])
    else:
        color_cycler = cycler('color', [cm(1. * (i - 0.5) / float(ncolors)) for i in xrange(ncolors)])
    return color_cycler

class BarrierCrossingMC(object):
    def __init__(self, common_pars, cloud_pars, oracle, cross_pars, cts):
        self.common_pars = common_pars
        self.cloud_pars = cloud_pars
        self.oracle = oracle
        self.cross_pars = cross_pars
        self.cts = cts
        self.mc_temperature = 1
        self.origin = common_pars["origin"]
        self.mc_potential = NullPotential()
        self.mc = MC(self.mc_potential, self.origin, self.mc_temperature, self.common_pars["mc_steps"])
        self.eq_steps = 0
        self.random_walk = RandomCoordsDisplacement(42, self.common_pars["mc_stepsize"], single=True, nparticles=1, bdim=1,
                                                    min_acc_ratio=0, max_acc_ratio=1, report_interval=42)
        self.mc.set_takestep(self.random_walk)
        self.cloud_test = CloudTest(44, 46, cloud_pars["nr_points"], cloud_pars["radius"])
        self.cloud_test.add_conf_test(self.oracle)
        self.mc.add_accept_test(self.cloud_test)
        self.mc.add_action(self.cts)
        self.mc.set_report_steps(self.eq_steps)
    def run(self):
        self.mc.set_print_progress()
        self.mc.run()
        

class BarrierCrossingPlot(BasicPlot):
    def __init__(self, common_pars, drop_numbers, cross_pars):
        super(BarrierCrossingPlot, self).__init__()
        self.common_pars = common_pars
        self.drop_numbers = drop_numbers
        self.cross_pars = cross_pars
        self.out_name = "barrier_crossing_instant_r2.pdf"
        # self.ax.title(r"Cloud sampling: transition state finding", fontsize=19)
        self.ax.set_xlabel("MC steps / {}".format(self.common_pars["record_every"]))
        self.ax.set_ylabel(r"$x / \sigma$")

    def setup(self):
        pass

    def save_and_close(self, loc=3):
        self.ax.legend(loc=loc, prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1,
                    markerscale=1, columnspacing=0.25, labelspacing=0.25, handlelength=0.6,
                    frameon=False, ncol=2)
        self.ax.set_ylim(ymin=-2.5, ymax=2.5)
        pdf = PdfPages(self.out_name)
        self.fig.savefig(pdf, format='pdf')
        pdf.close()
        plt.close()
    def run(self):
        self.find_transition_state()
        self.make_plot()
    def find_transition_state(self):
        self.instant_r2 = []
        for n in self.drop_numbers:
            oracle = CheckBarrierCrossing(ss=self.cross_pars["ss"], prefactor=self.cross_pars["prefactor"],
                                          min_energy=self.cross_pars["min_energy"], seed=44)
            cloud_pars = dict(nr_points=n, radius=self.common_pars["cloud_radius"])
            cts = RecordCoordsTimeseries(1, record_every=self.common_pars["record_every"])
            mc = BarrierCrossingMC(self.common_pars, cloud_pars, oracle, self.cross_pars, cts)
            mc.run()
            self.instant_r2.append(cts.get_time_series())
    def make_plot(self):
        sigma = np.sqrt(self.cross_pars["ss"])
        color_cycle = get_color_cycle(len(self.drop_numbers)+1, reverse=False)
        for i, n in enumerate(self.drop_numbers):
            self.ax.plot(xrange(1, len(self.instant_r2[i]) + 1), self.instant_r2[i]/sigma, "-",
                     label="$k = {}$".format(str(n)), color=color_cycle.next(), linewidth=1.8)
            self.ax.plot(xrange(1, len(self.instant_r2[0]) + 1), np.ones(len(self.instant_r2[0])), "--", color="k")
            self.ax.plot(xrange(1, len(self.instant_r2[0]) + 1), -np.ones(len(self.instant_r2[0])), "--", color="k")
            self.ax.plot(xrange(1, len(self.instant_r2[0]) + 1), np.zeros(len(self.instant_r2[0])), "-", color="k")
        axbox = self.ax.get_position()
        loc = (axbox.x0-0.1, axbox.y0-0.1)
        self.save_and_close(loc=loc)
        
        
if __name__ == "__main__":
    ss = 1
    barrier_height = 30 # in units of kT, hence it's pointless to set the temperature
    mc_stepsize = 0.25 * np.sqrt(ss)
    cloud_radius = 0.25 * np.sqrt(ss)
    origin_pos = 2 * np.sqrt(ss)
    min_energy = ss * barrier_height * np.exp(-0.5*origin_pos**2/ss) / np.sqrt(2*np.pi)
    common_pars = dict(mc_stepsize=mc_stepsize, mc_steps=200000,
                       cloud_radius=cloud_radius, origin=origin_pos * np.ones(1))
    common_pars["record_every"] = common_pars["mc_steps"] // 1000
    drop_numbers = np.asarray([1, 10, 25, 50, 75])
    cross_pars = dict(temperature=1, ss=ss, prefactor=barrier_height, min_energy=min_energy)
    bcp = BarrierCrossingPlot(common_pars, drop_numbers, cross_pars)
    bcp.run()