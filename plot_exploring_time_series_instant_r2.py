from __future__ import division

import numpy as np

from basinvolume.utils import *


class TSPlot(BasicPlot):
    def __init__(self):
        self.out_name = "explore_instant_r2_power_plot.pdf"
        plt.title(r"Cloud sampling: stochastic oracle ($r^{-" + str(5) + "}$)", fontsize=19)
        self.n = [1000, 100, 10, 1]
        self.data = [np.loadtxt("StochasticPowerAcceptancePlot_DropNumber_instant_r2_{}.txt".format(n)) for n in self.n]
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r"MC steps / 100", fontsize=18)
        plt.ylabel(r"Backbone point $r^2$", fontsize=18)
        plt.tick_params(labelsize=18)
    def plot(self):
        for d, n in zip(self.data, self.n):
            plt.plot(xrange(1, len(d) + 1), d, "-", label=r"$n_d=$" + str(n))
        self.save_and_close()

        
class TSPlotExp(TSPlot):
    def __init__(self):
        super(TSPlotExp, self).__init__()
        self.out_name = "explore_instant_r2_exp_plot.pdf"
        plt.title(r"Cloud sampling: stochastic oracle ($\exp(-r)$)", fontsize=19)
        self.data = [np.loadtxt("StochasticAcceptancePlot_DropNumber_instant_r2_{}.txt".format(n)) for n in self.n]


class TSPlotDet(TSPlot):
    def __init__(self):
        super(TSPlotDet, self).__init__()
        self.out_name = "explore_instant_r2_det_plot.pdf"
        plt.title(r"Cloud sampling: deterministic oracle", fontsize=19)
        self.data = [np.loadtxt("DeterministicAcceptancePlot_DropNumber_instant_r2_{}.txt".format(n)) for n in self.n]


if __name__ == "__main__":
    pp = TSPlot()
    pp.plot()
    pe = TSPlotExp()
    pe.plot()
    pd = TSPlotDet()
    pd.plot()
