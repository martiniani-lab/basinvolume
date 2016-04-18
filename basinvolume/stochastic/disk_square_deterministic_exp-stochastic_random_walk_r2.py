from __future__ import division

import copy as c
import numpy as np

from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import CloudTest
from mcpele.monte_carlo import NullPotential
from mcpele.monte_carlo import RandomCoordsDisplacement
from mcpele.monte_carlo import RecordCloudR2

from basinvolume.monte_carlo import CheckExponentiallyDecayingProfile
from basinvolume.monte_carlo import CheckHyperCubicContainer
from basinvolume.monte_carlo import CheckHyperSphericalContainer
from basinvolume.utils import *

def get_disk_mean_r2(radius):
    return radius ** 2 / 2
    
def get_disk_exp_mean_r2(u, d):
    return (24 * d ** 4 + 24 * d ** 3 * u + 12 * d ** 2 * u ** 2 + 4 * d * u ** 3 + u ** 4) / (4 * d ** 2 + 4 * d * u + 2 * u ** 2)
    
def get_square_mean_r2(side_length):
    return side_length ** 2 / 6
    
def get_square_exp_mean_r2(u, d):
    return 2 * (6 * d ** 3 + 6 * d ** 2 * u + 3 * d * u ** 2 + u ** 3) / (3 * (d + u))
    
    
class MC(_BaseMCRunner):
    def set_control(self, temp):
        self.set_temperature(temp)
    
    
class OracleMCR2(object):
    def __init__(self, common_pars, cloud_pars, oracle):
        self.nr_steps = common_pars["mc_steps"]
        self.temperature = 1
        self.origin = common_pars["origin"]
        self.mc_potential = NullPotential()
        self.mc = MC(self.mc_potential, self.origin, self.temperature, self.nr_steps)
        self.eq_steps = self.nr_steps // 2
        self.random_walk = RandomCoordsDisplacement(42, 1, single=True, nparticles=1, bdim=2)
        self.mc.set_takestep(self.random_walk)
        self.oracle = oracle
        self.cloud_test = CloudTest(44, 46, cloud_pars["nr_points"], cloud_pars["radius"])
        self.cloud_test.add_conf_test(self.oracle)
        self.mc.add_accept_test(self.cloud_test)
        self.cloud_measure_r2 = RecordCloudR2(self.eq_steps, self.origin)
        self.mc.add_action(self.cloud_measure_r2)
        self.mc.set_report_steps(self.eq_steps)
    def run(self):
        self.mc.run()
    def get_r2(self):
        return self.cloud_measure_r2.get_mean_r2()    
    
    
class DeterministicPlot(BasicPlot):
    def __init__(self, common_pars, cloud_pars):
        self.common_pars = common_pars
        self.cloud_pars = cloud_pars
        self.out_name = "deterministic_plot.pdf"
        plt.title(r"Cloud sampling: deterministic oracle", fontsize=19)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r"Radius or side length", fontsize=18)
        plt.ylabel(r"Mean squared displacement", fontsize=18)
        plt.tick_params(labelsize=18)
        self.labels = ["Disk, exact", "Square, exact", "Disk, MC", "Square, MC"]
    def run(self):
        self.compute_r2()
        self.make_plot()
    def compute_r2(self):
        self.mc_disk_r2 = np.ones(len(self.common_pars["disk_radii"]))
        self.mc_square_r2 = np.ones(len(self.common_pars["square_sides"]))
        self.run_disk_mc()
        self.run_square_mc()
    def run_disk_mc(self):
        for i, r in enumerate(self.common_pars["disk_radii"]):
            oracle = CheckHyperSphericalContainer(self.common_pars["origin"], r, 2)
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle)
            mc.run()
            self.mc_disk_r2[i] = mc.get_r2()
    def run_square_mc(self):
        for i, l in enumerate(self.common_pars["square_sides"]):
            oracle = CheckHyperCubicContainer(self.common_pars["origin"], l, 2)
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle)
            mc.run()
            self.mc_square_r2[i] = mc.get_r2()
    def make_plot(self):
        symbols = ["s", "o", "^", "v"]
        rs = self.common_pars["disk_radii"]
        ls = self.common_pars["square_sides"]
        rsp = np.linspace(np.amin(rs), np.amax(rs), 1000)
        lsp = np.linspace(np.amin(ls), np.amax(ls), 1000)
        plt.plot(rsp, get_disk_mean_r2(rsp), label=self.labels[0])
        plt.plot(lsp, get_square_mean_r2(lsp), label=self.labels[1])
        plt.plot(rs, self.mc_disk_r2, symbols[0], label=self.labels[2])
        plt.plot(ls, self.mc_square_r2, symbols[1], label=self.labels[3])
        self.save_and_close()


class StochasticPlot(DeterministicPlot):
    def __init__(self, common_pars, cloud_pars):
        super(StochasticPlot, self).__init__(common_pars, cloud_pars)
        self.out_name = "stochastic_plot.pdf"
        plt.title(r"Cloud sampling: stochastic oracle", fontsize=19)
        self.labels = ["Disk+exp, exact", "Square+exp, exact", "Disk+exp, MC", "Square+exp, MC"]
    def run_disk_mc(self):
        for i, r in enumerate(self.common_pars["disk_radii"]):
            oracle = CheckExponentiallyDecayingProfile(self.common_pars["origin"], r, self.common_pars["decay_length"], cubic=False)
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle)
            mc.run()
            self.mc_disk_r2[i] = mc.get_r2()
    def run_square_mc(self):
        for i, l in enumerate(self.common_pars["square_sides"]):
            oracle = CheckExponentiallyDecayingProfile(self.common_pars["origin"], l / 2, self.common_pars["decay_length"], cubic=True)
            mc = OracleMCR2(self.common_pars, self.cloud_pars, oracle)
            mc.run()
            self.mc_square_r2[i] = mc.get_r2()
            
    def make_plot(self):
        symbols = ["s", "o", "^", "v"]
        rs = self.common_pars["disk_radii"]
        ls = self.common_pars["square_sides"]
        rsp = np.linspace(np.amin(rs), np.amax(rs), 1000)
        lsp = np.linspace(np.amin(ls), np.amax(ls), 1000)
        plt.plot(rsp, np.asarray([get_disk_exp_mean_r2(r, self.common_pars["decay_length"]) for r in rsp]), label=self.labels[0])
        plt.plot(rsp, get_disk_mean_r2(rsp), "--", label="Disk, exact")
        plt.plot(lsp, np.asarray([get_square_exp_mean_r2(l / 2, self.common_pars["decay_length"]) for l in lsp]), label=self.labels[1])
        plt.plot(lsp, get_square_mean_r2(lsp), "--", label="Square, exact")
        plt.plot(rs, self.mc_disk_r2, symbols[0], label=self.labels[2])
        plt.plot(ls, self.mc_square_r2, symbols[1], label=self.labels[3])
        self.save_and_close()
    

if __name__ == "__main__":
    disc_radii = np.linspace(1, 10, 20)
    square_sides = np.linspace(1, 10, 20)
    common_pars = dict([("disk_radii", disc_radii),
        ("square_sides", square_sides),
        ("mc_steps", 10000),
        ("origin", np.zeros(2)),
        ("decay_length", 1)])
    cloud_pars = dict([("nr_points", 100),
        ("radius", 1)])
    dp = DeterministicPlot(common_pars, cloud_pars)
    dp.run()
    sp = StochasticPlot(common_pars, cloud_pars)
    sp.run()
