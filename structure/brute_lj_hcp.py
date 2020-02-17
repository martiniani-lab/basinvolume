from __future__ import division
from __future__ import print_function

from builtins import str
import argparse as ap
import numpy as np
import os

from pele.optimize import LBFGS_CPP
from pele.optimize import ModifiedFireCPP
from pele.potentials._lj_cpp import LJCut
from pele.potentials._lj_cpp import LJCutCellLists

from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import UniformRectangularSampling
from mcpele.monte_carlo import NullPotential

from basinvolume.monte_carlo import CheckMinimumIsHCP
from basinvolume.utils import BasicPlot
from basinvolume.utils import to_string

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pele.optimize import SteepestDescentCPP
except ImportError as err:
    print(err)

class MC(_BaseMCRunner):
    def set_control(self, tmp):
        self.set_temperature(tmp)
            
class BruteComputer(BasicPlot):
    def __init__(self, common_pars, Q4_pars, opt_pars):
        self.common_pars = common_pars
        self.Q4_pars = Q4_pars
        self.opt_pars = opt_pars
        self.boxvec = self.common_pars["boxvec"]
        if (0.5 * np.amin(self.boxvec) < 2.5):
            self.optimizer_potential = LJCut(boxvec=self.boxvec)
        else:
            self.optimizer_potential = LJCutCellLists(boxvec=self.boxvec)
        self.x_ini = np.ones(self.common_pars["nr_particles"] * 3)
        #self.optimizer = SteepestDescentCPP(self.x_ini, self.optimizer_potential, tol=self.opt_pars["tol"], nsteps=self.opt_pars["max_iter"])
        #self.optimizer = LBFGS_CPP(self.x_ini, self.optimizer_potential, tol=self.opt_pars["tol"], nsteps=self.opt_pars["max_iter"])
        self.optimizer = ModifiedFireCPP(self.x_ini, self.optimizer_potential, tol=self.opt_pars["tol"], nsteps=self.opt_pars["max_iter"], maxstep=self.opt_pars["maxstep"])
        self.conftest_check_minimum_is_hcp = CheckMinimumIsHCP(optimizer=self.optimizer,
            Q4tol=self.Q4_pars["tol"], boxvec=self.boxvec,
            rcut=self.Q4_pars["rcut"], verbose=self.Q4_pars["verbose"],
            fixed_distance_cutoff=self.Q4_pars["fixed_distance_cutoff"],
            record_q4_histogram=self.Q4_pars["record_histogram"],
            nr_bins=self.Q4_pars["nr_bins"])
        self.mc_potential = NullPotential()
        self.temperature = 1
        self.mc = MC(self.mc_potential, self.x_ini, self.temperature, self.common_pars["nr_samples"])
        self.mc.set_report_steps(0)
        self.step = UniformRectangularSampling(rseed=42, boxvec=self.boxvec)
        self.mc.set_takestep(self.step)
        self.mc.add_conf_test(self.conftest_check_minimum_is_hcp)
        self.mc.set_print_progress()
                
    def run_bv(self):
        self.mc.run()
        p = self.mc.get_accepted_fraction()
        print(("nr times hcp found", p * self.common_pars["nr_samples"]))
        self.volume = np.exp(np.log(p) + self.common_pars["log_accessible_volume"])
        self.log_volume = np.log(p) + self.common_pars["log_accessible_volume"]
        if p > 0:
            self.nr_attempts = 1. / p
        else:
            self.nr_attempts = None
        if self.Q4_pars["record_histogram"]:
            self.print_Q4_histogram()
            
    def print_Q4_histogram(self):
        hist_x = self.conftest_check_minimum_is_hcp.get_hist_x()
        hist_y = self.conftest_check_minimum_is_hcp.get_hist_y()
        hist_ey = self.conftest_check_minimum_is_hcp.get_hist_ey()
        basic_pars = "N_" + to_string(self.common_pars["nr_particles"], 0) + \
            "_samples_" + to_string(self.common_pars["nr_samples"], 0) + \
            "_sann_" + str(not self.Q4_pars["fixed_distance_cutoff"])
        np.savetxt(basic_pars + "_hist_x.txt", hist_x)
        np.savetxt(basic_pars + "_hist_y.txt", hist_y)
        np.savetxt(basic_pars + "_hist_ey.txt", hist_ey)
        if self.common_pars["plot"]:
            plt.errorbar(hist_x, hist_y, yerr=hist_ey, fmt="s--")
            plt.xlabel(r"Local bond order $Q_4$")
            plt.ylabel(r"PDF")
            # https://philbull.wordpress.com/2012/04/05/drawing-arrows-in-matplotlib/
            plt.arrow(7/72,  plt.axes().get_ylim()[1], 0, -0.75, fc="k", ec="k", head_width=0.007, head_length=0.2, label="hcp")
            plt.arrow(0.191, plt.axes().get_ylim()[1], 0, -0.75, fc="k", ec="k", head_width=0.007, head_length=0.2, label="fcc")
            self.out_name = basic_pars + "_histogram.pdf"
            self.save_and_close()

if __name__ == "__main__":
    r = 0.5 * (2 ** (1./6.))
    p = ap.ArgumentParser()
    p.add_argument("-Nroot3", type=int, help="N == nroot3 ** 3")
    p.add_argument("-nr_samples", type=int, help="number of trials")
    p.add_argument("--plot", action="store_true", default=False)
    args = p.parse_args()
    N = args.Nroot3 ** 3
    print(("r", r))
    print(("2r", 2 * r))
    print(("N", N))
    bv = np.asarray([2 * r, np.sqrt(3) * r, np.sqrt(6) * 2 / 3 * r]) * args.Nroot3
    common_pars = dict([("nr_samples", args.nr_samples),
        ("nr_particles", N), ("log_accessible_volume", N * np.log(np.prod(bv))),
        ("boxvec", bv), ("plot", args.plot)])
    opt_pars = dict([("tol", 1e-12), ("max_iter", 1e9), ("maxstep", 0.1)])
    Q4_pars = dict([("tol", 0.05), ("rcut", 1.3), ("verbose", False),
        ("fixed_distance_cutoff", False), ("record_histogram", True),
        ("nr_bins", 200)])
    c = BruteComputer(common_pars, Q4_pars, opt_pars)
    c.run_bv()
    print(("common_pars", common_pars))
    print(("Q4_pars", Q4_pars))
    print(("log_volume", c.log_volume))
    print(("nr_attempts", c.nr_attempts))
    print(to_string(N, 0) + " " + to_string(common_pars["log_accessible_volume"]) + " " + to_string(common_pars["nr_samples"], 0) + " " + to_string(Q4_pars["tol"]) + " " + to_string(c.log_volume) + " " + to_string(c.nr_attempts))
    
