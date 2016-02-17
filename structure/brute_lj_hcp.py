from __future__ import division

import numpy as np
import os

from pele.optimize import ModifiedFireCPP
from pele.optimize import LBFGS_CPP
from pele.potentials._lj_cpp import LJCut

from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import UniformRectangularSampling
from mcpele.monte_carlo import NullPotential

from basinvolume.monte_carlo import CheckMinimumIsHCP
from basinvolume.utils import to_string

class MC(_BaseMCRunner):
    def set_control(self, tmp):
        self.set_temperature(tmp)
            
class BruteComputer(object):
    def __init__(self, common_pars, Q4_pars, opt_pars):
        self.common_pars = common_pars
        self.Q4_pars = Q4_pars
        self.opt_pars = opt_pars
        self.boxvec = self.common_pars["boxvec"]
        self.optimizer_potential = LJCut(boxvec=self.boxvec)
        self.x_ini = np.ones(self.common_pars["nr_particles"] * 3)
        self.optimizer = LBFGS_CPP(self.x_ini, self.optimizer_potential, tol=self.opt_pars["tol"], nsteps=self.opt_pars["max_iter"])
        self.conftest_check_minimum_is_hcp = CheckMinimumIsHCP(optimizer=self.optimizer, Q4tol=self.Q4_pars["tol"], boxvec=self.boxvec, rcut=self.Q4_pars["rcut"], verbose=self.Q4_pars["verbose"])
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
        self.volume = np.exp(np.log(p) + self.common_pars["log_accessible_volume"])
        self.log_volume = np.log(p) + self.common_pars["log_accessible_volume"]
        if p > 0:
            self.nr_attempts = 1. / p
        else:
            self.nr_attempts = None

if __name__ == "__main__":
    #r = 0.5 * (2 ** (1./6.))
    r = 0.5
    log3N = 7
    N = log3N ** 3
    bv = np.asarray([2 * r, np.sqrt(3) * r, np.sqrt(6) * 2 / 3 * r]) * log3N
    common_pars = dict([("nr_samples", int(1e4)),
        ("nr_particles", N), ("log_accessible_volume", N * np.log(np.prod(bv))),
        ("boxvec", bv)])
    opt_pars = dict([("tol", 1e-12), ("max_iter", 1e9)])
    Q4_pars = dict([("tol", 0.2), ("rcut", 2.1), ("verbose", False)])
    c = BruteComputer(common_pars, Q4_pars, opt_pars)
    c.run_bv()
    print("common_pars", common_pars)
    print("Q4_pars", Q4_pars)
    print("log_volume", c.log_volume)
    print("nr_attempts", c.nr_attempts)
    print(to_string(N, 0) + " " + to_string(common_pars["log_accessible_volume"]) + " " + to_string(common_pars["nr_samples"], 0) + " " + to_string(Q4_pars["tol"]) + " " + to_string(c.log_volume) + " " + to_string(c.nr_attempts))
