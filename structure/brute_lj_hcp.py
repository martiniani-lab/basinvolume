from __future__ import division

import numpy as np
import os

from pele.optimize import ModifiedFireCPP
from pele.potentials._lj_cpp import LJCut

from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import UniformCubicSampling
from mcpele.monte_carlo import NullPotential

from basinvolume.monte_carlo import CheckMinimumIsHCP

class MC(_BaseMCRunner):
    def set_control(self, tmp):
        self.set_temperature(tmp)
            
class BruteComputer(object):
    def __init__(self, common_pars, Q4_pars):
        self.common_pars = common_pars
        self.Q4_pars = Q4_pars
        self.boxvec = self.common_pars["boxvec"]
        self.optimizer_potential = LJCut(boxvec=self.boxvec)
        self.x_ini = np.ones(self.common_pars["nr_particles"] * 3)
        self.optimizer = ModifiedFireCPP(self.x_ini, self.optimizer_potential)
        self.conftest_check_minimum_is_hcp = CheckMinimumIsHCP(optimizer=self.optimizer, Q4tol=self.Q4_pars["tol"], boxvec=self.boxvec, rcut=self.Q4_pars["rcut"])
        self.mc_potential = NullPotential()
        self.temperature = 1
        self.mc = MC(self.mc_potential, self.x_ini, self.temperature, self.common_pars["nr_samples"])
        self.mc.set_report_steps(0)
        self.step = UniformCubicSampling(rseed=42, delta=self.boxvec)
        self.mc.set_takestep(self.step)
        self.mc.add_conf_test(self.conftest_check_minimum_is_hcp)
                
    def run_bv(self):
        self.mc.run()
        p = self.mc.get_accepted_fraction()
        self.volume = np.exp(np.log(p) + self.common_pars["log_accessible_volume"])

if __name__ == "__main__":
    r = 1
    log3N = 3
    N = log3N ** 3
    bv = np.asarray([2 * r, np.sqrt(3) * r, np.sqrt(6) * 2 / 3 * r]) * log3N
    common_pars = dict([("nr_samples", int(1e5)),
        ("nr_particles", N), ("log_accessible_volume", N * np.log(np.prod(bv))),
        ("boxvec", bv)])
    Q4_pars = dict([("tol", 1e-10), ("rcut", 2.1)])
    c = BruteComputer(common_pars, Q4_pars)
    c.run_bv()
    print("common_pars", common_pars)
    print("Q4_pars", Q4_pars)
    print("volume", c.volume)
