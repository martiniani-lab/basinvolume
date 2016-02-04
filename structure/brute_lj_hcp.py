from __future__ import division

import numpy as np
import os

from pele.optimize import ModifiedFireCPP

from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import UniformCubicSampling
from mcpele.monte_carlo import NullPotential

from basinvolume.monte_carlo import CheckSameMinimum

class MC(_BaseMCRunner):
    def set_control(self, tmp):
        self.set_temperature(tmp)
            
class BruteComptuer(object):
    def __init__(self, common_pars):
        self.common_pars = common_pars
        self.optimizer_potential = 
        self.optimizer = 
        self.conftest_check_same_minimum = 
        self.mc_potential = NullPotential()
        self.mc = 
        self.step = 
        self.mc.set_takestep(self.step)
        self.mc.add_conf_test(self.conftest_check_same_minimum)
        
    def run_bv(self):
        self.mc.run()
        p = self.mc.get_accepted_fraction()
        self.volume = np.exp(np.log(p) + self.common_pars["log_accessible_volume"])

if __name__ == "__main__":
    common_pars = dict([("nr_samples", int(1e5)),
        ("nr_particles", 16), ("log_accessible_volume", 42)])
    c = BruteComputer(common_pars)
    c.run_bv()
    print("common_pars", common_pars)
    print("volume", c.volume)
