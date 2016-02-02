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
    def run(self, nr_iterations):
        for _ in xrange(nr_iterations):
            self.one_iteration()
            
def run_brute(common_pars):
    """
    Run brute force basin computation for LJ crystal.
    """
    run_basin_computer(BruteComputer, common_pars)

if __name__ == "__main__":
    common_pars = dict([("nr_samples", int(1e5)),
        ("nr_particles", 16)])
    run_brute()
