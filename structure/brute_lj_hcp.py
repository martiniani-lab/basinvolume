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
            
def run_brute(brute_parameters):
    return

if __name__ == "__main__":
    brute_parameters = dict([("nr_samples", int(1e5)),
        ("nr_particles", 16)])
    run_brute()
