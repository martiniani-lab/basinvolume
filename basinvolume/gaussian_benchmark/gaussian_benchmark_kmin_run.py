from __future__ import division
import numpy as np

from mcpele.monte_carlo import _BaseMCRunner

class GaussianBenchmarkKminRun(_BaseMCRunner):
    def __init__(self, pot_optimizer=None, origin=None, optimizer=None,
                 seeds=None, conftest_outer_sphere=None,
                 conftest_check_same_minimum=None,
                 action_record_displ=None):
        self.pot_optimizer = pot_optimizer
        self.origin = origin
        self.optimizer = optimizer
        self.conftest_outer_sphere = conftest_outer_sphere
        self.conftest_check_same_minimum = conftest_check_same_minimum
        self.action_record_displ = action_record_displ
        if self.pot_optimizer is None or self.origin is None or self.optimizer is None or self.conftest_outer_sphere is None or self.conftest_check_same_minimum is None or self.action_record_displ is None:
            raise Exception("GaussianBenchmarkKminRun: illegal input")
    def set_control(self, c):
        """set temperature, canonical control parameter"""
        self.k = c
        self.potential.set_k(c)
        self.reset_energy()
        
