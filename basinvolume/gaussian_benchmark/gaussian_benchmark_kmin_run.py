from __future__ import division
import numpy as np

from mcpele.monte_carlo import _BaseMCRunner

class GaussianBenchmarkKminRun(_BaseMCRunner):
    def __init__(self, pot_optimizer=None, origin=None, optimizer=None,
                 seeds=None, conftest_outer_sphere=None,
                 conftest_check_same_minimum=None,
                 action_record_displ=None,
                 adjustf_niter=None,
                 pt_eq_niter=None,
                 equilibration_steps=None,
                 seeds=None,
                 metropolis=None):
        self.pot_optimizer = pot_optimizer
        self.origin = origin
        self.optimizer = optimizer
        self.conftest_outer_sphere = conftest_outer_sphere
        self.conftest_check_same_minimum = conftest_check_same_minimum
        self.action_record_displ = action_record_displ
        self.adjustf_niter = adjustf_niter
        self.pt_eq_niter = pt_eq_niter
        self.equilibration_steps = equilibration_steps
        if self.pot_optimizer is None or self.origin is None or self.optimizer is None or self.conftest_outer_sphere is None or self.conftest_check_same_minimum is None or self.action_record_displ is None or metropolis is None:
            raise Exception("GaussianBenchmarkKminRun: illegal input")
        self.seeds = seeds
    def get_displ2_kmin(self):
        return self.action_record_displ.get_mean_variance()
    def set_control(self, c):
        """set temperature, canonical control parameter"""
        self.k = c
        self.potential.set_k(c)
        self.reset_energy()
        
