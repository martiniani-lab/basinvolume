from __future__ import division
import numpy as np

from mcpele.monte_carlo import _BaseMCRunner

class GaussianBenchmarkKminRun(_BaseMCRunner):
    def __init__(self,
                 pot_optimizer=None,
                 origin=None,
                 optimizer=None,
                 seeds=None,
                 conftest_outer_sphere=None,
                 conftest_check_same_minimum=None,
                 action_record_displ=None,
                 adjustf_niter=None,
                 pt_eq_niter=None,
                 equilibration_steps=None,
                 metropolis=None,
                 takestep=None,
                 niter=1e7,
                 potential=None,
                 nparticles=None):
        print("constructing GaussianBenchmarkKminRun")
        self.pot_optimizer = pot_optimizer
        self.origin = origin
        self.optimizer = optimizer
        self.conftest_outer_sphere = conftest_outer_sphere
        self.conftest_check_same_minimum = conftest_check_same_minimum
        self.action_record_displ = action_record_displ
        self.adjustf_niter = adjustf_niter
        self.pt_eq_niter = pt_eq_niter
        self.equilibration_steps = equilibration_steps
        self.metropolis = metropolis
        self.takestep = takestep
        self.niter = niter
        self.potential = potential
        self.nparticles = nparticles
        print("forwarded input")
        if self.pot_optimizer is None or self.origin is None or self.optimizer is None or self.conftest_outer_sphere is None or self.conftest_check_same_minimum is None or self.action_record_displ is None or self.metropolis is None or self.takestep is None or self.potential is None or self.nparticles is None:
            raise Exception("GaussianBenchmarkKminRun: illegal input")
        print("checked input")
        super(GaussianBenchmarkKminRun, self).__init__(self.potential,
                                            self.origin, 1, self.niter)
        print("constructed super")
        self.set_report_steps(self.adjustf_niter)
        self.set_control(0)
        self.add_action(self.action_record_displ)
        self.set_takestep(self.takestep)
        self.add_conf_test(self.conftest_outer_sphere)
        self.add_late_conf_test(self.conftest_check_same_minimum)
        self.add_accept_test(self.metropolis)
    def run_kmin(self):
        print("run kmin")
        print("coords initial", self.get_coords())
        self.set_print_progress()
        self.run()
        print("coords final", self.get_coords())
    def get_displ2_kmin(self):
        print("recorded steps for displ2", self.action_record_displ.get_count())
        return self.action_record_displ.get_mean_variance()
    def set_control(self, c):
        """set temperature, canonical control parameter"""
        self.k = c
        self.potential.set_k(c)
        self.reset_energy()
        
