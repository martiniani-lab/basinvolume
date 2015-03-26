from __future__ import division
import numpy as np

from mcpele.monte_carlo import _BaseMCRunner
from basinvolume.monte_carlo import RecordDisplacementTimeseries

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
                 nparticles=None,
                 bdim=1):
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
        self.bdim = bdim
        print("forwarded input")
        if self.pot_optimizer is None or self.origin is None or self.optimizer is None or self.conftest_outer_sphere is None or self.conftest_check_same_minimum is None or self.metropolis is None or self.takestep is None or self.potential is None or self.nparticles is None:
            raise Exception("GaussianBenchmarkKminRun: illegal input")
        print("checked input")
        super(GaussianBenchmarkKminRun, self).__init__(self.potential,
                                            self.origin, 1, self.niter)
        print("constructed super")
        self.red_origin = self.origin
        self.set_report_steps(self.adjustf_niter)
        self.set_control(0)
        if self.action_record_displ is not None:
            self.add_action(self.action_record_displ)
        self.set_takestep(self.takestep)
        self.add_conf_test(self.conftest_outer_sphere)
        self.add_late_conf_test(self.conftest_check_same_minimum)
        self.add_accept_test(self.metropolis)
        ts_niter = niter
        ts_freq = 1
        self.time_series = RecordDisplacementTimeseries(self.red_origin, self.bdim, ts_niter, ts_freq)
        self.add_action(self.time_series)
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
    def dump_timeseries(self, fname, clear=True):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.time_series.get_time_series())
        np.savetxt(fname, timeseries)
        if clear:
            self.time_series.clear()
        return timeseries
    def get_timeseries(self):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.time_series.get_time_series())
        return timeseries
    def check_convergence(self, nr_steps_to_check=10000, rel_std_threshold=0.05):
        return self.time_series.check_convergence(nr_steps_to_check=nr_steps_to_check,
                                                   rel_std_threshold=rel_std_threshold)
        
        
