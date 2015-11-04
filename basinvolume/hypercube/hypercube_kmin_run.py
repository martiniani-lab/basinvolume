from __future__ import division
import numpy as np

from mcpele.monte_carlo import _BaseMCRunner
from basinvolume.monte_carlo import RecordDisplacementTimeseries, CheckHyperCubicContainer, RecordStepsTimeseries
from mcpele.monte_carlo import MetropolisTest, RandomCoordsDisplacement

class HypercubeMCRunner(_BaseMCRunner):
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 sidelength=1, k=1.0, acceptance=0.2, adjustf=0.9,
                 adjustf_niter=1e4, adjustf_navg=100, pt_eq_niter=0,
                 ts_niter=None, ts_freq=1, seeds=None, 
                 record_steps_timeseries=False, record_steps_timeseries_every=[1], 
                 single=False):
        #construct base class
        super(HypercubeMCRunner, self).__init__(potential, full_coords, temperature, niter)
        
        self.nparticles = 1
        self.bdim = len(full_coords)
        self.ndof = self.bdim
        self.origin = np.array(origin)
        self.rattlers = np.ones(self.bdim)
        self.sidelength = sidelength
        self.set_control(k)
        self.equilibration_steps = adjustf_niter + pt_eq_niter
        if ts_niter is None:
            ts_niter = niter
        
        #compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max),
                    seed_metropolis=np.random.randint(i32max))
        self.seeds = seeds
        
        self.conftest = CheckHyperCubicContainer(self.origin, sidelength, self.bdim)
        self.action_record_displ = RecordDisplacementTimeseries(self.red_origin, self.bdim, ts_niter, ts_freq)
        self.metropolis = MetropolisTest(self.seeds['seed_metropolis'])
        
        self.set_report_steps(adjustf_niter)
        self.takestep = RandomCoordsDisplacement(self.seeds['seed_takestep'], stepsize, report_interval=adjustf_navg,
                                                  factor=adjustf, min_acc_ratio=acceptance, max_acc_ratio=acceptance,
                                                  single=single, nparticles=self.nparticles, bdim=self.bdim)
        
        #set up pele:MC
        self.set_takestep(self.takestep)
        self.add_accept_test(self.metropolis) #metropolis uses the harmonic potential
        self.add_late_conf_test(self.conftest)
        self.add_action(self.action_record_displ)
        if record_steps_timeseries:
            self.steps_timeseries_list = []
            self.record_steps_timeseries_every = record_steps_timeseries_every
            for freq in self.record_steps_timeseries_every:
                self.steps_timeseries_list.append(RecordStepsTimeseries(self.red_origin, self.rattlers, self.bdim, ts_niter, freq))
            for action in self.steps_timeseries_list:
                self.add_action(action)
        
    def set_control(self, c):
        """set temperature, canonical control parameter"""
        self.k = c
        self.potential.set_k(c)
        self.reset_energy()
    
    def get_stepsize(self):
        return self.takestep.get_stepsize()
    
    def run_kmin(self):
        print("run kmin")
        print("coords initial", self.get_coords())
        self.set_print_progress()
        self.run()
        print("coords final", self.get_coords())
    
    def get_displ2_kmin(self):
        print("recorded steps for displ2", self.action_record_displ.get_count())
        return self.action_record_displ.get_mean_variance()
    
    def dump_timeseries(self, fname, clear=True):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.action_record_displ.get_time_series())
        np.savetxt(fname, timeseries)
        if clear:
            self.action_record_displ.clear()
        return timeseries
    
    def get_timeseries(self):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.action_record_displ.get_time_series())
        return timeseries
    
    def check_convergence(self, nr_steps_to_check=10000, rel_std_threshold=0.05):
        return self.action_record_displ.check_convergence(nr_steps_to_check=nr_steps_to_check,
                                                   rel_std_threshold=rel_std_threshold)
        
        
