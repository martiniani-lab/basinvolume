from __future__ import division
import numpy as np
import copy
from pele.potentials import SumGaussianPot
from pele.optimize import ModifiedFireCPP
from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import NullPotential
from mcpele.monte_carlo import UniformSphericalSampling
from basinvolume.monte_carlo import CheckSameMinimumConfig
from basinvolume.utils import volume_nball

class EvalCounter(object):
    def __init__(self):
        self.count = 0

class MC(_BaseMCRunner):
    def set_control(self, temp):
        self.set_temperature(temp)
    def run(self, nr_iterations):
        for _ in xrange(nr_iterations):
            self.one_iteration()

class BruteForce2D(object):
    def __init__(self,
                 means=np.ones((10, 2)),
                 cov=8*np.ones((10, 2)),
                 minimum_index=0,
                 radius_container=10,
                 #max_nr_samples=1e14,
                 max_nr_samples=1e14,
                 #min_nr_samples=1e3,
                 min_nr_samples=1e3,
                 #nr_samples_increment=1e3,
                 nr_samples_increment=1e3,
                 #opt_dtmax=1,
                 opt_dtmax=1,
                 #opt_tol=1e-8,
                 opt_tol=1e-8,
                 #opt_nsteps=1e8,
                 opt_nsteps=1e8,
                 #opt_maxstep=0.01,
                 opt_maxstep=0.1,
                 #csm_dtol=1e-6,
                 csm_dtol=1,
                 #convergence_delta_threshold=1e-5
                 convergence_delta_threshold=1e-5
                 ):
        #
        self.means = means
        self.cov = cov
        self.minimum_index = int(minimum_index)
        self.radius_container = radius_container
        self.max_nr_samples = int(max_nr_samples)
        self.min_nr_samples = int(min_nr_samples)
        self.nr_samples_increment = int(nr_samples_increment)
        self.opt_dtmax = opt_dtmax
        self.opt_tol = opt_tol
        self.opt_nsteps = opt_nsteps
        self.opt_maxstep = opt_maxstep
        self.csm_dtol=csm_dtol
        self.convergence_delta_threshold = convergence_delta_threshold
        #
        self.nr_evaluations = EvalCounter()
        self.ngaussians = self.means.shape[0]
        self.gdim = self.means.shape[1]
        self.bdim = self.gdim
        self.pot_optimizer = SumGaussianPot(self.means, self.cov)
        self.optimizer = ModifiedFireCPP(self.means[self.minimum_index][:],
                                         self.pot_optimizer,
                                         dtmax=self.opt_dtmax,
                                         maxstep=self.opt_maxstep,
                                         tol=self.opt_tol, 
                                         nsteps=opt_nsteps,
                                         verbosity=0)
        self.find_origin()
        self.rattlers = np.ones(self.origin.size)
        self.use_cgd = False
        self.conftest_check_same_minimum = CheckSameMinimumConfig(self.pot_optimizer,
                                           self.origin,
                                           self.csm_dtol,
                                           opt=self.optimizer,
                                           opt_tol=self.opt_tol,
                                           opt_maxiter=self.opt_nsteps)
        self.temperature = 1
        self.potential = NullPotential()
        self.mc = MC(self.potential, self.origin, self.temperature, self.max_nr_samples)
        self.step = UniformSphericalSampling(42, self.radius_container)
        self.mc.set_takestep(self.step)
        self.mc.set_report_steps(0)
        self.mc.add_conf_test(self.conftest_check_same_minimum)
    def find_origin(self):
        self.origin = copy.deepcopy(self.get_local_minimum(mean_index=self.minimum_index))
        print("Gaussian center coords", self.means[self.minimum_index][:])
        print("corresponding mimimum position (origin)", self.origin)
    def get_local_minimum(self, mean_index=0):
        initial_position = self.means[self.minimum_index][:]
        print("initial_position", initial_position)
        self.optimizer.reset(initial_position)
        result = self.optimizer.run()
        print("initial optimization", result.success)
        origin_result = result.coords
        print("self.optimizer.get_niter()", self.optimizer.get_niter())
        print("self.optimizer.get_result().nfev", self.optimizer.get_result().nfev)
        #self.nr_evaluations.count += self.optimizer.get_result().nfev
        self.optimizer.reset(origin_result)
        print("reset: self.optimizer.get_result().nfev", self.optimizer.get_result().nfev)
        return origin_result 
    def compute_volume(self):
        print("compute volume")
        self.mc.set_print_progress()
        self.prev_basin_volume = 0
        keep_running = True
        while keep_running:
            self.mc.run(self.nr_samples_increment)
            print("self.mc.get_accepted_fraction()", self.mc.get_accepted_fraction())
            p = self.mc.get_accepted_fraction()
            print("self.conftest_check_same_minimum.get_nfev()", self.conftest_check_same_minimum.get_nfev())
            self.basin_volume = p * volume_nball(self.radius_container, self.bdim)
            self.error_basin_volume = np.sqrt(p * (1 - p) / self.mc.get_iterations_count()) * self.basin_volume
            print("self.basin_volume", self.basin_volume)
            print("self.error_basin_volume", self.error_basin_volume)
            keep_running = self.check_not_converged()
        self.nr_evaluations.count += self.conftest_check_same_minimum.get_nfev()
        self.nfev = self.nr_evaluations.count
        print("basin volume", self.basin_volume)
        print("error bar", self.error_basin_volume)
        print("nr evaluations", self.nfev)
        print("done")
    def check_not_converged(self):
        delta = np.absolute(self.basin_volume - self.prev_basin_volume) / self.basin_volume
        self.prev_basin_volume = self.basin_volume
        print("delta", delta)
        not_converged = None
        if self.basin_volume == 0:
            not_converged = True
        elif delta == 0:
            not_converged = True
        elif self.mc.get_iterations_count() < self.min_nr_samples:
            not_converged = True
        elif self.mc.get_iterations_count() >= self.max_nr_samples:
            not_converged = False
        else:
            not_converged = delta > self.convergence_delta_threshold
        return not_converged
