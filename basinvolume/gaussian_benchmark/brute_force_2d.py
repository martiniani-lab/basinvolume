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

class MC(_BaseMCRunner):
    def set_control(self, temp):
        self.set_temperature(temp)

class BruteForce2D(object):
    def __init__(self,
                 means=np.ones((10, 2)),
                 cov=8*np.ones((10, 2)),
                 minimum_index=0,
                 radius_container=10,
                 nr_samples=1e5,
                 opt_dtmax=1,
                 opt_tol=1e-7,
                 opt_nsteps=1e5,
                 opt_maxstep=1,
                 csm_dtol=1e-5
                 ):
        #
        self.means = means
        self.cov = cov
        self.minimum_index = minimum_index
        self.radius_container = radius_container
        self.nr_samples = nr_samples
        self.opt_dtmax = opt_dtmax
        self.opt_tol = opt_tol
        self.opt_nsteps = opt_nsteps
        self.opt_maxstep = opt_maxstep
        self.csm_dtol=csm_dtol
        #
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
                                         verbosity=1)
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
        self.mc = MC(self.potential, self.origin, self.temperature, self.nr_samples)
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
        self.optimizer.reset(origin_result)
        return origin_result 
    def compute_volume(self):
        print("compute volume")
        self.mc.set_print_progress()
        self.mc.run()
        print("self.mc.get_accepted_fraction()", self.mc.get_accepted_fraction())
        p = self.mc.get_accepted_fraction()
        self.basin_volume = p * volume_nball(self.radius_container, self.bdim)
        self.error_basin_volume = np.sqrt(p * (1 - p) / self.nr_samples) * self.basin_volume
        print("basin volume", self.basin_volume)
        print("error bar", self.error_basin_volume)
        print("done")
