from __future__ import division
import numpy as np
from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import CheckSphericalContainer
from mcpele.monte_carlo import MetropolisTest
from mcpele.monte_carlo import NullPotential
from mcpele.monte_carlo import SampleGaussian
from basinvolume.monte_carlo import CheckSameMinimum

class GaussianBenchmarkKmaxRun(_BaseMCRunner):
    def __init__(self, pot_optimizer=None, origin=None, dtol=1e-3, optimizer=None, seeds=None, conftest_outer_sphere=None, conftest_check_same_minimum=None, action_findk=None, niter=1e8):
        self.pot_optimizer = pot_optimizer
        self.origin = origin
        self.dtol = dtol
        self.avgcount = avgcount
        self.ktarget = ktarget
        self.knavg = knavg 
        self.ktol = ktol
        self.optimizer = optimizer
        self.conftest_outer_sphere = conftest_outer_sphere
        self.conftest_check_same_minimum = conftest_check_same_minimum
        self.action_findk = action_findk
        self.niter = niter
        #
        if self.pot_optimizer is None or self.origin is None or self.optimizer is None or self.conftest_outer_sphere is None or self.conftest_check_same_minimum is None or self.action_findk is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input")
        fake_potential = NullPotential()
        super(GaussianBenchmarkKmaxRun, self).__init__(fake_potential, self.origin, 1, self.niter)
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max))
        self.seeds = seeds
        self.takestep = SampleGaussian(self.seeds['seed_takestep'], stepsize, self.origin)
        self.rattlers = np.ones(self.origin.size())
        self.add_modules_to_mc()
    def add_modules_to_mc(self):
        self.set_takestep(self.takestep)
        self.add_conf_test(self.conftest_outer_sphere)
        self.add_conf_test(self.conftest_check_same_minimum)
        self.add_action(self.action_findk)
    def get_stepsize(self):
        return self.takestep.get_stepsize()
    def get_k(self):
        """ The MC potential is just a placeholder. 
        k is determined by takestep.
        """
        stepsize = self.get_stepsize()
        return stepsize ** -2
    
