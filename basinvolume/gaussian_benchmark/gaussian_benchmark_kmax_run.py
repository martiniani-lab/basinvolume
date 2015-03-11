from __future__ import division
import numpy as np
from mcpele.monte_carlo import SampleGaussian
from mcpele.monte_carlo import SampleGaussian
from mcpele.monte_carlo import MetropolisTest, CheckSphericalContainer
from basinvolume.monte_carlo import CheckSameMinimum

class GaussianBenchmarkKmaxRun(object):
    def __init__(self, pot_optimizer=None, origin=None, dtol=1e-3, optimizer=None, seeds=None, conftest_outer_sphere=None, conftest_check_same_minimum=None, action_findk=None):
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
        #
        if self.pot_optimizer is None or self.origin is None or self.optimizer is None or self.conftest_outer_sphere is None or self.conftest_check_same_minimum is None or self.action_findk is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input")
        #
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max))
        self.seeds = seeds
        self.takestep = SampleGaussian(self.seeds['seed_takestep'], stepsize, self.origin)
        self.rattlers = np.ones(self.origin.size())
