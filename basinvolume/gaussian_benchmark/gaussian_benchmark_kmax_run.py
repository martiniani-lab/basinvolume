from __future__ import division
import numpy as np
from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import CheckSphericalContainer
from mcpele.monte_carlo import MetropolisTest
from mcpele.monte_carlo import NullPotential
from mcpele.monte_carlo import SampleGaussian
from basinvolume.monte_carlo import CheckSameMinimum

class GaussianBenchmarkKmaxRun(_BaseMCRunner):
    def __init__(self,
                 potential,
                 coords,
                 temperature,
                 stepsize,
                 niter,
                 pot_optimizer=None,
                 origin=None,
                 optimizer=None,
                 seeds=None,
                 conftest_outer_sphere=None,
                 conftest_check_same_minimum=None,
                 action_findk=None,
                 avgcount=1e4):
        self.pot_optimizer = pot_optimizer
        self.origin = origin
        self.optimizer = optimizer
        self.conftest_outer_sphere = conftest_outer_sphere
        self.conftest_check_same_minimum = conftest_check_same_minimum
        self.action_findk = action_findk
        self.niter = niter
        self.avgcount = avgcount
        #
        if self.pot_optimizer is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input: pot_optimizer")
        if self.origin is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input: origin")
        if self.optimizer is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input: optimizer")
        if self.conftest_outer_sphere is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input: conftest_outer_sphere")
        if self.conftest_check_same_minimum is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input: conftest_check_same_minimum")
        if self.action_findk is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input: action_findk")
        fake_potential = NullPotential()
        super(GaussianBenchmarkKmaxRun, self).__init__(fake_potential,
                                            self.origin, 1, self.niter)
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max))
        self.seeds = seeds
        self.takestep = SampleGaussian(self.seeds['seed_takestep'],
                                        stepsize, self.origin)
        self.rattlers = np.ones(self.origin.size)
        self.add_modules_to_mc()
        #self.set_report_steps(self.niter - self.avgcount)
    def add_modules_to_mc(self):
        self.set_takestep(self.takestep)
        self.add_conf_test(self.conftest_outer_sphere)
        self.add_conf_test(self.conftest_check_same_minimum)
        self.add_action(self.action_findk)
    def get_stepsize(self):
        return self.takestep.get_stepsize()
    def get_displ2(self):
        displ_k_max, var_displ_k_max = self.action_findk.get_mean_variance()
        return displ_k_max
    def get_var_displ_kmax(self):
        displ_k_max, var_displ_k_max = self.action_findk.get_mean_variance()
        return var_displ_k_max
    def get_entries(self):
        return self.action_findk.get_entries()
    def get_prob_kmax(self):
        return self.action_findk.get_prob()
    def get_k(self):
        """ The MC potential is just a placeholder. 
        k is determined by takestep.
        """
        stepsize = self.get_stepsize()
        k = 1.0 / (stepsize * stepsize)
        ###
        print("k", k)
        ###
        return k
    def set_control(self, c):
        """set k"""
        print("WARNING: findk set control is not defined, spring constant is set through stepsize")
