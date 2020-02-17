from __future__ import division
from __future__ import print_function
import numpy as np
from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import CheckSphericalContainer
from mcpele.monte_carlo import MetropolisTest
from mcpele.monte_carlo import NullPotential
from mcpele.monte_carlo import SampleGaussian

class GaussianBenchmarkKmaxRun(_BaseMCRunner):
    def __init__(self, a, b, c, d):
        super(GaussianBenchmarkKmaxRun, self).__init__(a, b, c, d)
    def setup(self, pot_optimizer=None, origin=None, seeds=None,
        conftest_outer_sphere=None, conftest_check_same_minimum=None,
        action_findk=None, niter=None):
        self.pot_optimizer = pot_optimizer
        self.origin = origin
        self.conftest_outer_sphere = conftest_outer_sphere
        self.conftest_check_same_minimum = conftest_check_same_minimum
        self.action_findk = action_findk
        self.niter = niter
        if self.pot_optimizer is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input: pot_optimizer")
        if self.origin is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input: origin")
        if self.conftest_outer_sphere is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input: conftest_outer_sphere")
        if self.conftest_check_same_minimum is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input: conftest_check_same_minimum")
        if self.action_findk is None:
            raise Exception("GaussianBenchmarkKmaxRun: illegal input: action_findk")
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max))
        self.seeds = seeds
        stepsize = 1
        self.takestep = SampleGaussian(self.seeds['seed_takestep'],
                                        stepsize, self.origin)
        self.add_modules_to_mc()
        #self.set_report_steps(self.niter - self.avgcount)
        in_origin = self.origin
        used_origin = self.conftest_check_same_minimum.get_origin()
        print(("in_origin", in_origin))
        print(("used_origin", used_origin))
    def add_modules_to_mc(self):
        self.set_takestep(self.takestep)
        self.add_conf_test(self.conftest_outer_sphere)
        self.add_late_conf_test(self.conftest_check_same_minimum)
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
        print(("p", self.action_findk.get_prob()))
        return self.action_findk.get_prob()
    def get_k(self):
        """ The MC potential is just a placeholder.
        k is determined by takestep.
        """
        stepsize = self.get_stepsize()
        k = 1.0 / (stepsize * stepsize)
        ###
        print(("k", k))
        ###
        return k
    def set_control(self, c):
        """set k"""
        print("WARNING: findk set control is not defined, spring constant is set through stepsize")
