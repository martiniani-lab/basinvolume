from __future__ import division
import numpy as np
from pele.optimize import ModifiedFireCPP
from mcpele.monte_carlo import CheckSphericalContainer
from basinvolume.monte_carlo import CheckSameMinimum
from gaussian_benchmark_kmax_run import GaussianBenchmarkKmaxRun

class GaussianBenchmark(object):
    def __init__(self, means=None, cov=None, minimum_index=0, opt_dtmax=1, opt_maxstep=0.6, opt_tol=1e-4, opt_nsteps=1e5, radius_container=10, bdim=2):
        self.means = means
        self.cov = cov
        self.minimum_index = minimum_index
        self.opt_dtmax = opt_dtmax
        self.opt_maxstep = opt_maxstep
        self.opt_tol = opt_tol
        self.opt_nsteps = opt_nsteps
        self.radius_container = radius_container
        self.bdim = bdim
        #
        if self.means is None:
            raise Exception("GaussianBenchmark: illegal input: means")
        if self.cov is None:
            raise Exception("GaussianBenchmark: illegal input: cov")
    def set_up_potential(self):
        print("set up potential")
        self.pot_optimizer = SumGaussianPot(self.means, self.cov)
        self.origin = self.get_minimum_coords(index=self.minimum_index)
        self.optimizer = ModifiedFireCPP(self.origin, self.pot_optimizer, dtmax=self.opt_dtmax, maxstep=self.opt_maxstep, tol=self.opt_tol, nsteps=opt_nsteps)
        self.conftest_outer_sphere = CheckSphericalContainer(self.radius_container, self.bdim)
        self.rattlers = np.ones(self.origin.size())
        self.use_cgd = False
        self.conftest_check_same_minimum = CheckSameMinimum(self.pot_optimizer, self.origin, self.rattlers, self.dtol, opt=self.optimizer, opt_tol=opt_tol, opt_maxiter=opt_nsteps, bdim=self.bdim, use_cgd=self.use_cgd, perform_convergence_test=False, collect_minima_list=False)
    def find_kmax(self):
        print("find kmax")
        kmax_run = GaussianBenchmarkKmaxRun(pot_optimizer=self.pot_optimizer, origin=self.origin, optimizer=self.optimizer, conftest_outer_sphere=self.conftest_outer_sphere)
        kmax_run.run()
        self.kmax = kmax_run.kmax
    def run_kmin(self):
        print("run kmin")
        kmin_run = GaussianBenchmarkKminRun()
        kmin_run.run()
        self.displ2_kmin = kmin_run.displ2_kmin
    def run_PT(self):
        print("run PT")
        pt_run = GaussianBenchmarkPTRun()
        pt_run.run()
        self.k, self.displ2 = pt_run.get_k_displ2()
    def compute_volume(self):
        print("compute volume")
        print("k", self.k)
        print("displ2", self.displ2)

if __name__ == "__main__":
    means = [
    [-0.66188835, -4.90248303],
    [-2.50068746,  1.00984605],
    [ 2.79156189,  3.46309925],
    [ 4.7283517, -1.92263871],
    [-7.2936999,  -2.33272689],
    [-5.75772061,  6.56907688],
    [ 1.26304236,  8.80647431],
    [ 7.82900305,  2.89542514],
    [ 5.35866288, -7.17580499],
    [-5.16164399, -7.13075119]        
    ]
    cov = [
    [ 2.89579414,  2.89579414],
    [ 3.27805735,  3.27805735],
    [ 2.36765046,  2.36765046],
    [ 4.02608698,  4.02608698],
    [ 1.87897462,  1.87897462],
    [ 3.45861227,  3.45861227],
    [ 4.63020449,  4.63020449],
    [ 5.88827858,  5.88827858],
    [ 1.97514666,  1.97514666],
    [ 1.62236091,  1.62236091]
    ]
    bm = GaussianBenchmark(means=means, cov=cov, minimum_index=0)
    bm.set_up_potential()
    bm.find_kmax()
    bm.run_kmin()
    bm.run_PT()
    bm.compute_volume()
