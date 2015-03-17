from __future__ import division
import numpy as np
import copy
from pele.optimize import ModifiedFireCPP
from pele.potentials import SumGaussianPot
from pele.potentials import Harmonic
from mcpele.monte_carlo import CheckSphericalContainerConfig
from mcpele.monte_carlo import RandomCoordsDisplacement
from mcpele.monte_carlo import MetropolisTest
from basinvolume.monte_carlo import CheckSameMinimumConfig
from basinvolume.monte_carlo import Findk
from basinvolume.monte_carlo import RecordDisp2Histogram
from gaussian_benchmark_kmax_run import GaussianBenchmarkKmaxRun
from gaussian_benchmark_kmin_run import GaussianBenchmarkKminRun

class GaussianBenchmark(object):
    def __init__(self,
                 means=None,
                 cov=None,
                 minimum_index=0,
                 opt_dtmax=1,
                 opt_maxstep=0.6,
                 opt_tol=1e-4,
                 opt_nsteps=1e5,
                 radius_container=10,
                 bdim=1,
                 avgcount=1e6,
                 ktarget=0.75,
                 knavg=500,
                 ktol=0.05,
                 hmin=0,
                 hmax=1,
                 binsize=0.005,
                 dtol=1e-2,
                 adjustf_niter=1e4,
                 pt_eq_niter=1e3,
                 seeds=None):
        self.means = means
        self.cov = cov
        self.minimum_index = minimum_index
        self.opt_dtmax = opt_dtmax
        self.opt_maxstep = opt_maxstep
        self.opt_tol = opt_tol
        self.opt_nsteps = opt_nsteps
        self.radius_container = radius_container
        self.bdim = bdim
        self.avgcount = avgcount
        self.ktarget = ktarget
        self.knavg = knavg 
        self.ktol = ktol
        self.hmin = hmin
        self.hmax = hmax
        self.binsize = binsize
        self.dtol = dtol
        self.adjustf_niter = adjustf_niter
        self.pt_eq_niter = pt_eq_niter
        self.equilibration_steps = adjustf_niter + pt_eq_niter
        if self.means is None:
            raise Exception("GaussianBenchmark: illegal input: means")
        if self.cov is None:
            raise Exception("GaussianBenchmark: illegal input: cov")
        #####
        #self.pot_optimizer = Harmonic(np.ones(2), 42, bdim=1, com=False)
        self.pot_optimizer = SumGaussianPot(np.ones(self.means.shape), self.cov)
        #self.pot_optimizer = SumGaussianPot(self.means, self.cov)
        #####
        self.optimizer = ModifiedFireCPP(self.means[self.minimum_index][:], self.pot_optimizer, dtmax=self.opt_dtmax, maxstep=self.opt_maxstep, tol=self.opt_tol, nsteps=opt_nsteps)
        self.find_origin()
        print("self.origin.size", self.origin.size)
        self.rattlers = np.ones(self.origin.size)
        self.use_cgd = False
        self.conftest_outer_sphere = CheckSphericalContainerConfig(self.radius_container)
        self.conftest_check_same_minimum = CheckSameMinimumConfig(self.pot_optimizer, self.origin, self.dtol, opt=self.optimizer, opt_tol=opt_tol, opt_maxiter=opt_nsteps)
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max),
                    seed_metropolis=np.random.randint(i32max))
        self.seeds = seeds
        stepsize = 0.1
        adjustf_navg = 20
        acceptance = 0.2
        adjustf = 0.9
        single = False
        self.takestep = RandomCoordsDisplacement(self.seeds['seed_takestep'], stepsize, report_interval=adjustf_navg,
                                                  factor=adjustf, min_acc_ratio=acceptance, max_acc_ratio=acceptance,
                                                  single=single, bdim=self.bdim)
        self.metropolis = MetropolisTest(self.seeds['seed_metropolis'])
        k = 42
        harmonic_com_flag = True
        self.potential = Harmonic(self.origin, k, bdim=self.bdim, com=harmonic_com_flag)
    def find_origin(self):
        print("initial quench")
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
        self.optimizer.reset(origin_result)
        return origin_result
    def find_kmax(self):
        print("find kmax")
        hmin = 0
        hmax = 1
        hbinsize = 0.001
        action_record_displ_kmax = RecordDisp2Histogram(self.origin, self.rattlers, self.bdim, hmin, hmax, hbinsize, 0)
        action_findk = Findk(self.origin, self.rattlers, self.bdim, self.avgcount, self.ktarget, self.knavg, self.ktol, self.hmin, self.hmax, self.binsize)
        kmax_run = GaussianBenchmarkKmaxRun(pot_optimizer=self.pot_optimizer, origin=self.origin, optimizer=self.optimizer, conftest_outer_sphere=self.conftest_outer_sphere, conftest_check_same_minimum=self.conftest_check_same_minimum, action_findk=action_findk, action_record_displ_kmax=action_record_displ_kmax)
        kmax_run.run()
        self.kmax = kmax_run.get_k()
        self.kmax_displ2 = kmax_run.get_displ2()
        print("kmax", self.kmax)
        print("kmax_displ2", self.kmax_displ2)
        print("kmax_displ2 samples", action_record_displ_kmax.get_count())
    def run_kmin(self):
        print("run kmin")
        hmin = 0
        hmax = 1
        hbinsize = 0.001
        print("histogram parameters set")
        action_record_displ_kmin = RecordDisp2Histogram(self.origin,
                              self.rattlers, self.bdim, hmin, hmax,
                              hbinsize, self.equilibration_steps)
        print("histogram action constructed")
        kmin_run = GaussianBenchmarkKminRun(pot_optimizer=self.pot_optimizer,
                   origin=self.origin, optimizer=self.optimizer,
                   conftest_outer_sphere=self.conftest_outer_sphere,
                   conftest_check_same_minimum=self.conftest_check_same_minimum,
                   action_record_displ=action_record_displ_kmin,
                   adjustf_niter=self.adjustf_niter,
                   pt_eq_niter=self.pt_eq_niter,
                   equilibration_steps=self.equilibration_steps,
                   metropolis=self.metropolis,
                   takestep=self.takestep,
                   potential=self.potential)
        print("kmin run constructed")
        kmin_run.run_kmin()
        self.displ2_kmin_mean, self.displ2_kmin_variance = kmin_run.get_displ2_kmin()
        print("displ2_kmin", self.displ2_kmin_mean)
        print("displ2_kmin_variance", self.displ2_kmin_variance)
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
    means = np.asarray([
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
    ])
    cov = np.asarray([
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
    ])
    bm = GaussianBenchmark(means=means, cov=cov, minimum_index=0)
    bm.find_kmax()
    bm.run_kmin()
    #bm.run_PT()
    #bm.compute_volume()
