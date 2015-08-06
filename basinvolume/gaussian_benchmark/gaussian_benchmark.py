from __future__ import division
import os
import copy
import subprocess
import shlex
import shutil
import numpy as np
from pele.optimize import ModifiedFireCPP
from pele.potentials import SumGaussianPot
from pele.potentials import Harmonic
from mcpele.monte_carlo import CheckSphericalContainer
from mcpele.monte_carlo import RandomCoordsDisplacement
from mcpele.monte_carlo import MetropolisTest
import basinvolume
from basinvolume.monte_carlo import CheckSameMinimumConfig
from basinvolume.monte_carlo import Findk
from basinvolume.monte_carlo import RecordDisp2Histogram
from basinvolume.utils import trymakedir
from basinvolume.utils import ResultsFile
from basinvolume.utils import to_string
from basinvolume.utils import volume_nball
from basinvolume.utils import trymakedir
from basinvolume.post_processing import F_Basin_From_MC_Data
from basinvolume.post_processing import F_Basin_From_MC_Data_Free_COM
from basinvolume.post_processing import Gauss_Lobatto_abscissas
from brute_force_2d import BruteForce2D
try:
    from gaussian_benchmark_kmax_run import GaussianBenchmarkKmaxRun
    from gaussian_benchmark_kmin_run import GaussianBenchmarkKminRun
    from gaussian_benchmark_pt_run import GaussianBenchmarkPTRun
except:
    print("gaussian import failed")

class GaussianBenchmark(object):
    def __init__(self,
                 means=np.ones((10, 2)),
                 cov=8*np.ones((10, 2)),
                 minimum_index=0,
                 opt_dtmax=1,
                 opt_maxstep=1,
                 opt_tol=1e-7,
                 opt_nsteps=1e5,
                 radius_container=10,
                 bdim=1,
                 avgcount=1e5,
                 ktarget=0.85,
                 knavg=500,
                 ktol=0.05,
                 hmin=0,
                 hmax=1,
                 binsize=0.005,
                 dtol=1e-5,
                 adjustf_niter=1e3,
                 pt_eq_niter=1e2,
                 seeds=None,
                 nprocs=7,
                 totniter=2e5,
                 harmonic_com_flag=False,
                 kmin_niter=1e5,
                 harmonic_well=False,
                 kmax_niter=2e5,
                 simple_integrator=False):
        self.means = means
        self.cov = cov
        self.minimum_index = minimum_index
        self.opt_dtmax = opt_dtmax
        self.opt_maxstep = opt_maxstep
        self.opt_tol = opt_tol
        self.opt_nsteps = opt_nsteps
        self.radius_container = radius_container
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
        self.nprocs = nprocs
        self.totniter = totniter
        self.harmonic_com_flag = harmonic_com_flag
        self.kmin_niter = kmin_niter
        self.harmonic_well = harmonic_well
        self.kmax_niter = kmax_niter
        self.simple_integrator = simple_integrator
        if self.means is None:
            raise Exception("GaussianBenchmark: illegal input: means")
        if self.cov is None:
            raise Exception("GaussianBenchmark: illegal input: cov")
        self.basic_config_path = os.path.join(os.getcwd(), "gaussian_sum")
        self.ngaussians = self.means.shape[0]
        self.gdim = self.means.shape[1]
        self.bdim = self.gdim
        self.nparticles = 1
        #####
        if self.harmonic_well:
            self.pot_optimizer = Harmonic(np.asarray([0.0, 0.0]), 42, bdim=self.bdim, com=False)
        else:
            self.pot_optimizer = SumGaussianPot(self.means, self.cov)
        for minimum in self.means:
            print "Energy", self.pot_optimizer.getEnergy(minimum)
        print("ENERGY", self.pot_optimizer.getEnergy(np.asarray([10.0, 10.0])))
        #self.pot_optimizer = SumGaussianPot(self.means, self.cov)
        #####
        self.optimizer = ModifiedFireCPP(self.means[self.minimum_index][:],
                                         self.pot_optimizer,
                                         dtmax=self.opt_dtmax,
                                         maxstep=self.opt_maxstep,
                                         tol=self.opt_tol, 
                                         nsteps=opt_nsteps, verbosity=1)
        self.find_origin()
        print("self.origin.size", self.origin.size)
        self.rattlers = np.ones(self.origin.size)
        self.use_cgd = False
        self.conftest_outer_sphere = CheckSphericalContainer(self.radius_container, self.bdim)
        self.conftest_check_same_minimum = CheckSameMinimumConfig(self.pot_optimizer,
                                           self.origin, self.dtol,
                                           opt=self.optimizer, opt_tol=opt_tol,
                                           opt_maxiter=opt_nsteps)
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max),
                    seed_metropolis=np.random.randint(i32max))
        self.seeds = seeds
        stepsize = 5
        adjustf_navg = 100
        acceptance = 0.2
        adjustf = 0.9
        single = False
        self.takestep = RandomCoordsDisplacement(self.seeds['seed_takestep'], stepsize, report_interval=adjustf_navg,
                                                  factor=adjustf, min_acc_ratio=acceptance, max_acc_ratio=acceptance,
                                                  single=single, bdim=self.bdim)
        self.metropolis = MetropolisTest(self.seeds['seed_metropolis'])
        k = 42
        self.potential = Harmonic(self.origin, k, bdim=self.bdim, com=self.harmonic_com_flag)
        self.PES_energy_calls = 0
        self.harmonic_energy_calls = 0
        self.total_neval = 0
        self.print_gaussian_sum_config_file()
    def find_origin(self):
        print("initial quench")
        self.origin = copy.deepcopy(self.get_local_minimum(mean_index=self.minimum_index))
        print("Gaussian center coords", self.means[self.minimum_index][:])
        print("corresponding mimimum position (origin)", self.origin)
        self.print_minimum_coords_file(configuration_name="config0.gauss")
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
    def find_kmax(self):
        print("find kmax")
        print("self.optimizer.get_niter()", self.optimizer.get_niter())
        hmin = 0
        hmax = 1
        hbinsize = 0.001
        action_record_displ_kmax = RecordDisp2Histogram(self.origin,
                                                        self.rattlers,
                                                        self.bdim,
                                                        hmin,
                                                        hmax,
                                                        hbinsize,
                                                        self.kmax_niter - self.avgcount,
                                                        fix_com=self.harmonic_com_flag)
        action_findk = Findk(self.origin,
                             self.rattlers,
                             self.bdim,
                             self.avgcount,
                             self.ktarget,
                             self.knavg,
                             self.ktol,
                             self.hmin,
                             self.hmax,
                             self.binsize,
                             fix_com=self.harmonic_com_flag)
        kmax_run = GaussianBenchmarkKmaxRun(pot_optimizer=self.pot_optimizer,
                                           origin=self.origin,
                                           optimizer=self.optimizer,
                                           conftest_outer_sphere=self.conftest_outer_sphere,
                                           conftest_check_same_minimum=self.conftest_check_same_minimum,
                                           action_findk=action_findk,
                                           action_record_displ_kmax=action_record_displ_kmax,
                                           niter=self.kmax_niter,
                                           avgcount=self.avgcount)
        kmax_run.run()
        self.kmax = kmax_run.get_k()
        self.kmax_displ2 = kmax_run.get_displ2()
        self.prob_kmax = kmax_run.get_prob_kmax()
        self.var_displ_kmax = kmax_run.get_var_displ_kmax()
        self.kmax_displ2_nr_samples = action_record_displ_kmax.get_count()
        print("kmax", self.kmax)
        print("kmax_displ2", self.kmax_displ2)
        print("kmax_displ2 samples", action_record_displ_kmax.get_count())
        print("self.optimizer.get_niter()", self.optimizer.get_niter())
        print("kmax_run.get_nfev()", kmax_run.get_neval())
        self.total_neval += kmax_run.get_neval()
        self.harmonic_energy_calls += kmax_run.get_iterations_count()
        self.print_findk_config_file()
    def run_kmin(self):
        print("run kmin")
        print("self.optimizer.get_niter()", self.optimizer.get_niter())
        hmin = 0
        hmax = 1
        hbinsize = 0.1
        print("histogram parameters set")
        action_record_displ_kmin = RecordDisp2Histogram(self.origin,
                                                        self.rattlers,
                                                        self.bdim,
                                                        hmin,
                                                        hmax,
                                                        hbinsize,
                                                        self.equilibration_steps,
                                                        fix_com=self.harmonic_com_flag)
        print("histogram action constructed")
        kmin_run = GaussianBenchmarkKminRun(pot_optimizer=self.pot_optimizer,
                                            origin=self.origin,
                                            optimizer=self.optimizer,
                                            conftest_outer_sphere=self.conftest_outer_sphere,
                                            conftest_check_same_minimum=self.conftest_check_same_minimum,
                                            action_record_displ=action_record_displ_kmin,
                                            adjustf_niter=self.adjustf_niter,
                                            pt_eq_niter=self.pt_eq_niter,
                                            equilibration_steps=self.equilibration_steps,
                                            metropolis=self.metropolis,
                                            takestep=self.takestep,
                                            potential=self.potential,
                                            niter=self.kmin_niter,
                                            nparticles=self.nparticles)
        print("kmin run constructed")
        ###
        #kmin_run.set_control(self.kmax)
        ###
        kmin_run.run_kmin()
        self.displ2_kmin_mean, self.displ2_kmin_variance = kmin_run.get_displ2_kmin()
        print("displ2_kmin", self.displ2_kmin_mean)
        print("displ2_kmin_variance", self.displ2_kmin_variance)
        print("self.optimizer.get_niter()", self.optimizer.get_niter())
        self.total_neval += kmin_run.get_neval()
        self.harmonic_energy_calls += kmin_run.get_iterations_count()
        print("self.total_neval, kmin, kmax", self.total_neval)
        self.print_kmin_config_file()
    def run_PT(self):
        print("run PT")
        configuration_name="config0.gauss"
        dname = configuration_name[0:-6]
        base_pt_path = os.path.join(os.getcwd(), 'explore_bv_' + str(dname))
        full_path_to_pt_run_script = os.path.join(os.path.dirname(basinvolume.__file__), "gaussian_benchmark", "gaussian_benchmark_pt_run.py")
        cmd_base_str = "mpiexec -n {0} python " + full_path_to_pt_run_script + " {1} {2} {3} {4}"
        cmd = cmd_base_str.format(self.nprocs, "config0.gauss", base_pt_path, int(self.totniter), self.nparticles)
        if self.harmonic_well:
            cmd += " --harmonic_well"
        p = subprocess.call(shlex.split(cmd))
        if p != 0:
            raise Exception("gauss pt run failed")
    def compute_volume(self):
        print("compute volume")
        print("k", self.k)
        print("displ2", self.displ2)
    def print_nr_function_calls(self):
        #print("total nr function calls PES")
        #print("self.optimizer.get_niter()", self.optimizer.get_niter())
        print("self.total_neval", self.total_neval)
        print("self.harmonic_energy_calls", self.harmonic_energy_calls)
    def print_gaussian_sum_config_file(self):
        print("trymakedir", self.basic_config_path)
        trymakedir(self.basic_config_path)
        f = ResultsFile(os.path.join(self.basic_config_path, "gaussian_sum.config"))
        f.set_heading("GAUSSIAN_SUM")
        print("ngaussians", self.ngaussians)
        f.to_file_plain("ngaussians", self.ngaussians)
        f.to_file_plain("bdim", self.bdim)
        f.to_file_plain("gdim", self.gdim)
        f.to_file("radius_container", self.radius_container)
        f.to_file_plain("nparticles", self.nparticles)
        f.close()
        np.savetxt(os.path.join(self.basic_config_path, "gaussian_sum_means.config"), self.means)
        np.savetxt(os.path.join(self.basic_config_path, "gaussian_sum_cov.config"), self.cov)
    def print_findk_config_file(self, configuration_name="config0.gauss"):
        dname = configuration_name[0:-6]
        basic_findk_config_path = os.path.join(os.getcwd(), 'explore_bv_' + str(dname))
        trymakedir(basic_findk_config_path)
        findk_config_name = os.path.join(basic_findk_config_path, "findk_" + dname + ".config")
        print("findk_config_name", findk_config_name)
        f = ResultsFile(findk_config_name)
        f.set_heading("FINDK")
        f.to_file("kmax", self.kmax)
        f.to_file("prob", self.prob_kmax)
        f.to_file("displ_k_max", self.kmax_displ2)
        f.to_file("var_displ_k_max", self.var_displ_kmax)
        f.close()
    def print_kmin_config_file(self, configuration_name="config0.gauss"):
        dname = configuration_name[0:-6]
        basic_kmin_config_path = os.path.join(os.getcwd(), 'explore_bv_' + str(dname))
        trymakedir(basic_kmin_config_path)
        kmin_config_name = os.path.join(basic_kmin_config_path, "kmin_" + dname + ".config")
        f = ResultsFile(kmin_config_name)
        f.set_heading("KMIN")
        f.to_file("displ_k_min", self.displ2_kmin_mean)
        f.to_file("var_displ_k_min", self.displ2_kmin_variance)
        f.close()
    def print_minimum_coords_file(self, configuration_name="config0.gauss"):
        """
        /home/kjs73/projects/basinvolume/gaussian_sum/config0.gauss
        """
        trymakedir(self.basic_config_path)
        f = open(os.path.join(self.basic_config_path, configuration_name), "w")
        for x in self.origin:
            f.write(to_string(x) + "\n")
        f.close()
    def compute_volume(self, configuration_name="config0.gauss"):
        print("computing volume")
        """
        Set analysis base directory.
        """
        self.explore_dir = os.path.join(os.getcwd(), 'explore_bv_' + str(configuration_name[0:-6]))
        self.base_directory = os.path.join(os.path.join(os.getcwd(), 'explore_bv_' + str(configuration_name[0:-6])), "analysis")
        base_directory = self.base_directory
        trymakedir(base_directory)
        """
        Full volume computation, assuming that PT data is available
        """
        self._import_ks()
        self._import_u2_reverse()
        self._print_u2_vs_k()
        self._compute_volume()
        #self._plot_data()
        """
        Print basin volumes for further processing
        """
        self._print_volumes()
        print("computing volume -- done")
    def _import_ks(self):
        """
        must run before import u2
        """
        karray = [] 
        path = os.path.join(self.explore_dir, 'temperatures')
        f = open(path, "r")
        while True:
            k = f.readline()
            if not k: break
            karray.extend([float(k)])
        #prepend kmax
        karray.insert(0, self.kmax)
        self.karray = np.array(karray[::-1], dtype='d')
        print("self.karray", self.karray)
    def _import_u2_reverse(self):
        n = len(self.karray)-1
        self.u2_array = [0 for _ in xrange(n)]
        self.var_array = [0 for _ in xrange(n)] 
        self.std_error_array = [0 for _ in xrange(n)]
        for subdir, dirs, files in os.walk(self.explore_dir):
            for dir in dirs:
                if dir.isdigit():
                    path = os.path.join(self.explore_dir, dir + '/hist_mean')
                    fileHandle = open (path, "r")
                    lineList = fileHandle.readlines()
                    fileHandle.close()
                    niter, u2, var, std_err = lineList[-1].split()
                    self.u2_array[int(dir)] = u2
                    self.var_array[int(dir)] = var
                    self.std_error_array[int(dir)] = std_err
        #prepend u2 kmax
        self.displ_k_max = self.kmax_displ2
        self.var_displ_k_max = self.var_displ_kmax
        self.u2_array.insert(0, self.displ_k_max)
        self.var_array.insert(0, self.var_displ_k_max)
        self.std_error_kmax = np.sqrt(self.var_displ_k_max / self.kmax_displ2_nr_samples)
        self.std_error_array.insert(0, self.std_error_kmax)
        self.u2_array = np.array(self.u2_array[::-1], dtype='d')
        self.var_array = np.array(self.var_array[::-1], dtype='d')
        self.std_error_array = np.array(self.std_error_array[::-1], dtype='d')
    def _print_u2_vs_k(self):
        """writes <u2> and variance vs """
        dname = 'u2_vs_k'
        fname = '{}/{}'.format(self.base_directory,dname)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#{:>15}\t{:>15}\n'.format('<u2>', 'var(<u2>)'))
        for i in xrange(len(self.u2_array)):
            f.write('{:>15.15e}\t{:>15.15e}\n'.format(self.u2_array[i], self.var_array[i])) 
        f.close()
    def _compute_volume(self):
        """
        numerical volume obtained by integrating over the PT data
        Note that to function get_free_energy_F0, we need to pass the array of squared standard errors of the data points to get the correct error bars.
        This was not done previously, so the naming in the subsequent function calls can be confusing, suggesting that we are actually passing the array of variances of the displ2 points.
        """
        #sqared_std_errors = self.var_array # This line is just to illustrate how the code worked before.
        sqared_std_errors = self.std_error_array ** 2
        print("data for integral")
        print("self.u2_array", self.u2_array)
        print("self.karray", self.karray)
        print("self.prob_kmax", self.prob_kmax)
        print("data for integral -- END")
        self.F0unc, self.sigF0unc, self.farrayunc, self.sigfarrayunc = F_Basin_From_MC_Data_Free_COM(self.bdim,
                                                                                                    self.nparticles,
                                                                                                    self.karray,
                                                                                                    self.u2_array,
                                                                                                    self.prob_kmax,
                                                                                                    displ_k_min_trafo=self.displ2_kmin_mean,
                                                                                                    simple_integrator=self.simple_integrator).get_free_energy_F0(sqared_std_errors)
        self.tarray = Gauss_Lobatto_abscissas(len(self.u2_array))()
    def _print_volumes(self):
        dname = 'volume_data'
        fname = '{}/{}'.format(self.base_directory, dname)
        f = open(fname, 'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        def _to_file(name, value):
            f.write((name + ": {}\n").format(to_string(value)))
        f.write('[VOLUME_FULL_PT]\n')
        if hasattr(self, "F0unc"):
            _to_file("F0unc", self.F0unc)
            _to_file("sigF0unc", self.sigF0unc)
        f.close()

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

    def plot_potential(means, cov):    
        import matplotlib.pyplot as plt
        
        N = 200
        xx = np.linspace(-10, 10, N)
        U = np.zeros((N, N))
        pot = SumGaussianPot(means, cov)
        R = 10
        
        for i in xrange(0, N):
            for j in xrange(0, N):
                U[i, j] = pot.getEnergy(np.array([xx[j], xx[i]]))
    
        plot_axes = R + 1
        plt.figure(1)
        plt.clf()
        plt.axes(aspect='equal')
        plt.contourf(xx, xx, U, 30)
        plt.colorbar()
        plt.axis([-plot_axes, plot_axes, -plot_axes, plot_axes])
        plt.hold(True)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
    
        centre1 = np.array([0, 0])
        X_circ1 = np.linspace(centre1[0] - R, centre1[0] + R, N)
        Y_circ1_pos = centre1[1] + np.sqrt(R ** 2 - (X_circ1 - centre1[0]) ** 2)
        Y_circ1_neg = centre1[1] - np.sqrt(R ** 2 - (X_circ1 - centre1[0]) ** 2)
    
        plt.plot(X_circ1, Y_circ1_pos, 'c')
        plt.plot(X_circ1, Y_circ1_neg, 'c')
        plt.show()
        plt.savefig(str(means.shape[0]) + '-Gaussian_Potential.png', bbox_inches='tight')
    bm = GaussianBenchmark(means=means, cov=cov, minimum_index=0, simple_integrator=False)
    #bm = GaussianBenchmark(minimum_index=0, harmonic_well=False, simple_integrator=False)
    bm.find_kmax()
    bm.run_kmin()
    bm.run_PT()
    bm.compute_volume()
    bm.print_nr_function_calls()
    bf = BruteForce2D(means=means, cov=cov, minimum_index=0)
    bf.compute_volume()
