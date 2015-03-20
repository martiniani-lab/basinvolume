from __future__ import division
import os
import ConfigParser
import time
import cPickle as pickle
import numpy as np
from pele.optimize import ModifiedFireCPP
from pele.potentials import SumGaussianPot
from pele.potentials import Harmonic
from mcpele.monte_carlo import CheckSphericalContainerConfig
from mcpele.monte_carlo import RandomCoordsDisplacement
from mcpele.monte_carlo import MetropolisTest
from basinvolume.monte_carlo import CheckSameMinimumConfig
from basinvolume.monte_carlo import RecordDisp2Histogram
from basinvolume.utils import read_single_column_coords
from gaussian_benchmark_kmin_run import GaussianBenchmarkKminRun

class configure_bv_gauss_mcrunner(object):
    """
    Adapts configure_bv_mcrunner to perform PT for Gaussian benchmark.
    """
    def __init__(self, rank, nprocs):
        self.rank = rank
        self.nprocs = nprocs
    def __call__(self,
                 fname,
                 k=1.0,
                 temperature=1.0,
                 stepsize=1e-1,
                 niter=2e4,
                 dtol=1e-4,
                 eps=1.,
                 hmin=0, 
                 hmax=100,
                 hbinsize=1,
                 acceptance=0.2,
                 adjustf=0.9,
                 adjustf_niter=5e3,
                 adjustf_navg=100, 
                 pt_eq_niter=0,
                 ts_niter=None,
                 ts_freq=1,
                 opt_dtmax=1,
                 opt_maxstep=None, 
                 opt_tol=1e-5,
                 opt_nsteps=1e5,
                 perform_convergence_test=False,
                 collect_minima_list=False, 
                 single=False,
                 seeds=None,
                 use_cell_lists=False,
                 use_cgd=False,
                 record_histogram=False,
                 packings_dir='gaussian_sum',
                 base_dir=None,
                 verbose=False):
        self.fname = fname
        self._set_paths(base_dir, packings_dir)
        self._import_packing_config_files()
        self._import_packing_configuration()
        self.get_means_cov()
        hbinsize = self._get_histogram_bin(k)
        opt_maxstep = self._get_opt_maxstep(opt_maxstep)
        self.eps = eps
        self.mc_params = {'k':k,'temperature':temperature,'niter':niter,'stepsize':stepsize,'dtol':dtol,'eps':self.eps,
                          'hmin':hmin,'hmax':hmax,'hbinsize':hbinsize,'acceptance':acceptance,'adjustf':adjustf,
                          'adjustf_niter':adjustf_niter,'adjustf_navg':adjustf_navg,'pt_eq_niter':pt_eq_niter,
                          'ts_niter':ts_niter, 'ts_freq':ts_freq,'opt_dtmax':opt_dtmax,'opt_maxstep':opt_maxstep,
                          'opt_tol':opt_tol,'opt_nsteps':opt_nsteps,'perform_convergence_test':perform_convergence_test, 
                          'collect_minima_list':collect_minima_list, 'record_histogram':record_histogram,
                          'single':single, 'use_cell_lists':use_cell_lists, 'use_cgd':use_cgd}
        #add seeds dictionary to mc_params
        try:
            self.mc_params.update(seeds)
        except:
            print "WARNING:seeds not passed"
        self._initialise()
        self._requench_coords(dtol, opt_maxstep, verbose)
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
        self.pot_optimizer = SumGaussianPot(self.means, self.cov)
        self.optimizer = ModifiedFireCPP(self.origin,
                                    self.pot_optimizer,
                                    dtmax=self.opt_dtmax,
                                    maxstep=self.opt_maxstep,
                                    tol=self.opt_tol, 
                                    nsteps=opt_nsteps,
                                    verbosity=1)
        harmonic_com_flag = True
        self.potential = Harmonic(self.origin, k, bdim=self.bdim, com=harmonic_com_flag)
        self.conftest_outer_sphere = CheckSphericalContainerConfig(self.radius_container)
        self.conftest_check_same_minimum = CheckSameMinimumConfig(self.pot_optimizer,
                                           self.origin, self.dtol,
                                           opt=self.optimizer, opt_tol=opt_tol,
                                           opt_maxiter=opt_nsteps)
        self.action_record_displ_kmin = RecordDisp2Histogram(self.origin,
                                      self.rattlers, self.bdim, hmin, hmax,
                                      hbinsize, self.equilibration_steps)
        self.metropolis = MetropolisTest(self.seeds['seed_metropolis'])
        mcrunner = GaussianBenchmarkKminRun(
                     pot_optimizer=self.pot_optimizer,
                     origin=self.origin,
                     optimizer=self.optimizer,
                     conftest_outer_sphere=self.conftest_outer_sphere,
                     conftest_check_same_minimum=self.conftest_check_same_minimum,
                     action_record_displ=self.action_record_displ2_kmin,
                     adjustf_niter=self.adjustf_niter,
                     pt_eq_niter=self.pt_eq_niter,
                     equilibration_steps=self.equilibration_steps,
                     metropolis=self.metropolis,
                     takestep=self.takestep,
                     niter=self.pt_niter,
                     potential=self.potential
                     )
        return mcrunner
    def _set_paths(self, base_dir, packings_dir):
        """
        set base_directory, packings_directory and configpaths, configfile
        """
        dname = self.fname
        if dname.endswith('.gauss'):
            dname = dname[:-6]
        else:
            raise Exception("illegal file name")
        if base_dir is None:
            base_directory = os.path.join(os.getcwd(), 'explore_bv_' + str(dname))
            print("assert existence of base disrctory:", base_directory)
            assert(os.path.exists(base_directory))
        else:
            if not os.path.isabs(base_dir):
                base_directory = os.path.join(os.getcwd(), packings_dir)
        self.base_directory = base_directory
        print("self.base_directory", self.base_directory)
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(), packings_dir)
        self.packings_dir = packings_dir
        self.means_configpath = os.path.join(packings_dir, "gaussian_sum_means.config")
        self.cov_configpath = os.path.join(packings_dir, "gaussian_sum_cov.config")
        self.packing_configpath = os.path.join(packings_dir, 'gaussian_sum.config')
        self.findk_configpath = os.path.join(self.base_directory, 'findk_' + dname + '.config')  
        self.kmin_configpath = os.path.join(self.base_directory, 'kmin_' + dname + '.config')
        self.configfile = '{}/explore_{}.config'.format(self.base_directory, dname)
    def _import_packing_config_files(self):
        configf = ConfigParser.ConfigParser()
        print("attempting to read from config file at", self.packing_configpath)
        configf.read(str(self.packing_configpath))
        self.ngaussians = configf.getint('GAUSSIAN_SUM', 'ngaussians')
        self.bdim = configf.getint('GAUSSIAN_SUM', 'bdim')
        self.gdim = configf.getint("GAUSSIAN_SUM", "gdim")
        self.radius_container = configf.getfloat("GAUSSIAN_SUM", "radius_container")
        #self.ndim = self.nparticles * self.bdim
        print("self.findk_configpath", self.findk_configpath)
        configf.read(str(self.findk_configpath))
        self.kmax = configf.getfloat('FINDK', 'kmax')
        self.prob_kmax = configf.getfloat('FINDK', 'prob')
        self.displ_k_max = configf.getfloat('FINDK', 'displ_k_max')
        self.var_displ_k_max = configf.getfloat('FINDK', 'var_displ_k_max')
        configf.read(str(self.kmin_configpath))
        self.displ_k_min = configf.getfloat('KMIN', 'displ_k_min')
        self.var_displ_k_min = configf.getfloat('KMIN', 'var_displ_k_min')
    def _import_packing_configuration(self):
        """
        Import position of minimum, as determined earlier.
        This should replace the function with the same name in bv for jammed particles.
        """
        print("self.packings_dir", self.packings_dir)
        print("self.fname", self.fname)
        path = os.path.join(self.packings_dir, self.fname)
        self.coords = read_single_column_coords(path)
    def get_means_cov(self):
        """
        Import means and covs from file.
        """
        print("gdim", self.gdim)
        self.means = read_multi_column(self.means_configpath, self.gdim)
        self.cov = read_multi_column(self.cov_configpath, self.gdim)
    def _get_histogram_bin(self, k):
        """automatically estimate size of histogram"""
        """
        This is copied from bv config. (Needs to be changed?) 
        """
        hmax = self.displ_k_min * k #self.displ_k_max*self.kmax
        hbinsize = hmax * 0.0001 
        return hbinsize
    def _get_opt_maxstep(self, opt_maxstep):
        """returns opt max step"""
        if opt_maxstep is None:
            #opt_maxstep = self.boxv[0] * 0.01
            #opt_maxstep = self.sca * np.amin(self.red_radii)
            opt_maxstep = self.radius_container
        return opt_maxstep
