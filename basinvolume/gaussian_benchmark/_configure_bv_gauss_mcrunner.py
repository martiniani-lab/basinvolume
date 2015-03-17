from __future__ import division
import numpy as np
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
                 packings_dir='jammed_packings',
                 base_dir=None,
                 verbose=False):
        self.fname = fname
        self._set_paths(base_dir, packings_dir)
        self._import_packing_config_files()
        self._import_packing_configuration()
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
        ####
        potential = Harmonic(self.coords, k, bdim=self.bdim, com=True)
        mcrunner = GaussianBenchmarkKminRun()
        ##
        ##
        return mcrunner
        
