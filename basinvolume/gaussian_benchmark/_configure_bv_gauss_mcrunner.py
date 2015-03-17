from __future__ import division
import numpy as np

class configure_bv_gauss_mcrunner(object):
    """
    Adapts configure_bv_mcrunner to perform PT for Gaussian benchmark.
    """
    def __init__(self, rank, nprocs):
        self.rank = rank
        self.nprocs = nprocs
    def __call__(self, fname, k=1.0, temperature=1.0, stepsize=1e-1, niter=2e4, dtol=1e-4, eps=1., hmin=0, 
                 hmax=100, hbinsize=1, acceptance=0.2, adjustf=0.9, adjustf_niter = 5e3, adjustf_navg = 100, 
                 pt_eq_niter=0, ts_niter=None, ts_freq=1, opt_dtmax=1, opt_maxstep=None, 
                 opt_tol=1e-5, opt_nsteps=1e5, perform_convergence_test=False, collect_minima_list=False, 
                 single=False, seeds=None, use_cell_lists=False, use_cgd=False, record_histogram = False,
                 packings_dir='jammed_packings', base_dir=None, verbose = False):
