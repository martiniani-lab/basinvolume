from __future__ import division
import argparse
import numpy as np
from basinvolume.spheres import MPI_BV_PT_RLhandshake
from basinvolume.gaussian_benchmark import configure_bv_gauss_mcrunner
import time
from mpi4py import MPI
from basinvolume.utils import view_traceback

class GaussianBenchmarkPTRun(object):
    """
    Replicates the PT setup in bv_parallel_termpering.py.
    """
    def __init__(self,
                 configuration_name=None,
                 base_directory=None,
                 totniter=5e5,
                 nocell=True,
                 nocollectminima=True,
                 cgd=False,
                 verbose=True,
                 nparticles=None
                 ):
        print("construct: GaussianBenchmarkPTRun")
        self.configuration_name = configuration_name
        self.base_directory = base_directory
        self.totniter = totniter
        self.nocell = nocell
        self.nocollectminima = nocollectminima
        self.cgd = cgd
        self.nparticles = nparticles
        if self.configuration_name is None or self.base_directory is None:
            raise Exception("illegal input")
        path = self.base_directory
        fname = self.configuration_name
        single = True
        tot_niter = self.totniter
        #ptiter = int(tot_niter * 0.1) #10% PT swaps
        ptiter = int(tot_niter * 1e-2) #1% PT swaps
        niter = int((tot_niter - ptiter) / ptiter) #90% MCMC walk        
        adjustf_niter = int(tot_niter * 0.1) #equilibrate for the first 1/10th of total steps        
        nskip = int(adjustf_niter / niter) #don't swap while adjusting the step-size        
        # pt_eq_niter equilibrate pt for the following 4/10th of total steps (), this has an effect on histogram        
        # and on checksameminimum: it only starts recording the neighbouring minima when equilibration is reached        
        pt_eq_niter = 0 #set to 0         
        #the histogram starts recording the mean after adjustf_niter+pt_eq_niter steps        
        pfreq = int((ptiter - 1) * 0.1) #print every 1/10th of ptiter (this will give 5 snapshots) #this is also frequency of tests        
        ts_freq = 1        
        ts_niter = int(niter * pfreq / ts_freq)        
        perform_minimisation_convergence_test = False        
        test_convergence_ts = True        
        record_histogram = False        
        assert(record_histogram == False and pt_eq_niter == 0 and ts_freq == 1) #ts_freq must be 1 with current output implementation (all based on timeseries)        
        rel_std_err = 0.05 #relative standard error in the mean used by convergence test        
        min_window = int(0.5 * tot_niter) #minimum amount of data before trying to check convergence        
        max_eq_time = np.min([int(0.5 * tot_niter), 2.5e5]) #maximum amount of data to discard (throw away max the first 2.5e5 points, to avoid reading spurious features)        
        fast_ct = False #if false skip heuristic search for equilibration point        
        collect_minima_list = False       
        i32max = np.iinfo(np.int32).max
        seeds = dict(seed_takestep=np.random.randint(i32max),seed_metropolis=np.random.randint(i32max))
        print seeds
        # mc run setup
        comm = MPI.COMM_WORLD   
        nprocs = comm.Get_size()
        rank = comm.Get_rank()
        sim = configure_bv_gauss_mcrunner(rank, nprocs)
        mcrunner = sim(fname,
                       niter=niter,
                       stepsize=1e-1,
                       dtol=1e-4,
                       opt_tol=1e-5,
                       opt_nsteps=1e5,
                       hmin=0,
                       hmax=1000,
                       hbinsize=1e-1,
                       acceptance=0.2,
                       adjustf=0.9,
                       adjustf_niter=adjustf_niter,
                       adjustf_navg=100,
                       pt_eq_niter=pt_eq_niter,
                       ts_niter=ts_niter,
                       ts_freq=ts_freq,
                       use_cgd=self.cgd,
                       perform_convergence_test=perform_minimisation_convergence_test,
                       collect_minima_list=collect_minima_list,
                       seeds=seeds,
                       use_cell_lists=self.nocell,
                       single=single,
                       record_histogram=record_histogram)
        mcrunner.set_report_steps(adjustf_niter)
        #prepare PT runner
        kmin = 0
        displ_k_min = sim.displ_k_min
        var_displ_k_min = sim.displ_k_min
        kmax = sim.kmax
        ptrunner = MPI_BV_PT_RLhandshake(mcrunner,
                                         kmax,
                                         kmin,
                                         displ_k_min,
                                         max_ptiter=ptiter+1,
                                         pfreq=pfreq,
                                         skip=nskip,
                                         test_convergence=test_convergence_ts,
                                         fast_ct=fast_ct,
                                         rel_std_err=rel_std_err, 
                                         min_window=min_window,
                                         max_eq_time=max_eq_time,
                                         base_directory=path,
                                         verbose=verbose,
                                         bs_nodes=10)
        ptrunner.suppress_histogram = True
        assert ptrunner.rank == rank, "rank id do not match"
        assert ptrunner.nproc == nprocs, "number of cores do not match"        
        # run PT
        print("run gaussian pt")
        start = time.time()
        try:
            ptrunner.run()
            sim.print_success_all(True)
        except:
            view_traceback()
            try:
                sim.print_success_all(False)
            except:
                view_traceback()
        end = time.time()
        print 'core: {} ptiter: {} niter: {} adjustf_niter: {} skip: {} pfreq: {}'.format(rank, mcrunner.niter, 
                                                                                       ptrunner.ptiter, adjustf_niter, 
                                                                                       ptrunner.skip, ptrunner.pfreq)
        print ("elapsed time", end - start)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pt runs for gaussian bv benchmark")
#    parser.add_argument("configuration_name", type=str, default="config0.gauss")
    parser.add_argument("configuration_name", type=str)
#    parser.add_argument("base_directory", type=str, default="gauss_pt")
    parser.add_argument("base_directory", type=str)
    parser.add_argument("totniter", type=int)
    parser.add_argument("nparticles", type=int)
    args = parser.parse_args()
    print("args", args)
    GaussianBenchmarkPTRun(configuration_name=args.configuration_name,
                           base_directory=args.base_directory,
                           totniter=args.totniter,
                           nocell=True,
                           nocollectminima=True,
                           cgd=False,
                           verbose=True,
                           nparticles=args.nparticles)
