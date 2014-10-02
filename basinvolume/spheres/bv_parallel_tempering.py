from __future__ import division
import numpy as np
import argparse
from basinvolume.spheres import configure_bv_mcrunner, MPI_BV_PT_RLhandshake
import time
            
if __name__ == "__main__":
    """
    set Tmax to k_max
    set Tmin to k_min = 0
    set <u2>_min = mean of histogram from simulation done at k=0
    """
    parser = argparse.ArgumentParser(description="perform parallel tempering for basin volume method")
    parser.add_argument("jammed_packing_fname", type=str, help="name of xy[z]dr file")
    parser.add_argument("base_directory", type=str, help="directory in which to save results")
    parser.add_argument("-n","--totniter", type=int, help="number of energy evaluation per replica, default: 3e5. This sets a lower bound",default=3e5)
    parser.add_argument("--nocell", action='store_false', help="don't use cell lists, default: True",default=True)
    parser.add_argument("-v","--verbose", action='store_true', help="verbosity",default=False)
    parser.add_argument("--nocollectminima", action='store_false', help="don't collect database of minima",default=True)
    args = parser.parse_args()
    
    path = args.base_directory
    fname = args.jammed_packing_fname
    
    #Parallel Tempering
    tot_niter = args.totniter
    
    ptiter = int(tot_niter*0.1) #10% PT swaps
    niter = int((tot_niter-ptiter)/ptiter) #90% MCMC walk
    adjustf_niter = int(tot_niter*0.1) #equilibrate for the first 1/10th of total steps
    nskip = int(adjustf_niter/niter) #don't swap while adjusting the step-size
    # pt_eq_niter equilibrate pt for the following 4/10th of total steps (), this has an effect on histogram
    # and on checksameminimum: it only starts recording the neighbouring minima when equilibration is reached
    pt_eq_niter = 0 #set to 0 
    #the histogram starts recording the mean after adjustf_niter+pt_eq_niter steps
    pfreq = int((ptiter-1)*0.1) #print every 1/10th of ptiter (this will give 5 snapshots) #this is also frequency of tests
    ts_freq = 1
    ts_niter = int(niter*pfreq/ts_freq)
    perform_minimisation_convergence_test=False
    test_convergence_ts=True
    record_histogram=False
    rel_std_err= 0.05
    collect_minima_list=args.nocollectminima
    i32max = np.iinfo(np.int32).max
    seeds = dict(seed_takestep=np.random.randint(i32max),seed_metropolis=np.random.randint(i32max))
    print seeds
    
    assert(ts_freq == 1) #must be 1 with current output implementation (all based on timeseries)
    sim = configure_bv_mcrunner()
    mcrunner = sim(fname, niter=niter, stepsize=1e-1, dtol=1e-4, hmin=0, 
                 hmax=1000, hbinsize=1e-1, acceptance=0.2, adjustf=0.9, adjustf_niter = adjustf_niter, adjustf_navg = 100,
                 pt_eq_niter=pt_eq_niter, ts_niter=ts_niter, ts_freq=ts_freq, 
                 perform_convergence_test=perform_minimisation_convergence_test, collect_minima_list=collect_minima_list,
                 seeds=seeds, use_cell_lists=args.nocell, record_histogram=record_histogram)
    kmin = 0
    displ_k_min = sim.displ_k_min
    var_displ_k_min = sim.displ_k_min
    kmax = sim.kmax
        
    ptrunner = MPI_BV_PT_RLhandshake(mcrunner, kmax, kmin, displ_k_min, max_ptiter=ptiter+1, pfreq=pfreq, skip=nskip,
                                     test_convergence=test_convergence_ts, rel_std_err=rel_std_err, 
                                     base_directory=path, verbose=args.verbose)
    start=time.time()
    ptrunner.run()
    if collect_minima_list:
        mcrunner.dump_minima_list('{}/minima_list.sqlite'.format(ptrunner.rank))
    end=time.time()
    print 'tot_niter: {} ptiter: {} niter: {} adjustf_niter: {} nskip: {} pfreq: {}'.format(tot_niter, ptiter, niter, adjustf_niter, nskip, pfreq)
    print 'elapsed time',end-start
    
    
