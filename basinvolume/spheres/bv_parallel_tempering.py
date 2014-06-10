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
    args = parser.parse_args()
    
    path = args.base_directory
    fname = args.jammed_packing_fname
    
    #Parallel Tempering
    tot_niter = 1e5
    
    ptiter = int(tot_niter*0.1) #10% PT swaps
    niter = int((tot_niter-ptiter)/ptiter) #90% MCMC walk
    adjustf_niter = int(tot_niter*0.02) #equilibrate for the first 1/100th of total steps
    nskip = int(adjustf_niter/niter) #don't swap while adjusting the step-size
    pt_eq_niter = int(tot_niter*0.1) #equilibrate pt for the following 1/10th of total steps
    pfreq = int(ptiter*0.1) #print every 1/10th of ptiter (this will give 10 snapshots)
    ts_freq = 10
    ts_niter = int(niter*pfreq/ts_freq) 
        
    sim = configure_bv_mcrunner()
    mcrunner = sim(fname, niter=niter, stepsize=1e-1, dtol=1e-4, hmin=0, 
                 hmax=1000, hbinsize=1e-1, acceptance=0.2, adjustf=0.9, adjustf_niter = adjustf_niter, adjustf_navg = 100,
                 pt_eq_niter=pt_eq_niter, ts_niter=ts_niter, ts_freq=ts_freq)
    kmin = 0
    displ_k_min = sim.displ_k_min
    var_displ_k_min = sim.displ_k_min
    kmax = sim.kmax
        
    ptrunner = MPI_BV_PT_RLhandshake(mcrunner, kmax, kmin, displ_k_min, max_ptiter=ptiter+1, pfreq=pfreq, skip=nskip, base_directory=path, verbose=False)
    start=time.time()
    ptrunner.run()
    end=time.time()
    print 'tot_niter: {} ptiter: {} niter: {} adjustf_niter: {} nskip: {} pfreq: {}'.format(tot_niter, ptiter, niter, adjustf_niter, nskip, pfreq)
    print 'elapsed time',end-start
    
