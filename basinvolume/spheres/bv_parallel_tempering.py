from __future__ import division
import numpy as np
import argparse
from basinvolume.spheres import configure_bv_mcrunner, MPI_BV_PT_RLhandshake
            
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
    adjustf_niter = int(tot_niter*0.01) #equilibrate for the first 1/100th of steps
    nskip = int(adjustf_niter/niter)
    pfreq = int(ptiter*0.1) #print every 1/10th of ptiter (this will give 10 snapshots)
    print 'tot_niter: {} ptiter: {} niter: {} adjustf_niter: {} nskip: {} pfreq: {}'.format(tot_niter, ptiter, niter, adjustf_niter, nskip, pfreq)
    
    sim = configure_bv_mcrunner()
    mcrunner = sim(fname, niter=niter, stepsize=1e-1, dtol=1e-4, hmin=0, 
                 hmax=1000, hbinsize=1e-1, acceptance=0.2, adjustf=0.9, adjustf_niter = adjustf_niter, adjustf_navg = 100)
    kmin = 0
    displ_k_min = sim.displ_k_min
    var_displ_k_min = sim.displ_k_min
    kmax = sim.kmax
        
    ptrunner = MPI_BV_PT_RLhandshake(mcrunner, kmax, kmin, displ_k_min, max_ptiter=ptiter, pfreq=pfreq, skip=nskip, base_directory=path, verbose=False)
    ptrunner.run()
    
