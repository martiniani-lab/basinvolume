from __future__ import division
import numpy as np
import argparse
from basinvolume.spheres import configure_bv_mcrunner
from mcpele.parallel_tempering import MPI_PT_RLhandshake

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do nested sampling on a Lennard Jones cluster")
    parser.add_argument("base_directory", type=str, help="directory in which to save results")
    #parser.add_argument("-K", "--nreplicas", type=int, help="number of replicas", default=300)
    args = parser.parse_args()
    
    path = args.base_directory
    #Parallel Tempering
    sim = configure_bv_mcrunner()
    mcrunner = sim('jammed_packing0.xyzdr', niter=1e4, stepsize=1e-1, dtol=1e-4, hmin=0, 
                 hmax=100, hbinsize=1e-1, acceptance=0.2, adjustf=0.9, adjustf_niter = 5000, adjustf_navg = 100)
    ptrunner = MPI_PT_RLhandshake(mcrunner, 1,100, max_ptiter=2, pfreq=1, base_directory=path, verbose=True)
    ptrunner.run()