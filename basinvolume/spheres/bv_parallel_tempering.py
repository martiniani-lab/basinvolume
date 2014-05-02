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
    temperature=1.0
    stepsize=1
    niter=1e4
    sim = configure_bv_mcrunner()
    mcrunner = sim('jammed_packing0.xyzdr')
    ptrunner = MPI_PT_RLhandshake(mcrunner, 1,100, max_ptiter=2, pfreq=1, base_directory=path, verbose=True)
    ptrunner.run()