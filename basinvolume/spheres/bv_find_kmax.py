import numpy as np
import os
import argparse
import traceback
import copy
from _findk_mcrunner import _findk_mcrunner

def worker_findk(fname, kwargs):
    try:
        mcrunner = _findk_mcrunner(fname, **kwargs)
        mcrunner.run()
    except:
        print('find_k worker: %s' % (traceback.format_exc()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute kmax and minimum average displacement for kmin for all jammed packings")
    parser.add_argument("fname", type=str, help="packing file name")
    parser.add_argument("-p","--packingsdir", type=str, help="protocol to generate packings, assume in cwd", default="jammed_packings")
    parser.add_argument("-c","--cell", type=bool, help="use cell lists, default: True",default=True)
    args = parser.parse_args()
    
    fname = args.fname
    packings_dir = args.packingsdir
    if not os.path.isabs(packings_dir):
        packings_dir = os.path.join(os.getcwd(),packings_dir)
    
    findk_kwargs = dict(k=200, niter=1e8, avgcount=1e5, dtol=1e-4, eps=1., ktarget=0.9,
                        knavg=2000, ktol=0.025, opt_dtmax=1, opt_tol=1e-7, opt_nsteps=1e4,
                        packings_dir=packings_dir, use_cell_lists=args.cell)
    
    i32max = np.iinfo(np.int32).max
    #construct mcrunners in place and append them to pool
    seeds_dict = dict(seed_takestep=np.random.randint(i32max))
    seeds = dict(seeds=seeds_dict)
    findk_kwargs_s = copy.deepcopy(dict(findk_kwargs,**seeds)) 
    worker_findk(fname, findk_kwargs_s)
