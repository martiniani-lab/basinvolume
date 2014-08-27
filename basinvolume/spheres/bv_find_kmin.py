import numpy as np
import os
import argparse
import traceback
import copy
from _kmin_mcrunner import _kmin_mcrunner

def worker_kmin(fname, kwargs):
    try:
        mcrunner = _kmin_mcrunner(fname, **kwargs)
        mcrunner.run()
    except:
        print('kmin worker: %s' % (traceback.format_exc()))
        
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
    
    kmin_kwargs = dict(k=0, stepsize=1e-1, niter=1e5, dtol=1e-4, eps=1., hmin=0, hmax=1000, hbinsize=1, 
                       acceptance=0.2, adjustf=0.9, adjustf_niter = 1e4, adjustf_navg = 100,
                       opt_dtmax=1, opt_tol=1e-7, opt_nsteps=1e4, packings_dir=packings_dir, use_cell_lists=args.cell)
    
    i32max = np.iinfo(np.int32).max
    seeds_dict = dict(seed_takestep=np.random.randint(i32max),seed_metropolis=np.random.randint(i32max))
    seeds = dict(seeds=seeds_dict)
    kmin_kwargs_s = copy.deepcopy(dict(kmin_kwargs,**seeds))
    worker_kmin(fname,kmin_kwargs_s,)