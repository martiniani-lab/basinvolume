import numpy as np
import os
import argparse
import traceback
import copy
from _findk_mcrunner import _findk_mcrunner
from basinvolume.experiment_2d import _findk_exp_mcrunner

def worker_findk(fname, kwargs):
    try:
        if ".xydfr" in fname or ".xyzdfr" in fname:
            print "found experimental packing"
            mcrunner = _findk_exp_mcrunner(fname, **kwargs)
        else:
            print "found numerical packing"
            mcrunner = _findk_mcrunner(fname, **kwargs)
        mcrunner.run()
    except:
        print('find_k worker: %s' % (traceback.format_exc()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute kmax and average displacement for kmax for all jammed packings")
    parser.add_argument("fname", type=str, help="packing file name")
    parser.add_argument("-k","--kstart", type=float, help="initial guess for kmax, default: 500", default=500)
    parser.add_argument("-p","--packingsdir", type=str, help="protocol to generate packings, assume in cwd", default="jammed_packings")
    parser.add_argument("--nocell", action='store_false', help="don't use cell lists, default: True",default=True)
    parser.add_argument("--cgd", action='store_true', help="use CG_DESCENT, default: False",default=False)
    parser.add_argument("-v","--verbose", action='store_true', help="verbosity",default=False)
    args = parser.parse_args()
    print args
    fname = args.fname
    packings_dir = args.packingsdir
    if not os.path.isabs(packings_dir):
        packings_dir = os.path.join(os.getcwd(),packings_dir)
    
    findk_kwargs = dict(k=args.kstart, niter=1e8, avgcount=1e5, dtol=1e-4, eps=1., ktarget=0.9,
                        knavg=1e4, ktol=0.025, opt_dtmax=1, opt_tol=1e-5, opt_nsteps=1e5,
                        packings_dir=packings_dir, use_cell_lists=args.nocell, use_cgd=args.cgd,
                        verbose=args.verbose)
    
    i32max = np.iinfo(np.int32).max
    #construct mcrunners in place and append them to pool
    seeds_dict = dict(seed_takestep=np.random.randint(i32max))
    seeds = dict(seeds=seeds_dict)
    findk_kwargs_s = copy.deepcopy(dict(findk_kwargs,**seeds)) 
    worker_findk(fname, findk_kwargs_s)
