import numpy as np
import os
import argparse
import traceback
import copy
from _config_innersphere_mcrunner import _config_innersphere_mcrunner

def worker_innersphere(fname, kwargs):
    try:
        if ".xydfr" in fname or ".xyzdfr" in fname:
            print "found experimental packing"
            raise Exception("innersphere_mcurnner not implemented!")
        else:
            print "found numerical packing"
            mcrunner = _config_innersphere_mcrunner(fname, **kwargs)
        mcrunner.run()
    except:
        print('innersphere worker: %s' % (traceback.format_exc()))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute dos for inner sphere of basin")
    parser.add_argument("fname", type=str, help="packing file name")
    parser.add_argument("-p","--packingsdir", type=str, help="protocol to generate packings, assume in cwd", default="jammed_packings")
    parser.add_argument("--nocell", action='store_false', help="don't use cell lists, default: True",default=True)
    parser.add_argument("--cgd", action='store_true', help="use CG_DESCENT, default: False",default=False)
    parser.add_argument("-v","--verbose", action='store_true', help="verbosity",default=False)
    args = parser.parse_args()
    
    fname = args.fname
    packings_dir = args.packingsdir
    if not os.path.isabs(packings_dir):
        packings_dir = os.path.join(os.getcwd(),packings_dir)
    innersphere_kwargs = dict(niter=5e5, dtol=1e-4, eps=1., hmin=0, hmax=1000, hbinsize=1, 
                              opt_dtmax=1, opt_tol=1e-5, opt_nsteps=1e5, packings_dir=packings_dir,
                              use_cell_lists=args.nocell, use_cgd=args.cgd, verbose=args.verbose)
    
    i32max = np.iinfo(np.int32).max
    seeds_dict = dict(seed_takestep=np.random.randint(i32max))
    seeds = dict(seeds=seeds_dict)
    innersphere_kwargs_s = copy.deepcopy(dict(innersphere_kwargs,**seeds))
    worker_innersphere(fname,innersphere_kwargs_s,)