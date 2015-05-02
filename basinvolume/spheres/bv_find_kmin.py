import numpy as np
import os
import argparse
import traceback
import copy
from _kmin_mcrunner import _kmin_mcrunner
from basinvolume.experiment_2d import _kmin_exp_mcrunner
from basinvolume.utils import check_kmax_reasonable 

def worker_kmin(fname, kwargs):
    try:
        if ".xydfr" in fname or ".xyzdfr" in fname:
            print "found experimental packing"
            mcrunner = _kmin_exp_mcrunner(fname, **kwargs)
        else:
            print "found numerical packing"
            mcrunner = _kmin_mcrunner(fname, **kwargs)
        if check_kmax_reasonable(mcrunner.findk_configpath):
            mcrunner.run()
        else:
            print('bv_find_kmin.py: kmax is unreasonable, exiting')
    except:
        print('kmin worker: %s' % (traceback.format_exc()))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute kmax and minimum average displacement for kmin for all jammed packings")
    parser.add_argument("fname", type=str, help="packing file name")
    parser.add_argument("-p","--packingsdir", type=str, help="protocol to generate packings, assume in cwd", default="jammed_packings")
    parser.add_argument("-n","--niter", type=float, help="number of energy evaluation, default: 1e5",default=1e5)
    parser.add_argument("--nocell", action='store_false', help="don't use cell lists, default: True",default=True)
    parser.add_argument("--moveall", action='store_true', help="don't use cell lists, default: False",default=False)
    parser.add_argument("--cgd", action='store_true', help="use CG_DESCENT, default: False",default=False)
    parser.add_argument("--rsts", action='store_true', help="record steps timeseries for diffusion studies, default: False",default=False)
    parser.add_argument("--rsts-only", action='store_true', help="record steps timeseries for diffusion studies ONLY, default: False",default=False)
    parser.add_argument("-v","--verbose", action='store_true', help="verbosity",default=False)
    args = parser.parse_args()
    
    fname = args.fname
    packings_dir = args.packingsdir
    if not os.path.isabs(packings_dir):
        packings_dir = os.path.join(os.getcwd(),packings_dir)
    
    single = not args.moveall
    if args.rsts_only:
        args.rsts = True
    kmin_kwargs = dict(k=0, stepsize=1e-1, niter=args.niter, dtol=1e-4, eps=1., hmin=0, hmax=1000, hbinsize=1, 
                       acceptance=0.2, adjustf=0.9, adjustf_niter=1e4, adjustf_navg=100,
                       opt_dtmax=1, opt_tol=1e-5, opt_nsteps=1e5, packings_dir=packings_dir,
                       use_cell_lists=args.nocell, single=single, use_cgd=args.cgd, verbose=args.verbose,
                       record_steps_timeseries=args.rsts, record_steps_timeseries_every=[int(np.ceil(1.5**n)) for n in xrange(22)],
                       print_diffusion_only=args.rsts_only)
    
    i32max = np.iinfo(np.int32).max
    seeds_dict = dict(seed_takestep=np.random.randint(i32max),seed_metropolis=np.random.randint(i32max))
    seeds = dict(seeds=seeds_dict)
    kmin_kwargs_s = copy.deepcopy(dict(kmin_kwargs,**seeds))
    worker_kmin(fname,kmin_kwargs_s,)