import numpy as np
import multiprocessing as mp
import pele.utils.fix_multiprocessing
import os
import argparse
import traceback
import copy
from _findk_mcrunner import _findk_mcrunner
from _kmin_mcrunner import _kmin_mcrunner

def worker_findk(fname, kwargs):
    try:
        mcrunner = _findk_mcrunner(fname, **kwargs)
        mcrunner.run()
    except:
        print('find_k worker: %s' % (traceback.format_exc()))

def worker_kmin(fname, kwargs):
    try:
        mcrunner = _kmin_mcrunner(fname, **kwargs)
        mcrunner.run()
    except:
        print('kmin worker: %s' % (traceback.format_exc()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute kmax and minimum average displacement for kmin for all jammed packings")
    parser.add_argument("-n","--ncores", type=int, help="number of packings to produce",default=4)
    parser.add_argument("-p","--packingsdir", type=str, help="protocol to generate packings", default="jammed_packings")
    parser.add_argument("-c","--cell", type=bool, help="use cell lists, default: True",default=True)
    # potential arguments
    parser.add_argument("--opt-pot", type=str, help="optmizer's potential, 1) (default) hs_wca "
                                                    "2) inverse_power_stillinger", default='hs_wca')
    args = parser.parse_args()
    print args
    
    packings_dir = args.packingsdir
    if not os.path.isabs(packings_dir):
        packings_dir = os.path.join(os.getcwd(),packings_dir)
    
    ncores = args.ncores

    # potential type
    opt_pot_str = args.opt_pot
    extra_pot_kwargs = dict()
    if opt_pot_str == 'hs_wca':
        pass
    elif opt_pot_str == 'inverse_power_stillinger':
        extra_pot_kwargs.update(dict(pow=3, a=1))
        print 'setting inverse_power_stillinger parameters: ', extra_pot_kwargs
    else:
        raise NotImplementedError
    
    findk_kwargs = dict(k=600, niter=1e8, avgcount=1e5, dtol=1e-4, eps=1., ktarget=0.9,
                        knavg=2000, ktol=0.025, opt_dtmax=1, opt_tol=1e-7, opt_nsteps=1e4,
                        packings_dir=packings_dir, use_cell_lists=args.cell,
                        opt_pot_str=opt_pot_str, **extra_pot_kwargs)
    
    kmin_kwargs = dict(k=0, stepsize=1e-1, niter=1e5, dtol=1e-4, eps=1., hmin=0, hmax=1000, hbinsize=1, 
                       acceptance=0.2, adjustf=0.9, adjustf_niter = 1e4, adjustf_navg = 100,
                       opt_dtmax=1, opt_tol=1e-7, opt_nsteps=1e4, packings_dir=packings_dir,
                       use_cell_lists=args.cell, opt_pot_str=opt_pot_str, **extra_pot_kwargs)
    
    mypool = mp.Pool(ncores)
    
    i32max = np.iinfo(np.int32).max
    try:
        for fname in os.listdir(packings_dir):
            if ".xy" in fname:
                #construct mcrunners in place and append them to pool
                seeds_dict = dict(seed_takestep=np.random.randint(i32max))
                seeds = dict(seeds=seeds_dict)
                findk_kwargs_s = copy.deepcopy(dict(findk_kwargs,**seeds)) 
                mypool.apply_async(worker_findk, args=(fname,findk_kwargs_s,))
                
                seeds_dict = dict(seed_takestep=np.random.randint(i32max),seed_metropolis=np.random.randint(i32max))
                seeds = dict(seeds=seeds_dict)
                kmin_kwargs_s = copy.deepcopy(dict(kmin_kwargs,**seeds))
                mypool.apply_async(worker_kmin, args=(fname,kmin_kwargs_s,))
    except:
        mypool.terminate()
        mypool.join()
        raise
                
    mypool.close()
    mypool.join()

#        procs=[]
#        for mcrunner in jobs:
#            p = mp.Process(target=run_mcrunner, args=(mcrunner,))
#            p.start()
#            procs.append(p)
#        for p in procs:
#            p.join()        

    