import multiprocessing as mp
import pele.utils.fix_multiprocessing
import os
import argparse
import traceback
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
    args = parser.parse_args()
    print args
    
    packings_dir = args.packingsdir
    if not os.path.isabs(packings_dir):
        packings_dir = os.path.join(os.getcwd(),packings_dir)
    
    ncores = args.ncores
    
    findk_kwargs = dict(k=3e2, niter=1e8, avgcount=1e5, dtol=1e-4, eps=1., ktarget=0.9, kfactor=0.4,
                        knavg=2000, ktol=0.025, opt_dtmax=1, opt_tol=1e-7, opt_nsteps=1e4,
                        packings_dir=packings_dir)
    
    kmin_kwargs = dict(k=0, stepsize=1e-1, niter=1e5, dtol=1e-4, eps=1., hmin=0, hmax=1000, hbinsize=1, 
                       acceptance=0.2, adjustf=0.9, adjustf_niter = 1e4, adjustf_navg = 100,
                       opt_dtmax=1, opt_tol=1e-7, opt_nsteps=1e4, packings_dir=packings_dir)
    
    mypool = mp.Pool(ncores)
    
    try:
        for fname in os.listdir(packings_dir):
            if ".xy" in fname:
                #construct mcrunners in place and append them to pool
                mypool.apply_async(worker_findk, args=(fname,findk_kwargs,))
                mypool.apply_async(worker_kmin, args=(fname,kmin_kwargs,))
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

    