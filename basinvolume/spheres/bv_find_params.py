from _findk_mcrunner import _findk_mcrunner
from _kmin_mcrunner import _kmin_mcrunner
import argparse
import multiprocessing as mp
#import threading
import os

def run_mcrunner(mcrunner):
    mcrunner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute kmax and minimum average displacement for kmin for all jammed packings")
    parser.add_argument("-n","--ncores", type=int, help="number of packings to produce",default=4)
    parser.add_argument("-p","--packingsdir", type=str, help="protocol to generate packings", default="jammed_packings")
    args = parser.parse_args()
    print args
    
    packings_dir = args.packingsdir
    if not os.path.isabs(packings_dir):
        packings_dir = os.path.join(os.getcwd(),packings_dir)
    
    jobs = []
    for fname in os.listdir(packings_dir):
        #construct mcrunners in place and append them to pool
        if "xy" in fname:          
            jobs.append(_kmin_mcrunner(fname, k=0, temperature=1.0, stepsize=1e-1, niter=5e4, dtol=1e-4, eps=1., hmin=0,
                                           hmax=10, hbinsize=1e-1, acceptance=0.2, adjustf=0.9, adjustf_niter = 5e3, adjustf_navg = 100,
                                           opt_dtmax=1, opt_maxstep=0.5, opt_tol=1e-3, opt_nsteps=1e4, packings_dir=packings_dir))

            jobs.append(_findk_mcrunner(fname, k=1e2, temperature=1.0, niter=1e6, dtol=1e-4, eps=1., ktarget=0.75, kfactor=0.9, 
                                             knavg=1000, ktol=0.05, opt_dtmax=1, opt_maxstep=0.5, opt_tol=1e-3, opt_nsteps=1e4, 
                                             packings_dir=packings_dir))
    
    ncores = args.ncores
    #pool = mp.Pool(processes=ncores)
    
    try:
        for mcrunner in jobs:
            #construct mcrunners in place and append them to pool
            pool.apply_async(run_mcrunner, args=(mcrunner,))
    except:
        pool.terminate()
        pool.join()
        raise
            
    pool.close()
    pool.join()