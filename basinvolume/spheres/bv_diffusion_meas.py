import numpy as np
import multiprocessing as mp
import pele.utils.fix_multiprocessing
import os
import argparse
import traceback
import copy
from _kmin_mcrunner import _kmin_mcrunner
import glob
import re

def worker_kmin(fname, kwargs):
    try:
        mcrunner = _kmin_mcrunner(fname, **kwargs)
        mcrunner.run()
    except:
        print('kmin worker: %s' % (traceback.format_exc()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute kmax and minimum average displacement for kmin for all jammed packings")
    parser.add_argument("-j","--ncores", type=int, help="number of packings to produce",default=4)
    parser.add_argument("-n","--niter", type=int, help="number of iterations",default=int(1e7))
    parser.add_argument("-f","--force", action='store_true', help="run all", default=False)
    parser.add_argument("-v","--verbose", action='store_true', help="verbosity", default=False)
    # potential arguments
    parser.add_argument("--opt-pot", type=str, help="optmizer's potential, 1) (default) hs_wca "
                                                    "2) inverse_power_stillinger", default='hs_wca')
    args = parser.parse_args()

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

    ncores = int(args.ncores)
    niter = int(args.niter)
    mypool = mp.Pool(ncores)

    i32max = np.iinfo(np.int32).max

    try:
        workspace = os.getcwd()
        dir_signature='n*_phi*_phi*_*D*'
        listdir = glob.glob(os.path.join(workspace, dir_signature))
        for dir_path in listdir:
            dir_name = os.path.split(dir_path)[1]
            str_values = re.findall('\d+', dir_name)
            nparticles, hs_phi = float(str_values[0]), float('0.'+str_values[1])
            ss_phi, bdim = float('0.'+str_values[2]), float(str_values[3])
            packings_dir = os.path.join(dir_path, "jammed_packings")
            explore_dir_list = listdir = glob.glob(os.path.join(dir_path, "explore_bv_jammed_packing*"))
            for explore_dir_path in explore_dir_list:
                if args.force:
                    tst=[]
                else:
                    tst = glob.glob(os.path.join(explore_dir_path, "diffusion/StepsTimeSeries.{}*".format(niter)))
                if len(tst) < 1:
                    explore_dir_name = os.path.split(explore_dir_path)[1]
                    packing_number = re.findall('\d+', explore_dir_name)[0]
                    fname = explore_dir_name.replace("explore_bv_","")
                    if bdim == 3:
                        fname += ".xyzdr"
                    elif bdim == 2:
                        fname += ".xydr"
                    kmin_kwargs = dict(k=0, stepsize=1e-1, niter=niter, dtol=1e-4, eps=1., hmin=0, hmax=1000, hbinsize=1,
                                       acceptance=0.2, adjustf=0.9, adjustf_niter=1e5, adjustf_navg=100,
                                       opt_dtmax=1, opt_tol=1e-5, opt_nsteps=1e5, packings_dir=packings_dir,
                                       use_cell_lists=True, single=True, minimizer='fire', verbose=args.verbose,
                                       record_steps_timeseries=True,
                                       record_steps_timeseries_every=[int(np.ceil(1.5**n)) for n in xrange(28)],
                                       print_diffusion_only=True, workspace=dir_path,
                                       record_trajectory_npoints=int(1e4),
                                       opt_pot_str=opt_pot_str, **extra_pot_kwargs)
                    seeds_dict = dict(seed_takestep=np.random.randint(i32max),seed_metropolis=np.random.randint(i32max))
                    seeds = dict(seeds=seeds_dict)
                    kmin_kwargs_s = copy.deepcopy(dict(kmin_kwargs,**seeds))
                    mypool.apply_async(worker_kmin, args=(fname,kmin_kwargs_s,))
                    print "performing diffusion measurement for: \n", explore_dir_path
    except:
        mypool.terminate()
        mypool.join()
        raise

    mypool.close()
    mypool.join()
