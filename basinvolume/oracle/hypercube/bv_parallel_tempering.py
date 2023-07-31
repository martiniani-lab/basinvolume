from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
from basinvolume.spheres import MPI_BV_PT_RLhandshake
from basinvolume.hypercube import _hypercube_bv_mcrunner
import time
from mpi4py import MPI
from basinvolume.utils import (
    view_traceback,
    check_kmax_reasonable,
    import_pt_time_series,
)
import sys

if __name__ == "__main__":
    """
    set Tmax to k_max
    set Tmin to k_min = 0
    set <u2>_min = mean of histogram from simulation done at k=0
    """
    parser = argparse.ArgumentParser(
        description="perform parallel tempering for basin volume method"
    )
    parser.add_argument(
        "base_directory", type=str, help="directory in which to save results"
    )
    parser.add_argument(
        "-n",
        "--mintotniter",
        type=float,
        help="minimum number of energy evaluation per replica, \
                        before checking for convergence default: 5e5. This sets a lower bound",
        default=5e5,
    )
    parser.add_argument(
        "-m",
        "--maxtotniter",
        type=float,
        help="maximum number of energy evaluation per replica, \
                        This sets an upper bound default: 2e6",
        default=2e6,
    )
    parser.add_argument(
        "--numnegk",
        type=int,
        help="number of negative k's to use, default 0",
        default=0,
    )
    parser.add_argument(
        "--lownegk",
        type=float,
        help="lowest value of negative k's to use, default -2.5",
        default=-2.5,
    )
    parser.add_argument(
        "-s",
        "--relstderr",
        type=float,
        help="relative standard error to test convergence, default 0.05",
        default=0.05,
    )
    parser.add_argument(
        "--moveall",
        action="store_true",
        help="don't use cell lists, default: False",
        default=False,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbosity", default=False
    )
    args = parser.parse_args()

    base_dir = args.base_directory
    single = not args.moveall
    # Parallel Tempering
    min_tot_niter = int(args.mintotniter)
    max_tot_niter = int(args.maxtotniter)

    min_ptiter = int(
        min_tot_niter * 0.1
    )  # 10% PT swaps, this is the initial proposed maximum length of the run. at the end of min_ptiter convergence is checked
    niter = int((min_tot_niter - min_ptiter) / min_ptiter)  # 90% MCMC walk
    adjustf_niter = int(
        min_tot_niter * 0.1
    )  # equilibrate for the first 1/10th of total steps
    nskip = int(adjustf_niter / niter)  # don't swap while adjusting the step-size
    # pt_eq_niter equilibrate pt for the following 4/10th of total steps (), this has an effect on histogram
    # and on checksameminimum: it only starts recording the neighbouring minima when equilibration is reached
    pt_eq_niter = 0  # set to 0
    # the histogram starts recording the mean after adjustf_niter+pt_eq_niter steps
    pfreq = int(
        (min_ptiter - 1) * 0.1
    )  # print every 1/10th of min_ptiter (this will give 5 snapshots) #this is also frequency of tests
    ts_freq = 1
    ts_niter = int(niter * pfreq / ts_freq)
    test_convergence_ts = True
    record_histogram = False
    assert (
        record_histogram == False and pt_eq_niter == 0 and ts_freq == 1
    )  # ts_freq must be 1 with current output implementation (all based on timeseries)
    rel_std_err = (
        args.relstderr
    )  # relative standard error in the mean used by convergence test
    min_window = 2.5e5  # minimum amount of data before trying to check convergence
    max_eq_time = 2.5e5  # #maximum amount of data to discard (throw away max the first 2.5e5 points, to avoid reading spurious features)
    fast_ct = False  # if false skip euristic search for equilibration point
    i32max = np.iinfo(np.int32).max
    seeds = dict(
        seed_takestep=np.random.randint(i32max),
        seed_metropolis=np.random.randint(i32max),
    )
    print(seeds)

    # prepare MC runner
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    sim = _hypercube_bv_mcrunner(rank, nprocs)
    mcrunner = sim(
        base_dir,
        niter=niter,
        stepsize=5e-1,
        hmin=0,
        hmax=1,
        hbinsize=1e-4,
        acceptance=0.2,
        adjustf=0.9,
        adjustf_niter=adjustf_niter,
        adjustf_navg=100,
        pt_eq_niter=pt_eq_niter,
        ts_niter=ts_niter,
        ts_freq=ts_freq,
        seeds=seeds,
        single=single,
        record_histogram=record_histogram,
    )

    if not check_kmax_reasonable(sim.findk_configpath):
        print("bv_parallel_tempering: kmax is unreasonable, exiting")
        sys.exit()

    # prepare PT runner
    kmin = 0
    displ_k_min = sim.displ_k_min
    var_displ_k_min = sim.displ_k_min
    kmax = sim.kmax
    path = sim.base_directory
    ptrunner = MPI_BV_PT_RLhandshake(
        mcrunner,
        kmax,
        kmin,
        displ_k_min,
        max_ptiter=min_ptiter + 1,
        pfreq=pfreq,
        skip=nskip,
        test_convergence=test_convergence_ts,
        fast_ct=fast_ct,
        rel_std_err=rel_std_err,
        min_window=min_window,
        max_eq_time=max_eq_time,
        eq_max_ptiter=int(max_tot_niter / niter),
        numnegk=args.numnegk,
        lownegk=args.lownegk,
        base_directory=path,
        verbose=args.verbose,
        fix_com=False,
    )
    assert ptrunner.rank == rank, "rank id does not match"
    assert ptrunner.nprocs == nprocs, "number of processes does not match"

    # run simulation
    start = time.time()
    try:
        ptrunner.run()
        sim.print_success_all(True)
    except:
        view_traceback()
        try:
            sim.print_success_all(False)
        except:
            view_traceback()

    end = time.time()
    print(
        "core: {} ptiter: {} niter: {} adjustf_niter: {} skip: {} pfreq: {}".format(
            rank,
            mcrunner.niter,
            ptrunner.ptiter,
            adjustf_niter,
            ptrunner.skip,
            ptrunner.pfreq,
        )
    )
    print("convert timeseries to hdf5...")
    if rank == 0:
        # it is imperative that max_series_size=0 to avoid loss of raw data, the objective of this step is to
        # reduce the amount of occupied memory and i/o speed without loosing any information
        timeseries = import_pt_time_series(
            sim.base_directory,
            int(sim.mc_params["adjustf_niter"]),
            max_series_size=0,
            ncores=1,
            del_raw=False,
        )
    print("done")
    print("elapsed time", end - start)
