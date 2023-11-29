from __future__ import division
from future import standard_library

standard_library.install_aliases()
from builtins import range
import numpy as np
import random
import argparse
import logging
import time
import sys
import os
import pickle
from mpi4py import MPI
from basinvolume.spheres import (
    ConfigBVMCRunner,
    MPI_BV_PT_RLhandshake,
    PT_Worker,
    PT_Master,
    ExchangeScheme,
)
from basinvolume.experiment_2d import configure_bv_exp_mcrunner
from basinvolume.utils import (
    view_traceback,
    check_kmax_reasonable,
    import_pt_time_series,
)
from basinvolume.enums import Minimizer

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
        "jammed_packing_fname", type=str, help="name of xy[z]dr file"
    )
    parser.add_argument(
        "base_directory", type=str, help="directory in which to save results"
    )
    parser.add_argument(
        "-n",
        "--mintotniter",
        type=float,
        help="minimum number of energy evaluations per replica "
        "before checking for convergence default: 5e5. "
        "This sets a lower bound",
        default=5e5,
    )
    parser.add_argument(
        "-m",
        "--maxtotniter",
        type=float,
        help="maximum number of energy evaluations per replica. "
        "This sets an upper bound default: 2e6",
        default=2e6,
    )
    parser.add_argument(
        "--adjustf-niter",
        type=float,
        help="Number of steps to adjust the stepsize. "
        "Default: 0.1 * mintotniter",
        default=None,
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
        "--nocell",
        action="store_true",
        help="don't use cell lists, default: False",
        default=False,
    )
    parser.add_argument(
        "--moveall",
        action="store_true",
        help="Use global particle movements in the MC runner.",
        default=False,
    )
    parser.add_argument(
        "--adjustf-navg",
        type=int,
        help="Number of steps to average over when adjusting the stepsize. "
        "Default: 100",
        default=100,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="More verbose logging (debug level).",
        default=False,
    )
    parser.add_argument(
        "--collect-minima",
        action="store_true",
        help="Collect a database of minima.",
        default=False,
    )
    parser.add_argument(
        "-p",
        "--packings-dir",
        type=str,
        help="protocol to generate packings, assume in cwd",
        default="jammed_packings",
    )
    parser.add_argument(
        "--delraw",
        action="store_true",
        help="Delete raw timeseries textfiles "
        "and only use the HDF5 format.",
        default=False,
    )
    parser.add_argument(
        "--nreplicas",
        type=int,
        help="Number of PT replicas. Default: Number of MPI ranks",
        default=None,
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        help="Waiting time between MPI probes for the job queue master. "
        "Default: 0.0001 (100us)",
        default=0.0001,
    )
    parser.add_argument(
        "--exchange-scheme",
        type=str,
        help="Exchange scheme used in parallel tempering. "
        "Options: 'NEIGHBOR_EXCHANGE', 'INDEPENDENCE_SAMPLING'. "
        "Default: 'NEIGHBOR_EXCHANGE'",
        default="NEIGHBOR_EXCHANGE",
    )
    parser.add_argument(
        "--checkpoint-time",
        type=int,
        help="Minutes after which to create a checkpoint and stop.",
        default=None,
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        help="File from which to load a saved checkpoint.",
        default=None,
    )

    parser.add_argument(
        "--stepsize",
        type=float,
        default=1e-1,
        help="Step size for the MCRunner.",
    )
    parser.add_argument(
        "--opt_nsteps", type=float, default=1e5, help="Optimization runs"
    )
    parser.add_argument(
        "--hmin",
        type=float,
        default=0,
        help="Minimum H value for the MCRunner.",
    )
    parser.add_argument(
        "--hmax",
        type=float,
        default=1000,
        help="Maximum H value for the MCRunner.",
    )
    parser.add_argument(
        "--hbinsize",
        type=float,
        default=1e-1,
        help="H bin size for the MCRunner.",
    )
    parser.add_argument(
        "--acceptance",
        type=float,
        default=0.2,
        help="Acceptance value for the MCRunner.",
    )
    parser.add_argument(
        "--adjustf",
        type=float,
        default=0.9,
        help="Adjust F value for the MCRunner.",
    )
    parser.add_argument(
        "--k_spreading",
        type=str,
        default="linspace",
        help="K spreading method, options: \
        gausslobato, linspace, logspace, positionlinspace",
    )
    parser.add_argument(
        "--bias",
        type = str,
        default = "harmonic",
        help = "Biasing potentials used in umbrella sampling, options:\
            harmonic, radial_gaussian"
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    if args.nreplicas is None:
        nreplicas = nprocs
    else:
        nreplicas = args.nreplicas

    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: Rank {:>2}: %(message)s".format(
            rank
        ),
        datefmt="%d/%m/%Y %H:%M:%S",
        level=loglevel,
    )
    if rank == 0:
        logging.info(args)

    path = args.base_directory
    fname = args.jammed_packing_fname
    single = not args.moveall

    # Parallel Tempering
    min_tot_niter = int(args.mintotniter)
    max_tot_niter = int(args.maxtotniter)

    min_ptiter = int(
        min_tot_niter * 0.1
    )  # 10% PT swaps, this is the initial proposed maximum length of the run. at the end of min_ptiter convergence is checked
    niter = int((min_tot_niter - min_ptiter) / min_ptiter)  # 90% MCMC walk
    if args.adjustf_niter is None:
        adjustf_niter = int(
            min_tot_niter * 0.1
        )  # equilibrate for the first 1/10th of total steps
    else:
        adjustf_niter = int(args.adjustf_niter)
    nskip = int(
        adjustf_niter / niter
    )  # don't swap while adjusting the step-size
    # pt_eq_niter equilibrate pt for the following 4/10th of total steps (), this has an effect on histogram
    # and on checksameminimum: it only starts recording the neighbouring minima when equilibration is reached
    pt_eq_niter = 0
    # the histogram starts recording the mean after adjustf_niter + pt_eq_niter steps
    pfreq = int(
        (min_ptiter - 1) * 0.1
    )  # print every 1/10th of min_ptiter (this will give 5 snapshots) # this is also frequency of tests
    ts_freq = 1
    ts_niter = int(niter * pfreq / ts_freq)
    perform_minimisation_convergence_test = False
    test_convergence_ts = True
    record_histogram = False
    assert (
        record_histogram == False and pt_eq_niter == 0 and ts_freq == 1
    )  # ts_freq must be 1 with current output implementation (all based on timeseries)
    rel_std_err = (
        args.relstderr
    )  # relative standard error in the mean used by convergence test
    min_window = (
        min_tot_niter * 0.5  # minimum amount of data before trying to check convergence
    )
    max_eq_time = min_tot_niter * 0.5  # maximum amount of data to discard (throw away max the first 2.5e5 points, to avoid reading spurious features)
    fast_ct = False  # if false skip euristic search for equilibration point
    collect_minima_list = args.collect_minima
    i32max = np.iinfo(np.int32).max

    seeds = dict(
        seed_takestep=random.randint(0, i32max),
        seed_metropolis=random.randint(0, i32max),
    )
    logging.info(seeds)

    if args.exchange_scheme.upper() in ExchangeScheme.__members__:
        exchange_scheme = ExchangeScheme[args.exchange_scheme.upper()]
    else:
        raise ValueError(
            "Unknown exchange scheme: {}".format(args.exchange_scheme)
        )

    # prepare MC runner
    if ".xydfr" in fname or ".xyzdfr" in fname:
        if rank == 0:
            logging.info("found experimental packing")
        sim = configure_bv_exp_mcrunner(rank, nprocs)
    else:
        if rank == 0:
            logging.info("found numerical packing")
        sim = ConfigBVMCRunner(rank, nprocs)

    # Specific check overlap with cell lists and job queue doesn't work.
    # Use specific version without cell lists instead, since that's faster than
    # the non-specific cell lists version
    mcrunner_checkoverlap_cell_lists = False

    mcrunner = sim(
        fname,
        bias = args.bias,
        niter=niter,
        stepsize=args.stepsize,
        opt_nsteps=args.opt_nsteps,
        hmin=args.hmin,
        hmax=args.hmax,
        hbinsize=args.hbinsize,
        acceptance=args.acceptance,
        adjustf=args.adjustf,
        adjustf_niter=adjustf_niter,
        adjustf_navg=args.adjustf_navg,
        pt_eq_niter=pt_eq_niter,
        ts_niter=ts_niter,
        ts_freq=ts_freq,
        perform_convergence_test=perform_minimisation_convergence_test,
        collect_minima_list=collect_minima_list,
        seeds=seeds,
        use_cell_lists=not args.nocell,
        checkoverlap_cell_lists=mcrunner_checkoverlap_cell_lists,
        single=single,
        record_histogram=record_histogram,
        packings_dir=args.packings_dir,
        base_dir=path,
    )

    if not check_kmax_reasonable(sim.findk_configpath):
        logging.error("bv_parallel_tempering: kmax is unreasonable, exiting")
        sys.exit()

    # prepare PT runner
    kmin = 0
    displ_k_min = sim.displ_k_min
    var_displ_k_min = sim.var_displ_k_min
    kmax = sim.kmax

    start = time.time()
    exit_on_checkpoint = False
    logging.info("path: {}".format(path))
    if nprocs < nreplicas:
        if rank == 0:
            logging.info("Using job queue with {} workers.".format(nprocs - 1))

            try:
                if args.checkpoint_time is None:
                    checkpoint_time = None
                else:
                    checkpoint_time = 60 * args.checkpoint_time
                if args.load_checkpoint is None:
                    master = PT_Master(
                        nreplicas,
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
                        k_spreading=args.k_spreading,
                        bias = args.bias,
                        print_status=args.verbose,
                        base_directory=path,
                        sleep_seconds=args.sleep_seconds,
                        exchange_scheme=exchange_scheme,
                        checkpoint_time=checkpoint_time,
                    )
                else:
                    checkpoint_path = os.path.join(path, args.load_checkpoint)
                    with open(checkpoint_path, "rb") as infile:
                        master = pickle.load(infile)
                    master.init_state(
                        base_directory=path, checkpoint_time=checkpoint_time
                    )
                master.run()
                exit_on_checkpoint = master.created_checkpoint
                if args.load_checkpoint is not None and not exit_on_checkpoint:
                    os.remove(args.load_checkpoint)
                if not exit_on_checkpoint:
                    sim.print_success_all(True)
                logging.info(
                    "ptiter: {} niter: {} adjustf_niter: {} skip: {} pfreq: {}".format(
                        master.ptiter,
                        mcrunner.niter,
                        adjustf_niter,
                        master.skip,
                        master.pfreq,
                    )
                )
            except Exception:
                view_traceback()
                for iworker in range(1, nprocs):
                    comm.Isend(np.array([-1], dtype="d"), dest=iworker)
                sim.print_success_all(False)

        else:
            worker = PT_Worker(mcrunner)
            worker.run()
            if collect_minima_list:
                mcrunner.dump_minima_list("{}/minima_list.sqlite".format(rank))

    else:
        if rank == 0:
            logging.info("Using handshake with {} replicas.".format(nprocs))
        if exchange_scheme != ExchangeScheme.NEIGHBOR_EXCHANGE:
            raise ValueError(
                "Only the exchange scheme NEIGHBOR_EXCHANGE works with PT handshake."
            )
        if (
            args.checkpoint_time is not None
            or args.load_checkpoint is not None
        ):
            raise ValueError("Checkpointing does not work with PT handshake.")

        ptreplica = MPI_BV_PT_RLhandshake(
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
            k_spreading=args.k_spreading,
            bias=args.bias,
            base_directory=path,
        )
        assert ptreplica.rank == rank, "rank id does not match"
        assert ptreplica.nprocs == nprocs, "number of processes does not match"

        # run simulation
        try:
            ptreplica.run()
            if collect_minima_list:
                mcrunner.dump_minima_list("{}/minima_list.sqlite".format(rank))
            sim.print_success_all(True)
        except:
            view_traceback()
            try:
                sim.print_success_all(False)
            except:
                view_traceback()

        logging.info(
            "ptiter: {} niter: {} adjustf_niter: {} skip: {} pfreq: {}".format(
                ptreplica.ptiter,
                mcrunner.niter,
                adjustf_niter,
                ptreplica.skip,
                ptreplica.pfreq,
            )
        )

    if rank == 0:
        end = time.time()
        if not exit_on_checkpoint:
            logging.info("Convert timeseries to hdf5...")
            # it is imperative that max_series_size=0 to avoid loss of raw data,
            # the objective of this step is to reduce the amount of occupied memory
            # and i/o speed without losing any information
            timeseries = import_pt_time_series(
                sim.base_dir,
                int(sim.mc_params["adjustf_niter"]),
                max_series_size=0,
                ncores=1,
                del_raw=args.delraw,
            )
        logging.info("Done")
        logging.info("Elapsed time: {}".format(end - start))
