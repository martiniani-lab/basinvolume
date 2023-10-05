from __future__ import absolute_import
from builtins import range
import numpy as np
import random
import os
import argparse
import logging
import traceback
import copy
from _kmin_mcrunner import KminMCRunner
from basinvolume.experiment_2d import _kmin_exp_mcrunner
from basinvolume.utils import check_kmax_reasonable
from basinvolume.enums import Minimizer


def worker_kmin(fname, kwargs):
    try:
        if ".xydfr" in fname or ".xyzdfr" in fname:
            logging.info("Found experimental packing")
            mcrunner = _kmin_exp_mcrunner(fname, **kwargs)
        else:
            logging.info("Found numerical packing")
            mcrunner = KminMCRunner(fname, **kwargs)
        if check_kmax_reasonable(mcrunner.findk_configpath):
            mcrunner.run()
        else:
            logging.error("bv_find_kmin.py: kmax is unreasonable, exiting")
    except:
        logging.error("kmin worker: %s" % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="compute kmax and minimum average displacement for kmin for all jammed packings"
    )
    parser.add_argument("fname", type=str, help="packing file name")
    parser.add_argument(
        "-p",
        "--packings-dir",
        type=str,
        help="protocol to generate packings, assume in cwd",
        default="jammed_packings",
    )
    parser.add_argument(
        "--explore-dir",
        type=str,
        help="Start of the directory name for the output data. "
        "Default: 'explore_bv_jammed_packing'",
        default="explore_bv_jammed_packing",
    )
    parser.add_argument(
        "-n",
        "--niter",
        type=float,
        help="number of energy evaluation, default: 1e5",
        default=1e5,
    )
    parser.add_argument(
        "--adjustf-niter",
        type=float,
        help="number of steps to adjust stepsize, default: 1e4",
        default=1e4,
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
        help="don't use cell lists, default: False",
        default=False,
    )
    parser.add_argument(
        "--minimizer",
        type=str,
        help="Energy minimization algorithm "
        "used for quenching. Options: 'CG', 'FIRE', 'LBFGS'. "
        "Default: 'FIRE'",
        default="FIRE",
    )
    parser.add_argument(
        "--rsts",
        action="store_true",
        help="record steps timeseries for diffusion studies, default: False",
        default=False,
    )
    parser.add_argument(
        "--rsts-only",
        action="store_true",
        help="record steps timeseries for diffusion studies ONLY, default: False",
        default=False,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbosity", default=False
    )
    parser.add_argument(
        "--seed-takestep",
        type=int,
        help="Seed for the takestep method",
        default=None,
    )
    parser.add_argument(
        "--seed-metropolis",
        type=int,
        help="Seed for the metropolis algorithm",
        default=None,
    )

    # TODO: add description for these arguments
    parser.add_argument("--k", type=int, default=0, help="Description for k.")
    parser.add_argument(
        "--stepsize",
        type=float,
        default=1e-1,
        help="Description for stepsize.",
    )
    parser.add_argument(
        "--dtol", type=float, default=1e-4, help="Description for dtol."
    )
    parser.add_argument(
        "--eps", type=float, default=1.0, help="Description for eps."
    )
    parser.add_argument(
        "--hmin", type=int, default=0, help="Description for hmin."
    )
    parser.add_argument(
        "--hmax", type=int, default=1000, help="Description for hmax."
    )
    parser.add_argument(
        "--hbinsize", type=int, default=1, help="Description for hbinsize."
    )
    parser.add_argument(
        "--acceptance",
        type=float,
        default=0.2,
        help="Description for acceptance.",
    )
    parser.add_argument(
        "--adjustf", type=float, default=0.9, help="Description for adjustf."
    )
    parser.add_argument(
        "--opt_dtmax", type=int, default=1, help="Description for opt_dtmax."
    )
    parser.add_argument(
        "--opt_tol", type=float, default=1e-10, help="optimizer tolerance"
    )
    parser.add_argument(
        "--opt_nsteps",
        type=float,
        default=1e5,
        help="number of steps for optimizer",
    )
    parser.add_argument(
        "--record_trajectory_npoints",
        type=float,
        default=1e4,
        help="Description for record_trajectory_npoints.",
    )

    args = parser.parse_args()

    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=loglevel,
    )

    fname = args.fname
    packings_dir = args.packings_dir
    if not os.path.isabs(packings_dir):
        packings_dir = os.path.join(os.getcwd(), packings_dir)

    if args.minimizer.upper() in Minimizer.__members__:
        minimizer = Minimizer[args.minimizer.upper()]
    else:
        raise ValueError("Unknown minimizer: {}".format(args.minimizer))

    single = not args.moveall
    if args.rsts_only:
        args.rsts = True

    kmin_kwargs = dict(
        k=args.k,
        stepsize=args.stepsize,
        niter=args.niter,
        dtol=args.dtol,
        eps=args.eps,
        hmin=args.hmin,
        hmax=args.hmax,
        hbinsize=args.hbinsize,
        acceptance=args.acceptance,
        adjustf=args.adjustf,
        adjustf_niter=args.adjustf_niter,
        adjustf_navg=100,
        opt_dtmax=args.opt_dtmax,
        opt_tol=args.opt_tol,
        opt_nsteps=args.opt_nsteps,
        packings_dir=packings_dir,
        explore_dir=args.explore_dir,
        use_cell_lists=not args.nocell,
        single=single,
        minimizer=minimizer,
        verbose=args.verbose,
        record_steps_timeseries=args.rsts,
        record_steps_timeseries_every=[
            int(np.ceil(1.5**n)) for n in range(22)
        ],
        print_diffusion_only=args.rsts_only,
        record_trajectory_npoints=int(args.record_trajectory_npoints),
    )

    i32max = np.iinfo(np.int32).max
    if args.seed_takestep is None:
        seed_takestep = random.randint(0, i32max)
    else:
        seed_takestep = args.seed_takestep
    if args.seed_metropolis is None:
        seed_metropolis = random.randint(0, i32max)
    else:
        seed_metropolis = args.seed_metropolis
    seeds_dict = dict(
        seed_takestep=seed_takestep, seed_metropolis=seed_metropolis
    )
    seeds = dict(seeds=seeds_dict)
    kmin_kwargs_s = copy.deepcopy(dict(kmin_kwargs, **seeds))

    worker_kmin(
        fname,
        kmin_kwargs_s,
    )
