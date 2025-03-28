from __future__ import absolute_import
import numpy as np
import random
import os
import argparse
import logging
import traceback
import copy
from _findk_mcrunner import _findk_mcrunner
from basinvolume.enums import Minimizer
from time import time
import toml


def worker_findk(fname, kwargs):
    try:
        logging.info("Found numerical packing")
        mcrunner = _findk_mcrunner(fname, **kwargs)
        mcrunner.run()
    except:
        logging.error("find_k worker: %s" % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="compute kmax and average displacement for kmax for all jammed packings"
    )
    parser.add_argument("fname", type=str, help="packing file name")
    parser.add_argument(
        "-k",
        "--kstart",
        type=float,
        help="initial guess for kmax, default: 500",
        default=500,
    )
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
        "--nocell",
        action="store_true",
        help="don't use cell lists, default: False",
        default=False,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbosity", default=False)
    parser.add_argument(
        "--seed-takestep",
        type=int,
        help="Seed for the takestep method",
        default=None,
    )
    parser.add_argument("--niter", type=float, default=1e8, help="number of iterations")
    parser.add_argument(
        "--dtol",
        type=float,
        default=1e-4,
        help="for a minimum A, and a minimimization result B, if |d_A-d_B|^2<dtol, then A and B are considered the same ",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0,
        help="softness of the potential (but this is hacked around enough so it's not used for some potentials)",
    )
    parser.add_argument("--ktarget", type=float, default=0.9, help="Target acceptance ration")
    parser.add_argument(
        "--knavg",
        type=float,
        default=1e4,
        help="number of iterations to average for acceptance",
    )
    parser.add_argument("--ktol", type=float, default=0.025, help="tolerance for knavg")
    parser.add_argument(
        "--opt_nsteps",
        type=float,
        default=1e5,
        help="number of steps for optimizer",
    )
    parser.add_argument(
        "--minimizer",
        type=str,
        help="Energy minimization algorithm "
        "used for quenching. Options: 'CG', 'FIRE', 'LBFGS'. "
        "Default: 'FIRE'",
        default="FIRE",
    )
    parser.add_argument("--opt_tol", type=float, default=1e-5, help="tolerance for optimizer")
    parser.add_argument(
        "--opt_kwargs_file",
        type=str,
        default="",
        help="File containing a dictionary of optimizer-specific arguments, if any",
    )

    args = parser.parse_args()

    if args.opt_kwargs_file != "":
        opt_kwargs = toml.load(args.opt_kwargs_file)["opt_kwargs"]
    else:
        opt_kwargs = {}

    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=loglevel,
    )

    logging.info(args)

    fname = args.fname
    packings_dir = args.packings_dir
    if not os.path.isabs(packings_dir):
        packings_dir = os.path.join(os.getcwd(), packings_dir)

    findk_kwargs = dict(
        k=args.kstart,
        niter=args.niter,
        minimizer=args.minimizer,
        opt_tol=args.opt_tol,
        dtol=args.dtol,
        eps=args.eps,
        ktarget=args.ktarget,
        knavg=args.knavg,
        ktol=args.ktol,
        opt_nsteps=args.opt_nsteps,
        opt_kwargs=opt_kwargs,
        packings_dir=packings_dir,
        explore_dir=args.explore_dir,
        use_cell_lists=not args.nocell,
        verbose=args.verbose,
    )

    i32max = np.iinfo(np.int32).max
    if args.seed_takestep is None:
        seed_takestep = random.randint(0, i32max)
    else:
        seed_takestep = args.seed_takestep
    seeds_dict = dict(seed_takestep=seed_takestep)
    seeds = dict(seeds=seeds_dict)
    findk_kwargs_s = copy.deepcopy(dict(findk_kwargs, **seeds))
    start_time = time()
    worker_findk(fname, findk_kwargs_s)
    end_time = time()
    print("time ", end_time - start_time, " seconds")
