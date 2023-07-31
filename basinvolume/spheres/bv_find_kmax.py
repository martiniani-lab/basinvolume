from __future__ import absolute_import
import numpy as np
import random
import os
import argparse
import logging
import traceback
import copy
from _findk_mcrunner import _findk_mcrunner
from basinvolume.experiment_2d import _findk_exp_mcrunner
from basinvolume.enums import Minimizer


def worker_findk(fname, kwargs):
    try:
        if ".xydfr" in fname or ".xyzdfr" in fname:
            logging.info("Found experimental packing")
            mcrunner = _findk_exp_mcrunner(fname, **kwargs)
        else:
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
    parser.add_argument(
        "--minimizer",
        type=str,
        help="Energy minimization algorithm "
        "used for quenching. Options: 'CG', 'FIRE', 'LBFGS'. "
        "Default: 'FIRE'",
        default="FIRE",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbosity", default=False
    )
    parser.add_argument(
        "--seed-takestep", type=int, help="Seed for the takestep method", default=None
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

    logging.info(args)

    fname = args.fname
    packings_dir = args.packings_dir
    if not os.path.isabs(packings_dir):
        packings_dir = os.path.join(os.getcwd(), packings_dir)

    if args.minimizer.upper() in Minimizer.__members__:
        minimizer = Minimizer[args.minimizer.upper()]
    else:
        raise ValueError("Unknown minimizer: {}".format(args.minimizer))

    findk_kwargs = dict(
        k=args.kstart,
        niter=1e8,
        dtol=1e-4,
        eps=1.0,
        ktarget=0.9,
        knavg=1e4,
        ktol=0.025,
        opt_dtmax=1,
        opt_tol=1e-5,
        opt_nsteps=1e5,
        packings_dir=packings_dir,
        explore_dir=args.explore_dir,
        use_cell_lists=not args.nocell,
        minimizer=minimizer,
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
    worker_findk(fname, findk_kwargs_s)
