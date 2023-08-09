from __future__ import absolute_import
import random
import numpy as np
import os
import argparse
import logging
import traceback
import copy
from _config_innersphere_mcrunner import ConfigInnerSphereMCRunner
from basinvolume.enums import Minimizer


def worker_innersphere(fname, kwargs):
    try:
        if ".xydfr" in fname or ".xyzdfr" in fname:
            logging.info("Found experimental packing")
            raise NotImplementedError("innersphere_mcrunner not implemented!")
        else:
            logging.info("Found numerical packing")
            mcrunner = ConfigInnerSphereMCRunner(fname, **kwargs)
        mcrunner.run()
    except:
        logging.error("innersphere worker: %s" % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="compute dos for inner sphere of basin"
    )
    parser.add_argument(
        "fname",
        type=str,
        help="packing file name, example fname jammed_packing0.xydr",
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

    # when niter=None, niter is set equal to exact number of PT niter
    innersphere_kwargs = dict(
        niter=1e5,
        dtol=1e-4,
        eps=1.0,
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
    seeds_dict = dict(seed_takestep=random.randint(0, i32max))
    seeds = dict(seeds=seeds_dict)
    innersphere_kwargs_s = copy.deepcopy(dict(innersphere_kwargs, **seeds))
    worker_innersphere(
        fname,
        innersphere_kwargs_s,
    )
