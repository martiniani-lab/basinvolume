from __future__ import division
import numpy as np
import argparse
import os
import shutil
import logging
import traceback
import multiprocessing as mp
from generate_packing import HS_Generate_Packing
from generate_jammed_packing import HS_Generate_Jammed_Packing
from basinvolume.utils import import_packing, trymakedir


def worker_packing(kwargs, nparticles, start_iteration=0):
    try:
        gen_packing = HS_Generate_Packing(nparticles, start_iteration=start_iteration, **kwargs)
        gen_packing.run()
    except:
        logging.error('worker_packing worker: %s' % (traceback.format_exc()))


def worker_jammed_packing(kwargs, logging_tag, packing_nrs=None):
    try:
        gen_jammed_packing = HS_Generate_Jammed_Packing(packing_nrs=packing_nrs,
                                                        logging_tag=logging_tag,
                                                        **kwargs)
        return gen_jammed_packing.run()
    except:
        logging.error('worker_jammed_packing worker: %s' % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a sequence of packings "
                                     "with increasing shear.")
    # General arguments
    parser.add_argument("-s", "--step", type=float, help="Size of shearing steps. "
                        "Default: 0.001", default=0.001)
    parser.add_argument("-f", "--final_shear", type=float, help="Final shear. Default: 1.0", default=1.)
    parser.add_argument("--cell", action='store_true', help="Use cell lists. "
                        "Default: False", default=False)
    parser.add_argument("--npackings", type=int, help="Number of packings to produce. "
                        "Default: 1", default=1)
    parser.add_argument("-n", "--nparticles", type=int, help="Number of particles. "
                        "Default: 32", default=32)
    parser.add_argument("-d", "--boxdim", type=int, help="Box dimensions. Default: 2", default=2)
    parser.add_argument("--input_packings", type=str, help="Use precalculated "
                        "loose packings from directory.")
    parser.add_argument("--input_jammed", type=str, help="Use precalculated jammed "
                        "packings from directory.")
    parser.add_argument("-j", "--njobs", type=int, help="Number of jobs to run in parallel. "
                        "Default: 1 (serial)", default=1)

    # Arguments for generating packings
    parser.add_argument("-phs", "--density_hs", type=float, help="Target hard sphere packing fraction. "
                        "Default: Calculated from soft sphere packing fraction by phs = pss * 0.7/0.88",
                        default=None)
    parser.add_argument("--rmean", type=float, help="Mean particle radius. Default: 1.0",
                        default=1.0)
    parser.add_argument("--rsigma", type=float, help="Percent standard deviation. Default: 0.1",
                        default=0.1)
    parser.add_argument("--hsf-niter-dif", type=int, help="Step count for the "
                        "estimation of the decorrelation step count. Default: 1e9",
                        default=1e9)
    parser.add_argument("--dpath", type=str, help="Path to xy(z)d file from which to import diameters. "
                        "Default: None", default=None)
    parser.add_argument("--packing_moveall", action='store_true', help="Move all "
                        "particles at each hard sphere fluid MC step. Default: False",
                        default=False)
    parser.add_argument("--packing_method", type=str, help="Protocol for generating packings. Default: "
                        "'quench'", default="quench")

    # Arguments for generating jammed packings
    parser.add_argument("-pss", "--density_ss", type=float, help="Target soft sphere packing fraction. "
                        "Default: 0.85", default=0.85)
    parser.add_argument("--min_tol", type=float, help="RMS tolerance of the minimizer. Default: 1e-9",
                        default=1e-9)
    parser.add_argument("--minimizer", type=str, help="Energy minimization algorithm "
                        "used for quenching. Options: 'cg', 'fire'. Default: 'fire'",
                        default='fire')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set up thread pool
    if args.njobs > 1:
        mypool = mp.Pool(args.njobs)

    # Calculate hard sphere density
    if args.density_hs is None:
        density_hs = args.density_ss * 0.7/0.88
    else:
        density_hs = args.density_hs

    # Import radii from other configuration file
    dpath = args.dpath
    hs_radii = None
    if dpath:
        if not os.path.isabs(args.dpath):
            dpath = os.path.abspath(dpath)
        hs_radii = import_packing(dpath, False, args.boxdim)['hs_radii']

    # Generate packings at no shear
    pot_kwargs = {'shear': 0.0}
    if args.input_packings is not None:
        if not os.path.isdir(args.input_packings):
            raise IOError("The specified input packings-directory does not exist "
                          "({})!".format(args.input_packings))
        if os.path.isdir("packings"):
            if args.input_packings != "packings":
                logging.warning("The packings directory already exists.")
        else:
            shutil.copytree(args.input_packings, "packings")
    elif args.input_jammed is None:
        logging.info("Generating loose packings:")
        packing_kwargs = dict(method=args.packing_method,
                              bdim=args.boxdim, packing_frac=density_hs,
                              hs_radii=hs_radii, mu=args.rmean, sig=args.rsigma,
                              new_poly=False, hsf_niter_dif=args.hsf_niter_dif,
                              max_iter=args.npackings,
                              use_cell_lists=args.cell,
                              single=not args.packing_moveall,
                              distance_method='lees-edwards',
                              pot_kwargs=pot_kwargs)
        if args.njobs > 1:
            packing_kwargs['max_iter'] = 1
            worker_packing(packing_kwargs, args.nparticles)
            packing_kwargs['precalc_config_file'] = os.path.join('packings', 'packing0.config')
            results = []
            for packing_nr in xrange(1, args.npackings):
                results.append(mypool.apply_async(
                    worker_packing, args=(packing_kwargs, args.nparticles, packing_nr)))
            for result in results:
                result.get()
        else:
            worker_packing(packing_kwargs, args.nparticles)

    # Generate jammed packings at no shear
    jammed_kwargs = dict(target_packing_frac=args.density_ss,
                         tol=args.min_tol, use_cell_lists=args.cell,
                         show=False, opt_pot_str='hs_wca',
                         minimizer=args.minimizer)
    if args.input_jammed is not None:
        if not os.path.isdir(args.input_jammed):
            raise IOError("The specified input packings-directory does not exist "
                          "({})!".format(args.input_jammed))
        if os.path.isdir("shear_0.0"):
            if args.input_jammed != "shear_0.0":
                logging.warning("The shear_0.0 directory already exists.")
        else:
            shutil.copytree(args.input_jammed, "shear_0.0")
    else:
        logging.info("Generating jammed packings")
        unsheared_kwargs = dict(jammed_kwargs, packings_dir="packings",
                                outdir="shear_0.0")
        trymakedir(unsheared_kwargs['outdir'])
        if args.njobs > 1:
            results = []
            for packing_nr in xrange(args.npackings):
                results.append(mypool.apply_async(
                    worker_jammed_packing, args=(unsheared_kwargs,
                                                 "Shear 0.0, {}".format(packing_nr),
                                                 [packing_nr])))
            for result in results:
                result.get()
        else:
            worker_jammed_packing(unsheared_kwargs, "Shear 0.0")

    # Generate sheared packings
    unjammed_packings = []
    for shear in np.arange(0., args.final_shear - 0.5 * args.step, args.step) + args.step:
        pot_kwargs['shear'] = shear
        sheared_kwargs = dict(jammed_kwargs,
                              packings_dir="shear_{}".format(shear - args.step),
                              outdir="shear_{}".format(shear),
                              import_jammed=True,
                              override_pot_kwargs=pot_kwargs)
        trymakedir(sheared_kwargs['outdir'])
        if args.njobs > 1:
            results = []
            for packing_nr in xrange(args.npackings):
                results.append(mypool.apply_async(
                    worker_jammed_packing,
                    args=(sheared_kwargs,
                          "Shear {}, {}".format(shear, packing_nr),
                          [packing_nr])))
            successes = []
            for result in results:
                successes += result.get()
        else:
            successes = worker_jammed_packing(
                sheared_kwargs,
                "Shear {}".format(shear))

        # Check for failed (unjammed) packings and save them with packing number and current shear
        if not all(success for (_, success) in successes):
            unjammed_packings += [(int(fname[len("jammed_packing"):].split('.')[0]), shear)
                         for (fname, success) in successes if not success]

    # Check for unjammed packings
    if len(unjammed_packings) != 0:
        logging.warning("{} packing(s) unjammed:".format(len(unjammed_packings)))

        for packing, shear in unjammed_packings:
            logging.warning("Packing {} unjammed at shear {}".format(packing, shear))

    if args.njobs > 1:
        mypool.close()
        mypool.join()
