import numpy as np
import argparse
import os
import shutil
from generate_packing import HS_Generate_Packing
from generate_jammed_packing import HS_Generate_Jammed_Packing
from basinvolume.utils import import_packing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a sequence of packings with increasing shear.")
    # General arguments
    parser.add_argument("-s", "--step", type=float, help="Size of shearing steps. Default: 0.001", default=0.001)
    parser.add_argument("-f", "--final_shear", type=float, help="Final shear. Default: 1.0", default=1.)
    parser.add_argument("--cell", action='store_true', help="Use cell lists. Not yet supported! "
                        "Default: False", default=False)
    parser.add_argument("--npackings", type=int, help="Number of packings to produce. Default: 1", default=1)
    parser.add_argument("-n", "--nparticles", type=int, help="Number of particles. Default: 32", default=32)
    parser.add_argument("-d", "--boxdim", type=int, help="Box dimensions. Default: 2", default=2)
    parser.add_argument("--input_packings", type=str, help="Use precalculated loose packings from directory.")
    parser.add_argument("--input_jammed", type=str, help="Use precalculated jammed packings from directory.")

    # Arguments for generating packings
    parser.add_argument("-phs", "--density_hs", type=float, help="Target hard sphere packing fraction. "
                        "Default: Calculated from soft sphere packing fraction by phs = pss * 0.7/0.88",
                        default=None)
    parser.add_argument("--rmean", type=float, help="Mean particle radius. Default: 1.0",
                        default=1.0)
    parser.add_argument("--rsigma", type=float, help="Percent standard deviation. Default: 0.1",
                        default=0.1)
    parser.add_argument("--hsfniter", type=int, help="Number of hard sphere fluid MC steps "
                        "between 2 samples. Default: 1e6", default=1e6)
    parser.add_argument("--hsfstep", type=float, help="Stepsize for hard sphere fluid MC "
                        "simulation. Default: 1e-3", default=1e-3)
    parser.add_argument("--dpath", type=str, help="Path to xy(z)d file from which to import diameters. "
                        "Default: None", default=None)
    parser.add_argument("--packing_moveall", action='store_true', help="Move all particles at each hard "
                        "sphere fluid MC step. Default: False",default=False)
    parser.add_argument("--packing_method", type=str, help="Protocol for generating packings. Default: "
                        "'quench'", default="quench")

    # Arguments for generating jammed packings
    parser.add_argument("-pss", "--density_ss", type=float, help="Target soft sphere packing fraction. "
                        "Default: 0.85", default=0.85)
    parser.add_argument("--min_tol", type=float, help="RMS tolerance of the minimizer. Default: 1e-9",
                        default=1e-9)
    parser.add_argument("--minimizer", type=str, help="Energy minimization algorithm used for quenching. "
                        "Options: 'cg', 'fire'. Default: 'fire'", default='fire')

    args = parser.parse_args()

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
            print("The packings directory already exists!")
        else:
            shutil.copytree(args.input_packings, "packings")
    elif args.input_jammed is None:
        print("\n--------- Generating loose packings ---------")
        gen_packing = HS_Generate_Packing(args.nparticles, method=args.packing_method,
                                          bdim=args.boxdim, packing_frac=density_hs,
                                          hs_radii=hs_radii, mu=args.rmean, sig=args.rsigma,
                                          new_poly=False, hsf_niter=args.hsfniter,
                                          hsf_stepsize=args.hsfstep, max_iter=args.npackings,
                                          use_cell_lists=args.cell, single=not args.packing_moveall,
                                          start_iteration=0, distance_method='lees-edwards',
                                          pot_kwargs=pot_kwargs)
        gen_packing.run()

    # Generate jammed packings at no shear
    if args.input_jammed is not None:
        if not os.path.isdir(args.input_jammed):
            raise IOError("The specified input packings-directory does not exist "
                          "({})!".format(args.input_jammed))
        if os.path.isdir("shear_0.0"):
            print("The shear_0.0 directory already exists!")
        else:
            shutil.copytree(args.input_jammed, "shear_0.0")
    else:
        print("\n--------- Generating jammed packings ---------")
        gen_jammed_packing = HS_Generate_Jammed_Packing(target_packing_frac=args.density_ss,
                                                        packings_dir="packings", outdir="shear_0.0",
                                                        tol=args.min_tol, use_cell_lists=args.cell,
                                                        show=False, opt_pot_str='hs_wca',
                                                        minimizer=args.minimizer)
        gen_jammed_packing.run()

    # Generate sheared packings
    unjammed_packings = []
    for shear in np.arange(0., args.final_shear - 0.5 * args.step, args.step) + args.step:
        pot_kwargs['shear'] = shear
        print("\n--------- Shear: {} ---------".format(shear))
        gen_jammed_packing = HS_Generate_Jammed_Packing(target_packing_frac=args.density_ss,
                                                        packings_dir="shear_{}".format(shear - args.step),
                                                        import_jammed=True,
                                                        outdir="shear_{}".format(shear),
                                                        tol=args.min_tol, use_cell_lists=args.cell,
                                                        show=False, opt_pot_str='hs_wca',
                                                        override_pot_kwargs=pot_kwargs,
                                                        minimizer=args.minimizer)
        successes = gen_jammed_packing.run()

        # Check for failed (unjammed) packings and save them with packing number and current shear
        if not all(success for (_, success) in successes):
            unjammed_packings += [(int(fname[len("jammed_packing"):].split('.')[0]), shear)
                         for (fname, success) in successes if not success]

    # Check for unjammed packings
    if len(unjammed_packings) != 0:
        print("\n{} packing(s) unjammed:".format(len(unjammed_packings)))

        for packing, shear in unjammed_packings:
            print("Packing {} unjammed at shear {}".format(packing, shear))
