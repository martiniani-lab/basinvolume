import numpy as np
import argparse
from generate_packing import HS_Generate_Packing
from generate_jammed_packing import HS_Generate_Jammed_Packing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a sequence of packings with increasing shear.")
    # General arguments
    parser.add_argument("-s", "--step", type=float, help="Size of shearing steps. Default: 0.001", default=0.001)
    parser.add_argument("-f", "--final_shear", type=float, help="Final shear. Default: 1.0", default=1.)
    parser.add_argument("--cell", action='store_true', help="Use cell lists. Not yet supported! "
                        "Default: False", default=False)
    # TODO: Useful features
    # parser.add_argument("-n","--npackings", type=int, help="number of packings to produce", default=1)

    # Arguments for generating packings
    parser.add_argument("-n", "--nparticles", type=int, help="Number of particles. Default: 32", default=32)
    parser.add_argument("-d", "--boxdim", type=int, help="Box dimensions. Default: 3", default=2)
    parser.add_argument("-phs", "--density_hs", type=float, help="Target hard sphere packing fraction. "
                        "Default: 0.68", default=0.68)
    parser.add_argument("--rmean", type=float, help="Mean particle radius. Default: 1.0",
                        default=1.0)
    parser.add_argument("--rsigma", type=float, help="Percent standard deviation. Default: 0.1",
                        default=0.1)
    parser.add_argument("--hsfniter", type=int, help="Number of hard sphere fluid MC steps "
                        "between 2 samples. Default: 1e6", default=1e6)
    parser.add_argument("--hsfstep", type=float, help="Stepsize for hard sphere fluid MC "
                        "simulation. Default: 1e-3", default=1e-3)
    parser.add_argument("--dpath", type=str, help="Path to xy(z)d path from where to import diameters. "
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
    args = parser.parse_args()

    #import radii from other configuration file
    dpath = args.dpath
    hs_radii = None
    if dpath:
        if not os.path.isabs(args.dpath):
            dpath = os.path.abspath(dpath)
        if args.boxdim == 2:
            coords, hs_diameters = read_xyd(dpath)
        else:
            coords, hs_diameters = read_xyzd(dpath)
        hs_radii = hs_diameters/2

    # Generate packing at no shear
    pot_kwargs = {'shear': 0.0}
    gen_packing = HS_Generate_Packing(args.nparticles, method=args.packing_method, bdim=args.boxdim,
                                      packing_frac=args.density_hs, hs_radii=hs_radii,
                                      mu=args.rmean, sig=args.rsigma, new_poly=False,
                                      hsf_niter=args.hsfniter, hsf_stepsize=args.hsfstep, max_iter=1,
                                      use_cell_lists=args.cell, single=not args.packing_moveall,
                                      start_iteration=0, distance_method='lees-edwards',
                                      pot_kwargs=pot_kwargs)
    gen_packing.run()

    # Generate jammed packing at no shear
    gen_jammed_packing = HS_Generate_Jammed_Packing(packing_frac=args.density_ss,
                                     packings_dir="packings", outdir="shear_0.0",
                                     tol=args.min_tol, use_cell_lists=args.cell,
                                     show=False, opt_pot_str='hs_wca', pot_kwargs=pot_kwargs)
    gen_jammed_packing.run()

    for shear in np.arange(0., args.final_shear, args.step) + args.step:
        gen_jammed_packing = HS_Generate_Jammed_Packing(packing_frac=args.density_ss,
                                         packings_dir="shear_{}".format(shear - args.step),
                                         import_jammed=True, outdir="shear_{}".format(shear),
                                         tol=args.min_tol, use_cell_lists=args.cell,
                                         show=False, opt_pot_str='hs_wca', override_shear=shear)
        print("shear: {}".format(shear))
        gen_jammed_packing.run()
