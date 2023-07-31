from __future__ import division
from __future__ import absolute_import
import numpy as np
import os
import argparse
import multiprocessing as mp
import logging
from .bond_orientational_order import worker_boo
from .density_of_states import worker_dos
from .displacement import worker_disp
from .inversion_symmetry import worker_invsym
from .neighbors import worker_neighbors
from .pressure_tensor import worker_pressure


def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute structural properties " "for jammed packings."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run for all packing subdirectories.",
        default=False,
    )
    parser.add_argument(
        "-j", "--ncores", type=int, help="Threads for parallel execution.", default=7
    )

    parser.add_argument(
        "-d",
        "--workspace-dir",
        type=str,
        help="Top-level dir containing " "the packings, e.g. 'n32_phi88_2D'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force to run on all packings.",
        default=False,
    )
    parser.add_argument(
        "--nonex",
        action="store_false",
        help="Run also for packings "
        "for which there are no work folders ('explore_bv_[...]', "
        "created e.g. by parallel tempering).",
        default=True,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix for the work directory. " "Default: 'explore_bv_'",
        default="explore_bv_",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing the " "jammed packings. Default: 'jammed_packings'",
        default="jammed_packings",
    )
    parser.add_argument(
        "--nocell",
        action="store_true",
        help="Don't use cell lists. " "Default: False",
        default=False,
    )

    # bond-orientational order
    parser.add_argument(
        "--solid",
        action="store_true",
        help="Use solid angle method " "to find and weight neighbors.",
        default=False,
    )

    # displacement
    parser.add_argument(
        "--packings-old",
        type=str,
        help="Directory containing the "
        "jammed packings with the particle positions to calculate the "
        "displacement from. Displacement calculation is turned off by default.",
        default=None,
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Don't subtract the " "centre of mass displacement.",
        default=False,
    )
    parser.add_argument(
        "--shear",
        type=float,
        help="Difference in shear between "
        "the two packings. Setting this triggers the additional "
        "calculation of non-affine displacements.",
        default=None,
    )

    # neighbors
    parser.add_argument(
        "--restrict-neighbors",
        type=str,
        help="Prefix leading to "
        "a neighbor lists file. This string is analogous to the normal prefix. "
        "Only neighbors in these lists are considered.",
        default=None,
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        help="Multiple of particle radii "
        "defining the maximum neighbor distance. Default: 1",
        default=1.0,
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    ncores = args.ncores
    kwargs = dict(
        verbose=True,
        force=args.force,
        existing_only=args.nonex,
        jammed_packings_dir=args.input_dir,
        prefix=args.prefix,
        use_cell_lists=not args.nocell,
    )

    structural_props = []

    # bond-orientational order
    boo_kwargs = dict(kwargs)
    if args.solid:
        boo_kwargs.update(solid_angle_weighted=args.solid)
    structural_props.append((worker_boo, boo_kwargs))

    # density of states
    dos_kwargs = dict(kwargs)
    structural_props.append((worker_dos, dos_kwargs))

    # displacement
    if args.packings_old is not None:
        disp_kwargs = dict(
            kwargs,
            packings_old=args.packings_old,
            shear=args.shear,
            sub_centre_mass=not args.drift,
        )
        structural_props.append((worker_disp, disp_kwargs))

    # local inversion symmetry
    invsym_kwargs = dict(kwargs)
    structural_props.append((worker_invsym, invsym_kwargs))

    # neighbors
    neighbors_kwargs = dict(
        kwargs, restrict_neighbors=args.restrict_neighbors, cutoff=args.cutoff
    )
    structural_props.append((worker_neighbors, neighbors_kwargs))

    # pressure tensor
    pressure_kwargs = dict(kwargs)
    structural_props.append((worker_pressure, pressure_kwargs))

    if not args.all:
        if not args.workspace_dir:
            workspace_dir = os.getcwd()
        else:
            workspace_dir = os.path.abspath(args.workspace_dir)
        for prop in structural_props:
            prop[0](workspace_dir, prop[1])
    else:
        mypool = mp.Pool(ncores)
        if not args.workspace_dir:
            workspace_dir = os.getcwd()
        else:
            workspace_dir = os.path.abspath(args.workspace_dir)
        subdirs = get_immediate_subdirectories(workspace_dir)
        try:
            for folder in subdirs:
                if folder[1].isdigit() and "phi" in folder and "D" in folder:
                    for prop in structural_props:
                        mypool.apply_async(
                            prop[0],
                            args=(
                                os.path.abspath(folder),
                                prop[1],
                            ),
                        )
        except Exception:
            mypool.terminate()
            mypool.join()
            raise
        mypool.close()
        mypool.join()
