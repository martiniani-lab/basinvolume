from __future__ import division
from __future__ import absolute_import
from future import standard_library

standard_library.install_aliases()
from builtins import str
from builtins import range
import configparser
import os
import ast
import traceback
import numpy as np
import argparse
import logging
from basinvolume.utils import trymakedir
from ._structural_analysis import StructuralAnalysis


class InversionSymmetry(StructuralAnalysis):
    def __init__(
        self,
        workspace,
        jammed_packings_dir="jammed_packings",
        analysis_dir="analysis",
        force=False,
        existing_only=True,
        prefix="explore_bv_",
        verbose=True,
        use_cell_lists=True,
    ):
        super(InversionSymmetry, self).__init__(
            workspace,
            jammed_packings_dir=jammed_packings_dir,
            analysis_dir=analysis_dir,
            force=force,
            existing_only=existing_only,
            prefix=prefix,
            verbose=verbose,
            use_cell_lists=use_cell_lists,
        )
        self.analysis_name = "inversion_symmetry"

    # Returns the affine force of a pair of particles
    # Tensor (numpy-array):
    # depth index is the direction perpendicular to the sheared boundary (beta)
    # column index is the affine force component
    # row index is the shear direction (alpha)
    def _affine_force_interaction(self, distance, atomi, atomj):
        # Get hessian in interaction direction
        dist_norm = np.linalg.norm(distance)
        hess_radial = self.potential.getInteractionHessian(dist_norm, atomi, atomj)

        # Transform into coordinate system
        dist_dir = distance / dist_norm
        if self.bdim == 2:
            # Transformation defined by base vectors dist_dir and its normal
            rot_matrix = np.array([[dist_dir[0], dist_dir[1]], [-dist_dir[1], dist_dir[0]]])
        elif self.bdim == 3:
            # Transformation defined by base vectors dist_dir,
            # its normal in the xy-plane and their cross product
            rot_matrix = np.array(
                [
                    [dist_dir[0], dist_dir[1], dist_dir[2]],
                    [-dist_dir[1], dist_dir[0], 0],
                    [
                        -dist_dir[0] * dist_dir[2],
                        dist_dir[1] * dist_dir[2],
                        dist_dir[0] ** 2 + dist_dir[1] ** 2,
                    ],
                ]
            )
        else:
            raise NotImplementedError
        if self.bdim == 2:
            hessian_particle_system = np.array([[hess_radial, 0], [0, 0]])
        elif self.bdim == 3:
            hessian_particle_system = np.array([[hess_radial, 0, 0], [0, 0, 0], [0, 0, 0]])
        hessian = np.dot(rot_matrix.T, np.dot(hessian_particle_system, rot_matrix))

        result = np.empty((self.bdim, self.bdim, self.bdim))
        for i in range(self.bdim):
            result[i, :, :] = distance[i] * hessian

        return result

    def _affine_force_particle(self, index, distances, neighbors):
        affine_force_particle = np.zeros((self.bdim, self.bdim, self.bdim))
        for i in range(len(neighbors)):
            affine_force = self._affine_force_interaction(distances[i], index, neighbors[i])
            affine_force_particle += affine_force
        return affine_force_particle

    def _sum_affine_forces(self, neighbor_distancess, neighbor_lists):
        affine_forces = 0
        for i in range(self.nparticles):
            affine_force = self._affine_force_particle(
                i, neighbor_distancess[i], neighbor_lists[i]
            )
            affine_forces += np.sum(affine_force**2)
        return affine_forces

    def _affine_force_interaction_sym_broken(self, distance, atomi, atomj):
        dist_norm = np.linalg.norm(distance)
        dist_dir = np.array(distance / dist_norm)
        hess_radial = self.potential.getInteractionHessian(dist_norm, atomi, atomj)
        return hess_radial * dist_norm * np.outer(dist_dir, dist_dir)

    def _sum_affine_forces_sym_broken(self, neighbor_distancess, neighbor_lists):
        affine_forces_isb = 0
        for i in range(len(neighbor_lists)):
            for j in range(len(neighbor_lists[i])):
                affine_forces_isb += np.sum(
                    self._affine_force_interaction_sym_broken(
                        neighbor_distancess[i][j], i, neighbor_lists[i][j]
                    )
                    ** 2
                )
        return affine_forces_isb

    @staticmethod
    def read(invsym_fname):
        configf = configparser.ConfigParser()
        configf.read(invsym_fname)
        invsym_dict = {}
        invsym_dict["inversion_symmetry"] = configf.getfloat(
            "INVERSION_SYMMETRY", "inversion_symmetry"
        )
        return invsym_dict

    def _calculate(self, invsym_fname, packing_name, input_fname):
        if self.verbose:
            logging.info(
                "Calculating local inversion symmetry: {}".format(self.prefix + str(packing_name))
            )

        # Read coordinates
        (
            self.coords,
            self.hs_radii,
            self.ss_radii,
            _,
        ) = self._import_packing_configuration(input_fname)

        # Create potential
        self._initialise_potential()

        # Compute distances to neighbors
        neighbor_lists, neighbor_distancess = self.potential.getNeighbors(self.coords)

        # Compute local inversion symmetry
        affine_forces_sum = self._sum_affine_forces(neighbor_distancess, neighbor_lists)
        affine_forces_isb = self._sum_affine_forces_sym_broken(neighbor_distancess, neighbor_lists)
        inv_sym = 1 - affine_forces_sum / affine_forces_isb

        # Output inversion symmetry to file
        with open(invsym_fname, "w") as f:
            f.write("#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n")
            f.write("[INVERSION_SYMMETRY]\n")
            f.write("inversion_symmetry: {:.16f}\n".format(inv_sym))


def worker_invsym(workspace, kwargs):
    try:
        invsym = InversionSymmetry(workspace, **kwargs)
        invsym.run()
    except Exception:
        logging.error("worker_invsym worker: %s" % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute local inversion symmetry " "for jammed packings."
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
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set up arguments
    kwargs = dict(
        force=args.force,
        existing_only=args.nonex,
        jammed_packings_dir=args.input_dir,
        prefix=args.prefix,
        use_cell_lists=not args.nocell,
    )

    # Create workspace directory name
    if not args.workspace_dir:
        workspace_dir = os.getcwd()
    else:
        workspace_dir = os.path.abspath(args.workspace_dir)

    # Start local inversion symmetry computation
    invsym = InversionSymmetry(workspace_dir, **kwargs)
    invsym.run()
