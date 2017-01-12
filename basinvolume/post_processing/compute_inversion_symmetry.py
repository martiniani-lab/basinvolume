import ConfigParser
import os
import ast
import numpy as np
import argparse
from pele.potentials import HS_WCA
from compute_structural_properties import StructuralAnalysis
from basinvolume.utils import trymakedir, read_xydr, read_xyzdr, find_neighbours


class InversionSymmetry(StructuralAnalysis):
    def __init__(self, workspace, jammed_packings_dir='jammed_packings',
                 analysis_dir='analysis', force=False, existing_only=True,
                 prefix='explore_bv_'):
        super(InversionSymmetry, self).__init__(workspace, jammed_packings_dir=jammed_packings_dir,
                                                analysis_dir=analysis_dir, force=force,
                                                existing_only=existing_only, prefix=prefix)


    # Returns the affine force of a pair of particles
    # Indices:
    # List index is direction perpendicular to the sheared boundary (beta)
    # Matrix (numpy-array):
    # row index is the shear direction (alpha)
    # column index is the affine force component
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
            rot_matrix = np.array([[dist_dir[0], dist_dir[1], dist_dir[2]], [-dist_dir[1], dist_dir[0], 0], [-dist_dir[0]*dist_dir[2], dist_dir[1] * dist_dir[2], dist_dir[0]**2 + dist_dir[1]**2]])
        else:
            raise NotImplementedError
        hessian_particle_system = np.array([[hess_radial, 0], [0, 0]])
        hessian = np.dot(rot_matrix.T, np.dot(hessian_particle_system, rot_matrix))

        return [hessian * d for d in distance]


    def _affine_force_particle(self, index, distances, neighbours):
        affine_force_particle = [np.zeros((self.bdim, self.bdim)) for _ in range(self.bdim)]
        for i in range(len(neighbours)):
            affine_force = self._affine_force_interaction(distances[i], index, neighbours[i])
            for j in range(self.bdim):
                affine_force_particle[j] += affine_force[j]
        return affine_force_particle


    def _sum_affine_forces(self, neighbour_distancess, neighbour_lists):
        affine_forces = [np.zeros((self.bdim, self.bdim)) for _ in range(self.bdim)]
        for i in range(self.nparticles):
            affine_force = self._affine_force_particle(i, neighbour_distancess[i], neighbour_lists[i])
            for i in range(self.bdim):
                affine_forces[i] += affine_force[i] ** 2
        return np.sum(sum(affine_forces))


    def _affine_force_interaction_sym_broken(self, distance, atomi, atomj, shear_direction, shear_perpendicular):
        dist_norm = np.linalg.norm(distance)
        dist_dir = distance / dist_norm
        hess_radial = self.potential.getInteractionHessian(dist_norm, atomi, atomj)
        return hess_radial * dist_norm * dist_dir[shear_direction] \
               * dist_dir[shear_perpendicular]


    def _sum_affine_forces_sym_broken(self, neighbour_distancess, neighbour_lists):
        affine_forces_isb = 0
        for alpha in range(self.bdim):
            for beta in range(self.bdim):
                for i in range(len(neighbour_lists)):
                    for j in range(len(neighbour_lists[i])):
                        affine_forces_isb += \
                            self._affine_force_interaction_sym_broken(neighbour_distancess[i][j],
                                                                      i, neighbour_lists[i][j],
                                                                      alpha, beta) ** 2
        return affine_forces_isb


    def run(self):
        for fname in os.listdir(self.jammed_packings_dir):
            if 'xyzdr' in fname or 'xydr' in fname:
                dname = self._get_dname(fname)

                # Get configuration
                configpath = os.path.join(self.jammed_packings_dir, dname + '.config')
                self._import_packing_config_file(configpath)

                # Check if the work directory exists
                base_directory_path = os.path.join(self.workspace, self.prefix + str(dname))
                if os.path.isdir(base_directory_path) or not self.existing_only:
                    trymakedir(base_directory_path)

                    # Check if this packing has already been analysed
                    analysis_dir_path = os.path.join(base_directory_path, self.analysis_dir)
                    invsym_fname = os.path.join(analysis_dir_path,'inversion_symmetry')
                    compute = False
                    try:
                        configf = ConfigParser.ConfigParser()
                        configf.read(invsym_fname)
                        configf.getfloat('INVERSION_SYMMETRY', 'inversion_symmetry')
                    except Exception:
                        compute = True

                    if compute or self.force:
                        print("Calculating local inversion symmetry: {}".format(self.prefix + str(dname)))
                        trymakedir(analysis_dir_path)

                        # Read coordinates and compute distances to neighbours
                        self.coords, self.hs_radii, self.ss_radii, _ = self._import_packing_configuration(fname)
                        neighbour_distancess, neighbour_lists = \
                            find_neighbours(self.coords, self.ss_radii, self.bdim, self.boxv,
                                            self.distance_method, self.pot_kwargs)

                        # Create potential
                        self.potential = HS_WCA(eps=self.eps, sca=self.sca,
                                                radii=self.hs_radii, boxvec=self.boxv, ndim=self.bdim,
                                                distance_method=self.distance_method, pot_kwargs=self.pot_kwargs)

                        # Compute local inversion symmetry
                        affine_forces_sum = self._sum_affine_forces(neighbour_distancess,
                                                                    neighbour_lists)
                        affine_forces_isb = self._sum_affine_forces_sym_broken(neighbour_distancess,
                                                                               neighbour_lists)
                        inv_sym = 1 - affine_forces_sum / affine_forces_isb

                        # Output inversion symmetry to file
                        with open(invsym_fname, 'w') as f:
                            f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
                            f.write('[INVERSION_SYMMETRY]\n')
                            f.write('inversion_symmetry: {:.16f}\n'.format(inv_sym))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute local inversion symmetry "
                                     "for jammed packings.")
    parser.add_argument("-d", "--workspace_dir", type=str, help="Top-level dir containing "
                        "the packings, e.g. 'n32_phi88_2D'.")
    parser.add_argument("--force", action='store_true', help="Force to run on all packings.",
                        default=False)
    parser.add_argument("--nonex", action='store_false', help="Run also for packings "
                        "for which there are no work folders ('explore_bv_[...]', "
                        "created e.g. by parallel tempering).", default=True)
    parser.add_argument("--prefix", type=str, help="Prefix for the work directory. "
                        "Default: 'explore_bv_'", default='explore_bv_')
    parser.add_argument("--input_dir", type=str, help="Directory containing the "
                        "jammed packings. Default: 'jammed_packings'", default='jammed_packings')
    args = parser.parse_args()

    # Set up arguments
    kwargs = dict(force=args.force, existing_only=args.nonex,
                  jammed_packings_dir=args.input_dir, prefix=args.prefix)

    # Create workspace directory name
    if not args.workspace_dir:
        workspace_dir = os.getcwd()
    else:
        workspace_dir = os.path.abspath(args.workspace_dir)

    # Start local inversion symmetry computation
    invsym = InversionSymmetry(workspace_dir, **kwargs)
    invsym.run()
