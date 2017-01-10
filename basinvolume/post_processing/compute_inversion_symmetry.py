import ConfigParser
import os
import ast
import numpy as np
import argparse
from pele.distance import get_distance
from compute_structural_properties import StructuralAnalysis
from basinvolume.utils import trymakedir, read_xydr, read_xyzdr


class InversionSymmetry(StructuralAnalysis):
    def __init__(self, workspace, jammed_packings_dir='jammed_packings',
                 analysis_dir='analysis', force=False, existing_only=True,
                 prefix='explore_bv_'):
        super(InversionSymmetry, self).__init__(workspace, jammed_packings_dir=jammed_packings_dir,
                                                analysis_dir=analysis_dir, force=force,
                                                existing_only=existing_only, prefix=prefix)
        self.potential = HS_WCA(eps=self.eps, sca=self.sca,
                                radii=self.hs_radii, boxvec=self.boxv, ndim=self.bdim,
                                distance_method=self.distance_method, pot_kwargs=self.pot_kwargs)


    def _distance (self, coord1, coord2):
        if self.distance_method == 'lees-edwards':
            return get_distance(coord1, coord2, self.bdim, self.distance_method,
                                box=self.boxv, shear=self.pot_kwargs['shear'])
        else:
            return get_distance(coord1, coord2, self.bdim, self.distance_method, box=self.boxv)


    def _find_nearest_neighbours(self, coords, radii):
        nparticles = radii.size
        neighbour_distancess = [[] for _ in xrange(nparticles)]
        neighbour_indicess = [[] for _ in xrange(nparticles)]

        # Loop over all unique pairs of different particles
        for i in xrange(nparticles - 1):
            for j in xrange(i + 1, nparticles):

                # Calculate distance
                dij = self._distance(coords[i * self.bdim : (i + 1) * self.bdim],
                                         coords[j * self.bdim : (j + 1) * self.bdim])
                dijnorm = np.linalg.norm(dij)

                # Check if this particle lies within neighbour range
                dmax = radii[i] + radii[j]
                if dijnorm <= dmax:
                    neighbour_distancess[i].append(dij)
                    neighbour_distancess[j].append(-dij)
                    neighbour_indicess[i].append(j)
                    neighbour_indicess[j].append(i)

        return neighbour_distancess, neighbour_indicess

    # Returns the affine force of a pair of particles
    # Indices:
    # List index is direction perpendicular to the sheared boundary (beta)
    # Matrix (numpy-array):
    # row index is the shear direction (alpha)
    # column index is the affine force component
    def _affine_force_pair(distance):
        hessian = self.potential.getHessian(distance)
        return [hessian * d for d in distance]


    def _affine_forces_total(neighbour_distancess):
        affine_forces_total = [np.zeros((self.bdim, self.bdim)) for _ in range(self.bdim)]

        # Sum up all affine forces
        for distances in neighbour_distancess:
            for distance in distances:
                affine_force = _affine_force_pair(distance)
                for i in range(self.bdim):
                    affine_forces_total[i] = affine_forces_total[i] + affine_force[i]

        return affine_forces_total


    def _sum_affine_forces(neighbour_distancess):
        affine_forces = _affine_forces_total(neighbour_distancess)
        for i in range(self.bdim):
            affine_forces[i] = affine_forces[i] ** 2
        return np.sum(sum(affine_forces))


    def _affine_force_sym_broken_pair(distance, shear_direction, shear_perpendicular):
        distance_norm = np.linalg.norm(distance)
        distance_direction = distance / distance_norm
        grad_norm = self.potential.getGradient(np.array([distance_norm, 0, 0]))[0]
        return grad_norm * distance_direction[shear_direction] \
               * distance_direction[shear_perpendicular]


    def _sum_affine_forces_sym_broken(neighbour_distancess):
        affine_forces_isb = 0
        for alpha in range(self.bdim):
            for beta in range(self.bdim):
                for distances in neighbour_distancess:
                    for distance in distances:
                        affine_forces_isb += _affine_force_sym_broken_pair(distance, alpha, beta) ** 2
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
                        configf.getfloat('INVERSION_SYMMETRY', 'inversion_symmetry'))
                    except Exception:
                        compute = True

                    if compute or self.force:
                        print "local inversion symmetry ", dname
                        trymakedir(analysis_dir_path)

                        # Read coordinates and compute distances to neighbours
                        self.coords, _, self.ss_radii, _ = self._import_packing_configuration(fname)
                        neighbour_distancess, _ = self._find_nearest_neighbours(self.coords,
                                                                                self.ss_radii)

                        # Compute local inversion symmetry
                        affine_forces_sum = _sum_affine_forces(neighbour_distancess)
                        affine_forces_isb = _sum_affine_forces_sym_broken(neighbour_distancess)
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
