from __future__ import division
import copy
import numpy as np
import os
import traceback
import ConfigParser
import logging
from scipy.special import sph_harm
from basinvolume.utils import trymakedir
from basinvolume.post_processing.simple_solid_angle_neighbors import SimpleSolidAngleNeighbors
from _structural_analysis import StructuralAnalysis
from pele.potentials import HS_WCA


class BondOrientationalOrder(StructuralAnalysis):

    def __init__(self, workspace, jammed_packings_dir='jammed_packings',
                 analysis_dir='analysis', force=False, existing_only=True,
                 solid_angle_weighted=False, prefix='explore_bv_', verbose=True,
                 use_cell_lists=True):
        super(BondOrientationalOrder,self).__init__(workspace,
                                                    jammed_packings_dir=jammed_packings_dir,
                                                    analysis_dir=analysis_dir, force=force,
                                                    existing_only=existing_only, prefix=prefix,
                                                    verbose=verbose, use_cell_lists=use_cell_lists)
        self.solid_angle_weighted = solid_angle_weighted
        if self.verbose:
            logging.debug("self.solid_angle_weighted: {}".format(self.solid_angle_weighted))

    @staticmethod
    def read(boo_fname):
        configf = ConfigParser.ConfigParser()
        configf.read(boo_fname)
        boo_dict = {}
        boo_dict['Z'] = configf.getfloat('Z', 'Z')
        boo_items = configf.items('BOO')
        for i, boo in enumerate(boo_items):
            if i > 0:
                raise IOError("Too many items in bond-orientational order file "
                              "(expected: 1): {}".format(boo_items))
            boo_dict['BOO'] = (boo[0].upper(), float(boo[1]))
        return boo_dict

    def run(self, deg=6, pinit=True):
        """compute boo for packings. we exclude rattlers from the computation of the global structure factors
        exisisting_only: bool
            run on already existing packings only
        pinit : bool
            initialise printing
        """
        for fname in os.listdir(self.jammed_packings_dir):
            if 'xyzd' in fname or 'xyd' in fname:
                compute = False
                packing_name = os.path.splitext(fname)[0]
                base_directory_path = os.path.join(self.workspace, self.prefix + str(packing_name))
                configpath = os.path.join(self.jammed_packings_dir, packing_name + '.config')
                self._import_packing_config_file(configpath)
                if os.path.isdir(base_directory_path) or not self.existing_only:
                    trymakedir(base_directory_path)
                    analysis_dir_path = os.path.join(base_directory_path, self.analysis_dir)
                    boo_fname = os.path.join(analysis_dir_path, 'boo_deg{}'.format(deg))
                    global_boo_fname = os.path.join(analysis_dir_path, 'glob_boo')
                    try:
                        self.read(str(global_boo_fname))
                        if not os.path.isfile(boo_fname):
                            raise Exception
                    except Exception:
                        compute = True
                    if compute or self.force:
                        if self.verbose:
                            logging.info("Calculating bond orientational order: {}"
                                         .format(self.prefix + str(packing_name)))
                        trymakedir(analysis_dir_path)
                        coords, hs_radii, ss_radii, stable_atoms = \
                            self._import_packing_configuration(fname)
                        if self.use_cell_lists:
                            self.potential = HS_WCA(use_cell_lists=True, eps=self.eps, sca=self.sca,
                                                    radii=hs_radii, boxvec=self.boxv,
                                                    reference_coords=coords, ndim=self.bdim,
                                                    ncellx_scale=1.0, distance_method=self.distance_method,
                                                    pot_kwargs=self.pot_kwargs)
                        else:
                            self.potential = HS_WCA(eps=self.eps, sca=self.sca, radii=hs_radii,
                                                    boxvec=self.boxv, ndim=self.bdim,
                                                    distance_method=self.distance_method,
                                                    pot_kwargs=self.pot_kwargs)
                        boo_list, z_list = self.bond_orientation_order_all(coords,
                                                                           ss_radii,
                                                                           stable_atoms,
                                                                           ndim=self.bdim,
                                                                           deg=deg)
                        with open(boo_fname, 'w') as f:
                            f.write('#Q{} \t Z\n'.format(deg))
                            for q, z in zip(boo_list, z_list):
                                f.write('{:.16f} \t {}\n'.format(q, z))
                        opt = 'w' if pinit else 'a'
                        with open(global_boo_fname, opt) as f:
                            if pinit:
                                f.write('[Z] \n')
                                f.write('Z: {:.16f} \n'.format(np.sum(z_list) / (z_list > 1e-12).sum()))
                                f.write('[BOO] \n')
                            f.write('Q{}: {:.16f} \n'.format(deg, np.sum(boo_list) / (boo_list > 1e-12).sum() ))

    def run_all(self, deg_list=[4,6,8,10,12]):
        if any('xyzd' in fname for fname in os.listdir(self.jammed_packings_dir)):
            for i, deg in enumerate(deg_list):
                self.run(deg, pinit=i<1)
                assert self.bdim == 3
        elif any('xyd' in fname for fname in os.listdir(self.jammed_packings_dir)):
            self.run(6, pinit=True)
            assert self.bdim == 2

    def _cartesian_to_polar3d(self, vector):
        vector = np.array(vector)
        r = np.linalg.norm(vector)
        theta = np.arctan2(vector[1], vector[0]) + np.pi    #[0, 2*pi]
        phi = np.arccos(vector[2]/r)                        #[0, pi]
        return r, theta, phi

    def _cartesian_to_polar2d(self, vector):
        vector = np.array(vector)
        r = np.linalg.norm(vector)
        theta = np.arctan2(vector[1], vector[0]) + np.pi
        return r, theta

    def _qsum(self, nnatoms_vec, order, ndim=3, deg=6, weights=None):
        """
        this method compute the qsum, necessary for computing
        nn_atoms: array
            array of indexes of the nearest neighbours
        ndim: int
            dimensionality of space (box)
        order: int
            order of the spherical harmonic (m)
        deg: int
            degree of the spherical harmonic (l)
        weights : array (optional)
            weight of each neighbor in the sum. In case the solid-angle
            weighted method is used, this will come from the
            nn-search-algorithm. For fixed-distance cutoff, the weights
            will all be the same.
        """
        n = len(nnatoms_vec)
        if weights is None:
            weights = np.ones(n)
        qsum = np.complex(0.,0.)
        if ndim == 3:
            for i, vector in enumerate(nnatoms_vec):
                r, theta, phi = self._cartesian_to_polar3d(vector)
                Y = sph_harm(order, deg, theta, phi) #theta, phi
                qsum += Y * weights[i]
        elif ndim == 2:
            assert deg == 6, "boo only meaningful for exhatic phase in 2d"
            for i, vector in enumerate(nnatoms_vec):
                r, theta = self._cartesian_to_polar2d(vector)
                Y = np.exp(np.complex(0.,deg*theta))
                qsum += Y * weights[i]
        else:
            raise Exception('ndim not implemented')
        return qsum / np.sum(weights)

    def _bond_orientational_order3d(self, nnatoms_vec, deg=6, weights=None):
        q = 0.
        for m in xrange(-deg,deg+1):
            c = self._qsum(nnatoms_vec, m, ndim=3, deg=deg, weights=weights)
            q += np.absolute(c)**2
        return np.sqrt(q * 4 * np.pi / (2*deg+1))

    def _bond_orientational_order2d(self, nnatoms_vec, deg=6, weights=None):
        c = self._qsum(nnatoms_vec, 0, ndim=2, deg=deg, weights=weights)
        return np.absolute(c)

    def _bond_orientational_order(self, nnatoms_vec, ndim=3, deg=6, weights=None):
        if ndim == 3:
            return self._bond_orientational_order3d(nnatoms_vec, deg=deg, weights=weights)
        elif ndim == 2:
            return self._bond_orientational_order2d(nnatoms_vec, deg=deg, weights=weights)
        else:
            raise Exception('ndim not implemented')

    def find_nearest_neighbors_solid_angle(self, coords, ss_radii):
        nparticles = ss_radii.size
        nnatoms_list = [[] for _ in xrange(nparticles)]
        weights_all = copy.deepcopy(nnatoms_list)
        for i in xrange(nparticles):
            """
            Note that if i has neighbor j it is not obvious that j has
            neighbor i, in contrast to fixed distance cutoff.
            Note: SimpleSolidAngleNeighbors does not exclude rattlers from the particles shells
            """
            sann = SimpleSolidAngleNeighbors(i, coords, nparticles, self.boxv)
            for j in xrange(sann.nr_neighbors):
                nnatoms_list[i].append(sann.nn_vector[j])
                weights_all[i].append(sann.weight[j])
        return nnatoms_list, weights_all

    def bond_orientation_order_single(self, coords, ss_radii, stable_atoms, atom_index, ndim=3, deg=6):
        _, nnatoms_list = self.potential.getNeighbours(coords, include_atoms=stable_atoms)
        nnatoms_vec = nnatoms_list[atom_index]
        return self._bond_orientational_order(nnatoms_vec, ndim=ndim, deg=deg)

    def bond_orientation_order_all(self, coords, ss_radii, stable_atoms, ndim=3, deg=6):
        """
        boo_list : array
            list of bond orientational order
        z_list : array
            list of coordination number for each particle
        """
        nnatoms_list = None
        weights_all = None
        _, contacts_list = self.potential.getNeighbours(coords, include_atoms=stable_atoms)
        if not self.solid_angle_weighted:
            nnatoms_list = contacts_list
        else:
            nnatoms_list, weights_all = self.find_nearest_neighbors_solid_angle(coords, ss_radii)

        boo_list = []
        z_list = []
        for i in xrange(ss_radii.size):
            contacts_vec = contacts_list[i]
            nnatoms_vec = nnatoms_list[i]
            weights = None
            if weights_all is not None:
                weights = weights_all[i]
            if len(nnatoms_vec) > 0 and stable_atoms[i]:
                boo = self._bond_orientational_order(nnatoms_vec, ndim=ndim, deg=deg, weights=weights)
                boo_list.append(boo)
            else:
                #rattlers
                boo_list.append(0)
            if len(contacts_vec) > 0 and stable_atoms[i]:
                z_list.append(len(contacts_vec))
            else:
                #rattlers
                z_list.append(0)
        return np.array(boo_list), np.array(z_list)


def worker_boo(workspace, kwargs):
    try:
        boo = BondOrientationalOrder(workspace, **kwargs)
        boo.run_all()
    except:
        logging.error('worker_boo worker: %s' % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute bond-orientational order "
                                     "for jammed packings.")
    parser.add_argument("-d", "--workspace-dir", type=str, help="Top-level dir containing "
                        "the packings, e.g. 'n32_phi88_2D'.")
    parser.add_argument("--force", action='store_true', help="Force to run on all packings.",
                        default=False)
    parser.add_argument("--solid", action="store_true", help="Use solid angle method "
                        "to find and weight neighbors.", default=False)
    parser.add_argument("--nonex", action='store_false', help="Run also for packings "
                        "for which there are no work folders ('explore_bv_[...]', "
                        "created e.g. by parallel tempering).", default=True)
    parser.add_argument("--prefix", type=str, help="Prefix for the work directory. "
                        "Default: 'explore_bv_'", default='explore_bv_')
    parser.add_argument("--input-dir", type=str, help="Directory containing the "
                        "jammed packings. Default: 'jammed_packings'", default='jammed_packings')
    parser.add_argument("--nocell", action='store_true', help="Don't use cell lists. "
                        "Default: False", default=False)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.INFO)

    kwargs = dict(force=args.force, existing_only=args.nonex,
                  jammed_packings_dir=args.input_dir, prefix=args.prefix,
                  use_cell_lists=not args.nocell)
    if args.solid:
        kwargs.update(solid_angle_weighted=args.solid)

    if not args.workspace_dir:
        workspace_dir = os.getcwd()
    else:
        workspace_dir = os.path.abspath(args.workspace_dir)

    boo = BondOrientationalOrder(workspace_dir, **kwargs)
    boo.run_all()
