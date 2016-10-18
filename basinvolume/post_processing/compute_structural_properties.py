from __future__ import division
import copy
import numpy as np
from scipy.special import sph_harm
from basinvolume.utils import *
from pele.utils._pressure_tensor import pressure_tensor
import abc
from pele.potentials import HS_WCA, InversePowerStillingerCut
import argparse
import multiprocessing as mp
from simple_solid_angle_neighbors import SimpleSolidAngleNeighbors
from pele.optimize._quench import modifiedfire_cpp

class StructuralAnalysis(object):
    __metaclass__ = abc.ABCMeta
    #@abc.abstractmethod
    
    def __init__(self, workspace, packings_dir='packings', jammed_packings_dir='jammed_packings', analysis_dir='analysis',
                 force=False, existing_only=True):
        if not os.path.isabs(workspace):
            workspace = os.path.abspath(workspace)
        self.workspace = workspace
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(self.workspace, packings_dir)
        if not os.path.isabs(jammed_packings_dir):
            jammed_packings_dir = os.path.join(self.workspace, jammed_packings_dir)
        self.packings_dir = packings_dir
        self.jammed_packings_dir = jammed_packings_dir
        self.analysis_dir = analysis_dir
        self.iteration = 0
        self.eps = 1.
        self.frozen = False
        self.force = force
        self.existing_only = existing_only
    
    def _import_packing_config_file(self, configpath):
        configf = ConfigParser.ConfigParser()
        configf.read(str(configpath))
        self.nparticles = configf.getint('JAMMED_PACKING','nparticles')
        self.bdim = configf.getint('JAMMED_PACKING','boxdim')
        assert self.bdim==2 or self.bdim==3, "bdim={} not implemented".format(self.bdim)
        self.ndof = self.nparticles * self.bdim
        boxv = configf.get('JAMMED_PACKING','boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        if self.frozen:
            self.vcavity = configf.getfloat('JAMMED_PACKING', 'vcavity')
        else:
            self.vcavity = np.prod(self.boxv)
        self.packing_frac = configf.getfloat('JAMMED_PACKING','packing_fraction')
        self.sca = configf.getfloat('JAMMED_PACKING','sca')
        
    def _import_packing_configuration(self, fname):
        path = os.path.join(self.jammed_packings_dir, fname)
        if self.bdim == 2:
            coords, hs_diameters, rattlers = read_xydr(path)
        elif self.bdim == 3:
            coords, hs_diameters, rattlers = read_xyzdr(path)
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        hs_radii = hs_diameters/2
        ss_radii = hs_radii * (1+self.sca)
        return coords, hs_radii, ss_radii, rattlers
    
    def _get_dname(self, dname):
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        return dname
    
class BondOrientationalOrder(StructuralAnalysis):
    def __init__(self, workspace, packings_dir='packings',
        jammed_packings_dir='jammed_packings', analysis_dir='analysis', 
        force=False, existing_only=True, solid_angle_weighted=False):
        super(BondOrientationalOrder,self).__init__(workspace, packings_dir=packings_dir, jammed_packings_dir=jammed_packings_dir, 
                                                    analysis_dir=analysis_dir, force=force, existing_only=existing_only)
        self.solid_angle_weighted = solid_angle_weighted
        print("self.solid_angle_weighted", self.solid_angle_weighted)
        
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
                dname = self._get_dname(fname)
                base_directory_path = os.path.join(self.workspace,'explore_bv_'+str(dname))
                configpath = os.path.join(self.jammed_packings_dir, dname + '.config')
                self._import_packing_config_file(configpath)
                if os.path.isdir(base_directory_path) or not self.existing_only:
                    trymakedir(base_directory_path)
                    analysis_dir_path = os.path.join(base_directory_path, self.analysis_dir)
                    boo_fname = os.path.join(analysis_dir_path,'boo_deg{}'.format(deg))
                    global_boo_fname = os.path.join(analysis_dir_path,'glob_boo')
                    try:
                        configf = ConfigParser.ConfigParser()
                        configf.read(str(global_boo_fname))
                        test_z = configf.getfloat('Z','Z')
                        test_boo = configf.getfloat('BOO','Q{}'.format(deg))
                        if not os.path.isfile(boo_fname):
                            raise Exception
                    except Exception:
                        compute = True
                    if compute or self.force:
                        print "boo ", dname
                        trymakedir(analysis_dir_path)
                        coords, hs_radii, ss_radii, rattlers = self._import_packing_configuration(fname)
                        boo_list, z_list = self.bond_orientation_order_all(coords, ss_radii, rattlers, ndim=self.bdim, deg=deg)
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
            for i,deg in enumerate(deg_list):
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
    
    def find_nearest_neighbors(self, coords, hs_radii, rattlers=None):
        """
        this function ignores rattlers
        """
        nparticles = hs_radii.size
        nnatoms_list = [[] for _ in xrange(nparticles)]
        if rattlers is None:
            atom_labels = range(nparticles)
        else:
            atom_labels = np.array(range(nparticles))[np.array(rattlers[::self.bdim], dtype='int') == 1]
        for i in atom_labels:
            for j in atom_labels:
                if j > i:
                    dij = np.zeros(self.bdim)
                    for k in xrange(self.bdim):
                        #use distances to nearest image convention
                        dij[k] = ((coords[j*self.bdim+k] - coords[i*self.bdim+k]) -
                                           cround((coords[j*self.bdim+k] - coords[i*self.bdim+k]) / self.boxv[k]) * self.boxv[k])
                    dijnorm = np.linalg.norm(dij)
                    dmin = hs_radii[i] + hs_radii[j]
                    if dijnorm <= dmin:
                        nnatoms_list[i].append(dij)
                        nnatoms_list[j].append(-dij)
        return nnatoms_list
    
    def find_nearest_neighbors_solid_angle(self, coords, hs_radii):
        nparticles = hs_radii.size
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
    
    def bond_orientation_order_single(self, coords, hs_radii, rattlers, atom_index, ndim=3, deg=6):
        nnatoms_list = self.find_nearest_neighbors(coords, hs_radii, rattlers=rattlers)
        nnatoms_vec = nnatoms_list[atom_index]
        return self._bond_orientational_order(nnatoms_vec, ndim=ndim, deg=deg)
    
    def bond_orientation_order_all(self, coords, hs_radii, rattlers, ndim=3, deg=6):
        """
        boo_list : array
            list of bond orientational order
        z_list : array
            list of coordination number for each particle
        """
        nnatoms_list = None
        weights_all = None
        contacts_list = self.find_nearest_neighbors(coords, hs_radii, rattlers=rattlers)
        if not self.solid_angle_weighted:
            nnatoms_list = contacts_list
        else:
            nnatoms_list, weights_all = self.find_nearest_neighbors_solid_angle(coords, hs_radii)

        boo_list = []
        z_list = []
        for i in xrange(hs_radii.size):
            contacts_vec = contacts_list[i]
            nnatoms_vec = nnatoms_list[i]
            weights = None
            if weights_all is not None:
                weights = weights_all[i]
            if len(nnatoms_vec) > 0 and int(rattlers[i*self.bdim]) == 1:
                boo = self._bond_orientational_order(nnatoms_vec, ndim=ndim, deg=deg, weights=weights)
                boo_list.append(boo)
            else:
                #rattlers
                boo_list.append(0)
            if len(contacts_vec) > 0 and int(rattlers[i*self.bdim]) == 1:
                z_list.append(len(contacts_vec))
            else:
                #rattlers
                z_list.append(0)
        return np.array(boo_list), np.array(z_list)

class PressureTensor(StructuralAnalysis):
    def __init__(self, workspace, packings_dir='packings', jammed_packings_dir='jammed_packings', analysis_dir='analysis', 
                 force=False, existing_only=True, opt_pot_str='hs_wca', **extra_pot_kwargs):
        super(PressureTensor,self).__init__(workspace, packings_dir=packings_dir, jammed_packings_dir=jammed_packings_dir, 
                                            analysis_dir=analysis_dir, force=force, existing_only=existing_only)
        self.opt_pot_str = opt_pot_str
        self.extra_pot_kwargs = extra_pot_kwargs

    def run(self):
        """compute boo for packings
        exisisting_only: bool
            run on already existing packings only
        pinit : bool
            initialise printing
        """
        for fname in os.listdir(self.jammed_packings_dir):
            if 'xyzd' in fname or 'xyd' in fname:
                compute = False
                dname = self._get_dname(fname)
                base_directory_path = os.path.join(self.workspace,'explore_bv_'+str(dname))
                configpath = os.path.join(self.jammed_packings_dir, dname + '.config')
                self._import_packing_config_file(configpath)
                if os.path.isdir(base_directory_path) or not self.existing_only:
                    trymakedir(base_directory_path)
                    analysis_dir_path = os.path.join(base_directory_path, self.analysis_dir) 
                    pressure_fname = os.path.join(analysis_dir_path,'pressure_data')
                    try:
                        configf = ConfigParser.ConfigParser()
                        configf.read(pressure_fname)
                        test_p = configf.getfloat('PRESSURE','P')
                        test_ptensor = configf.get('PRESSURE','Ptensor')
                        test_e = configf.get('ENERGY', 'E')
                    except Exception:
                        compute = True
                    if compute or self.force:
                        print "pressure ", dname
                        trymakedir(analysis_dir_path)
                        self.coords, self.hs_radii, self.ss_radii, self.rattlers = self._import_packing_configuration(fname)
                        potential = self.get_potential()
                        if self.packing_frac < 0.835:
                            fire_maxstep = np.amin(self.hs_radii) * self.sca
                            res = modifiedfire_cpp(self.coords, potential, maxstep=fire_maxstep,
                                                   nsteps=1e6, tol=1e-11, iprint=-1)
                            self.coords = res.coords
                        p, ptensor = pressure_tensor(potential, self.coords, self.vcavity, self.bdim)
                        energy = potential.getEnergy(self.coords)
                        with open(pressure_fname, 'w') as f:
                            f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND \n')
                            f.write('[PRESSURE]\n')
                            f.write('P: {:.16f}\n'.format(p))
                            f.write('Ptensor: ')
                            for val in ptensor:
                                f.write('{:.16f} '.format(val))
                            f.write('\n')
                            f.write('[ENERGY]\n')
                            f.write('E: {:.16f}\n'.format(energy))

    def get_potential(self):
        # here put a flag and pick potential
        if self.opt_pot_str.lower() == 'hs_wca':
            pot = HS_WCA(use_periodic=True, eps=self.eps, sca=self.sca,
                         radii=self.hs_radii, boxvec=self.boxv, ndim=self.bdim)
        elif self.opt_pot_str.lower() == 'inverse_power_stillinger':
            pow = self.extra_pot_kwargs['pow']
            rcut = self.extra_pot_kwargs["rcut"]
            pot_optimizer = InversePowerStillingerCut(pow,
                self.stillinger_a_radii, ndim=self.bdim,
                boxvec=self.boxv, rcut=rcut, use_cell_lists=True)
        else:
            raise NotImplementedError
        return pot

def worker_boo(workspace, kwargs):
    try:
        boo = BondOrientationalOrder(workspace, **kwargs)
        boo.run_all()
    except:
        print('worker_boo worker: %s' % (traceback.format_exc()))

def worker_pts(workspace, kwargs):
    try:
        pts = PressureTensor(workspace, **kwargs)
        pts.run()
    except:
        print('worker_pts worker: %s' % (traceback.format_exc()))

def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

if __name__ == "__main__":
    
#    #boo.run(deg=12)
#    boo.run_all()
#    pts = PressureTensor()
#    pts.run()

    parser = argparse.ArgumentParser(description="Compute volumes from PT data, use either MBAR or TINT methods")
    parser.add_argument("-d", "--workspace_dir", type=str, help="top-level dir containing the packings, e.g. n32_phi88_2D")
    parser.add_argument("--all", action='store_true', help="run for all packing subdirectories", default=False)
    parser.add_argument("-j","--ncores", type=int, help="threads for prallel execution", default=7)
    parser.add_argument("--force", action='store_true', help="force to run on all packings", default=False)
    parser.add_argument("--solid", action="store_true", help="use solid angle method to find and weight neighbors", default=False)
    parser.add_argument("--nonex", action='store_false', help="run also the non packings for which there aren't working folders", default=True)
    # potential arguments
    parser.add_argument("--opt-pot", type=str, help="optmizer's potential, 1) (default) hs_wca "
                                                    "2) inverse_power_stillinger", default='hs_wca')
    args = parser.parse_args()

    # potential type
    opt_pot_str = args.opt_pot
    extra_pot_kwargs = dict()
    if opt_pot_str == 'hs_wca':
        pass
    elif opt_pot_str == 'inverse_power_stillinger':
        extra_pot_kwargs.update(dict(pow=3, rcut=1.5))
        print 'setting inverse_power_stillinger parameters: ', extra_pot_kwargs
    else:
        raise NotImplementedError

    ncores = args.ncores
    kwargs = dict(force=args.force, existing_only=args.nonex)
    if args.solid:
        kwargs.update(solid_angle_weighted=args.solid)

    pts_kwargs = dict(opt_pot_str=opt_pot_str, **extra_pot_kwargs)
    pts_kwargs.update(kwargs)
    
    if not args.all:
        if not args.workspace_dir:
            workspace_dir = os.getcwd()
        else:
            workspace_dir = os.path.abspath(args.workspace_dir)
        worker_boo(workspace_dir, kwargs)
        worker_pts(workspace_dir, pts_kwargs)
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
                    mypool.apply_async(worker_boo, args=(os.path.abspath(folder),kwargs,))
                    mypool.apply_async(worker_pts, args=(os.path.abspath(folder),pts_kwargs,))
        except:
            mypool.terminate()
            mypool.join()
            raise
                    
        mypool.close()
        mypool.join()
