from __future__ import division
import numpy as np
from scipy.special import sph_harm
import os
from basinvolume.utils import *
from pele.utils._pressure_tensor import pressure_tensor
import abc
from pele.potentials import HS_WCA

class StructuralAnalysis(object):
    __metaclass__ = abc.ABCMeta
    #@abc.abstractmethod
    
    def __init__(self, packings_dir='packings', jammed_packings_dir='jammed_packings', analysis_dir='analysis'):
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        if not os.path.isabs(jammed_packings_dir):
            packings_dir = os.path.join(os.getcwd(), jammed_packings_dir)
        self.packings_dir = packings_dir
        self.jammed_packings_dir = jammed_packings_dir
        self.analysis_dir = analysis_dir
        self.iteration = 0
        self.eps = 1.
        self.frozen = False
    
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
        path = os.path.join(self.packings_dir, fname)
        if self.bdim == 2:
            coords, hs_diameters, rattlers = read_xydr(path)
        elif self.bdim == 3:
            coords, hs_diameters, rattlers = read_xyzdr(path)
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        hs_radii = hs_diameters/2
        ss_radii = hs_radii * (1+self.sca)
        return coords, hs_radii, ss_radii
    
    def _get_dname(self, dname):
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        return dname

class BondOrientationalOrder(StructuralAnalysis):
    def __init__(self, packings_dir='packings', jammed_packings_dir='jammed_packings', analysis_dir='analysis'):
        super(BondOrientationalOrder,self).__init__(packings_dir=packings_dir, jammed_packings_dir=jammed_packings_dir, 
                                                    analysis_dir=analysis_dir)
    
    def run(self, deg=6, pinit=True, existing_only=True):
        """compute boo for packings. we exclude rattlers from the computation of the global structure factors
        exisisting_only: bool
            run on already existing packings only
        pinit : bool
            initialise printing
        """
        for fname in os.listdir(self.packings_dir):
            if 'xyzd' in fname or 'xyd' in fname:
                dname = self._get_dname(fname)
                base_directory_path = os.path.join(os.getcwd(),'explore_bv_'+str(dname))
                configpath = os.path.join(self.jammed_packings_dir, dname + '.config')
                self._import_packing_config_file(configpath)
                if os.path.isdir(base_directory_path) or not existing_only:
                    print dname
                    trymakedir(base_directory_path)
                    analysis_dir_path = os.path.join(base_directory_path, self.analysis_dir)
                    trymakedir(analysis_dir_path)
                    coords, hs_radii, ss_radii = self._import_packing_configuration(fname)
                    boo_list, z_list = self.bond_orientation_order_all(coords, ss_radii, ndim=self.bdim, deg=deg)
                    boo_fname = os.path.join(analysis_dir_path,'boo_deg{}'.format(deg))
                    with open(boo_fname, 'w') as f:
                        f.write('#Q{} \t Z\n'.format(deg))
                        for q, z in zip(boo_list, z_list):
                            f.write('{:.16f} \t {}\n'.format(q, z))
                    global_boo_fname = os.path.join(analysis_dir_path,'glob_boo')
                    opt = 'w' if pinit else 'a'
                    with open(global_boo_fname, opt) as f:
                        if pinit:
                            f.write('[Z] \n')
                            f.write('Z: {:.16f} \n'.format(np.sum(z_list) / (z_list > 1e-12).sum()))
                            f.write('[BOO] \n')
                        f.write('Q{}: {:.16f} \n'.format(deg, np.sum(boo_list) / (boo_list > 1e-12).sum() ))
    
    def run_all(self, deg_list=[4,6,8,10,12], existing_only=True):
        for i,deg in enumerate(deg_list):
            self.run(deg, pinit=i<1, existing_only=existing_only)
    
    def _cartesian_to_polar(self, vector):
        vector = np.array(vector)
        r = np.linalg.norm(vector)
        theta = np.arctan2(vector[1], vector[0])
        phi = np.arccos(vector[2]/r)
        return r, theta, phi

    def _qsum(self, nnatoms_vec, order, ndim=3, deg=6):
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
        """
        assert ndim == 3, "2 dimensional case not implemented yet"
        n = len(nnatoms_vec)
        qsum = np.complex(0.,0.)
        for vector in nnatoms_vec:
            r, theta, phi = self._cartesian_to_polar(vector)
            Y = sph_harm(order, deg, theta, phi)
            qsum += Y
        return qsum / n
    
    def find_nearest_neighbors(self, coords, hs_radii):
        nparticles = hs_radii.size
        nnatoms_list = [[] for _ in xrange(nparticles)]
        for i in xrange(nparticles):
            for j in xrange(i, nparticles):
                if i != j:
                    dij = np.zeros(3)
                    for k in xrange(self.bdim):
                        #use distances to nearest image convention
                        dij[k] = ((coords[j*self.bdim+k] - coords[i*self.bdim+k]) -
                                           cround((coords[j*self.bdim+k] - coords[i*self.bdim+k]) / self.boxv[k]) * self.boxv[k])
                    dijnorm = np.linalg.norm(dij)
                    dmin = hs_radii[i]+ hs_radii[j]
                    if dijnorm <= dmin:
                        nnatoms_list[i].append(dij)
                        nnatoms_list[j].append(-dij)
        return nnatoms_list
    
    def _bond_orientational_order(self, nnatoms_vec, ndim=3, deg=6):
        q = 0.
        for m in xrange(-deg,deg):
            c = self._qsum(nnatoms_vec, m, ndim=ndim, deg=deg)
            q += np.absolute(c)**2
        return np.sqrt(q * 4 * np.pi / (2*deg+1))
    
    def bond_orientation_order_single(self, coords, hs_radii, atom_index, ndim=3, deg=6):
        nnatoms_list = self.find_nearest_neighbors(coords, hs_radii)
        nnatoms_vec = nnatoms_list[atom_index]
        return self._bond_orientational_order(nnatoms_vec, ndim=ndim, deg=deg)
    
    def bond_orientation_order_all(self, coords, hs_radii, ndim=3, deg=6):
        """
        boo_list : array
            list of bond orientational order
        z_list : array
            list of coordination number for each particle
        """
        nnatoms_list = self.find_nearest_neighbors(coords, hs_radii)
        boo_list = []
        z_list = []
        for i in xrange(hs_radii.size):
            nnatoms_vec = nnatoms_list[i]
            if len(nnatoms_vec) > 0:
                boo = self._bond_orientational_order(nnatoms_vec, ndim=ndim, deg=deg)
                boo_list.append(boo)
                z_list.append(len(nnatoms_vec))
            else:
                #rattlers
                z_list.append(0)
                boo_list.append(0)
        return np.array(boo_list), np.array(z_list)

class PressureTensor(StructuralAnalysis):
    def __init__(self, packings_dir='packings', jammed_packings_dir='jammed_packings', analysis_dir='analysis'):
        super(PressureTensor,self).__init__(packings_dir=packings_dir, jammed_packings_dir=jammed_packings_dir, 
                                                    analysis_dir=analysis_dir)
    
    def run(self, existing_only=True):
        """compute boo for packings
        exisisting_only: bool
            run on already existing packings only
        pinit : bool
            initialise printing
        """
        for fname in os.listdir(self.packings_dir):
            if 'xyzd' in fname or 'xyd' in fname:
                dname = self._get_dname(fname)
                base_directory_path = os.path.join(os.getcwd(),'explore_bv_'+str(dname))
                configpath = os.path.join(self.jammed_packings_dir, dname + '.config')
                self._import_packing_config_file(configpath)
                if os.path.isdir(base_directory_path) or not existing_only:
                    trymakedir(base_directory_path)
                    analysis_dir_path = os.path.join(base_directory_path, self.analysis_dir)
                    trymakedir(analysis_dir_path)
                    coords, hs_radii, ss_radii = self._import_packing_configuration(fname)
                    potential = HS_WCA(use_periodic=True, eps=self.eps, sca=self.sca, 
                                       radii=hs_radii, boxvec=self.boxv, ndim=self.bdim)
                    p, ptensor = pressure_tensor(potential, coords, self.vcavity, self.bdim) 
                    fname = os.path.join(analysis_dir_path,'pressure_data')
                    with open(fname, 'w') as f:
                        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND \n')
                        f.write('[PRESSURE]\n')
                        f.write('P: {:.16f}\n'.format(p))
                        f.write('Ptensor: ')
                        for val in ptensor:
                            f.write('{:.16f} '.format(val))
                        f.write('\n')

if __name__ == "__main__":
    boo = BondOrientationalOrder()
    #boo.run(deg=12)
    boo.run_all()
    pts = PressureTensor()
    pts.run()