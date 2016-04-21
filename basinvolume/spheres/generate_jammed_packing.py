from __future__ import division
import numpy as np
import abc
from pele.potentials import HS_WCA
from pele.potentials import InversePowerStillingerCut
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.utils import *
import ConfigParser
import re
import argparse
try:
    import pylab
except:
    pass


def cartesian_to_polar2d(vector):
    vector = np.array(vector)
    r = np.linalg.norm(vector)
    theta = np.arctan2(vector[1], vector[0]) + np.pi
    return r, theta

def sum_neighbor_angles2d(neigh_vec):
    sum_ = 0.
    for idx in xrange(len(neigh_vec) - 1):
        sum_ += np.arccos(np.dot(neigh_vec[idx], neigh_vec[idx + 1]) / \
                         (np.linalg.norm(neigh_vec[idx]) * np.linalg.norm(neigh_vec[idx + 1])))
    sum_ += np.arccos(np.dot(neigh_vec[-1], neigh_vec[0]) / \
                     (np.linalg.norm(neigh_vec[-1]) * np.linalg.norm(neigh_vec[0])))
    return sum_

class _Generate_Jammed_Packing(object):
    """
    this is an abstract class that implements the basic components of a generate packing class,
    and declares a number of abstract methods which should be implemented in all inheriting classes
    *method to generate packing, this could be for example direct sampling, 
    sequential sampling,quench or LSA
    *nparticles: number of particles
    *bdim: dimensionality of the box
    *ndim: dimensionality of the problem (i.e. size of the coordinates array)
    *packing_frac: target jammed packing fraction
    *boxv: an array of size bdim that contains the vectors defining the box
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, packing_frac=0.65, packings_dir='packings'):
        self.packing_frac = packing_frac
        self.base_directory = os.path.join(os.getcwd(),'jammed_packings')
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir
        self.iteration = 0
        self.sca = -1
        self.eps = 1.
    
    def _import_single_packing_config_file(self, fname):
        dname = fname
        if dname.endswith('.xyzd'):
            dname = dname[:-5]
        elif dname.endswith('.xyd'):
            dname = dname[:-4]
        self.configpath = os.path.join(self.packings_dir, dname+'.config')
        self._import_packing_config_file()
          
    def _import_packing_config_file(self):
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.configpath))
        self.nparticles = configf.getint('PACKING','nparticles')
        self.bdim = configf.getint('PACKING','boxdim')
        assert self.bdim==2 or self.bdim==3, "bdim={} not implemented".format(self.bdim)
        self.ndim = self.nparticles * self.bdim
        boxv = configf.get('PACKING','boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        self.imp_packing_frac = configf.getfloat('PACKING','packing_fraction')
    
    @abc.abstractmethod
    def _initialise(self):
        """initialisation function"""
        self.configpath = os.path.join(self.packings_dir,'packings.config')
        assert(os.path.isfile(self.configpath))
        self._import_packing_config_file()
    
    @abc.abstractmethod
    def _import_packing_configuration(self, fname):
        """imports the coordinates and data relative to the shape of the particles
            this should be run in initialise()
        """
    @abc.abstractmethod
    def _generate_packing_coords(self):
        """function that generates the packing"""
    
    @abc.abstractmethod
    def _write_opengl_input(self, n):
        """writes a opengl input file, n is the unique identifier of the structure"""
        
    @abc.abstractmethod
    def _dump_configuration(self, n):
        """writes a configuration file, e.g .xyzd, n is the unique identifier of the structure"""
            
    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
    
    def _print_parameters(self, n):
        """writes the simulation parameters"""
        fname = '{}/jammed_packing{}.config'.format(self.base_directory,n)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Generate_Jammed_Packings base class input parameters\n')
        f.write('[JAMMED_PACKING]\n')
        f.write('nparticles: {}\n'.format(self.nparticles))
        f.write('packing_fraction: {}\n'.format(self.packing_frac))
        f.write('boxdim: {}\n'.format(self.bdim))
        f.write('ndim: {}\n'.format(self.ndim))
        f.write('boxv: ')
        for val in self.boxv:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        assert(self.sca > 0)
        f.write('sca: {:.16f}\n'.format(self.sca))
        f.write('\n')
        #print software version
        f.write('[CODEVERSION]\n')
        f.write('basinvolume_version: {}\n'.format(get_git_version('basinvolume')))
        f.write('mcpele_version: {}\n'.format(get_git_version('mcpele')))
        f.write('pele_version: {}\n'.format(get_git_version('pele')))
        f.write('python_version: {}\n'.format(get_python_version()))
        f.write('cython_version: {}\n'.format(get_cython_version()))
        f.close()
        
    def _print(self, n):
        """dump configuration and opengl input to packings directory
            n is the unique identifier of the structure
        """
        self._print_parameters(n)
        self._dump_configuration(n)
        self._write_opengl_input(n)
    
    # @abc.abstractmethod
    # def _histogram_eigenvalues(self):
    #     """ method to plot eigenvalues histograms
    #     """
    
    @abc.abstractmethod
    def one_iteration(self,fname):
        """perform one iteration
        """
    def run(self):
        """run generate packings"""
        self._initialise()
        for fname in os.listdir(self.packings_dir):
            if ('xyzd' in fname) or ('xyd' in fname):
                print "\n",fname
                self.one_iteration(fname)
        # self._histogram_eigenvalues()
            
class HS_Generate_Jammed_Packing(_Generate_Jammed_Packing):
    """
    *this class generates packings and identifies rattlers by computing the hessian eigenvalues for each particle
    *in the equilibrium jammed structure. A .xyzdr file is produced that contains the 3 system coordinates, the particle 
    * diameter and if not it's a rattler (0 if a rattler, 1 otherwise)
    *PARAMETERS
    *hs_radii: array with the radii of the particles, if none sample particle sizes from a normal distribution
    *mu: average particle size, passable to normal distribution
    *sig: standard deviaton of normal distribution from which to sample particles
    *sca: determines % by which the hs is inflated
    *eps: LJ interaction energy of WCA part of the HS potential
    """    
    def __init__(self, packing_frac=0.7,
        packings_dir='packings', use_cell_lists=False, show=False,
        opt_pot_str='hs_wca', extra_pot_kwargs=None):
        super(HS_Generate_Jammed_Packing,self).__init__(packing_frac=packing_frac, packings_dir=packings_dir)
        
        self.opt_pot_str = opt_pot_str
        self.extra_pot_kwargs = extra_pot_kwargs
        self.use_cell_lists = use_cell_lists
        ##constants#
        ############
    
    def _initialise(self):
        self._print_initialise()
    
    def one_iteration(self,fname):
        """perform one iteration
        """
        self._import_single_packing_config_file(fname)
        self.rattlers = np.empty(self.nparticles,dtype='d')
        self.rattlers_draw = np.empty(self.nparticles,dtype='d')
        
        self._import_packing_configuration(fname)
        self.max_nrattlers = int(self.nparticles*0.2)
        
        #assert that largest soft particle is not > 1/2 of smallest box size
        if np.amax(self.hs_radii) * 2 * (1 + self.sca) >= np.amin(self.boxv) / 2:
            print "WARNING: max soft diameter >= 1/2 box side!"
        if np.amax(self.hs_radii) * 2 * (1 + self.sca) >= np.amin(self.boxv):
            raise Exception("WARNING: particle does not fit the box")
        
        ###potential needs to be called because self.coords is an input argument of HS_WCAPeriodicCellLists
        rcut = np.amax(self.hs_radii) * 2.0 * (1.0 + self.sca) #rcut set to largest particle diameter
        if self.use_cell_lists:
            if np.amin(self.boxv) // rcut <= 3:
                self.use_cell_lists = False
        if self.opt_pot_str.lower() == "hs_wca":
            if self.use_cell_lists:
                self.potential = HS_WCA(use_periodic=True,
                    use_cell_lists=True, eps=self.eps, sca=self.sca,
                    radii=self.hs_radii, boxvec=self.boxv,
                    reference_coords=self.coords, rcut=rcut,
                    ndim=self.bdim, ncellx_scale=1.0)
            else:
                self.potential = HS_WCA(use_periodic=True, eps=self.eps,
                    sca=self.sca, radii=self.hs_radii, boxvec=self.boxv,
                    ndim=self.bdim)
        elif self.opt_pot_str.lower() == "inverse_power_stillinger":
            self.stillinger_a_radii = self.hs_radii * (1 + self.sca)
            pow = self.extra_pot_kwargs["pow"]
            rcut = self.extra_pot_kwargs["rcut"]
            self.potential = InversePowerStillingerCut(pow,
                self.stillinger_a_radii, ndim=self.bdim,
                boxvec=self.boxv, rcut=rcut, use_cell_lists=True)
        else:
            raise NotImplementedError
        
        success = self._generate_packing_coords() #returns false if saddle
        
        if success:
            n = int(re.search(r'\d+',fname).group())
            self._print(n)
        
        self.iteration+=1

    # def _find_rattlers(self):
    #     """
    #     finish this, I need to remove the rattler and break. Also need to get compare to existing jammed_packing option
    #     :return:
    #     """
    #     if self.bdim == 2:
    #         zmin = 3
    #     elif self.bdim == 3:
    #         zmin = 4
    #     else:
    #         raise NotImplemented
    #
    #     def get_index(x):
    #         # x is a 3 array with the coordinates of the particles
    #         dij = np.zeros(self.bdim)
    #         dmin = np.amin(self.hs_radii)/10.
    #         for j in xrange(self.nparticles):
    #             for k in xrange(self.bdim):
    #                 #use distances to nearest image convention
    #                 dij[k] = ((self.coords[j*self.bdim+k] - x[k]) -
    #                           cround((self.coords[j*self.bdim+k] - x[k]) / self.boxv[k]) * self.boxv[k])
    #             if np.linalg.norm(dij) < dmin:
    #                 return j
    #
    #     # tesselate packing
    #     if self.bdim == 2:
    #         cells = pyvoro.compute_2d_voronoi(coords, limits, dispersion, radii=radii)
    #     elif self.bdim == 3:
    #         cells = pyvoro.compute_voronoi(coords, limits, dispersion, radii=radii)
    #     else:
    #         raise NotImplementedError("pyvoro bdim={} not implemented".format(self.bdim))
    #     assert (len(cells) == int(len(self.coords) / self.bdim))
    #
    #     coords = np.array(self.coords)
    #     hs_radii = np.array(self.hs_radii)
    #     block_evalues = np.empty((self.nparticles, self.bdim))
    #     look = True
    #     nratls = 0
    #     while look:
    #         print "restarting loop"
    #         found_rattler = False
    #         if nratls > self.max_nrattlers:
    #             return False
    #         potential = HS_WCA(use_periodic=True, eps=self.eps, sca=self.sca,
    #                            radii=hs_radii, boxvec=self.boxv, ndim=self.bdim)
    #         hess = potential.getHessian(coords)
    #         radii = hs_radii*(1.+self.sca)
    #         contact_list = self._find_nearest_neighbors(coords, radii)
    #         for i in xrange(len(hs_radii)):
    #             i1 = self.bdim*i
    #             no_neighbors = len(contact_list[i])
    #             # print "no_neighbors", no_neighbors
    #             if no_neighbors < zmin:
    #                 hess_block = hess[i1:i1+self.bdim, i1:i1+self.bdim]
    #                 w, v = np.linalg.eig(hess_block)
    #                 print no_neighbors, w
    #                 w = np.zeros(self.bdim)
    #                 # print "particle {} is not isostatic".format(i)
    #             else:
    #                 hess_block = hess[i1:i1+self.bdim, i1:i1+self.bdim]
    #                 w, v = np.linalg.eig(hess_block)
    #                 w = np.real(w)
    #             #here assign correct index by searchin for the corresponding atom
    #             j = get_index(coords[i1:i1+self.bdim])
    #             self.rattlers[j] = np.amin(w)
    #             self.rattlers_draw[j] = float(self.rattlers[j] >= self.rattler_eval_tol)
    #             block_evalues[j] = w
    #             if self.rattlers_draw[j] < self.rattler_eval_tol:
    #                 nratls += 1
    #                 coords = np.delete(coords, [i1+k for k in xrange(self.bdim)]) #remove particle from array
    #                 hs_radii = np.delete(hs_radii, [i]) #remove particle from array
    #                 # print 'zero eigenvalue, particle {}'.format(j)
    #                 # print w
    #                 found_rattler = True
    #                 break
    #         look = True if found_rattler else False
    #     #now look at validity of the packing, first check that it's a minimum
    #     w, v = np.linalg.eig(hess)
    #     w = np.real(w)
    #     if np.any(w < -1e-7):
    #         print 'e: {} eigenvalue < -1e-7'.format(np.amin(w))
    #         return False
    #     #check that the hessian has the correct number of 0 eigenvalues
    #     full0evals = [x for x in w if np.abs(x) < 1e-7]
    #     if len(full0evals) > self.bdim:
    #         print 'hessian 0s mismatch bdim 0s, found ', len(full0evals), full0evals
    #         return False
    #     self.block_evalues.extend(block_evalues.flatten())
    #     self.whole_evalues.extend(w)
    #     return True

    # force = np.zeros(self.bdim)
    # for j, dij in zip(neighbors_index_list[i], contact_list[i]):
    #     j1 = self.bdim * j
    #     # DEBUG: this needs to be able to use any particular potential
    #     pair_pot = HS_WCA(use_periodic=True, eps=self.eps, sca=self.sca,
    #                       radii=np.array([hs_radii[i], hs_radii[j]]),
    #                       boxvec=self.boxv, ndim=self.bdim)
    #     x = np.append(coords[i1:i1 + self.bdim], coords[j1:j1 + self.bdim])
    #     f = - dij * np.linalg.norm(pair_pot.getEnergyGradient(x)[1]) / np.linalg.norm(dij)
    #     force += f
    # print "|f| {}, nn {}".format(np.linalg.norm(force), no_neighbors)
    # found_rattler = np.linalg.norm(force) > self.force_tol

    def _find_rattlers(self):
        """
        finish this, I need to remove the rattler and break. Also need to get compare to existing jammed_packing option
        :return:
        """
        if self.bdim == 2:
            zmin = 3
        elif self.bdim == 3:
            raise NotImplementedError
            #zmin = 4
        else:
            raise NotImplementedError

        def get_index(x):
            # x is a 3 array with the coordinates of the particles
            dij = np.zeros(self.bdim)
            dmin = np.amin(self.hs_radii)/10.
            for j in xrange(self.nparticles):
                for k in xrange(self.bdim):
                    #use distances to nearest image convention
                    dij[k] = ((self.coords[j*self.bdim+k] - x[k]) -
                              cround((self.coords[j*self.bdim+k] - x[k]) / self.boxv[k]) * self.boxv[k])
                if np.linalg.norm(dij) < dmin:
                    return j

        coords = np.array(self.coords)
        hs_radii = np.array(self.hs_radii)
        look = True
        nratls = 0
        while look:
            # print "restarting loop"
            found_rattler = False
            if nratls > self.max_nrattlers:
                return False
            radii = hs_radii * (1. + self.sca)
            contact_list, neighbors_index_list = self._find_nearest_neighbors(coords, radii)
            for i in xrange(len(hs_radii)):
                i1 = self.bdim * i
                no_neighbors = len(contact_list[i])
                # print "no_neighbors", no_neighbors
                if no_neighbors < zmin:
                    found_rattler = True
                    print "particle {} is not isostatic".format(i)
                else:
                    angles = [cartesian_to_polar2d(dij)[1] for dij in contact_list[i]]
                    neigh_vec = [x for (y, x) in sorted(zip(angles, contact_list[i]))]
                    sum_ = sum_neighbor_angles2d(neigh_vec)
                    found_rattler =  np.abs(2*np.pi - sum_) > 1e-10
                    if found_rattler:
                        print "asymmetric contact rattler, 2pi - theta = {}".format(2*np.pi - sum_)
                #here assign correct index by searchin for the corresponding atom
                j = get_index(coords[i1:i1+self.bdim])
                self.rattlers[j] = 0 if found_rattler else 1000
                self.rattlers_draw[j] = float(not found_rattler)
                if found_rattler:
                    coords = np.delete(coords, [i1 + k for k in xrange(self.bdim)])  # remove particle from array
                    hs_radii = np.delete(hs_radii, [i]) #remove particle from array
                    nratls+=1
                    break
            look = True if found_rattler else False
        print "n rattlers ", nratls
        return True

    def _find_nearest_neighbors(self, coords, radii):
        nparticles = radii.size
        nnatoms_list = [[] for _ in xrange(nparticles)]
        nnatoms_index_list = [[] for _ in xrange(nparticles)]
        for i in xrange(nparticles):
            for j in xrange(i, nparticles):
                if i != j:
                    dij = np.zeros(self.bdim)
                    for k in xrange(self.bdim):
                        #use distances to nearest image convention
                        dij[k] = ((coords[j*self.bdim+k] - coords[i*self.bdim+k]) -
                                           cround((coords[j*self.bdim+k] - coords[i*self.bdim+k]) / self.boxv[k]) * self.boxv[k])
                    dijnorm = np.linalg.norm(dij)
                    dmin = radii[i] + radii[j]
                    if dijnorm <= dmin:
                        nnatoms_list[i].append(dij)
                        nnatoms_list[j].append(-dij)
                        nnatoms_index_list[i].append(j)
                        nnatoms_index_list[j].append(i)
        return nnatoms_list, nnatoms_index_list
    
    
    def _generate_packing_coords(self):
        """
        perform quench and run tests
        """
        success = self._generate_packing_coords_iteration(tol=1e-9)
        return success
    
    def _generate_packing_coords_iteration(self, tol=1e-9, iprint=-1):
        """quenches the imported structure using FIRE"""
        fire_maxstep = np.amin(self.hs_radii)*self.sca
        res = modifiedfire_cpp(self.coords, self.potential, maxstep=fire_maxstep, nsteps=1e6, tol=tol, iprint=iprint)
        if not res.success:
            print 'quench failed'
            return False
        
        self.coords = res.coords
        self.energy = res.energy
        
        #test that on ri-minimisation the structure does not change
        res2 = modifiedfire_cpp(self.coords, self.potential, maxstep=fire_maxstep, nsteps=1e6, tol=tol)
        if res2.nfev > 1:
            print 'quench failed (structure changed at second minimisation)'
            return False
        
        #asserts that none of the hard sphere is overlapping
        no_overlap = self._check_no_overlaps()
        if not no_overlap:
            print 'overlap found'
            return False

        return self._find_rattlers()

    def _get_particles_volume(self):
        """returns volume of n=self.bdim dimensional sphere"""
        volumes = volume_nball(self.hs_radii,self.bdim)
        vtot = np.sum(volumes)
        return vtot
    
    def _import_packing_configuration(self, fname):
        path = os.path.join(self.packings_dir, fname)
        if self.bdim == 2:
            self.coords, hs_diameters = read_xyd(path)
        elif self.bdim == 3:
            self.coords, hs_diameters = read_xyzd(path)
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        self.hs_radii = hs_diameters/2
        self._compute_sca()
    
    def _compute_sca(self):
        ##test##
        vol_part = self._get_particles_volume()
        vol_box = np.prod(self.boxv)
        phi = vol_part/vol_box #instanteneous pack frac
        assert(phi - self.imp_packing_frac < 1e-4)
        ##endtest##
        ###r_soft = r_hs*(1+sca)
        self.sca = np.power(self.packing_frac/self.imp_packing_frac,1./self.bdim) - 1
        
    def _check_no_overlaps(self):
        """check that no two particles are overlapping (using nearest image convention)"""
        no_overlap = True
        for i in xrange(self.nparticles):
            if no_overlap == True:
                for j in xrange(self.nparticles):
                    dij = 0
                    for k in xrange(self.bdim):
                        #use distances to nearest image convention
                        dij += np.square((self.coords[i*self.bdim+k] - self.coords[j*self.bdim+k]) -
                                          cround((self.coords[i*self.bdim+k] - self.coords[j*self.bdim+k]) / self.boxv[k]) * self.boxv[k])
                    if i != j:
                        dij = np.sqrt(dij)
                        dmin = self.hs_radii[i]+self.hs_radii[j]
                        if dij - dmin <= 0:
                            print 'invalid configuration'
                            print 'atoms {} {} are overlapping'.format(i,j)
                            print 'real distance {}'.format(dij)
                            print 'min distance {}'.format(dmin)
                            no_overlap = False
                            break
            else:
                break
        return no_overlap
            
    def _correct_coords(self):
        """this function returns the nearest images in the central box, useful for dumping the configurations"""
        coords = self.coords.copy()
        put_in_box(coords,self.boxv)
        return coords
    
    def _dump_configuration(self,n):
        """write coordinates to file .xyzdr"""
        directory = self.base_directory
        #compare to existing file (dirty hack)
        compare = True
        if compare:
            try:
                if self.bdim == 2:
                    fname = "{0}/jammed_packing{1}.xydr".format(directory, n)
                    coords, hs_diameters, rattlers = read_xydr(fname)
                elif self.bdim == 3:
                    fname = "{0}/jammed_packing{1}.xyzdr".format(directory, n)
                    coords, hs_diameters, rattlers = read_xyzdr(fname)
                n_old_nratls = (len(rattlers) - np.count_nonzero(rattlers))//self.bdim
                n_new_nratls = len(self.rattlers) - np.count_nonzero(self.rattlers)
                if n_new_nratls != n_old_nratls:

                        with open("{0}/mismatching_rattlers.txt".format(directory), 'a') as f:
                            f.write('jammed_packing{}\n'.format(n))
            except Exception, e:
                print e
        #dump configuration
        if not compare:
            coords = self._correct_coords()
            if self.bdim == 2:
                fname = "{0}/jammed_packing{1}.xydr".format(directory,n)
                f = open(fname,'w')
                for i in xrange(self.nparticles):
                    f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                                              self.hs_radii[i]*2,self.rattlers[i]))
            elif self.bdim == 3:
                fname = "{0}/jammed_packing{1}.xyzdr".format(directory,n)
                f = open(fname,'w')
                for i in xrange(self.nparticles):
                    f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                                              coords[i*self.bdim+2],self.hs_radii[i]*2,self.rattlers[i]))
            else:
                raise NotImplementedError("bdim={} not implemented".format(self.bdim))
            f.close()
    
    def _write_opengl_input(self,n):
        """write opengl input file"""
        coords = self._correct_coords()
        boxv = self.boxv
        colour = 14
        directory = self.base_directory
        fname = "{0}/jammed_packing{1}.dat".format(directory,n)
        f = open(fname,'w')
        f.write('{}\n'.format(self.nparticles))
        if self.bdim == 2:
            f.write('{} {} {}\n'.format(-boxv[0]/2,-boxv[1]/2, - np.amax(self.hs_radii)))
            f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
            f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
            f.write('0.0 \t 0.0 \t {}\n'.format(np.amax(self.hs_radii)*2))
            for i in xrange(self.nparticles):
                for j in xrange(self.bdim):
                    f.write('{}\t'.format(coords[i*self.bdim+j]))
                f.write('{}\t'.format(0.0))
                f.write('{}\t'.format(self.hs_radii[i]*2*(1.+self.sca)))
                f.write('{}\n'.format(colour-int(self.rattlers_draw[i])))
        elif self.bdim == 3:
            f.write('{} {} {}\n'.format(-boxv[0]/2,-boxv[1]/2,-boxv[2]/2))
            f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
            f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
            f.write('0.0 \t 0.0 \t {}\n'.format(boxv[2]))
            for i in xrange(self.nparticles):
                for j in xrange(self.bdim):
                    f.write('{}\t'.format(coords[i*self.bdim+j]))
                f.write('{}\t'.format(self.hs_radii[i]*2*(1.+self.sca)))
                f.write('{}\n'.format(colour-int(self.rattlers_draw[i])))
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        f.close()
    
    # def _histogram_eigenvalues(self):
    #     #self.eigenvalues = np.array(self.eigenvalues,dtype='d')
    #     self.block_evalues = np.real(self.block_evalues)
    #     pylab.figure()
    #     self.block_histogram, bins = np.histogram(self.block_evalues ,bins=self.nbins)
    #     width = bins[1] - bins[0]
    #     center = (bins[:-1] + bins[1:]) / 2
    #     pylab.bar(center, self.block_histogram, align='center', width=width)
    #     pylab.savefig(os.path.join(self.base_directory,'blocks_histogram.eps'))
    #     if self.show:
    #         pylab.show()
    #     pylab.figure()
    #     self.block_histogram_low, bins = np.histogram(self.block_evalues ,bins=self.nbins_low, range=self.low_range)
    #     width = bins[1] - bins[0]
    #     center = (bins[:-1] + bins[1:]) / 2
    #     pylab.bar(center, self.block_histogram_low, align='center', width=width)
    #     pylab.savefig(os.path.join(self.base_directory, 'blocks_histogram_low{}.eps'.format(self.low_range[1])) )
    #     if self.show:
    #         pylab.show()
    #
    #     self.whole_evalues = np.real(self.whole_evalues)
    #     pylab.figure()
    #     self.whole_histogram, bins = np.histogram(self.whole_evalues ,bins=self.nbins)
    #     width = bins[1] - bins[0]
    #     center = (bins[:-1] + bins[1:]) / 2
    #     pylab.bar(center, self.whole_histogram, align='center', width=width)
    #     pylab.savefig(os.path.join(self.base_directory,'whole_histogram.eps'))
    #     if self.show:
    #         pylab.show()
    #     pylab.figure()
    #     self.whole_histogram_low, bins = np.histogram(self.whole_evalues ,bins=self.nbins_low, range=self.low_range)
    #     width = bins[1] - bins[0]
    #     center = (bins[:-1] + bins[1:]) / 2
    #     pylab.bar(center, self.whole_histogram_low, align='center', width=width)
    #     pylab.savefig(os.path.join(self.base_directory,'whole_histogram_low{}.eps'.format(self.low_range[1])))
    #     if self.show:
    #         pylab.show()

            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="generate 2/3-D hard disks/spheres packings")
    parser.add_argument("-p","--density", type=float, help="target packing fraction",default=0.7)
    parser.add_argument("--nocell", action='store_false', help="don't use cell lists, default: True",default=True)
    parser.add_argument("--packingsdir", type=str, help="name of directory with packings, must be in cwd", default="packings")
    parser.add_argument("--show", action='store_true', help="show histograms", default=False)
    # potential arguments
    parser.add_argument("--opt_pot", type=str, help="optmizer's potential, 1) (default) hs_wca "
                                                    "2) inverse_power_stillinger", default='hs_wca')
    args = parser.parse_args()
    print args
    
    # potential type
    opt_pot_str = args.opt_pot
    extra_pot_kwargs = dict()
    if opt_pot_str.lower() == 'hs_wca':
        pass
    elif opt_pot_str.lower() == 'inverse_power_stillinger':
        extra_pot_kwargs = dict(pow=8, rcut=4.5)
        print 'setting inverse_power_stillinger parameters: ', extra_pot_kwargs
    else:
        raise NotImplementedError
    
    print("extra_pot_kwargs", extra_pot_kwargs)
    sim = HS_Generate_Jammed_Packing(packing_frac=args.density,
        packings_dir=args.packingsdir,
        use_cell_lists=args.nocell, show=args.show,
        opt_pot_str=args.opt_pot, extra_pot_kwargs=extra_pot_kwargs)
    sim.run()
    
    
        
                
            
              
                
                
                
