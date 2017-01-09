from __future__ import division
import numpy as np
import abc
import os
from pele.distance import get_distance, put_in_box
from pele.potentials import HS_WCA
from pele.potentials import InversePowerStillingerCut
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.utils import trymakedir, get_git_version, get_python_version, get_cython_version, cround
from basinvolume.utils import volume_nball, in_hull, read_xyd, read_xyzd, read_xydr, read_xyzdr
import ConfigParser
import re
import argparse
import subprocess
import shlex
import glob
import ast
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

    def __init__(self, packing_frac=0.65, packings_dir='packings', import_jammed=False,
                 outdir='jammed_packings', override_pot_kwargs=None):
        self.packing_frac = packing_frac
        self.base_directory = os.path.join(os.getcwd(), outdir)
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir
        self.import_jammed = import_jammed
        self.override_pot_kwargs = override_pot_kwargs
        self.iteration = 0
        self.sca = -1
        self.eps = 1.

    def _import_single_packing_config_file(self, fname):
        dname = fname
        if dname.endswith('.xyzd'):
            dname = dname[:-5]
        elif dname.endswith('.xyd'):
            dname = dname[:-4]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        elif dname.endswith('.xyzdr'):
            dname = dname[:-6]
        self.configpath = os.path.join(self.packings_dir, dname+'.config')
        self._import_packing_config_file("JAMMED_PACKING" if self.import_jammed else "PACKING")

    def _import_packing_config_file(self, section):
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.configpath))
        self.nparticles = configf.getint(section,'nparticles')
        self.bdim = configf.getint(section,'boxdim')
        assert self.bdim==2 or self.bdim==3, "bdim={} not implemented".format(self.bdim)
        self.ndim = self.nparticles * self.bdim
        boxv = configf.get(section,'boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        if self.import_jammed:
            imp_sca = configf.getfloat(section, 'sca')
            self.imp_packing_frac = self.packing_frac / (1 + imp_sca)**2
        else:
            self.imp_packing_frac = configf.getfloat(section,'packing_fraction')
        self.distance_method = configf.get(section, 'distance_method')
        if hasattr(self, 'pot_kwargs') and self.pot_kwargs is not None:
            self.pot_kwargs.update(ast.literal_eval(configf.get(section, 'pot_kwargs')))
        else:
            self.pot_kwargs = ast.literal_eval(configf.get(section, 'pot_kwargs'))
        if self.override_pot_kwargs is not None:
            self.pot_kwargs.update(self.override_pot_kwargs)



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
        f.write('packing_fraction: {:.16f}\n'.format(self.packing_frac))
        f.write('boxdim: {}\n'.format(self.bdim))
        f.write('ndim: {}\n'.format(self.ndim))
        f.write('boxv: ')
        for val in self.boxv:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        f.write('distance_method: {}\n'.format(self.distance_method))
        f.write('pot_kwargs: {}\n'.format(self.pot_kwargs))
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
        successes = []
        for fname in os.listdir(self.packings_dir):
            if self.import_jammed:
                if ('xyzdr' in fname) or ('xydr' in fname):
                    print(fname)
                    success = self.one_iteration(fname)
                    successes.append((fname, success))
                    print("")
            else:
                if ('xyzd' in fname) or ('xyd' in fname):
                    print(fname)
                    success = self.one_iteration(fname)
                    successes.append((fname, success))
                    print("")
        return successes
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
    *tol: rms tolerance for the minimizer
    """
    def __init__(self, packing_frac=0.7, tol=1e-9,
        packings_dir='packings', import_jammed=False, outdir='jammed_packings',
        use_cell_lists=False, show=False,
        opt_pot_str='hs_wca', pot_kwargs=None, override_pot_kwargs=None):
        super(HS_Generate_Jammed_Packing,self).__init__(packing_frac=packing_frac,
                                                        packings_dir=packings_dir,
                                                        import_jammed=import_jammed,
                                                        outdir=outdir, override_pot_kwargs=override_pot_kwargs)

        self.opt_pot_str = opt_pot_str
        self.pot_kwargs = pot_kwargs
        self.use_cell_lists = use_cell_lists
        self.tol = tol
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
        self.max_nrattlers = int(self.nparticles*0.5)

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
                self.potential = HS_WCA(use_cell_lists=True, eps=self.eps, sca=self.sca,
                                        radii=self.hs_radii, boxvec=self.boxv,
                                        reference_coords=self.coords, ndim=self.bdim,
                                        ncellx_scale=1.0, distance_method=self.distance_method,
                                        pot_kwargs=self.pot_kwargs)
            else:
                self.potential = HS_WCA(eps=self.eps, sca=self.sca, radii=self.hs_radii,
                                        boxvec=self.boxv, ndim=self.bdim,
                                        distance_method=self.distance_method,
                                        pot_kwargs=self.pot_kwargs)
        elif self.opt_pot_str.lower() == "inverse_power_stillinger":
            self.stillinger_a_radii = self.hs_radii * (1 + self.sca)
            pow = self.pot_kwargs["pow"]
            rcut = self.pot_kwargs["rcut"]
            self.potential = InversePowerStillingerCut(pow,
                self.stillinger_a_radii, ndim=self.bdim,
                boxvec=self.boxv, rcut=rcut, use_cell_lists=True)
        else:
            raise NotImplementedError

        success = self._generate_packing_coords() #returns false if saddle

        n = int(re.search(r'\d+', fname).group())
        if success:
            self._print(n)
        else:
            path_list = glob.glob("{0}/jammed_packing{1}.*".format(self.base_directory, n))
            if len(path_list) > 0:
                with open("{0}/mismatching_rattlers.txt".format(self.base_directory), 'a') as f:
                            f.write('jammed_packing{}\n'.format(n))
            for path_ in path_list:
                p = subprocess.call(shlex.split("rm {}".format(path_)))
        self.iteration += 1
        return success

    def _distance (self, coord1, coord2):
        if self.distance_method == 'lees-edwards':
            return get_distance(coord1, coord2, self.bdim, self.distance_method,
                                box=self.boxv, shear=self.pot_kwargs['shear'])
        else:
            return get_distance(coord1, coord2, self.bdim, self.distance_method, box=self.boxv)


    def _find_rattlers(self):
        """
        finish this, I need to remove the rattler and break. Also need to get compare to existing jammed_packing option
        :return:
        """
        if self.bdim < 4:
            zmin = self.bdim + 1
        else:
            raise NotImplementedError

        def get_index(x):
            # x is a 3 array with the coordinates of the particles
            dmin = np.amin(self.hs_radii)/10.
            for j in xrange(self.nparticles):
                dij = np.linalg.norm(self._distance(self.coords[j * self.bdim : (j + 1) * self.bdim], x))
                if dij < dmin:
                    return j

        coords = np.array(self.coords)
        hs_radii = np.array(self.hs_radii)
        look = True
        nratls = 0
        while look:
            found_rattler = False
            if nratls > self.max_nrattlers:
                print "Too many rattlers. Discarding packing."
                return False
            radii = hs_radii * (1. + self.sca)
            contact_list, neighbors_index_list = self._find_nearest_neighbors(coords, radii)
            for i in xrange(len(hs_radii)):
                i1 = self.bdim * i
                j = get_index(coords[i1:i1+self.bdim])
                no_neighbors = len(contact_list[i])
                if no_neighbors < zmin:
                    found_rattler = True
                    print "Particle {} is not isostatic.".format(j)
                else:
                    # angles = [cartesian_to_polar2d(dij)[1] for dij in contact_list[i]]
                    # neigh_vec = [x for (y, x) in sorted(zip(angles, contact_list[i]))]
                    # sum_ = sum_neighbor_angles2d(neigh_vec)
                    # found_rattler =  np.abs(2*np.pi - sum_) > 1e-10
                    p = np.zeros(self.bdim)
                    hull = np.asarray(contact_list[i]).reshape((-1,self.bdim))
                    found_rattler = not in_hull(p, hull)
                    if found_rattler:
                        # print "asymmetric contact rattler, 2pi - theta = {}".format(2*np.pi - sum_)
                        print "Particle {} is not in contacts' convex hull.".format(j)
                #here assign correct index by searching for the corresponding atom
                self.rattlers[j] = 0 if found_rattler else 1000
                self.rattlers_draw[j] = float(not found_rattler)
                if found_rattler:
                    coords = np.delete(coords, [i1 + k for k in xrange(self.bdim)])  # remove particle from array
                    hs_radii = np.delete(hs_radii, [i]) #remove particle from array
                    nratls+=1
                    break
            look = True if found_rattler else False

        # test that number of contacts is sufficient for bulk modulus to be positive,
        # see eq 4 in http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.095704
        # see eq 19 in arXiv:1406.1529
        N_contacts = int(np.sum([len(contacts) for contacts in contact_list]))
        no_stable = len(contact_list)
        N_min = int(2*(self.bdim * (no_stable - 1) + 1))
        print "N_min: {} N_contacts: {}".format(N_min, N_contacts)
        assert (self.nparticles - no_stable) == nratls
        print "n rattlers ", nratls
        if N_contacts >= N_min:
            return True
        else:
            print "Packing is not globally stable, N_min: {} N_contacts: {}".format(N_min, N_contacts)
            return False

    def _find_nearest_neighbors(self, coords, radii):
        nparticles = radii.size
        nnatoms_list = [[] for _ in xrange(nparticles)]
        nnatoms_index_list = [[] for _ in xrange(nparticles)]
        for i in xrange(nparticles - 1):
            for j in xrange(i + 1, nparticles):
                dij = self._distance(coords[i * self.bdim : (i + 1) * self.bdim],
                                         coords[j * self.bdim : (j + 1) * self.bdim])
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
        success = self._generate_packing_coords_iteration(tol=self.tol)
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
        if self.import_jammed:
            if self.bdim == 2:
                self.coords, hs_diameters, _ = read_xydr(path)
            elif self.bdim == 3:
                self.coords, hs_diameters, _ = read_xyzdr(path)
            else:
                raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        else:
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
                    if i != j:
                        dij = np.linalg.norm(self._distance(
                            self.coords[i * self.bdim : (i + 1) * self.bdim],
                            self.coords[j * self.bdim : (j + 1) * self.bdim]))
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
        if self.distance_method == 'lees-edwards':
            return put_in_box(self.coords, self.bdim, self.distance_method, self.boxv, self.pot_kwargs['shear'])
        else:
            return put_in_box(self.coords, self.bdim, self.distance_method, self.boxv)

    def _dump_configuration(self,n):
        """write coordinates to file .xyzdr"""
        directory = self.base_directory
        if self.bdim == 2:
            fname = "{0}/jammed_packing{1}.xydr".format(directory, n)
        elif self.bdim == 3:
            fname = "{0}/jammed_packing{1}.xyzdr".format(directory, n)

        #dump configuration
        coords = self._correct_coords()
        if self.bdim == 2:
            f = open(fname,'w')
            for i in xrange(self.nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(
                    coords[i*self.bdim],coords[i*self.bdim+1],
                    self.hs_radii[i]*2,self.rattlers[i]))
        elif self.bdim == 3:
            f = open(fname,'w')
            for i in xrange(self.nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(
                    coords[i*self.bdim],coords[i*self.bdim+1],
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
    parser.add_argument("--nocell", action='store_true', help="don't use cell lists, default: False",default=False)
    parser.add_argument("--packingsdir", type=str, help="name of directory with packings, must be in cwd", default="packings")
    parser.add_argument("--import_jammed", action='store_true', help="Take a jammed packing as input "
                        "instead of an unjammed one.", default=False)
    parser.add_argument("-o", "--outdir", type=str, help="Directory to save jammed packings in. "
                        "Default: 'jammed_packings'", default='jammed_packings')
    parser.add_argument("--show", action='store_true', help="show histograms", default=False)
    parser.add_argument("-t", "--tol", type=float, help="rms tolerance of the minimizer", default=1e-9)
    # potential arguments
    parser.add_argument("--opt_pot", type=str, help="optmizer's potential, 1) (default) hs_wca "
                                                    "2) inverse_power_stillinger", default='hs_wca')

    args = parser.parse_args()
    print args

    # potential type
    opt_pot_str = args.opt_pot
    override_pot_kwargs = dict()
    if opt_pot_str.lower() == 'hs_wca':
        pass
    elif opt_pot_str.lower() == 'inverse_power_stillinger':
        override_pot_kwargs.update(pow=8, rcut=4.5)
        print 'setting inverse_power_stillinger parameters: ', override_pot_kwargs
    else:
        raise NotImplementedError

    sim = HS_Generate_Jammed_Packing(packing_frac=args.density,
                                     packings_dir=args.packingsdir, import_jammed=args.import_jammed,
                                     outdir=args.outdir, tol=args.tol,
                                     use_cell_lists=not args.nocell, show=args.show,
                                     opt_pot_str=args.opt_pot, override_pot_kwargs=override_pot_kwargs)
    sim.run()
