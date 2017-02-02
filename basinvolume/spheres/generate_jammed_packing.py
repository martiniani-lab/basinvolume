from __future__ import division
import numpy as np
import abc
import os
from pele.distance import put_in_box
from pele.potentials import HS_WCA
from pele.potentials import InversePowerStillingerCut
from pele.optimize._quench import modifiedfire_cpp, lbfgs_cpp
from PyCG_DESCENT import CGDescent
from basinvolume.utils import trymakedir, get_git_version, get_python_version
from basinvolume.utils import volume_nball, import_packing, find_neighbours, calc_distance
from basinvolume.utils import get_cython_version, cround, in_hull, origin_in_hull_2d
from basinvolume.spheres.generate_packing import read_packing_config
import ConfigParser
import re
import argparse
import subprocess
import shlex
import glob
import ast
import logging
# try:
#     import pylab
# except:
#     pass


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

def read_jammed_packing_config(configpath, frozen=False):
    configf = ConfigParser.ConfigParser()
    configf.read(str(configpath))
    parameters = {}
    parameters['nparticles'] = configf.getint('JAMMED_PACKING','nparticles')
    parameters['packing_frac'] = configf.getfloat('JAMMED_PACKING','packing_fraction')
    parameters['bdim'] = configf.getint('JAMMED_PACKING','boxdim')
    assert parameters['bdim'] == 2 or parameters['bdim'] == 3, \
        "bdim={} not implemented".format(parameters['bdim'])
    parameters['ndim'] = parameters['nparticles'] * parameters['bdim']
    boxv = configf.get('JAMMED_PACKING','boxv')
    parameters['boxv'] = np.array([float(x) for x in boxv.split()])
    if frozen:
        parameters['vcavity'] = configf.getfloat('JAMMED_PACKING', 'vcavity')
    else:
        parameters['vcavity'] = np.prod(parameters['boxv'])
    parameters['distance_method'] = configf.get('JAMMED_PACKING', 'distance_method')
    parameters['pot_kwargs'] = ast.literal_eval(configf.get('JAMMED_PACKING', 'pot_kwargs'))
    parameters['sca'] = configf.getfloat('JAMMED_PACKING','sca')
    return parameters

class _Generate_Jammed_Packing(object):
    """
    this is an abstract class that implements the basic components of a generate packing class,
    and declares a number of abstract methods which should be implemented in all inheriting classes
    *method to generate packing, this could be for example direct sampling,
    sequential sampling,quench or LSA
    *nparticles: number of particles
    *bdim: dimensionality of the box
    *ndim: dimensionality of the problem (i.e. size of the coordinates array)
    *target_packing_frac: target jammed packing fraction
    *boxv: an array of size bdim that contains the vectors defining the box
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, target_packing_frac=0.65, packings_dir='packings', packing_nrs=None,
                 import_jammed=False, outdir='jammed_packings', override_pot_kwargs=None,
                 minimizer='fire', logging_tag=""):
        self.target_packing_frac = target_packing_frac
        self.base_directory = os.path.join(os.getcwd(), outdir)
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir
        self.packing_nrs = packing_nrs
        self.import_jammed = import_jammed
        self.override_pot_kwargs = override_pot_kwargs
        self.iteration = 0
        self.sca = -1
        self.eps = 1.
        self.minimizer = minimizer
        self.logging_tag = logging_tag

    def _import_single_packing_config_file(self, fname):
        dname = os.path.splitext(fname)[0]
        self.configpath = os.path.join(self.packings_dir, dname+'.config')
        if self.import_jammed:
            imp_packing = read_jammed_packing_config(str(self.configpath))
            self.nparticles = imp_packing['nparticles']
            self.packing_frac = imp_packing['packing_frac']
            self.bdim = imp_packing['bdim']
            self.ndim = imp_packing['ndim']
            self.boxv = imp_packing['boxv'].copy()
            self.vcavity = imp_packing['vcavity']
            self.distance_method = imp_packing['distance_method']
            if hasattr(self, 'pot_kwargs') and self.pot_kwargs is not None:
                self.pot_kwargs.update(imp_packing['pot_kwargs'])
            else:
                self.pot_kwargs = imp_packing['pot_kwargs'].copy()
            if self.override_pot_kwargs is not None:
                self.pot_kwargs.update(self.override_pot_kwargs)
            self.sca = imp_packing['sca']
            self.packing_frac = self.target_packing_frac / (1 + self.sca)**2
        else:
            self._import_packing_config_file(str(self.configpath))


    @abc.abstractmethod
    def _initialise(self):
        """initialisation function"""
        self.configpath = os.path.join(self.packings_dir,'packings.config')
        assert(os.path.isfile(self.configpath))
        self._import_packing_config_file(str(self.configpath))

    def _import_packing_config_file(self, configpath):
        imp_packing = read_packing_config(configpath)
        self.nparticles = imp_packing['nparticles']
        self.packing_frac = imp_packing['packing_frac']
        self.bdim = imp_packing['boxdim']
        self.ndim = imp_packing['ndim']
        self.hs_mean = imp_packing['radii_mean']
        self.hs_stddev = imp_packing['radii_stddev']
        self.boxv = imp_packing['boxv'].copy()
        self.vcavity = imp_packing['vcavity']
        self.distance_method = imp_packing['distance_method']
        if hasattr(self, 'pot_kwargs') and self.pot_kwargs is not None:
            self.pot_kwargs.update(imp_packing['pot_kwargs'])
        else:
            self.pot_kwargs = imp_packing['pot_kwargs'].copy()
        if self.override_pot_kwargs is not None:
            self.pot_kwargs.update(self.override_pot_kwargs)

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
        trymakedir(self.base_directory)

    def _print_parameters(self, n):
        """writes the simulation parameters"""
        fname = '{}/jammed_packing{}.config'.format(self.base_directory,n)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Generate_Jammed_Packings base class input parameters\n')
        f.write('[JAMMED_PACKING]\n')
        f.write('nparticles: {}\n'.format(self.nparticles))
        f.write('packing_fraction: {:.16f}\n'.format(self.target_packing_frac))
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

    def _log(self, message):
        if self.logging_tag == None or self.logging_tag == "":
            return "{}".format(message)
        else:
            return "{}: {}".format(self.logging_tag, message)

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
        # list of lists of strings. One string out of each list of strings must be in the filename
        filter_stringss = []
        if self.import_jammed:
            filter_stringss.append(['xyzdr', 'xydr'])
        else:
            filter_stringss.append(['xyzd', 'xyd'])
        if self.packing_nrs is not None:
            filter_stringss.append([str(nr) + '.' for nr in self.packing_nrs])
        for fname in os.listdir(self.packings_dir):
            if all([any([filter_str in fname for filter_str in filter_strings])
                    for filter_strings in filter_stringss]):
                logging.debug("")
                logging.info(self._log(fname))
                success = self.one_iteration(fname)
                successes.append((fname, success))
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
    def __init__(self, target_packing_frac=0.7, tol=1e-9, packings_dir='packings',
                 packing_nrs=None, import_jammed=False, outdir='jammed_packings',
                 use_cell_lists=False, show=False, opt_pot_str='hs_wca',
                 override_pot_kwargs=None, minimizer="fire", logging_tag=""):
        super(HS_Generate_Jammed_Packing, self).__init__(target_packing_frac=target_packing_frac,
                                                         packings_dir=packings_dir,
                                                         packing_nrs=packing_nrs,
                                                         import_jammed=import_jammed,
                                                         outdir=outdir, minimizer=minimizer,
                                                         override_pot_kwargs=override_pot_kwargs,
                                                         logging_tag=logging_tag)
        self.opt_pot_str = opt_pot_str
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
            logging.warning(self._log("Max soft diameter >= 1/2 box side!"))
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

    def _find_rattlers(self):
        """
        finish this, I need to remove the rattler and break. Also need to get compare to existing jammed_packing option
        :return:
        """
        if self.bdim < 4:
            zmin = self.bdim + 1
        else:
            raise NotImplementedError

        check_again = range(len(self.hs_radii))
        self.ss_radii = self.hs_radii * (1. + self.sca)
        neighbour_indicess, neighbour_distancess = self.potential.getNeighbours(self.coords)
        nrattlers = 0
        if self.bdim != 2:
            origin = np.zeros(self.bdim)
        while len(check_again) > 0:
            check_inds = list(check_again)
            check_again = []
            if nrattlers > self.max_nrattlers:
                logging.warning(self._log("Too many rattlers. Discarding packing."))
                return False
            for atomi in check_inds:
                found_rattler = False
                no_neighbors = len(neighbour_indicess[atomi])
                if no_neighbors < zmin:
                    found_rattler = True
                    logging.debug(self._log("Particle {} is not isostatic.".format(atomi)))
                else:
                    if self.bdim == 2:
                        found_rattler = not origin_in_hull_2d(neighbour_distancess[atomi])
                    else:
                        points = np.asarray(neighbour_distancess[atomi]).reshape((-1,self.bdim))
                        found_rattler = not in_hull(origin, points)
                    if found_rattler:
                        logging.debug(self._log("Particle {} is not in contacts' convex hull.".format(atomi)))
                self.rattlers[atomi] = 0 if found_rattler else 1000
                self.rattlers_draw[atomi] = float(not found_rattler)
                if found_rattler:
                    if atomi in check_again:
                        check_again.remove(atomi)
                    for atomj in neighbour_indicess[atomi]:
                        check_again.append(atomj)
                        i_in_j = neighbour_indicess[atomj].index(atomi)
                        del neighbour_indicess[atomj][i_in_j]
                        del neighbour_distancess[atomj][i_in_j]
                    nrattlers += 1

        # test that number of contacts is sufficient for bulk modulus to be positive,
        # see eq 4 in http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.095704
        # see eq 19 in arXiv:1406.1529
        no_stable = self.nparticles - nrattlers
        total_contacts = sum([len(neighbour_indices) for neighbour_indices in neighbour_indicess])
        N_min = int(2*(self.bdim * (no_stable - 1) + 1))
        logging.debug(self._log("N_min: {} total_contacts: {}".format(N_min, total_contacts)))
        logging.debug(self._log("Number of rattlers: {}".format(nrattlers)))
        if total_contacts >= N_min:
            return True
        else:
            logging.warning(self._log("Packing is not globally stable, N_min: {} "
                                      "total_contacts: {}".format(N_min, total_contacts)))
            return False


    def _generate_packing_coords(self):
        """
        perform quench and run tests
        """
        success = self._generate_packing_coords_iteration(tol=self.tol)
        return success

    def _generate_packing_coords_iteration(self, tol=1e-9, iprint=-1):
        """quenches the imported structure"""

        #asserts that none of the hard sphere is overlapping before quenching
        if __debug__:
            no_overlap = self._check_no_overlaps()
            if not no_overlap:
                logging.warning(self._log("Overlap found before quenching"))
                return False

        if self.minimizer == "fire":
            fire_maxstep = np.amin(self.hs_radii)*self.sca
            res = modifiedfire_cpp(self.coords, self.potential, maxstep=fire_maxstep,
                                   nsteps=1e6, tol=tol, iprint=iprint)
        elif self.minimizer == "cg":
            optimizer = CGDescent(self.coords, self.potential, tol=tol,
                                  nsteps=1e6, print_level=iprint)
            res = optimizer.run()
        elif self.minimizer == "lbfgs":
            res = lbfgs_cpp(self.coords, self.potential, tol=tol, nsteps=1e6,
                            iprint=iprint)
        else:
            raise NotImplementedError

        if not res.success:
            logging.warning(self._log("Quench failed"))
            return False

        self.coords = res.coords
        self.energy = res.energy

        #test that on re-minimisation the structure does not change
        if __debug__:
            if self.minimizer == "fire":
                res2 = modifiedfire_cpp(self.coords, self.potential,
                                        maxstep=fire_maxstep, nsteps=1e6, tol=tol)
            elif self.minimizer == "cg":
                optimizer = CGDescent(self.coords, self.potential, tol=tol,
                                      nsteps=1e6, print_level=iprint)
                res2 = optimizer.run()
            elif self.minimizer == "lbfgs":
                res2 = lbfgs_cpp(self.coords, self.potential, tol=tol, nsteps=1e6,
                                iprint=iprint)
            else:
                raise NotImplementedError
            if res2.nfev > 1:
                logging.warning(self._log("Quench failed (structure changed at second minimisation)"))
                return False

        #asserts that none of the hard sphere is overlapping
        if __debug__:
            no_overlap = self._check_no_overlaps()
            if not no_overlap:
                logging.warning(self._log("Overlap found after quenching"))
                return False

        return self._find_rattlers()

    def _get_particles_volume(self):
        """returns volume of n=self.bdim dimensional sphere"""
        volumes = volume_nball(self.hs_radii,self.bdim)
        vtot = np.sum(volumes)
        return vtot

    def _import_packing_configuration(self, fname):
        path = os.path.join(self.packings_dir, fname)
        packing = import_packing(path, self.import_jammed, self.bdim)
        self.coords = packing['coords']
        self.hs_radii = packing['hs_radii']
        self._compute_sca()

    def _compute_sca(self):
        ##test##
        vol_part = self._get_particles_volume()
        vol_box = np.prod(self.boxv)
        phi = vol_part/vol_box #instanteneous pack frac
        assert(phi - self.packing_frac < 1e-4)
        ##endtest##
        ###r_soft = r_hs*(1+sca)
        self.sca = np.power(self.target_packing_frac/self.packing_frac,1./self.bdim) - 1

    def _check_no_overlaps(self):
        """check that no two particles are overlapping (using nearest image convention)"""
        no_overlap = True
        for i in xrange(self.nparticles):
            if no_overlap == True:
                for j in xrange(self.nparticles):
                    if i != j:
                        dij = np.linalg.norm(calc_distance(
                            self.coords[i * self.bdim : (i + 1) * self.bdim],
                            self.coords[j * self.bdim : (j + 1) * self.bdim],
                            self.bdim, self.distance_method, self.boxv, self.pot_kwargs))
                        dmin = self.hs_radii[i]+self.hs_radii[j]
                        if dij - dmin <= 0:
                            logging.warning(self._log("Invalid configuration"))
                            logging.warning(self._log("Atoms {} {} are overlapping".format(i,j)))
                            logging.warning(self._log("Real distance {}".format(dij)))
                            logging.warning(self._log("Min distance {}".format(dmin)))
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
    parser.add_argument("--nocell", action='store_true', help="don't use cell lists, "
                        "default: False",default=False)
    parser.add_argument("--packingsdir", type=str, help="name of directory with packings, "
                        "must be in cwd", default='packings')
    parser.add_argument("--packing-nrs", type=int, nargs='*', help="Restrict the "
                        "packings to jam by a list of packing numbers.", default=None)
    parser.add_argument("--import-jammed", action='store_true', help="Take a jammed packing as input "
                        "instead of an unjammed one.", default=False)
    parser.add_argument("-o", "--outdir", type=str, help="Directory to save jammed packings in. "
                        "Default: 'jammed_packings'", default='jammed_packings')
    parser.add_argument("--show", action='store_true', help="show histograms", default=False)
    parser.add_argument("-t", "--tol", type=float, help="rms tolerance of the minimizer", default=1e-9)
    parser.add_argument("--minimizer", type=str, help="Energy minimization algorithm "
                        "used for quenching. Options: 'cg', 'fire', 'lbfgs'. "
                        "Default: 'fire'", default='fire')
    # potential arguments
    parser.add_argument("--opt_pot", type=str, help="optmizer's potential, 1) (default) hs_wca "
                                                    "2) inverse_power_stillinger", default='hs_wca')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.INFO)
    logging.info(args)

    # potential type
    opt_pot_str = args.opt_pot
    override_pot_kwargs = dict()
    if opt_pot_str.lower() == 'hs_wca':
        pass
    elif opt_pot_str.lower() == 'inverse_power_stillinger':
        override_pot_kwargs.update(pow=8, rcut=4.5)
        logging.info("Setting inverse_power_stillinger parameters: {}".format(override_pot_kwargs))
    else:
        raise NotImplementedError

    sim = HS_Generate_Jammed_Packing(target_packing_frac=args.density,
                                     packings_dir=args.packingsdir,
                                     packing_nrs=args.packing_nrs,
                                     import_jammed=args.import_jammed,
                                     outdir=args.outdir, tol=args.tol,
                                     use_cell_lists=not args.nocell, show=args.show,
                                     opt_pot_str=args.opt_pot, minimizer=args.minimizer,
                                     override_pot_kwargs=override_pot_kwargs)
    sim.run()
