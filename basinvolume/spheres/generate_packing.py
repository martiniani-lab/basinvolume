from __future__ import division
import numpy as np
import abc
import os
from basinvolume.spheres import HS_MCrunner, HS_MCrunnerOptDiffusion
from pele.potentials import HS_WCA
from pele.optimize._quench import lbfgs_cpp
from basinvolume.utils import trymakedir, get_git_version, get_python_version, get_cython_version, cround
from basinvolume.utils import volume_nball, read_xyd, read_xyzd, put_in_box
from numpy.random import RandomState
from mcpele.monte_carlo import NullPotential
import argparse

class _Generate_Packing(object):
    """
    this is an abstract class that implements the basic components of a generate packing class,
    and declares a number of abstract methods which should be implemented in all inheriting classes
    *method to generate packing, this could be for example direct sampling,
    sequential sampling,quench or LSA
    *nparticles: number of particles
    *bdim: dimensionality of the box
    *ndim: dimensionality of the problem (i.e. size of the coordinates array)
    *packing_frac: target packing fraction
    *boxv and boxl: note that in this implementation we aim to set the particle size and
    rescale the size of the box containing the particles to meet the target packing_fraction.
    Therefore it might not be entirely obvious why one should set a boxlengths vector. The reason is that
    one might not want a cubic box. In that case one sets boxv to have different relative rations,
    for instance if one want a parallelepiped. The rescaling maintains these relative ratios while
    meeting the target packing fraction.
    **boxv: an array of size bdim that contains the vectors defining the box
    **boxl: box side length, this is converted by the the class to a boxv array
    *single defines whether we should take single particle steps
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, nparticles, bdim=3, boxv=None, packing_frac=0.4, max_iter=1, use_cell_lists=False,
                 start_iteration=0):
        assert bdim==2 or bdim==3, "bdim={} not implemented".format(bdim)
        self.nparticles = nparticles
        self.bdim = bdim
        self.ndof = self.nparticles * self.bdim
        if boxv is None:
            self.boxv = np.array([1.0 for _ in xrange(self.bdim)],dtype='d')
        else:
            assert(len(boxv) == self.bdim)
            self.boxv = np.array(boxv,dtype='d')
        self.packing_frac = packing_frac
        self.base_directory = os.path.join(os.getcwd(),'packings')
        self.use_cell_lists = use_cell_lists
        self.start_iteration = start_iteration
        self.iteration = start_iteration
        self.max_iter = max_iter
        self.box_resized = False
        self.initialised = False

        ##constants#
        self.eps = 1. #energy unit
        ############

    @abc.abstractmethod
    def _initialise(self):
        """initialisation function"""

    @abc.abstractmethod
    def _get_particles_volume(self):
        """returns the total volume of the particles"""

    @abc.abstractmethod
    def _generate_packing_coords(self):
        """function that generates the packing"""

    @abc.abstractmethod
    def _write_opengl_input(self):
        """writes a opengl input file"""

    @abc.abstractmethod
    def _dump_configuration(self):
        """writes a configuration file, e.g .xyzd"""

    def _resize_box(self):
        """adjust the box size to meet the target packing fraction"""
        vol_part = self._get_particles_volume()
        vol_box = np.prod(self.boxv)
        phi = vol_part/vol_box #instanteneous pack frac
        a = np.power(phi/self.packing_frac,1./self.bdim)
        self.boxv *= a
        ###test###
        vol_box = np.prod(self.boxv)
        phi = vol_part/vol_box
        assert(phi - self.packing_frac < 1e-4)
        ##endtest##
        self.box_resized = True

    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)

    def _print_parameters(self):
        """writes the simulation parameters"""
        fname = '{}/packing{}.config'.format(self.base_directory, self.iteration)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Generate_Packings base class input parameters\n')
        f.write('[PACKING]\n')
        for key, value in self.seeds.iteritems() :
            f.write('{}: {}\n'.format(key,value))
        f.write('method: {}\n'.format(self.method))
        f.write('nparticles: {}\n'.format(self.nparticles))
        f.write('packing_fraction: {:.16f}\n'.format(self.packing_frac))
        f.write('boxdim: {}\n'.format(self.bdim))
        f.write('ndim: {}\n'.format(self.ndof))
        f.write('max_iter: {}\n'.format(self.max_iter))
        assert(self.box_resized)
        f.write('boxv: ')
        for val in self.boxv:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        #print software version
        f.write('[CODEVERSION]\n')
        f.write('basinvolume_version: {}\n'.format(get_git_version('basinvolume')))
        f.write('mcpele_version: {}\n'.format(get_git_version('mcpele')))
        f.write('pele_version: {}\n'.format(get_git_version('pele')))
        f.write('python_version: {}\n'.format(get_python_version()))
        f.write('cython_version: {}\n'.format(get_cython_version()))
        f.close()

    def _print(self):
        """dump configuration and opengl input to packings directory"""
        self._print_parameters()
        self._dump_configuration()
        self._write_opengl_input()

    def one_iteration(self):
        """perform one iteration"""
        self._initialise()
        success = self._generate_packing_coords()
        if success:
            self._print()
            self.iteration+=1
        print 'iteration ',self.iteration

    def run(self):
        """run generate packings"""
        while (self.iteration-self.start_iteration) < self.max_iter:
            self.one_iteration()


class HS_Generate_Packing(_Generate_Packing):
    """
    hs_radii: array
        array with the radii of the particles, if none sample particle sizes from a normal distribution
    mu: float
        average particle size, passable to normal distribution
    sig: float
        % standard deviaton of normal distribution from which to sample particles (this value is multiplied by the mean mu)
    sca: float
        determines % by which the hs is inflated
    eps: float
        LJ interaction energy of WCA part of the HS potential, here irrelevant because 'sca' is set to 0
    hsf:
        stands for hard sphere fluid
    new_poly : bool
        set to true to resample polidispersity at each new iteration
    seeds: array
        set seed to something other than none to remove randomness between instances of the class
    """
    def __init__(self, nparticles, method='quench', bdim=3, boxv=None, packing_frac=0.4, hs_radii=None,
                 mu = 1, sig = 0.1, new_poly=False, hsf_niter=1e6, hsf_stepsize = 1e-3, max_iter = 10,
                 use_cell_lists=False, single=False, seeds=None, start_iteration=0,
                 use_leesedwards=False, shear=0.0):
        super(HS_Generate_Packing,self).__init__(nparticles, bdim=bdim, boxv = boxv,
                                                 packing_frac=packing_frac, max_iter=max_iter,
                                                 use_cell_lists=use_cell_lists, start_iteration=start_iteration)
        self.method = method
        self.new_poly = new_poly
        assert not (self.method == 'quench' and self.new_poly is True)
        #give a random seed to random state or assign passed seed
        self.rng = RandomState()
        if seeds:
            assert('seed_takestep' in seeds and 'seed_generate_packing' in seeds and 'seed_swap' in seeds
                   and 'seed_probability_step_pattern' in seeds)
            self.seeds = seeds
        else:
            inf32 = np.iinfo(np.int32).max*2
            self.seeds = dict(seed_takestep=np.random.randint(0, inf32),
                              seed_swap=np.random.randint(0, inf32),
                              seed_generate_packing=np.random.randint(0, inf32),
                              seed_probability_step_pattern=np.random.randint(0, inf32))
        self.rng.seed(int(self.seeds['seed_generate_packing']))
        self.single = single
        self.sca = 0. #this must be 0 for hard spheres
        self.mu = mu
        self.sig = sig * mu
        self.hsf_niter = hsf_niter #number of iteration for each hs fluid configuration
        self.hsf_stepsize = hsf_stepsize
        self.hs_radii = hs_radii
        self.use_leesedwards = use_leesedwards
        self.shear = shear

    def _initialise(self):
        if self.initialised is False:
            self._sample_hs_radii(new_poly=False)
            self._resize_box()
            self.null_potential = NullPotential()
            if self.method == 'quench':
                #this is necessary to initialise the radii if using the quench routine
                self._initialise_coords_quench()
                rcut = np.amax(self.hs_radii) * 2.0 * (1.0 + self.sca) #rcut set to largest particle diameter
                if self.use_cell_lists:
                    if np.amin(self.boxv) // rcut <= 3:
                        self.use_cell_lists = False
                if self.use_cell_lists:
                    self.potential = HS_WCA(use_periodic=True,
                                     use_cell_lists=True, eps=self.eps,
                                     sca=self.sca, radii=self.hs_radii,
                                     boxvec=self.boxv,
                                     reference_coords=self.coords,
                                     rcut=rcut, ndim=self.bdim, ncellx_scale=1.0,
                                     use_leesedwards=self.use_leesedwards, shear=self.shear)
                else:
                    self.potential = HS_WCA(use_periodic=True, eps=self.eps,
                                     sca=self.sca, radii=self.hs_radii,
                                     boxvec=self.boxv, ndim=self.bdim,
                                     use_cell_lists=False,
                                     use_leesedwards=self.use_leesedwards, shear=self.shear)
            else:
                self._initialise_coords_crystal()
            self._print_initialise()
            self.initialised = True
        elif self.method != 'quench' and self.new_poly:
            self._sample_hs_radii(new_poly=self.new_poly)
            self._resize_box()
            self._initialise_coords_crystal()

    def _get_particles_volume(self):
        """returns volume of n=self.bdim dimensional sphere"""
        volumes = volume_nball(self.hs_radii,self.bdim)
        vtot = np.sum(volumes)
        return vtot

#    def _rescale_radii(self):
#        """rescale radii to meet target packing fraction"""
#        vol_box = np.power(self.boxl,self.bdim)
#        vol_part = np.sum(volume_nball(self.hs_radii,self.bdim))
#        phi = vol_part/vol_box
#        self.hs_radii *= np.power(self.packing_frac/phi,1/self.bdim)
#        #test
#        vol_part = np.sum(volume_nball(self.hs_radii,self.bdim))
#        phi = vol_part/vol_box
#        assert(phi - self.packing_frac < 1e-4)
#        #endtest

    def _sample_hs_radii(self, new_poly=False):
        if (self.hs_radii is None or new_poly) and self.sig > 1e-8:
            print "sampling hs_radii"
            self.hs_radii = self.rng.normal(self.mu,self.sig,self.nparticles)
        elif (self.hs_radii is None or new_poly) and self.sig <= 1e-8:
            print "sampling hs_radii, setting to ones because sig <= 1e-8"
            self.hs_radii = np.ones(self.nparticles)*self.mu
        else:
            self.hs_radii = np.array(self.hs_radii,dtype='d')
        assert(np.all(self.hs_radii > 0))

#    def _sample_hs_radii_from_area(self):
#        if self.hs_radii is None:
#            areas = self.rng.normal(self.mu,self.sig,self.nparticles)
#            self.hs_radii = np.sqrt(areas)
#        else:
#            self.hs_radii = np.array(self.hs_radii,dtype='d')
#        assert(self.hs_radii.all() > 0)

    def _distance (self, ind1, ind2):
        if self.use_leesedwards:
            d12 = [0 , 0]

            # Calculate Lees-Edwards distance for x- and y-dimensions
            d12[0] = self.coords[ind1 * self.bdim] - self.coords[ind2 * self.bdim]
            d12[1] = self.coords[ind1 * self.bdim + 1] - self.coords[ind2 * self.bdim + 1]

            round_y = cround(d12[1] / self.boxv[1])
            tmp12 = [d12[0] - round_y * self.shear * self.boxv[0],
                     d12[1] - round_y * self.boxv[1]]

            d12[0] -= cround(d12[0] / self.boxv[0]) * self.boxv[0]
            tmp12[0] -= cround(tmp12[0] / self.boxv[0]) * self.boxv[0]

            if d12[0] ** 2 + d12[1] ** 2 > tmp12[0] ** 2 + tmp12[1] ** 2:
                d12[0] = tmp12[0]
                d12[1] = tmp12[1]

            # Calculate other dimensions
            for dim in xrange(2, self.bdim):
                # use distances to nearest image convention
                d12.append((self.coords[ind1 * self.bdim + dim] - self.coords[ind2 * self.bdim + dim]) -
                          cround((self.coords[ind1 * self.bdim + dim]
                                  - self.coords[ind2 * self.bdim + dim]) / self.boxv[dim]) * self.boxv[dim])

            # Sum up elements
            return np.sqrt(sum([x ** 2 for x in d12]))
        else:
            d12 = 0
            for dim in xrange(self.bdim):
                # use distances to nearest image convention
                d12 += np.square((self.coords[ind1 * self.bdim + dim] - self.coords[ind2 * self.bdim + dim]) -
                          cround((self.coords[ind1 * self.bdim + dim]
                                  - self.coords[ind2 * self.bdim + dim]) / self.boxv[dim]) * self.boxv[dim])
            return np.sqrt(d12)

    def _check_no_overlaps(self):
        """check that no two particles are overlapping (using nearest image convention)"""
        no_overlap = True
        for i in xrange(self.nparticles):
            if no_overlap == True:
                for j in xrange(i, self.nparticles):
                    dij = self._distance(i, j)
                    if i != j:
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

    def _sample_random_coords(self):
        """returns random coordinates for the particles uniformly distributed in the box"""
        coords =  np.empty(self.ndof)
        for i in xrange(self.nparticles):
            for j in xrange(self.bdim):
                coords[i*self.bdim+j] = (self.rng.rand())*self.boxv[j]
        return coords

    def _build_distance_matrix(self):
        distances = np.empty([self.nparticles,self.nparticles])
        for i in xrange(self.nparticles):
            for j in xrange(i,self.nparticles):
                dij = 0
                for k in xrange(self.bdim):
                    #use distances to closest image
                    dij += np.square((self.coords[i*self.bdim+k] - self.coords[j*self.bdim+k]) -
                                      cround((self.coords[i*self.bdim+k] - self.coords[j*self.bdim+k]) / self.boxv[k]) * self.boxv[k])
                distances[i,j] = np.sqrt(dij)
                if i != j:
                    distances[j,i] = distances[i,j]
        return distances

    def _generate_packing_coords(self):
        if self.method == 'quench':
            self._generate_packing_coords_quench()
        elif self.method == 'direct':
            self._generate_packing_coords_direct()
        else:
            self._generate_coords_crystal()
        success = self._check_no_overlaps()
        return success

    def _generate_packing_coords_quench(self):
        """do a MCMC walk using the quenched coordinates. Here we do not satisfy detailed balance and we set the number
        of steps over which the stepsize is adjusted equal to the total number of steps. The value of the temperature should
        not matter as these are hard spehres and the difference in energy between valid configurations is 0. We set it high
        to be on the safe side."""
        if (self.iteration == self.start_iteration):
            temperature = 1.0
            dif_mcrunner = HS_MCrunnerOptDiffusion(self.null_potential, self.coords, temperature,
                                                   self.hsf_stepsize, 1e9, self.hs_radii, self.boxv,
                                                   adjustf = 0.9, acceptance=0.15, adjustf_niter = 1e6,
                                                   single=self.single, seeds = self.seeds,
                                                   use_cell=self.use_cell_lists,
                                                   use_leesedwards=self.use_leesedwards,
                                                   shear=self.shear)
            dif_mcrunner.run()
            self.hsf_stepsize = dif_mcrunner.get_stepsize()
            hsf_niter = dif_mcrunner.get_nr_decorrelation_steps()
            self.hsf_niter = max(hsf_niter, 2*self.nparticles)
            print "stepsize {} niter {}".format(self.hsf_stepsize, self.hsf_niter)
            self.coords, self.energy = dif_mcrunner.get_config()
            self.mcrunner = HS_MCrunner(self.null_potential, self.coords, temperature,
                                        self.hsf_stepsize, self.hsf_niter, self.hs_radii,
                                        self.boxv, adjustf = 0.9, acceptance=0.15,
                                        adjustf_niter = 0, single=self.single,
                                        seeds = self.seeds, use_cell=self.use_cell_lists,
                                        use_leesedwards=self.use_leesedwards, shear=self.shear)
        self.mcrunner.set_config(self.coords, self.energy)
        self.mcrunner.run()
        self.coords, self.energy = self.mcrunner.get_config()

    def _initialise_coords_quench(self):
        """
        it generates an initial set of coordinates from a LJ quench,
        the LJ particles are then substitued by HS based on the size of
        the gap
        coordinates are generated until a valid configuration is foun
        """
        #sigma =  min(self.boxv) / np.power(2,1./6) #set sigma such that the the wca radius is the same as the box smallest side length
        #pot = WCA(sig=sigma,boxvec=self.boxv,ndim=self.bdim) # choice of sigma might have to be different
        pot = HS_WCA(use_periodic=True, eps=self.eps, sca=0.05, radii=self.hs_radii, boxvec=self.boxv,
                     ndim=self.bdim, use_leesedwards=self.use_leesedwards, shear=self.shear)

        overlap = True
        while overlap == True:
            coords = self._sample_random_coords()
            res = lbfgs_cpp(coords, pot, nsteps=1e5, tol=1e-8)
            #res = lbfgs_cpp(coords,pot,nsteps=10000)
            #assert(res.success is True) #checks that a minimum configuration has been found
            self.coords = np.array(res.coords)
#            print "generated new start coords "
#            sort radii in cavity
#            self._sort_radii_in_cavities()
            #check that no two particles are overlapping (using nearest image convention)
            overlap = not self._check_no_overlaps()
            print "overlap",overlap

    def _generate_packing_coords_direct(self):
        """
        it generates an initial set of coordinates from a HSWCA quench,
        the HSWCA particles are then substitued by HS
        """
        pot = HS_WCA(use_periodic=True, eps=self.eps, sca=0.05, radii=self.hs_radii, boxvec=self.boxv,
                     ndim=self.bdim, use_cell_lists=True, use_leesedwards=self.use_leesedwards, shear=self.shear)
        overlap = True
        while overlap == True:
            coords = self._sample_random_coords()
            res = lbfgs_cpp(coords, pot, nsteps=1e4, tol=1e-5, iprint=0)
            self.coords = np.array(res.coords)
            #check that no two particles are overlapping (using nearest image convention)
            if res.success:
                overlap = not self._check_no_overlaps()
            else:
                overlap = True
            print "overlap",overlap

    def _initialise_coords_crystal(self):
        pass

    def _generate_coords_crystal(self):
        """
        place particles on a hegonal lattice
        """
        self.coords = np.ones(self.nparticles*self.bdim)
        if self.method == 'fcc':
            self._generate_coords_fcc_lattice()
        elif self.method == 'bcc':
            self._generate_coords_bcc_lattice_3d()
        else:
            raise Exception("_generate_coords_crystal: {} method not implemented".format(self.method))

    def _generate_coords_fcc_lattice(self):
        if self.bdim == 2:
            self._generate_coords_fcc_lattice_2d()
        elif self.bdim == 3:
            self._generate_coords_fcc_lattice_3d()
        #align centre of mass
        for i in xrange(self.bdim):
            self.coords[i::self.bdim] -= np.mean(self.coords[i::self.bdim])

    def _generate_coords_bcc_lattice(self):
        if self.bdim == 2:
            self._generate_coords_fcc_lattice_2d()
        elif self.bdim == 3:
            self._generate_coords_bcc_lattice_3d()
        #align centre of mass
        for i in xrange(self.bdim):
            self.coords[i::self.bdim] -= np.mean(self.coords[i::self.bdim])

    def _generate_coords_fcc_lattice_2d(self):
        """
        Put discs in triangular lattice.
        """
        n = int(np.power(self.nparticles/2,1./self.bdim))
        assert ( n - np.power(int(n),self.bdim)) < 1e-8, "Nparticles is not (N/2)^2"
        boxx = self.boxv[0]
        boxy = self.boxv[1]
        if boxx / boxy != 1:
            print "_generate_packing_coords_lattice_2d: warning: works best for aspect ratio unity"
        maximum_radius = np.amax(self.hs_radii)
        minimum_spacing_x = 2 * maximum_radius
        minimum_spacing_y = np.sqrt(3) * 0.5 * minimum_spacing_x
        LX = int(boxx / minimum_spacing_x)
        LY = int(boxy / minimum_spacing_y)
        max_placable_discs = LX * LY
        if self.nparticles > max_placable_discs:
            raise Exception("_generate_packing_coords_lattice_2d: discs can not be placed on lattice")
        while ((LX - 1) * (LY - 1)) >= self.nparticles:
            LX -= 1
            LY -= 1
        spacing_x = boxx / LX
        spacing_y = boxy / LY
        for i in xrange(self.nparticles):
            xi = self.bdim * i
            xint = i % LX
            yint = int(i / LX)
            self.coords[xi] = (xint + 0.5 * (yint % 2)) * spacing_x
            self.coords[xi + 1] = yint * spacing_y

    def _generate_coords_fcc_lattice_3d(self):
        """
        Put spheres in FCC lattice.
        See e.g. here: Frenkel and Smit: Understanding Molecular Simulation, page 252
        http://www.uic.edu/eng/ems/MEng/ChEME494/pdf/L8pt2.pdf
        """
        n = int(np.power(self.nparticles/4,1./self.bdim))
        assert ( n - np.power(int(n),self.bdim)) < 1e-8, "Nparticles is not (N/4)^3"
        #assuming that box is cubic
        L_cube = int((self.nparticles/4) ** (1/3))
        NX = L_cube
        NY = L_cube
        NZ = L_cube
        #print L_cube
        dx = self.boxv[0] / NX
        dy = self.boxv[1] / NY
        dz = self.boxv[2] / NZ
        d = [dx, dy, dz]
        if np.amax(self.hs_radii) > np.amax(d):
            raise Exception("_generate_packing_coords_lattice_3d: spheres can not be placed on lattice")
        coords=[]
        for iz in xrange(NZ):
            for iy in xrange(NY):
                for ix in xrange(NX):
                    coords.extend([ix*d[0],iy*d[1],iz*d[2]])
                    coords.extend([(ix+0.5)*d[0],(iy+0.5)*d[1],iz*d[2]])
                    coords.extend([ix*d[0],(iy+0.5)*d[1],(iz+0.5)*d[2]])
                    coords.extend([(ix+0.5)*d[0],iy*d[1],(iz+0.5)*d[2]])
        self.coords = np.array(coords)

    def _generate_coords_bcc_lattice_3d(self):
        """
        Put spheres in FCC lattice.
        See e.g. here: Frenkel and Smit: Understanding Molecular Simulation, page 252
        """
        n = int(np.power(self.nparticles/2,1./self.bdim))
        assert ( n - np.power(int(n),self.bdim)) < 1e-8, "Nparticles is not (N/2)^3"
        #assuming that box is cubic
        L_cube = int((self.nparticles/2) ** (1/3))
        NX = L_cube
        NY = L_cube
        NZ = L_cube
        print L_cube
        dx = self.boxv[0] / NX
        dy = self.boxv[1] / NY
        dz = self.boxv[2] / NZ
        d = [dx, dy, dz]
        if np.amax(self.hs_radii) > np.amax(d):
            raise Exception("_generate_packing_coords_lattice_3d: spheres can not be placed on lattice")
        coords=[]
        for iz in xrange(NZ):
            for iy in xrange(NY):
                for ix in xrange(NX):
                    coords.extend([ix*d[0],iy*d[1],iz*d[2]])
                    coords.extend([(ix+0.5)*d[0],(iy+0.5)*d[1],(iz+0.5)*d[2]])
        self.coords = np.array(coords)

#    def _initialise_coords_hcp_lattice_3d(self):
#        """
#        Put spheres in FCC lattice.
#        See e.g. here: Frenkel and Smit: Understanding Molecular Simulation, page 252
#        http://micro.stanford.edu/wiki/M02_Making_a_Perfect_Crystal
#        """
#        n = int(np.power(self.nparticles/4,1./self.bdim))
#        assert ( n - np.power(int(n),self.bdim)) < 1e-8, "Nparticles is not N^3/4"
#        #assuming that box is cubic
#        L_cube = int((self.nparticles/4) ** (1/3))
#        NX = L_cube
#        NY = L_cube
#        NZ = L_cube
#        print L_cube
#        dx = self.boxv[0] / NX
#        dy = self.boxv[1] / NY
#        dz = self.boxv[2] / NZ
#        d = [dx, dy, dz]
#        if np.amax(self.hs_radii) > np.amax(d):
#            raise Exception("_generate_packing_coords_lattice_3d: spheres can not be placed on lattice")
#        coords=[]
#        for iz in xrange(NZ):
#            for iy in xrange(NY):
#                for ix in xrange(NX):
#                    coords.extend([ix*d[0],iy*d[1],iz*d[2]])
#                    coords.extend([(ix+0.5)*d[0],(iy+0.5)*d[1],iz*d[2]])
#                    coords.extend([(ix+0.5),(iy+1./6)*d[1],(iz+0.5)*d[2]])
#                    coords.extend([ix*d[0],(iy+2/3)*d[1],(iz+0.5)*d[2]])
#        self.coords = np.array(coords)

#    def _initialise_coords_hcp_lattice_3d(self):
#        L_cube = int((self.nparticles/4) ** (1/3))
#        NX = L_cube
#        NY = L_cube
#        NZ = L_cube
#        print L_cube
#        a1 = (np.prod(self.boxv) / (NX * NY * NZ)) ** (1/3)
#        for iz in xrange(NZ):
#            for iy in xrange(NY):
#                for ix in xrange(NX):
#                    i = (ix + iy*NX + iz*NX*NY)*self.bdim
#                    self.coords[i] = (2*ix+((iy+iz)%2))*a1
#                    self.coords[i + 1] = (np.sqrt(3)*(iy+(iz%2)/3))*a1
#                    self.coords[i + 2] = (2*np.sqrt(6)*iz/3)*a1

    def _sort_radii_in_cavities(self):
        """
        this function sorts the radii according to the cavity sizes
        this is currently unused
        """
        #build a matrix with the distances between particles i and j
        distances = self._build_distance_matrix()
        #build an array with the weighted distance to neighbours, the shortest distance is 10 times heavier than the largest
        dmin = np.sort(distances,axis=1)

        if (self.nparticles > 8):
            neighbours = 8
        else:
            neighbours = self.nparticles-2

        CTE = np.exp( np.log(12) / (neighbours-1))
        weight = [CTE**i for i in xrange(neighbours)]
        weight = weight[::-1]
        weight.extend([0 for i in xrange(self.nparticles-neighbours)])
        #print 'weights',weight
        dmin = np.average(dmin,axis=1,weights=weight)
        #sort and return a map of indices in descending order
        dmap = np.argsort(dmin)[::-1]
        #order particle sizes so that they are associated to coordinates with appropriate gaps
        #print 'old radii',self.hs_radii
        hs_radii = np.zeros(self.nparticles)
        sorted_radii = np.sort(self.hs_radii)[::-1]
        for i in xrange(self.nparticles):
            hs_radii[dmap[i]] = sorted_radii[i]
        self.hs_radii = hs_radii.copy()
        #print 'new radii',self.hs_radii

    def _correct_coords(self):
        """this function returns the nearest images in the central box, useful for dumping the configurations"""
        coords = self.coords.copy()
        put_in_box(coords,self.boxv)
        return coords

    def _dump_configuration(self):
        """write coordinates to file .xyzd"""
        coords = self._correct_coords()
        directory = self.base_directory
        if self.bdim == 2:
            fname = "{0}/packing{1}.xyd".format(directory,self.iteration)
            f = open(fname,'w')
            for i in xrange(self.nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                                  self.hs_radii[i]*2))
        elif self.bdim == 3:
            fname = "{0}/packing{1}.xyzd".format(directory,self.iteration)
            f = open(fname,'w')
            for i in xrange(self.nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                               coords[i*self.bdim+2],self.hs_radii[i]*2))
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        f.close()

    def _write_opengl_input(self):
        """write opengl input file"""
        coords = self._correct_coords()
        boxv = self.boxv
        colour = 13
        directory = self.base_directory
        fname = "{0}/packing{1}.dat".format(directory,self.iteration)
        f = open(fname,'w')
        f.write('{}\n'.format(self.nparticles))

        if self.bdim == 2:
            f.write('{} {} {}\n'.format(-boxv[0]/2,-boxv[1]/2, -np.amax(self.hs_radii)))
            f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
            f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
            f.write('0.0 \t 0.0 \t {}\n'.format(np.amax(self.hs_radii)*2))
            for i in xrange(self.nparticles):
                for j in xrange(self.bdim):
                    f.write('{}\t'.format(coords[i*self.bdim+j]))
                f.write('{}\t'.format(0))
                f.write('{}\t'.format(self.hs_radii[i]*2))
                f.write('{}\n'.format(colour))
        elif self.bdim == 3:
            f.write('{} {} {}\n'.format(-boxv[0]/2,-boxv[1]/2,-boxv[2]/2))
            f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
            f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
            f.write('0.0 \t 0.0 \t {}\n'.format(boxv[2]))
            for i in xrange(self.nparticles):
                for j in xrange(self.bdim):
                    f.write('{}\t'.format(coords[i*self.bdim+j]))
                f.write('{}\t'.format(self.hs_radii[i]*2))
                f.write('{}\n'.format(colour))
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        f.close()

    def _print_parameters(self):
        """writes the simulation parameters"""
        fname = '{}/packing{}.config'.format(self.base_directory, self.iteration)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Generate_Packings base class input parameters\n')
        f.write('[PACKING]\n')
        for key, value in self.seeds.iteritems() :
            f.write('{}: {}\n'.format(key,value))
        f.write('method: {}\n'.format(self.method))
        f.write('nparticles: {}\n'.format(self.nparticles))
        f.write('packing_fraction: {}\n'.format(self.packing_frac))
        f.write('boxdim: {}\n'.format(self.bdim))
        f.write('ndim: {}\n'.format(self.ndof))
        f.write('radii_mean: {}\n'.format(self.mu))
        f.write('radii_stdev: {}\n'.format(self.sig))
        f.write('max_iter: {}\n'.format(self.max_iter))
        assert(self.box_resized)
        f.write('boxv: ')
        for val in self.boxv:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        #print software version
        f.write('[CODEVERSION]\n')
        f.write('basinvolume_version: {}\n'.format(get_git_version('basinvolume')))
        f.write('mcpele_version: {}\n'.format(get_git_version('mcpele')))
        f.write('pele_version: {}\n'.format(get_git_version('pele')))
        f.write('python_version: {}\n'.format(get_python_version()))
        f.write('cython_version: {}\n'.format(get_cython_version()))
        f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="generate 2/3-D hard disks/spheres packings")
    parser.add_argument("nparticles", type=int, help="number of particles")
    parser.add_argument("-n","--npackings", type=int, help="number of packings to produce",default=1)
    parser.add_argument("-d","--boxdim", type=int, help="box dimensions",default=3)
    parser.add_argument("-p","--density", type=float, help="target packing fraction",default=0.5)
    parser.add_argument("-u","--rmean", type=float, help="mean particle radius",default=1.0)
    parser.add_argument("-s","--rsigma", type=float, help="percent standard deviation",default=0.05)
    parser.add_argument("-m","--hsfniter", type=int, help="number of hard sphere fluid MC steps between 2 samples",default=1e6)
    parser.add_argument("-t","--hsfstep", type=float, help="stepsize for hard sphere fluid MC simulation",default=1e-3)
    parser.add_argument("-i", "--start-iter", type=int, help="starting label iteration, default=0", default=0)
    parser.add_argument("--newpoly", action='store_true', help="resample polidispersity at each iteration, default: False",default=False)
    parser.add_argument("--dpath", type=str, help="path to xy(z)d path from where to import diameters",default=None)
    parser.add_argument("--nocell", action='store_true', help="don't use cell lists, default: False",default=False)
    parser.add_argument("--moveall", action='store_true', help="move all particles at each step, default: False",default=False)
    parser.add_argument("--method", type=str, help="protocol to generate packings", default="quench")
    parser.add_argument("--shear", type=float, help="Use Lees-Edwards boundary conditions to simulate given amount of shear.", default=None)
    args = parser.parse_args()
    print args
    single = not args.moveall

    if args.shear == None:
        use_leesedwards = False
        shear = 0.0
    else:
        use_leesedwards = True
        shear = args.shear

    #import radii from other configuration file
    dpath = args.dpath
    hs_radii = None
    if dpath:
        if not os.path.isabs(args.dpath):
            dpath = os.path.abspath(dpath)
        if args.boxdim == 2:
            coords, hs_diameters = read_xyd(dpath)
        else:
            coords, hs_diameters = read_xyzd(dpath)
        hs_radii = hs_diameters/2

    sim = HS_Generate_Packing(args.nparticles, method=args.method, bdim=args.boxdim, packing_frac=args.density,
                              hs_radii=hs_radii, mu = args.rmean, sig = args.rsigma, new_poly=args.newpoly,
                              hsf_niter=args.hsfniter, hsf_stepsize = args.hsfstep, max_iter =args.npackings,
                              use_cell_lists=not args.nocell, single=single, start_iteration=args.start_iter, use_leesedwards=use_leesedwards, shear=shear)
    sim.run()
