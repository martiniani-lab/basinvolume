from __future__ import division
import numpy as np
import abc
import os
import sys
from scipy.special import gamma
from mcrunner import HS_MCrunner, HS_MCrunnerOptDiffusion
from pele.potentials import HS_WCA, WCA, HS_WCAPeriodicCellLists
from pele.optimize._quench import lbfgs_cpp
from basinvolume.utils import *
from numpy.random import RandomState
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
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, method, nparticles, bdim=3, boxv = None, packing_frac=0.4, max_iter = 1, use_cell_lists=False,
                 seeds=None):
        self.method = method
        assert(bdim==2 or bdim==3) #currently PBC only implemented for 3d case
        self.nparticles = nparticles
        self.bdim = bdim
        self.ndim = self.nparticles * self.bdim
        if boxv is None:
            self.boxv = np.array([1.0 for _ in xrange(self.bdim)],dtype='d')
        else:
            assert(len(boxv) == self.bdim)
            self.boxv = np.array(boxv,dtype='d')
        self.packing_frac = packing_frac
        self.base_directory = os.path.join(os.getcwd(),'packings')
        self.use_cell_lists = use_cell_lists
        self.iteration = 0
        self.max_iter = max_iter
        self.box_resized = False
        self.initialised = False
        #give a random seed to random state or assign passed seed
        self.rng = RandomState()
        if seeds:
            assert('seed_takestep' in seeds and 'seed_generate_packing' in seeds)
            self.seeds = seeds
        else:
            self.seeds = dict(seed_takestep=np.random.randint(0, sys.maxint), 
                              seed_generate_packing=np.random.randint(0, sys.maxint))
        self.rng.seed(int(self.seeds['seed_generate_packing']))
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
        self._print_parameters()
    
    def _print_parameters(self):
        """writes the simulation parameters"""
        fname = '{}/packings.config'.format(self.base_directory)
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
        f.write('ndim: {}\n'.format(self.ndim))
        f.write('max_iter: {}\n'.format(self.max_iter))
        assert(self.box_resized)
        f.write('boxv: ')
        for val in self.boxv:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        f.close()
        
    def _print(self):
        """dump configuration and opengl input to packings directory"""
        self._dump_configuration()
        self._write_opengl_input()
    
    def one_iteration(self):
        """perform one iteration"""
        if self.initialised is not True:
            self._initialise()
        self._generate_packing_coords()
        self._print()
        self.iteration+=1
        print 'iteration ',self.iteration
        
    def run(self):
        """run generate packings"""
        while self.iteration < self.max_iter:
            self.one_iteration()
            

class HS_Generate_Packing(_Generate_Packing):
    """
    *PARAMETERS
    *hs_radii: array with the radii of the particles, if none sample particle sizes from a normal distribution
    *mu: average particle size, passable to normal distribution
    *sig: % standard deviaton of normal distribution from which to sample particles (this value is multiplied by the mean mu)
    *sca: determines % by which the hs is inflated
    *eps: LJ interaction energy of WCA part of the HS potential, here irrelevant because 'sca' is set to 0
    *hsf stands for hard sphere fluid
    *set seed to something other than none to remove randomness between instances of the class
    """    
    def __init__(self, nparticles, method='quench', bdim=3, boxv=None, packing_frac=0.4, hs_radii=None, 
                 mu = 1, sig = 0.2, hsf_niter=1e6, hsf_stepsize = 1e-3, max_iter = 10, use_cell_lists=False, 
                 seeds=None):
        super(HS_Generate_Packing,self).__init__(method, nparticles, bdim=bdim, boxv = boxv, 
                                                 packing_frac=packing_frac, max_iter = max_iter, 
                                                 use_cell_lists = use_cell_lists, seeds = seeds)
        
        self.sca = 0. #this must be 0 for hard spheres
        self.mu = mu
        self.sig = sig * mu
        self.hsf_niter = hsf_niter #number of iteration for each hs fluid configuration
        self.hsf_stepsize = hsf_stepsize
        self.hs_radii = hs_radii
        self._sample_hs_radii() #outcome of sample radii depends on hs_radii. hence if initialise is called twice, the second
                                #time it will not resample the radii, hence it must be kept in __init__
        self._resize_box()
                                    
    def _initialise(self):
        if self.method is 'quench':
            #this is necessary to initialise the radii if using the quench routine
            self._initialise_coords_quench()
        if self.use_cell_lists:
            rcut = np.amax(self.hs_radii) * 2.0 * (1.0 + self.sca) #rcut set to largest particle diameter
            print 'rcut', rcut
            self.potential = HS_WCAPeriodicCellLists(self.eps, self.sca, self.hs_radii, self.boxv, self.coords, 
                                                     rcut, ndim=self.bdim, ncellx_scale = 1.0, frozen_atoms = None)
        else:
            self.potential = HS_WCA(self.eps, self.sca, self.hs_radii, boxvec=self.boxv, ndim=self.bdim)
        self._print_initialise()
        self.initialised = True     
    
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
    
    def _sample_hs_radii(self):
        if self.hs_radii is None:
            self.hs_radii = self.rng.normal(self.mu,self.sig,self.nparticles)
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
    
    def _check_overlaps(self):
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
    
    def _sample_random_coords(self):
        """returns random coordinates for the particles uniformly distributed in the box"""
        coords =  np.empty(self.ndim)
        for i in xrange(self.nparticles):
            for j in xrange(self.bdim):
                coords[i*self.bdim+j] = (self.rng.rand())*self.boxv[j]
        return coords
    
    def _build_distance_matrix(self):
        distances = np.empty([self.nparticles,self.nparticles])
        for i in xrange(self.nparticles):
            for j in xrange(self.nparticles):
                dij = 0
                for k in xrange(self.bdim):
                    #use distances to closest image
                    dij += np.square((self.coords[i*self.bdim+k] - self.coords[j*self.bdim+k]) -
                                      cround((self.coords[i*self.bdim+k] - self.coords[j*self.bdim+k]) / self.boxv[k]) * self.boxv[k])
                distances[i,j] = np.sqrt(dij)
        return distances
    
    def _generate_packing_coords(self):
        if self.method is 'quench':
            self._generate_packing_coords_quench()
        elif self.method is 'direct':
            self._generate_packing_coords_direct()
    
    def _generate_packing_coords_quench(self):
        """do a MCMC walk using the quenched coordinates. Here we do not satisfy detailed balance and we set the number
        of steps over which the stepsize is adjusted equal to the total number of steps. The value of the temperature should
        not matter as these are hard spehres and the difference in energy between valid configurations is 0. We set it high
        to be on the safe side."""
        if (self.iteration == 0):
            temperature = 1.0
            dif_mcrunner = HS_MCrunnerOptDiffusion(self.potential, self.coords, temperature, self.hsf_stepsize, 1e8,
                                        self.hs_radii, self.boxv, adjustf = 0.9, acceptance=0.2, adjustf_niter = 50000,
                                        seeds = self.seeds)
            dif_mcrunner.run()
            self.hsf_stepsize = dif_mcrunner.get_stepsize()
            self.hsf_niter = dif_mcrunner.get_nr_decorrelation_steps()
            print "stepsize {} niter {}".format(self.hsf_stepsize, self.hsf_niter)
            self.coords, self.energy = dif_mcrunner.get_config()
            self.mcrunner = HS_MCrunner(self.potential, self.coords, temperature, self.hsf_stepsize, self.hsf_niter,
                                        self.hs_radii,self.boxv, adjustf = 0.9, acceptance=0.2, adjustf_niter = 0,
                                        seeds = self.seeds)
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
        no_overlap = False
        sigma =  min(self.boxv) / np.power(2,1./6) #set sigma such that the the wca radius is the same as the box smallest side length
        pot = WCA(sig=sigma,boxvec=self.boxv,ndim=self.bdim) # choice of sigma might have to be different
                
        while no_overlap == False:
            no_overlap = True             
            coords = self._sample_random_coords()
            res = lbfgs_cpp(coords,pot,nsteps=1000)
            #assert(res.success is True) #checks that a minimum configuration has been found
            self.coords = np.array(res.coords)
            print "generated new start coords "
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
            #check that no two particles are overlapping (using nearest image convention)
            no_overlap = self._check_overlaps()
             
    def _generate_packing_coords_direct(self):
        """
        it generates an initial set of coordinates from a LJ quench,
        the LJ particles are then substitued by HS based on the size of
        the gap 
        """
        no_overlap = False
        while no_overlap == False:
            no_overlap = True 
            self.coords = self._sample_random_coords()
            print "generated new7 start coords "
            #check that no two particles are overlapping (using nearest image convention)
            no_overlap = self._check_overlaps()
    
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
        else:
            fname = "{0}/packing{1}.xyzd".format(directory,self.iteration)
            f = open(fname,'w')
            for i in xrange(self.nparticles):
                f.write('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n'.format(coords[i*self.bdim],coords[i*self.bdim+1],
                                                               coords[i*self.bdim+2],self.hs_radii[i]*2))
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
        else:
            f.write('{} {} {}\n'.format(-boxv[0]/2,-boxv[1]/2,-boxv[2]/2))
            f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
            f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
            f.write('0.0 \t 0.0 \t {}\n'.format(boxv[2]))
            for i in xrange(self.nparticles):
                for j in xrange(self.bdim):
                    f.write('{}\t'.format(coords[i*self.bdim+j]))
                f.write('{}\t'.format(self.hs_radii[i]*2))
                f.write('{}\n'.format(colour))
        f.close()
        
    def _print_parameters(self):
        """writes the simulation parameters"""
        fname = '{}/packings.config'.format(self.base_directory)
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
        f.write('ndim: {}\n'.format(self.ndim))
        f.write('radii_mean: {}\n'.format(self.mu))
        f.write('radii_stdev: {}\n'.format(self.sig))
        f.write('max_iter: {}\n'.format(self.max_iter))
        assert(self.box_resized)
        f.write('boxv: ')
        for val in self.boxv:
            f.write('{:.16f} '.format(val))
        f.write('\n')
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
    parser.add_argument("-t","--hsfstep", type=float, help="stepsize for hard sphere fluid MC simulation",default=1e-4)
    parser.add_argument("--method", type=str, help="protocol to generate packings", default="quench")
    args = parser.parse_args()
    print args
    
    sim = HS_Generate_Packing(args.nparticles, method=args.method, bdim=args.boxdim, packing_frac=args.density,
                              mu = args.rmean, sig = args.rsigma, hsf_niter=args.hsfniter, hsf_stepsize = args.hsfstep, max_iter =args.npackings)
    sim.run()    
                
            
              
                
                
                
