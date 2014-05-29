from __future__ import division
import numpy as np
#from pele.utils.rotations import vector_random_uniform_hypersphere
from pele.potentials import Harmonic
import unittest
import logging
from mcpele.monte_carlo import _BaseMCRunner, RandomCoordsDisplacement, MetropolisTest 
from mcpele.monte_carlo import AdjustStep, RecordEnergyHistogram, CheckSphericalContainer
import pylab as plt
import copy

class Metropolis_MCrunner(_BaseMCRunner):
    """This class is derived from the _base_MCrunner abstract
     method and performs Metropolis Monte Carlo. This particular implementation of the algorithm: 
     * runs niter steps per run call 
     * takes steps by sampling a random vector in a n dimensional hypersphere (n is the number of coordinates);
     * adjust the step size for the first adjustf_niter steps (averaging the acceptance for adjust_navg steps
       and adjusting the stepsize by a factor of 'adjustf') to meet some target acceptance 'acceptance'.
     * configuration test: accept if within a spherical box of radius 'radius'
     * acceptance test: metropolis for some particular temperature
     * record energy histogram (the energy histogram is resizable, but the bounds are defined by hEmin and hEmax,
       furthermore the bin size is set with hbinsize. Care must be taken because the array is resizable, if the step size
       is small and extremely high or low energies are sampled the memory for the histogram will be reallocated and this 
       might cause a badalloc error, if trying to allocate a huge array. If you are sampling unwanted extremely high or low energies
       then you might want to add a pele::EnergyWindow test that guarantees to keep you within a specific energy range and/or 
       make the stepsize larger or you might want to re-think about your simulation. Generally you shouldn't be 
       spanning energies that differ by several orders of magnitude, if that is the case, resizable or not resizable arrays are
       not the problem, you'd be incurring in memory issues no matter what you do, unless you write to disk at every iteration)
     * NOTE: some of the modules (e.g. take step and acceptance tests) require to be seeded. Users are free to do this as they think
     * is best, here we generate a random integer in [0,i32max) where i32max is the largest signed integer, for each seed. Each module
     * has a separate rng engine, therefore it's best if each receives a different randomly sampled seed
    """
    def __init__(self, potential, coords, temperature, stepsize, niter, 
                 k=0, hEmin=0, hEmax=100, hbinsize=0.01, radius=2.5,
                 acceptance=0.5, adjustf=0.9, adjustf_niter = 1e4, adjustf_navg = 100, bdim=3):
        #construct base class
        super(Metropolis_MCrunner,self).__init__(potential, coords, temperature,
                                                  stepsize, niter)
                               
        #construct test/action classes       
        i32max = np.iinfo(np.int32).max
        
        self.set_control(k)
        self.binsize = hbinsize
        self.histogram = RecordEnergyHistogram(hEmin,hEmax,self.binsize, adjustf_niter)
        self.adjust_step = AdjustStep(acceptance, adjustf, adjustf_niter, adjustf_navg)
        self.step = RandomCoordsDisplacement(123)#np.random.randint(i32max)
        self.metropolis = MetropolisTest(123)
        self.conftest = CheckSphericalContainer(radius, bdim)
        
        #set up pele:MC
        self.set_takestep(self.step)
        self.add_accept_test(self.metropolis)
        self.add_conf_test(self.conftest)
        self.add_action(self.histogram)
        self.add_action(self.adjust_step)
        
    def set_control(self, c):
        """set temperature, canonical control parameter"""
        self.k = c
        self.potential.set_k(c)
        self.reset_energy()
    
    def dump_histogram(self, fname):
        """write histogram to fname"""
        Emin, Emax = self.histogram.get_bounds_val()
        histl = self.histogram.get_histogram()
        hist = np.array(histl)
        Energies, step = np.linspace(Emin,Emax,num=len(hist),endpoint=False,retstep=True)
        assert(abs(step - self.binsize) < self.binsize/100)
        np.savetxt(fname, np.column_stack((Energies,hist)), delimiter='\t')
        mean, variance = self.histogram.get_mean_variance()
        return mean, variance
    
    def get_histogram(self):
        """returns a energy list and a histogram list"""
        Emin, Emax = self.histogram.get_bounds_val()
        histl = self.histogram.get_histogram()
        hist = np.array(histl)
        Energies, step = np.linspace(Emin,Emax,num=len(hist),endpoint=False,retstep=True)
        assert(abs(step - self.binsize) < self.binsize/100)
        return Energies, hist
    
    def show_histogram(self):
        """shows the histogram"""
        hist = self.histogram.get_histogram()
        val = [i*self.binsize for i in xrange(len(hist))]
        plt.hist(val, weights=hist,bins=len(hist))
        plt.show()

class TestHarmonic(unittest.TestCase):
        
    def test_heat_capacity(self):
        nparticles=5
        bdim = 3
        self.ndim = nparticles*bdim
        self.k=0
        self.origin = np.zeros(self.ndim)
        self.Emax = 2 #this choice is fundamentally arbitrary, it's only used to generate the initial configuration
        #start_coords = vector_random_uniform_hypersphere(self.ndim) * np.sqrt(2*self.Emax) #coordinates sampled from Pow(ndim)
        
        start_coords = np.array([ 1.31598679, 0.06608363, -0.74418328, -0.15057428, 0.35884697, 0.04370566, -0.23210831,-0.28712944,\
                                  -0.52125984, 0.60719266, 0.09329589, 0.42409246,-0.03688934, 0.24029019, -0.63964171])
        
        temperatures = [0.2,0.27,0.362,0.487,0.65,0.88,1.18,1.6]
                
        #print potential.get_k()
        
        
        for T in temperatures:
            temperature=T
            stepsize=0.5
            niter=3e6
            potential = Harmonic(self.origin,0,bdim=bdim,com=True)
            #potential.set_k(0.0)
            mcrunner = Metropolis_MCrunner(potential, start_coords, temperature, stepsize, niter, hEmax = 100, adjustf = 0.9, 
                                           k=1.0,adjustf_niter = 10000, radius=10000000)
            #mcrunner.potential.get_k()
            
            #MCMC 
            mcrunner.run()
            
            #collect the results

            mean, variance = mcrunner.dump_histogram('histogram')
            
            cv = variance/T**2
            
            print cv
            self.assertLess(abs(cv-(self.ndim-3)/2.0),1e-1,'failed for temperature {}, cv = {}'.format(T,cv))

if __name__ == "__main__":
    logging.basicConfig(filename='Metropolis_mcrunner.log',level=logging.DEBUG)
    unittest.main()
