import numpy as np
import time
from pele.potentials import Harmonic
from mcpele.monte_carlo import _BaseMCRunner, RandomCoordsDisplacement, MetropolisTest 
from mcpele.monte_carlo import AdjustStep, GaussianCoordsDisplacement, CheckSphericalContainer
from basinvolume.monte_carlo import RecordDisp2Histogram, CheckHyperSphericalContainer, Findk

"""
pele::MCrunner

Specific implementations of MCrunners, generally they should follow this pattern:
* construct _base_MCrunner
* construct takestep, accept test, configuration test, action classes
* add these to the pele::MC class
* write a set_control function, for example you may want to set the temperature 
  (this is done this way to be compatible with the MPI replica exchange/parallel tempering
  implementation)
* add other functionalities that you may find desirable, e.g. dump histogram to file
"""

class ES_MCrunner(_BaseMCRunner):
    """Basin Volume MCrunner
    *coords: initial coordinates, can be the same as origin
    *origin: jammed minimised structure
    *hs_radii: array of the radii of the particles
    *boxv: array with the box size lengths
    *rattlers: array of rattlers, if not rattler: 1 -> jammed dof
                                                  0 -> rattler dof
    *k: spring constant
    *temperature
    *niter: number of MC takesteps to perform
    *stepsize
    *Etol: tolerance with which a minimised structure is accepted
     when compared to origin energy
    *dtol: tolerance on the rms displacement of the minimised structure
     with respect to the origin coordinates
    """
    def __init__(self, potential, coords, temperature, stepsize, niter,
                  origin, k=1.0, dtol=1e-3, eps=1., hmin=0, hmax=100, hbinsize=0.01, acceptance=0.2, 
                  adjustf=0.9, adjustf_niter = 1e4, adjustf_navg = 100, opt_dtmax=1, opt_maxstep=0.5, 
                  opt_tol=1e-4, opt_nsteps=1e5, hyperradius = 2.0):
        #construct base class
        #potential = Harmonic(origin,k,boxv) set in _configure_bv_mcrunner
        super(ES_MCrunner,self).__init__(potential, coords, temperature, stepsize, niter)
        
        self.origin = origin
        self.k = k
        self.dtol = dtol
        self.eps = eps
        
        #construct gradient optimizer
#        self.pot_optimizer = HS_WCA(self.eps, self.sca, self.hs_radii, boxvec=self.boxv)
#        self.optimizer = ModifiedFireCPP(self.start_coords, self.pot_optimizer, dtmax=opt_dtmax, maxstep=opt_maxstep, 
#                                         tol=opt_tol, nsteps=opt_nsteps)
                
        #construct test/action classes      
        i32max = np.iinfo(np.int32).max
        self.rattlers = np.array([1.0 for _ in xrange(self.ndim)])
        self.binsize = hbinsize
        self.histogram = RecordDisp2Histogram(self.origin, self.rattlers, hmin, hmax, self.binsize, adjustf_niter)
        self.conf = CheckHyperSphericalContainer(self.origin,hyperradius)
        #self.conftest = CheckSphericalContainer(1.0)
        self.adjust_step = AdjustStep(acceptance, adjustf, adjustf_niter, adjustf_navg)
        self.step = RandomCoordsDisplacement(self.ndim, np.random.randint(i32max))
        #self.step = GaussianCoordsDisplacement(self.ndim, np.random.randint(i32max))
        self.metropolis = MetropolisTest(np.random.randint(i32max))
        
        #set up pele:MC
        self.set_takestep(self.step)
        self.add_accept_test(self.metropolis)
        #self.add_conf_test(self.conftest)
        self.add_conf_test(self.conf)
        #self.add_late_conf_test(self.conftest) #conf_test will happen after accept test because it is much cheaper
        self.add_action(self.histogram)
        self.add_action(self.adjust_step)
        
    def set_control(self, c):
        """set temperature, canonical control parameter"""
        self.k = c
        self.potential.set_k(c)
    
    def dump_histogram(self, fname):
        """write histogram to fname"""
        Emin, Emax = self.histogram.get_bounds_val()
        histl = self.histogram.get_histogram()
        hist = np.array(histl)
        Energies, step = np.linspace(Emin,Emax,num=len(hist),endpoint=False,retstep=True)
        assert(abs(step - self.binsize) < self.binsize/100)
        np.savetxt(fname, np.column_stack((Energies,hist)), delimiter='\t')
        mean = self.histogram.get_mean()
        return mean
    
    def show_histogram(self):
        """shows the histogram"""
        import pylab as plt
        hist = self.histogram.get_histogram()
        val = [i*self.binsize for i in xrange(len(hist))]
        plt.hist(val, weights=hist,bins=len(hist))
        plt.show()

class ES_Findk_MCrunner(_BaseMCRunner):
    """Findk MCrunner
    *coords: initial coordinates, can be the same as origin
    *origin: jammed minimised structure
    *hs_radii: array of the radii of the particles
    *boxv: array with the box size lengths
    *rattlers: array of rattlers, if not rattler: 1 -> jammed dof
                                                  0 -> rattler dof
    *k: spring constant
    *temperature
    *niter: number of MC takesteps to perform
    *stepsize
    *Etol: tolerance with which a minimised structure is accepted
     when compared to origin energy
    *dtol: tolerance on the rms displacement of the minimised structure
     with respect to the origin coordinates
    *ktarget: target acceptance associated to kmax
    *kfactor: the factor by which k is decreased at each iteration, it must be in (0,1)
    *knavg: number of steps over findk averages the acceptance
    *ktol: when acceptance-ktarget<ktol the search for k terminates 
    """
    def __init__(self, potential, coords, temperature, stepsize, niter,
                  origin, dtol=1e-3, eps=1., k=1.0, ktarget = 0.75, kfactor=0.9, knavg=500, ktol=0.05, 
                  opt_dtmax=1, opt_maxstep=0.5, opt_tol=1e-4, opt_nsteps=1e5, hyperradius = 2.0):
        #construct base class
        super(ES_Findk_MCrunner,self).__init__(potential, coords, temperature, stepsize, niter)
        
        self.origin = origin
        self.k = k
        self.dtol = dtol
        self.eps = eps
        
        #findk parameters
        self.ktarget = ktarget
        self.kfactor=kfactor 
        self.knavg=knavg 
        self.ktol=ktol
        
        #construct test/action classes      
        i32max = np.iinfo(np.int32).max
        
        self.step = GaussianCoordsDisplacement(self.ndim, np.random.randint(i32max))
        self.conftest = CheckHyperSphericalContainer(self.origin,hyperradius)
        self.findk = Findk(self.origin, self.ktarget, self.kfactor, self.knavg, self.ktol)
                
        #set up pele:MC
        self.set_takestep(self.step)
        self.add_conf_test(self.conftest)
        self.add_action(self.findk)
        
    def set_control(self, c):
        """set k"""
        self.k = c
        self.potential.set_k(c)
    
    def get_k(self):
        """in findk, potential is pretty much fictitious, k is adjusted through the stepsize"""
        stepsize = self.get_stepsize()
        k = 1.0/(stepsize*stepsize)
        return k

  
if __name__ == "__main__":
    from basinvolume.post_processing import spring_constants_variable_transform as vt
    from basinvolume.post_processing import F_Basin_From_MC_Data, F_Basin_From_MC_Data_Free_COM, Gauss_Lobatto_abscissas
    from basinvolume.utils import log_volume_nball
    
    #SYSTEM PARAMETERS
    k0=0.0
    r = 2.0 #hyperradius
    n = 3   #number of particles along edge
    nr_particles=n*n*n
    dimension=3
    nr_points=8
        
    #SIMULATION PARAMETERS
    stepsize = 10.0
    niter = 1e6
    
    #===========================================================================
    # BUILD ORIGIN
    #===========================================================================
    
    assert(nr_particles>1)
    pos = np.linspace(-0.5,0.5,n)
    origin = []
    for x in pos:
        for y in pos:
            for z in pos:
                origin.append([x,y,z])
                
    origin = np.array(origin).flatten()

    #===========================================================================
    # POTENTIAL
    #===========================================================================
    
    potential = Harmonic(origin,k0,com=True)
        
    #===========================================================================
    # COMPUTE <U2> FOR K0 (required to compute karray)0
    #===========================================================================
       
    mcrunner = ES_MCrunner(potential, origin, 1.0, stepsize, niter, origin, k=k0, adjustf_niter=1e5, hmin=0, hmax=10, hbinsize=0.01,hyperradius=r)
    mcrunner.run()
    end=time.time()
    #print end-start
    status = mcrunner.get_status()
    print status
    print 'meanu2 k=0 and variance ',mcrunner.histogram.get_mean_variance()
    #mcrunner.show_histogram()
        
    displ_k_min, var_displ_k_min =mcrunner.histogram.get_mean_variance()
        
    #===========================================================================
    # FIND K_MAX
    #===========================================================================
    kstart = 100
    ktarget = 0.95
    
    mcrunner = ES_Findk_MCrunner(potential, origin, 1.0, stepsize, niter, origin, k=kstart, ktarget=ktarget, hyperradius=r)
    mcrunner.set_control(0) #potential is entirely fictitious, there is no energy test
    mcrunner.run()
    k_max = mcrunner.get_k()
    prob = mcrunner.findk.get_prob()
    print 'kmax ',k_max
    print 'prob ',prob
    #k_max= nr_particles*dimension/(r*r) = 20.25
    
    #===========================================================================
    # COMPUTE k ARRAY
    #===========================================================================
    
    karray = vt(nr_points, k_max, displ_k_min, nr_particles, dimension, k_min=0.0)
    
    #===========================================================================
    # COMPUTE <U2> FOR k ARRAY
    #===========================================================================
        
    meanu2 = []
    var_meanu2 = []
    for k in karray:
        mcrunner = ES_MCrunner(potential, origin, 1.0, stepsize, niter, origin, k=k, adjustf_niter=1e5, hmin=0, hmax=10, hbinsize=0.01,hyperradius=r)
        mcrunner.set_control(k)
        mcrunner.run()
        status = mcrunner.get_status()
        #print status
        #mcrunner.show_histogram()
        mean, var = mcrunner.histogram.get_mean_variance()
        meanu2.append(mean)
        var_meanu2.append(var)
    meanu2 = np.array(meanu2)
    var_meanu2 = np.array(var_meanu2)
    print meanu2
    print var_meanu2
        
    #===========================================================================
    # COMPUTE VOLUMES
    #===========================================================================
      
    prob = prob
    vol = 1.0
       
    #analytical meanu2
    karray = np.array(karray)
    meanu2_analytical = (karray + (nr_particles*dimension)/displ_k_min) / (nr_particles*dimension)
    meanu2_analytical = 1.0/meanu2_analytical
    
    F0, farray = F_Basin_From_MC_Data(dimension, nr_particles, karray, meanu2, vol, prob).get_free_energy_F0()
    aF0, afarray = F_Basin_From_MC_Data(dimension, nr_particles, karray, meanu2_analytical, vol, prob).get_free_energy_F0()
    
    print 'meanu2 corrected vol',F0 
    print 'meanu2 uncorrected vol', F_Basin_From_MC_Data_Free_COM(dimension, nr_particles, karray, meanu2, prob).get_free_energy_F0()[0]
    print 'analytical meanu2 corrected vol', aF0
    print 'analytical meanu2 uncorrected vol', F_Basin_From_MC_Data_Free_COM(dimension, nr_particles, karray, meanu2_analytical, prob).get_free_energy_F0()[0]
    print 'hypersphere vol',log_volume_nball(r,nr_particles*dimension)
    from pylab import *
    
    print len(farray)
    
    
    tarray = Gauss_Lobatto_abscissas(nr_points)()
    plot(tarray,farray)
    plot(tarray,afarray)
    show()
    
    