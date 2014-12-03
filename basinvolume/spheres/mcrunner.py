from __future__ import print_function
import numpy as np
import sys
from pele.potentials import Harmonic, HS_WCA
from pele.optimize import ModifiedFireCPP
from pele.storage import Database
from pele.storage.database import Minimum
from mcpele.monte_carlo import _BaseMCRunner, RandomCoordsDisplacement
from mcpele.monte_carlo import MetropolisTest, CheckSphericalContainer 
from mcpele.monte_carlo import GaussianCoordsDisplacement
from mcpele.monte_carlo import ParticlePairSwap, TakeStepProbabilities
from basinvolume.monte_carlo import CheckSameMinimum, RecordDisp2Histogram
from basinvolume.monte_carlo import Findk
from basinvolume.monte_carlo import FindNrDecorrelationSteps
from basinvolume.monte_carlo import CheckOverlapPeriodic, CheckOverlapCartesian 
from basinvolume.monte_carlo import RecordDisplacementTimeseries
from basinvolume.monte_carlo import CheckOverlapCartesianCellLists
from basinvolume.monte_carlo import CheckOverlapPeriodicCellLists
from basinvolume.gui import HSWCASystem
from basinvolume.utils import reduce_coordinates, full_coordinates

#for plotting histogram
from itertools import cycle
from scipy.integrate import quad

try:
    import matplotlib.pyplot as plt
    #more stuff for plotting histogram and comparing to prediction
    #######################SET LATEX OPTIONS###################                            
    plt.rc('text', usetex=True)
    plt.rc('font',**{'family':'serif','serif':['Computer Modern']})
    #rc('text.latex',preamble=r'\usepackage{times}')                                       
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['xtick.major.pad'] = 8
    plt.rcParams['ytick.major.pad'] = 8
    ##########################################################                             
    ####SET COLOUR MAP######                                                               
    cm = plt.get_cmap('Dark2')
    ########################                                                               
    #####################LINE STYLE CYCLER####################                             
    lines = ["-","--","-."]
    linecycler = cycle(lines)
    color_cycle=[cm(1. * i / 6) for i in xrange(6)]
    ##########################################################
except ImportError as err:
    print(err)
    
def analytical_d2(x, k, N, boxdim=3):
    f = float(k * x) / 2
    g = float(boxdim * N - boxdim) / 2 - 1
    return np.exp(-f) * np.power(f, g)

vec_analytical_d2 = np.vectorize(analytical_d2)
#end: things for histogram

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

class HS_MCrunner(_BaseMCRunner):
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
       might cause a badalloc error, if trying to allocate a #potential = Harmonic(origin,k,boxv) set in _configure_bv_mcrunnerhuge array. If you are sampling unwanted extremely high or low energies
       then you might want to add a pele::EnergyWindow test that guarantees to keep you within a specific energy range and/or 
       make the stepsize larger or you might want to re-think about your simulation. Generally you shouldn't be 
       spanning energies that differ by several orders of magnitude, if that is the case, resizable or not resizable arrays are
       not the problem, you'd be incurring in memory issues no matter what you do, unless you write to disk at every iteration)
     * NOTE: some of the modules (e.g. take step and acceptance tests) require to be seeded. Users are free to do this as they think
     * is best, here we generate a random integer in [0,i32max) where i32max is the largest signed integer, for each seed. Each module
     * has a separate rng engine, therefore it's best if each receives a different randomly sampled seed
     * this class requires 1 seed for takestep
    """
    def __init__(self, potential, coords, temperature, stepsize, niter,
                  hs_radii, boxvec, acceptance=0.2, adjustf=0.9, adjustf_niter=1e4, 
                  adjustf_navg=100, single=False, seeds=None):
        #construct base class
        super(HS_MCrunner,self).__init__(potential, coords, temperature, niter)
        self.hs_radii = hs_radii
        self.boxv = boxvec
        self.bdim = len(boxvec)
        self.nparticles = len(hs_radii)
        
        #compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max),
                    seed_swap=np.random.randint(i32max),
                    seed_probability_step_pattern=np.random.randint(i32max))
        self.seeds=seeds
                
        #construct test/action classes  
        self.set_report_steps(adjustf_niter)
        self.takestep_displacement = RandomCoordsDisplacement(self.seeds['seed_takestep'], stepsize, 
                                                                     report_interval=adjustf_navg, factor=adjustf, 
                                                                     min_acc_ratio=0.2, max_acc_ratio=0.5,
                                                                     single=single, nparticles=self.nparticles, 
                                                                     bdim=self.bdim)
        self.takestep_particle_pair_swap = ParticlePairSwap(self.seeds['seed_swap'], self.nparticles)
        self.takestep = TakeStepProbabilities(self.seeds['seed_probability_step_pattern'])
        self.takestep.add_step(self.takestep_displacement, 1)
        self.takestep.add_step(self.takestep_particle_pair_swap, 1e-3)
        ##########################################
        #NOTE
        #should add an option to use cell lists, it shouldn't be the default behaviour
        rcut = np.amax(self.hs_radii) * 2
        if rcut < 0.5 * np.amin(boxvec):
            self.checkoverlap = CheckOverlapPeriodicCellLists(hs_radii, boxvec, rcut, use_frozen=False)
        else:
            self.checkoverlap = CheckOverlapPeriodic(hs_radii, boxvec)
        #set up pele:MC
        self.set_takestep(self.takestep)
        self.add_conf_test(self.checkoverlap)
        
    def set_control(self, T):
        """set temperature, canonical control parameter"""
        self.temperature = T
        self.set_temperature(T)
    
    def get_stepsize(self):
        return self.takestep_displacement.get_stepsize()

class HS_MCrunnerOptDiffusion(HS_MCrunner):
    """HS_MCrunnerOptDiffusion
    * this class requires 1 seed for takestep
    """
    def __init__(self, potential, coords, temperature, stepsize, niter,
                  hs_radii, boxvec, nr_samples_avergage=10, acceptance=0.2, 
                  adjustf=0.9, adjustf_niter=1e4, adjustf_navg=100, 
                  desired_mean_rsm_displ=None, single=False, seeds=None):
        #construct base class
        super(HS_MCrunnerOptDiffusion,self).__init__(potential, coords, temperature,
                                         stepsize, niter, hs_radii, boxvec, acceptance=acceptance, 
                                         adjustf=adjustf, adjustf_niter=adjustf_niter, 
                                         adjustf_navg=adjustf_navg, single=single, seeds=seeds)
        if not desired_mean_rsm_displ:
            desired_mean_rsm_displ = np.amax(self.hs_radii) * 2
        self.initial_stepsize = stepsize
        
        self.diffusion = FindNrDecorrelationSteps(desired_mean_rsm_displ, adjustf_niter, nr_samples_avergage,
                                                  coords, self.bdim)
        self.add_action(self.diffusion)
    
    def get_nr_decorrelation_steps(self):
        n = self.diffusion.get_nr_decorrelation_steps()
        return n
    
    def get_stepsize(self):
        stepsize = self.takestep_displacement.get_stepsize()
        #print("self.initial_stepsize:", self.initial_stepsize)
        #print("stepsize:", stepsize)
        #assert np.abs(self.initial_stepsize - stepsize) < 1e-10
        return stepsize
        
    
class BV_MCrunner(_BaseMCRunner):
    """
    Basin volume MC runner
    
    Parameters
    ----------
    potential : pele potential
        Harmonic potential used in the thermodynamic integration.
        These are the harmonic springs that tie each particle to its original
        position during the walk.
    coords : array
        Initial coordinates, can be the same as origin. These must be the full coordinates
    temperature : double
        Temperature is irrelevant here and probably set to unity.
    stepsize : double
        Initial stepsize of random_coords_displacement. This is adapted to
        match a desired acceptance ratio of steps before data is recorded.
    niter : integer
        Total number of MC steps.
    origin : array
        Coordinates of the minimum.
    hs_radii : array
        Radii of particle hard core radii for HS-WCA potential.
    boxvec : array
        List of box edge lengths.
    sca : double
        The thickness of the wca shell is sca * R where R is the hard core
        radius of the sphere.
    rattlers : array of bool
        Array of rattler status if degrees of freedom. If dof does not belong to
        rattler, 1, if dof does belong to rattler, 0.
    k : double
        Sping constant for harmonic potential.
    dtol : double
        Tolerance on the rms distance of the minimised structure to the origin.
    eps : double
        WCA parameter
    hmin : double
        Initial value of displ2 histogram lower bound.
    hmax : double
        Initial value of displ2 histogram upper bound.
    hbinsize : double
        Displ2 histogram bin size.
    acceptance : double
        Target step acceptance ratio.
    adjustf : double
        Factor for step size adaptation.
    adjustf_niter : integer
        Number of steps for step size adaptation.
    adjustf_navg : integer
        Number of steps from which to compute the step acceptance ratio during
        step size adaptation.
    pt_eq_niter : integer
        ?
    ts_niter : inteteger
        ?
    ts_freq : integer
        ?
    opt_dtmax : double
        DeltaT_max parameter of modified fire optimiser.
    opt_maxstep : double
        MaxStep parameter of modified fire optimiser.
    opt_tol : double
        Tolerance of modified fire optimiser.
    opt_nsteps : integer
        Numer of iterations of modified fire optimiser.
    perform_convergence_test : bool
        ?
    collect_minima_list : bool
        ?
    seeds : dict
        Seeds for random number generators.
    use_cell_lists : bool
        Flag indicating if cell lists are used.
    record_histogram : bool
        Flag indicating if Displ2 histogram is recorded and stored.
    single : bool
        Flag indicating if single particle moves are performed rather than global moves.
    use_periodic : bool
        Flag indicating if periodic boundary conditions are used.
    use_frozen : bool
        Flag indicating if there are frozen degrees of freedom.
    frozen_atoms : array
        List of labels of frozen (immobile) particles. Note: This is not the
        list of frozen degrees of freedom.
    rcontainer : double
        typically halfway between the outer and inner radius of the frozen shell
        forbids jumps outside out the frozen shell
    """
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 hs_radii, boxv, sca, rattlers=None, k=1.0, dtol=1e-3, eps=1.,
                 hmin=0, hmax=1, hbinsize=0.001, acceptance=0.2, adjustf=0.9,
                 adjustf_niter=1e4, adjustf_navg=100, pt_eq_niter=0,
                 ts_niter=None, ts_freq=1, opt_dtmax=1, opt_maxstep=0.5,
                 opt_tol=1e-4, opt_nsteps=1e5, perform_convergence_test=False,
                 collect_minima_list=False, seeds=None, use_cell_lists=True,
                 record_histogram=False, single=False, use_periodic=True,
                 use_frozen=False, frozen_atoms=None, rcontainer=None):
        #construct base class
        assert not (use_frozen and use_periodic)
        if use_frozen:
            assert not use_periodic and frozen_atoms is not None
            red_coords = reduce_coordinates(full_coords, frozen_atoms, len(boxv))
        else:
            red_coords = full_coords
        super(BV_MCrunner, self).__init__(potential, red_coords, temperature, niter)
        
        self.boxv = boxv
        self.bdim = len(boxv)
        self.origin = np.array(origin)
        self.red_origin = np.array(origin)
        self.hs_radii = np.array(hs_radii)
        self.red_radii = np.array(hs_radii)
        if use_frozen:            
            self.red_radii = np.delete(self.red_radii, frozen_atoms)
            self.red_origin = reduce_coordinates(self.red_origin, frozen_atoms, self.bdim)
            assert len(self.red_radii) == (len(self.hs_radii) - len(frozen_atoms))
            assert len(self.red_origin) == self.ndim
            assert rcontainer is not None
        self.sca = sca
        self.set_control(k)
        self.dtol = dtol
        self.eps = eps
        self.nparticles = len(self.red_radii)
        self.use_cell_lists = use_cell_lists
        self.use_frozen = use_frozen
        self.frozen_atoms = frozen_atoms
        self.use_periodic = use_periodic
        self.rcontainer = rcontainer
        self.equilibration_steps = adjustf_niter + pt_eq_niter
        if ts_niter is None:
            ts_niter = niter
        
        #manage array of rattlers, if not rattler: 1 -> jammed dof
        #                                          0 -> rattler dof 
        if (rattlers is None):
            self.rattlers = np.array([1. for _ in xrange(self.ndim)], dtype='d')
        else:
            self.rattlers = np.array(rattlers, dtype='d')
        if self.use_frozen:
            self.rattlers = reduce_coordinates(self.rattlers, frozen_atoms, self.bdim)
        assert(len(self.rattlers) == self.ndim)
        assert(self.rattlers.all() >= 0 and self.rattlers.all() <= 1)
                   
        #construct optimizer potential
        #rcut set to largest particle diameter
        self.rcut = np.amax(self.hs_radii) * 2.0 * (1.0 + self.sca)
        if self.use_cell_lists:
            if self.rcut > 0.5 * np.amin(self.boxv):
                print "warning: use_cell_lists flag was set, rcut is too large though"
                print "setting use_cell_lists to False"
                self.use_cell_lists = False
        self.ncellx_scale = 1.0
        self.pot_optimizer = HS_WCA(use_periodic=self.use_periodic,
                             use_cell_lists=use_cell_lists,
                             use_frozen=use_frozen, eps=self.eps, sca=self.sca,
                             radii=self.hs_radii, boxvec=self.boxv,
                             reference_coords=self.origin, rcut=self.rcut,
                             ndim=self.bdim, ncellx_scale=self.ncellx_scale,
                             frozen_atoms=self.frozen_atoms)
        
        #construct gradient optimizer    
        self.optimizer = ModifiedFireCPP(self.start_coords, self.pot_optimizer,
                                         dtmax=opt_dtmax, maxstep=opt_maxstep,
                                         tol=opt_tol, nsteps=opt_nsteps)
        
        #compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max),
                    seed_metropolis=np.random.randint(i32max))
        self.seeds = seeds
        
        #construct test/action classes
        if record_histogram:
            self.binsize = hbinsize
            self.histogram = RecordDisp2Histogram(self.red_origin, self.rattlers, self.bdim, hmin, hmax,
                                                  self.binsize, self.equilibration_steps)
            self.add_action(self.histogram)
        
        if use_periodic:
            if use_cell_lists:
                self.conftest1 = CheckOverlapPeriodicCellLists(self.hs_radii,
                                 self.boxv, self.rcut, ncellx_scale=self.ncellx_scale,
                                 use_frozen=self.use_frozen, frozen_atoms=self.frozen_atoms,
                                 reference_coords=self.origin) 
            
            else:
                self.conftest1 = CheckOverlapPeriodic(self.hs_radii,
                                 self.boxv, use_frozen=self.use_frozen,
                                 reference_coords=self.origin,
                                 frozen_atoms=self.frozen_atoms)
        else: 
            if use_cell_lists:
                self.conftest1 = CheckOverlapCartesianCellLists(self.hs_radii,
                                 self.boxv, self.rcut, ncellx_scale=self.ncellx_scale,
                                 use_frozen=self.use_frozen,
                                 frozen_atoms=self.frozen_atoms,
                                 reference_coords=self.origin)
            else:
                self.conftest1 = CheckOverlapCartesian(self.hs_radii,
                                 self.bdim, use_frozen=self.use_frozen,
                                 reference_coords=self.origin,
                                 frozen_atoms=self.frozen_atoms)
            
        #CheckSameMinimum MUST have use_periodic=False
        self.conftest2 = CheckSameMinimum(self.optimizer, self.pot_optimizer, self.red_origin, self.red_radii, 
                                          self.rattlers, self.dtol, bdim=self.bdim,
                                          eqsteps=self.equilibration_steps,
                                          perform_convergence_test=perform_convergence_test, 
                                          collect_minima_list=collect_minima_list, use_periodic=False)
        self.time_series = RecordDisplacementTimeseries(self.red_origin, self.bdim, ts_niter, ts_freq)
        self.metropolis = MetropolisTest(self.seeds['seed_metropolis'])
        
        self.set_report_steps(adjustf_niter)
        self.takestep = RandomCoordsDisplacement(self.seeds['seed_takestep'], stepsize, report_interval=adjustf_navg,
                                                  factor=adjustf, min_acc_ratio=acceptance, max_acc_ratio=acceptance,
                                                  single=single, nparticles=self.nparticles, bdim=self.bdim)
        
        #set up pele:MC
        self.set_takestep(self.takestep)
        if self.use_frozen:
            self.conftest0 = CheckSphericalContainer(self.rcontainer, self.bdim)
            self.add_conf_test(self.conftest0)
        self.add_accept_test(self.metropolis) #metropolis uses the harmonic potential
        self.add_late_conf_test(self.conftest1)
        self.add_late_conf_test(self.conftest2) #conf_test will happen after accept test because it is much cheaper
        self.add_action(self.time_series)
        
    def set_control(self, c):
        """set temperature, canonical control parameter"""
        self.k = c
        self.potential.set_k(c)
        self.reset_energy()
    
    def get_stepsize(self):
        return self.takestep.get_stepsize()
    
    def dump_histogram(self, fname):
        """write histogram to fname"""
        Emin, Emax = self.histogram.get_bounds_val()
        histl = self.histogram.get_histogram()
        hist = np.array(histl)
        Energies, step = np.linspace(Emin, Emax, num=len(hist), endpoint=False, retstep=True)
        Energies += 0.5 * step
        assert(abs(step - self.binsize) < self.binsize / 100)
        np.savetxt(fname, np.column_stack((Energies,hist)), delimiter='\t')
        mean, variance = self.histogram.get_mean_variance()
        return mean, variance
    
    def dump_timeseries(self, fname, clear=True):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.time_series.get_time_series())
        np.savetxt(fname, timeseries)
        if clear:
            self.time_series.clear()
        return timeseries
    
    def check_convergence(self, nr_steps_to_check=10000, rel_std_threshold=0.05):
        return self.time_series.check_convergence(nr_steps_to_check=nr_steps_to_check,
                                                   rel_std_threshold=rel_std_threshold)
        
    def dump_minima_list(self, fname):
        """write minima list to pele database"""
        system = HSWCASystem(self.eps, self.sca, self.hs_radii, self.boxv, 
                             bdim=self.bdim, dtol=self.dtol, etol=1)
        db = system.create_database(fname)
        minima_dicts = []
        #add origin to database, with _id == 0, to make post processing possible
        #for origin: set count to zero, but it does not have meaning, since we are only recording minima when quench took us to neighbor
        #distance should be zero because it is distance to itself
        mindict0 = dict(energy=self.pot_optimizer.getEnergy(self.red_origin),
                   coords=self.origin, user_data=dict(count=0, distance=0))
        minima_dicts.append(mindict0)
        #add neighboring minima to database
        self.conftest2.dump_minima(minima_dicts)
        #add spring constant to user_data
        for m in minima_dicts:
            m['user_data'].update(k=self.k)
            if self.use_frozen:
                redcoords = m['coords']
                m['coords'] = full_coordinates(redcoords, self.origin,
                              self.frozen_atoms, self.bdim)
        assert(len(minima_dicts) == self.conftest2.ml_nr_distinct_minima() + 1)
        print(len(minima_dicts))
        db.engine.execute(Minimum.__table__.insert(), minima_dicts)
        db.session.commit()
        
    def show_histogram(self):
        hist = self.histogram.get_histogram()
        val = np.array([i * self.binsize for i in xrange(len(hist))]) + 0.5 * self.binsize
        plt.hist(val, weights=hist, bins=len(hist))
        plt.show()
    
    def show_histogram_kmax(self):
        """
        shows the histogram against the analytical curve when k=kmax
        this function is useful for testing
        """
        hist = self.histogram.get_histogram()
        val = np.array([i * self.binsize for i in xrange(len(hist))]) + 0.5 * self.binsize
        n, bins, patches = plt.hist(val, weights=hist,bins=len(hist), normed=1,
                                    alpha=0.4, edgecolor=color_cycle[0], color=color_cycle[0])
        ###analytical
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        and2 = vec_analytical_d2(val,self.k, self.nparticles) / quad(vec_analytical_d2, bincenters[0], bincenters[-1], args=(self.k, self.nparticles))[0]
        plt.plot(bincenters, and2, linewidth=2.5, ls='--', color=color_cycle[-1])
        #plt.xlim(0,1)
        plt.xlabel(r'$|{\bf r}-{\bf r}_0|^2$')
        plt.ylabel(r'frequency $\times 10$')
        plt.tight_layout()
        plt.savefig('kmax_histogram.eps')
        plt.show()
        
class Findk_MCrunner(_BaseMCRunner):
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
    *avgcount is the number of steps over which the mean square displacement is
    *    averaged once kmax has been found
    *
    *stepsize
    *Etol: tolerance with which a minimised structure is accepted
     when compared to origin energy
    *dtol: tolerance on the rms displacement of the minimised structure
     with respect to the origin coordinates
    *ktarget: target acceptance associated to kmax
    *knavg: number of steps over findk averages the acceptance
    *ktol: when acceptance-ktarget<ktol the search for k terminates 
    * this class requires 1 seed
    """
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 hs_radii, boxv, sca, rattlers=None, avgcount=1e6, dtol=1e-3,
                 eps=1., ktarget = 0.75, knavg=500, ktol=0.05, opt_dtmax=1,
                 opt_maxstep=0.6, opt_tol=1e-4, opt_nsteps=1e5, hmin=0, hmax=1,
                 binsize=0.005, perform_convergence_test=False,
                 collect_minima_list=False, seeds=None, use_cell_lists=False,
                 single=False, use_periodic=True, use_frozen=False,
                 frozen_atoms=None, rcontainer=None):
        #construct base class
        assert not (use_frozen and use_periodic)
        if use_frozen:
            assert not use_periodic and frozen_atoms is not None
            red_coords = reduce_coordinates(full_coords, frozen_atoms, len(boxv))
        else:
            red_coords = full_coords
        super(Findk_MCrunner,self).__init__(potential, red_coords, temperature, niter)
        
        self.boxv = boxv
        self.bdim = len(boxv)
        self.origin = np.array(origin)
        self.red_origin = np.array(origin)
        self.hs_radii = np.array(hs_radii)
        self.red_radii = np.array(hs_radii)
        if use_frozen:            
            self.red_radii = np.delete(self.red_radii,frozen_atoms)
            self.red_origin = reduce_coordinates(self.red_origin, frozen_atoms, self.bdim)
            assert len(self.red_radii) == (len(self.hs_radii) - len(frozen_atoms))
            assert len(self.red_origin) == self.ndim
            assert rcontainer is not None
        self.sca = sca
        self.dtol = dtol
        self.eps = eps
        self.nparticles = len(self.red_radii)
        self.use_cell_lists = use_cell_lists
        self.use_frozen = use_frozen
        self.frozen_atoms = frozen_atoms
        self.use_periodic = use_periodic
        self.rcontainer = rcontainer
        
        #findk parameters
        self.avgcount = avgcount
        self.ktarget = ktarget
        self.knavg=knavg 
        self.ktol=ktol
        
        #manage array of rattlers, if not rattler: 1 -> jammed dof
        #                                          0 -> rattler dof 
        if (rattlers is None):
            self.rattlers = np.array([1. for _ in xrange(self.ndim)],dtype='d')
        else:
            self.rattlers = np.array(rattlers,dtype='d')
        if self.use_frozen:
            self.rattlers = reduce_coordinates(self.rattlers, self.frozen_atoms, self.bdim)
        assert(len(self.rattlers) == self.ndim)
        assert(self.rattlers.all() >= 0 and self.rattlers.all() <= 1)
        
        #construct optimizer potential
        #rcut set to largest particle diameter
        self.rcut = np.amax(self.hs_radii) * 2.0 * (1.0 + self.sca)
        if self.use_cell_lists:
            if self.rcut > 0.5 * np.amin(self.boxv):
                print "warning: use_cell_lists flag was set, but rcut is too large"
                print "setting use_cell_lists to False"
                self.use_cell_lists = False
        self.ncellx_scale = 1.0
        self.pot_optimizer = HS_WCA(use_periodic=self.use_periodic,
                             use_cell_lists=use_cell_lists,
                             use_frozen=use_frozen, eps=self.eps, sca=self.sca,
                             radii=self.hs_radii, boxvec=self.boxv,
                             reference_coords=self.origin, rcut=self.rcut,
                             ndim=self.bdim, ncellx_scale=self.ncellx_scale,
                             frozen_atoms=self.frozen_atoms)
        
        #construct gradient optimizer
        self.optimizer = ModifiedFireCPP(self.start_coords, self.pot_optimizer,
                         dtmax=opt_dtmax, maxstep=opt_maxstep, tol=opt_tol,
                         nsteps=opt_nsteps)
                
        #compute seeds
        #compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max))
        self.seeds=seeds
        
        #construct test/action classes      
        self.takestep = GaussianCoordsDisplacement(self.seeds['seed_takestep'], stepsize)
        
        if use_periodic:
            if use_cell_lists:
                self.conftest1 = CheckOverlapPeriodicCellLists(self.hs_radii,
                                 self.boxv, self.rcut, ncellx_scale=self.ncellx_scale,
                                 use_frozen=self.use_frozen, frozen_atoms=self.frozen_atoms,
                                 reference_coords=self.origin) 
            
            else:
                self.conftest1 = CheckOverlapPeriodic(self.hs_radii,
                                 self.boxv, use_frozen=self.use_frozen,
                                 reference_coords=self.origin,
                                 frozen_atoms=self.frozen_atoms)
        else: 
            if use_cell_lists:
                self.conftest1 = CheckOverlapCartesianCellLists(self.hs_radii,
                                 self.boxv, self.rcut, ncellx_scale=self.ncellx_scale,
                                 use_frozen=self.use_frozen,
                                 frozen_atoms=self.frozen_atoms,
                                 reference_coords=self.origin)
            else:
                self.conftest1 = CheckOverlapCartesian(self.hs_radii,
                                 self.bdim, use_frozen=self.use_frozen,
                                 reference_coords=self.origin,
                                 frozen_atoms=self.frozen_atoms)
        
        #CheckSameMinimum MUST have use_periodic=False
        self.conftest2 = CheckSameMinimum(self.optimizer, self.pot_optimizer, self.red_origin, self.red_radii, 
                                          self.rattlers, self.dtol, bdim = self.bdim,
                                          perform_convergence_test=perform_convergence_test, 
                                          collect_minima_list=collect_minima_list, use_periodic=False)
        self.hmin = hmin
        self.hmax = hmax
        self.binsize = binsize
        self.findk = Findk(self.red_origin, self.rattlers, self.bdim, self.avgcount, self.ktarget,
                           self.knavg, self.ktol, self.hmin, self.hmax, self.binsize)
        
        #set up pele:MC
        self.set_takestep(self.takestep)
        if self.use_frozen:
            self.conftest0 = CheckSphericalContainer(self.rcontainer, self.bdim)
            self.add_conf_test(self.conftest0)
        self.add_conf_test(self.conftest1)
        self.add_conf_test(self.conftest2)
        self.add_action(self.findk)
        #self.add_action(self.histogram)
        
    def set_control(self, c):
        """set k"""
        print("WARNING: findk set control is not defined, spring constant is set through stepsize", file=sys.stderr)
    
    def get_stepsize(self):
        return self.takestep.get_stepsize()
    
    def get_k(self):
        """in findk, potential is pretty much fictitious, k is adjusted through the stepsize"""
        stepsize = self.get_stepsize()
        k = 1.0 / (stepsize * stepsize)
        #k = self.bdim*len(self.hs_radii)/(stepsize*stepsize)##############
        return k
    
    def get_entries(self):
        return self.findk.get_entries()
    
    def dump_minima_list(self, fname):
        """write minima list to pele database"""
        system = HSWCASystem(self.eps, self.sca, self.hs_radii, self.boxv, 
                             bdim=self.bdim, dtol=self.dtol, etol=1)
        db = system.create_database(fname)
        minima_dicts = []
        #add origin to database, with _id == 0, to make post processing possible
        #for origin: set count to zero, but it does not have meaning, since we are only recording minima when quench took us to neighbor
        #distance should be zero because it is distance to itself
        mindict0 = dict(energy=self.pot_optimizer.getEnergy(self.red_origin),
                   coords=self.origin, user_data=dict(count=0, distance=0))
        minima_dicts.append(mindict0)
        #add neighboring minima to database
        self.conftest2.dump_minima(minima_dicts)
        #add spring constant to user_data
        for m in minima_dicts:
            m['user_data'].update(k=self.k)
            if self.use_frozen:
                redcoords = m['coords']
                m['coords'] = full_coordinates(redcoords, self.origin, self.frozen, self.bdim)
        assert(len(minima_dicts) == self.conftest2.ml_nr_distinct_minima() + 1)
        print(len(minima_dicts))
        db.engine.execute(Minimum.__table__.insert(), minima_dicts)
        db.session.commit()
        
    
    def show_histogram(self):
        """shows the histogram"""
        hist = self.findk.get_histogram()
        val = np.array([i * self.binsize for i in xrange(len(hist))]) + 0.5*self.binsize
        n, bins, patches = plt.hist(val, weights=hist,bins=len(hist), normed=1,
                                    alpha=0.4, edgecolor=color_cycle[0], color=color_cycle[0])
        ###analytical
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        and2 = vec_analytical_d2(val,self.get_k(), self.nparticles) / quad(vec_analytical_d2, bincenters[0], bincenters[-1], args=(self.get_k(), self.nparticles))[0]
        plt.plot(bincenters, and2, linewidth=2.5, ls='--', color=color_cycle[-1])
        #plt.xlim(0,1)
        plt.xlabel(r'$|{\bf r}-{\bf r}_0|^2$')
        plt.ylabel(r'frequency $\times 10$')
        plt.tight_layout()
        plt.savefig('findk_histogram.eps')
        plt.show()
    
if __name__ == "__main__":
    #to run harmonic potential go to tests
    
    from pele.utils.rotations import vector_random_uniform_hypersphere
    from pele.optimize._quench import modifiedfire_cpp
    import time
    
    nparticles = 1
    ndim = nparticles * 3
    origin = np.array([0, 0, 0], dtype='d')
    #build start configuration
    Emax = 0.1
    start_coords = vector_random_uniform_hypersphere(ndim) * np.sqrt(2 * Emax) #coordinates sampled from Pow(ndim)
    #Harmonic(origin,1)
    res = modifiedfire_cpp(start_coords,Harmonic(origin, 1))
    print(res)
    
#    print res.coords
    
    #Parallel Tempering
#   test = BV_MCrunner(start_coords, origin, temperature=1, k=1, niter=1e5, hEmin=0,hEmax=100,
#                       stepsize=0.5, adjustf = 0.9, adjustf_niter = 5000, radius=100)
    #test.set_control(1)
    start = time.time()
    #test.run()
    end = time.time()
    print(end - start)
    #test.show_histogram()
    
