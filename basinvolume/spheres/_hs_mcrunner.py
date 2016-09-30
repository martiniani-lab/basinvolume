from __future__ import print_function
import numpy as np
from mcpele.monte_carlo import _BaseMCRunner, RandomCoordsDisplacement
from mcpele.monte_carlo import ParticlePairSwap, TakeStepProbabilities
from basinvolume.monte_carlo import FindNrDecorrelationSteps
from basinvolume.monte_carlo import CheckOverlapPeriodic
from basinvolume.monte_carlo import CheckOverlapPeriodicCellLists

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
        self.takestep.add_step(self.takestep_displacement, 0.9)
        self.takestep.add_step(self.takestep_particle_pair_swap, 0.1) #1e-3
        ##########################################
        #NOTE
        #should add an option to use cell lists, it shouldn't be the default behaviour
        if np.amin(boxvec) // (2 * np.amax(hs_radii)) <= 3:
            self.checkoverlap = CheckOverlapPeriodic(hs_radii, boxvec)
        else:
            self.checkoverlap = CheckOverlapPeriodicCellLists(hs_radii, boxvec, use_frozen=False)
        #set up pele:MC
        self.set_takestep(self.takestep)
        self.add_conf_test(self.checkoverlap)

    def set_control(self, T):
        """set temperature, canonical control parameter"""
        self.temperature = T
        self.set_temperature(T)

    def get_stepsize(self):
        return self.takestep_displacement.get_stepsize()

    def get_status(self):
        """
        overloading the base class method to include stepsize
        """
        status = super(HS_MCrunner, self).get_status()
        status.stepsize = self.get_stepsize()
        return status

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