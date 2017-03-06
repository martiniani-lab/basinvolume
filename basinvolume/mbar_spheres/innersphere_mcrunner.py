from __future__ import print_function
import numpy as np
import sys
from pele.distance import Distance
from basinvolume.spheres import SpheresMCRunner
from basinvolume.monte_carlo import RecordDisplacementTimeseries
from basinvolume.monte_carlo import SampleUniformSphereGaussian
from basinvolume.enums import Minimizer

#for plotting histogram
from itertools import cycle

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

class BVInnerSphereMCrunner(SpheresMCRunner):
    """
    Basin volume Sphere MC runner

    Parameters
    ----------
    potential : pele potential
        this should be NullPotential
    coords : array
        Initial coordinates, can be the same as origin. These must be the full coordinates
    temperature : double
        Temperature is irrelevant here and probably set to unity.
    stepsize : double
        Initial stepsize of random_coords_displacement. This is set from
        the inferred spring constant stepsize = 1/sqrt(1/displ_k0)
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
    ts_niter : integer
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
    distance_method : Distance
        Specifies which distance method is used.
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
                 hs_radii, boxv, sca, rattlers=None, dtol=1e-3, eps=1.,
                 hmin=0, hmax=1, hbinsize=0.001,
                 ts_niter=None, ts_freq=1, opt_dtmax=1, opt_maxstep=0.5,
                 opt_tol=1e-5, opt_nsteps=1e5, perform_convergence_test=False,
                 collect_minima_list=False, seeds=None, use_cell_lists=True,
                 record_histogram=False, distance_method=Distance.PERIODIC,
                 use_frozen=False, frozen_atoms=None, rcontainer=None,
                 minimizer=Minimizer.FIRE, opt_pot_str='hs_wca', **extra_pot_kwargs):

        self.k = 1.0 / (stepsize * stepsize)
        # actions parameters
        if ts_niter is None:
            ts_niter = niter
        self.ts_niter = ts_niter
        self.ts_freq = ts_freq

        super(BVInnerSphereMCrunner, self).__init__(potential, full_coords, temperature, stepsize, niter, origin,
                                                    hs_radii, boxv, sca, rattlers=rattlers, k=self.k, dtol=dtol,
                                                    eps=eps, hmin=hmin, hmax=hmax, hbinsize=hbinsize,
                                                    report_steps=0, pt_eq_niter=0, opt_dtmax=opt_dtmax,
                                                    opt_maxstep=opt_maxstep, opt_tol=opt_tol, opt_nsteps=opt_nsteps,
                                                    perform_convergence_test=perform_convergence_test,
                                                    collect_minima_list=collect_minima_list,
                                                    seeds=seeds, use_cell_lists=use_cell_lists,
                                                    record_histogram=record_histogram, distance_method=distance_method,
                                                    use_frozen=use_frozen, frozen_atoms=frozen_atoms,
                                                    rcontainer=rcontainer, minimizer=minimizer,
                                                    opt_pot_str=opt_pot_str, **extra_pot_kwargs)
        assert self.equilibration_steps == 0

    def _set_actions(self):
        self.time_series = RecordDisplacementTimeseries(self.red_origin, self.bdim, self.ts_niter, self.ts_freq)
        self.add_action(self.time_series)

    def _set_takestep(self, stepsize):
        self.takestep = SampleUniformSphereGaussian(self.seeds['seed_takestep'], stepsize, self.origin)
        self.set_takestep(self.takestep)

    def _set_accept_tests(self):
        pass

    def set_control(self, c):
        """set k"""
        print("WARNING: set control is not defined, spring constant is set through stepsize", file=sys.stderr)

    def get_k(self):
        """in findk, potential is pretty much fictitious, k is adjusted through the stepsize"""
        stepsize = self.get_stepsize()
        k = 1.0 / (stepsize * stepsize)
        #k = self.bdim*len(self.hs_radii)/(stepsize*stepsize)##############
        return k

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

    def get_timeseries(self):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.time_series.get_time_series())
        return timeseries

    def check_convergence(self, nr_steps_to_check=10000, rel_std_threshold=0.05):
        return self.time_series.check_convergence(nr_steps_to_check=nr_steps_to_check,
                                                   rel_std_threshold=rel_std_threshold)

    def show_histogram(self):
        hist = self.histogram.get_histogram()
        val = np.array([i * self.binsize for i in xrange(len(hist))]) + 0.5 * self.binsize
        plt.hist(val, weights=hist, bins=len(hist))
        plt.show()

    def show_histogram_analytical(self):
        """
        shows the histogram against the analytical curve when k=kmax
        this function is useful for testing
        """
        timeseries = self.get_timeseries()
        n, bins, patch = plt.hist(timeseries, bins=500, range=(np.amin(timeseries), np.amax(timeseries)), normed=True,
                           alpha=0.4, edgecolor=color_cycle[0], color=color_cycle[0])
        ###analytical
        k = self.k * self.nparticles / (self.nparticles-1) #adjust for fixed com
        #and2 = np.exp(-0.5 * k * bincenters) * np.sqrt(k) / np.sqrt(2*np.pi*bincenters)
        and2 = n[0] * np.exp(-0.5 * k * bins[:-1]**2)
        plt.plot(bins[:-1], and2, linewidth=2.5, ls='--', color=color_cycle[-1])
        #plt.xlim(0,1)
        plt.xlabel(r'$|{\bf r}-{\bf r}_0|^2$')
        plt.ylabel(r'frequency $\times 10$')
        plt.tight_layout()
        plt.savefig('innersphere_histogram.eps')
        plt.show()
