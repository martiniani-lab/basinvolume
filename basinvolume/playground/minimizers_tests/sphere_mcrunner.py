from __future__ import print_function
import numpy as np
import sys
from pele.potentials import Harmonic, HS_WCA
from pele.optimize import ModifiedFireCPP, LBFGS_CPP
from pele.storage import Database
from pele.storage.database import Minimum
from pele.distance import Distance
from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import CheckSphericalContainer
from basinvolume.monte_carlo import CheckSameMinimum, RecordDisp2Histogram
from basinvolume.monte_carlo import CheckOverlapPeriodic, CheckOverlapCartesian
from basinvolume.monte_carlo import RecordDisplacementTimeseries
from basinvolume.monte_carlo import CheckOverlapCartesianCellLists
from basinvolume.monte_carlo import CheckOverlapPeriodicCellLists
from basinvolume.monte_carlo import SampleUniformSphereGaussian
from basinvolume.gui import HSWCASystem
from basinvolume.utils import reduce_coordinates, full_coordinates
from basinvolume.enums import Minimizer

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

def analytical_d2(x, k, N, boxdim=2):
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

#from pele.potentials._pythonpotential import as_cpp_potential
#from pele.potentials._pele import BasePotential
#
#class NullPotential(BasePotential):
#    def getEnergy(self):
#        return 0

class BVSphereMCrunner(_BaseMCRunner):
    """
    Basin volume Sphere MC runner

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
                 minimizer=Minimizer.FIRE):
        #construct base class
        if use_frozen:
            assert distance_method is Distance.CARTESIAN and frozen_atoms is not None
            red_coords = reduce_coordinates(full_coords, frozen_atoms, len(boxv))
        else:
            red_coords = full_coords
        #potential = as_cpp_potential(NullPotential())
        super(BVSphereMCrunner, self).__init__(potential, red_coords, temperature, niter)

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
        self.dtol = dtol
        self.eps = eps
        self.k = 1.0 / (stepsize * stepsize)
        self.nparticles = len(self.red_radii)
        self.use_cell_lists = use_cell_lists
        self.use_frozen = use_frozen
        self.minimizer = minimizer
        self.frozen_atoms = frozen_atoms
        self.distance_method = distance_method
        self.rcontainer = rcontainer
        self.equilibration_steps = 0
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
            if np.amin(self.boxv) // self.rcut <= 3:
                print ("warning: use_cell_lists flag was set, rcut is too large though")
                print ("setting use_cell_lists to False")
                self.use_cell_lists = False
        self.ncellx_scale = 1.0
        self.pot_optimizer = HS_WCA(distance_method=self.distance_method,
                             use_cell_lists=self.use_cell_lists,
                             use_frozen=use_frozen, eps=self.eps, sca=self.sca,
                             radii=self.hs_radii, boxvec=self.boxv,
                             reference_coords=self.origin,
                             ndim=self.bdim, ncellx_scale=self.ncellx_scale,
                             frozen_atoms=self.frozen_atoms)

        #construct gradient optimizer
        if self.minimizer is Minimizer.LBFGS:
            self.optimizer = LBFGS_CPP(self.start_coords,
                                       self.pot_optimizer,
                                       tol=opt_tol,
                                       nsteps=opt_nsteps,
                                       maxstep=opt_maxstep)
        else:
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

        if self.distance_method is Distance.PERIODIC:
            if self.use_cell_lists:
                self.conftest1 = CheckOverlapPeriodicCellLists(self.hs_radii,
                                 self.boxv, ncellx_scale=self.ncellx_scale,
                                 use_frozen=self.use_frozen, frozen_atoms=self.frozen_atoms,
                                 reference_coords=self.origin)

            else:
                self.conftest1 = CheckOverlapPeriodic(self.hs_radii,
                                 self.boxv, use_frozen=self.use_frozen,
                                 reference_coords=self.origin,
                                 frozen_atoms=self.frozen_atoms)
        elif self.distance_method is Distance.CARTESIAN:
            if self.use_cell_lists:
                self.conftest1 = CheckOverlapCartesianCellLists(self.hs_radii,
                                 self.boxv, ncellx_scale=self.ncellx_scale,
                                 use_frozen=self.use_frozen,
                                 frozen_atoms=self.frozen_atoms,
                                 reference_coords=self.origin)
            else:
                self.conftest1 = CheckOverlapCartesian(self.hs_radii,
                                 self.bdim, use_frozen=self.use_frozen,
                                 reference_coords=self.origin,
                                 frozen_atoms=self.frozen_atoms)
        elif self.distance_method is Distance.LEES_EDWARDS:
            if self.use_cell_lists:
                self.conftest1 = CheckOverlapLeesEdwardsCellLists(
                    self.hs_radii, self.boxv,
                    shear=self.pot_kwargs['shear'],
                    ncellx_scale=self.ncellx_scale,
                    use_frozen=self.use_frozen,
                    frozen_atoms=self.frozen_atoms,
                    reference_coords=self.origin)
            else:
                self.conftest1 = CheckOverlapLeesEdwards(
                    self.hs_radii, self.boxv,
                    shear=self.pot_kwargs['shear'],
                    use_frozen=self.use_frozen,
                    reference_coords=self.origin,
                    frozen_atoms=self.frozen_atoms)
        else:
            raise NotImplementedError("Specified distance method "
                                      "not implemented.")
        use_cgd = self.minimizer is Minimizer.CG
        self.conftest2 = CheckSameMinimum(self.pot_optimizer, self.red_origin,
                                          self.rattlers, self.dtol,
                                          opt=self.optimizer, opt_tol=opt_tol, opt_maxiter=opt_nsteps,
                                          bdim=self.bdim, eqsteps=self.equilibration_steps,
                                          use_cgd=use_cgd,
                                          perform_convergence_test=perform_convergence_test,
                                          collect_minima_list=collect_minima_list)
        self.time_series = RecordDisplacementTimeseries(self.red_origin, self.bdim, ts_niter, ts_freq)

        self.set_report_steps(0)
        self.takestep = SampleUniformSphereGaussian(self.seeds['seed_takestep'], stepsize, self.origin)

        #set up pele:MC
        self.set_takestep(self.takestep)
        if self.use_frozen:
            self.conftest0 = CheckSphericalContainer(self.rcontainer, self.bdim)
            self.add_conf_test(self.conftest0)
        self.add_late_conf_test(self.conftest1)
        self.add_late_conf_test(self.conftest2) #conf_test will happen after accept test because it is much cheaper
        self.add_action(self.time_series)

    def set_control(self, c):
        """set k"""
        print("WARNING: set control is not defined, spring constant is set through stepsize", file=sys.stderr)

    def get_stepsize(self):
        return self.takestep.get_stepsize()

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
        #and2 = vec_analytical_d2(val,self.k, self.nparticles) / quad(vec_analytical_d2, bincenters[0], bincenters[-1], args=(self.k, self.nparticles))[0]
        k = self.k * self.ndim / (self.ndim-1) #adjust for fixed com
        and2 = np.exp(-0.5 * k * bincenters) * np.sqrt(k) / np.sqrt(2*np.pi*bincenters)
        plt.plot(bincenters, and2, linewidth=2.5, ls='--', color=color_cycle[-1])
        #plt.xlim(0,1)
        plt.xlabel(r'$|{\bf r}-{\bf r}_0|^2$')
        plt.ylabel(r'frequency $\times 10$')
        plt.tight_layout()
        plt.savefig('kmax_histogram.eps')
        plt.show()

if __name__ == "__main__":
    #to run harmonic potential go to tests

    from pele.utils.rotations import vector_random_uniform_hypersphere
    from pele.optimize._quench import modifiedfire_cpp
    import time

    natoms = 4
    L = 4.
    boxvec = np.ones(3) * L
    x0 = np.random.uniform(0, L, 3*natoms)
    #x0 = _subtract_com(x0)
    #build start configuration

#    test = BV_MCrunner(start_coords, origin, temperature=1, k=1, niter=1e5, hEmin=0,hEmax=100,
#                       stepsize=0.5, adjustf = 0.9, adjustf_niter = 5000, radius=100)
    #test.set_control(1)
    start = time.time()
    #test.run()
    end = time.time()
    print(end - start)
    #test.show_histogram()
