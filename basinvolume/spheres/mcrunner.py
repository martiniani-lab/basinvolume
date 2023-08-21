from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np
import sys
import warnings
import logging
from pele.potentials import (
    Harmonic,
    HS_WCA,
    InversePowerStillingerCut,
    InversePower,
)
from pele.optimize import ModifiedFireCPP, LBFGS_CPP, CVODEBDFOptimizer
from pele.optimize._quench import modifiedfire_cpp
from pele.storage.database import Minimum
from pele.distance import Distance
from mcpele.monte_carlo import (
    RandomCoordsDisplacement,
    MetropolisTest,
    SampleGaussian,
    CheckSphericalContainer,
)
from basinvolume.gui import HSWCASystem
from basinvolume.utils import full_coordinates, write_2d_array_to_hdf5
from basinvolume.spheres import BaseSpheresMCrunner
from basinvolume.monte_carlo import (
    CheckSameMinimum,
    Findk,
    RecordDisplacementTimeseries,
    RecordStepsTimeseries,
    CheckOverlapPeriodic,
    CheckOverlapPeriodicCellLists,
    CheckOverlapCartesian,
    CheckOverlapCartesianCellLists,
    CheckOverlapLeesEdwards,
    CheckOverlapLeesEdwardsCellLists,
)
from basinvolume.enums import Minimizer, Interaction
from basinvolume.utils import INVERSE_POWER_CVODE_95_ACC


try:
    from mcpele.monte_carlo import RecordCoordsTimeseries
except Exception as e:
    print(e)

# for plotting histogram
from itertools import cycle
from scipy.integrate import quad

try:
    import matplotlib.pyplot as plt

    # more stuff for plotting histogram and comparing to prediction
    #######################SET LATEX OPTIONS###################
    plt.rc("text", usetex=True)
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    # rc('text.latex',preamble=r'\usepackage{times}')
    plt.rcParams.update({"font.size": 20})
    plt.rcParams["xtick.major.pad"] = 8
    plt.rcParams["ytick.major.pad"] = 8

    ##########################################################
    ####SET COLOUR MAP######
    cm = plt.get_cmap("Dark2")
    ########################
    #####################LINE STYLE CYCLER####################
    lines = ["-", "--", "-."]
    linecycler = cycle(lines)
    color_cycle = [cm(old_div(1.0 * i, 6)) for i in range(6)]
    ##########################################################
except ImportError as err:
    print(err)


def analytical_d2(x, k, N, boxdim=2):
    f = float(k * x) / 2
    g = float(boxdim * N - boxdim) / 2 - 1
    return np.exp(-f) * np.power(f, g)


vec_analytical_d2 = np.vectorize(analytical_d2)
# end: things for histogram

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

# add potential extra keyword arguments, like **potkwargs


class SpheresMCRunner(BaseSpheresMCrunner):
    def __init__(
        self,
        potential,
        full_coords,
        temperature,
        stepsize,
        niter,
        origin,
        hs_radii,
        boxv,
        sca,
        rattlers=None,
        k=1.0,
        dtol=1e-3,
        eps=1.0,
        hmin=0,
        hmax=1,
        hbinsize=0.001,
        avgcount=int(1e4),
        report_steps=0,
        pt_eq_niter=0,
        opt_dtmax=1,
        opt_maxstep=0.5,
        opt_tol=1e-5,
        opt_nsteps=1e5,
        perform_convergence_test=False,
        collect_minima_list=False,
        seeds=None,
        use_cell_lists=True,
        checkoverlap_cell_lists=None,
        record_histogram=False,
        distance_method=Distance.PERIODIC,
        use_frozen=False,
        frozen_atoms=None,
        rcontainer=None,
        minimizer=Minimizer.FIRE,
        interaction=Interaction.HS_WCA,
        pot_kwargs={},
    ):
        self.minimizer = minimizer
        # optimizer parameters
        self.opt_dtmax = opt_dtmax
        self.opt_maxstep = opt_maxstep
        self.opt_tol = opt_tol
        self.opt_nsteps = opt_nsteps
        self.interaction = interaction
        self.pot_kwargs = pot_kwargs
        self.distance_method = distance_method
        # check same minimum parameters
        self.perform_convergence_test = perform_convergence_test
        self.collect_minima_list = collect_minima_list
        self.checkoverlap_cell_lists = checkoverlap_cell_lists
        super(SpheresMCRunner, self).__init__(
            potential,
            full_coords,
            temperature,
            stepsize,
            niter,
            origin,
            hs_radii,
            boxv,
            sca,
            avgcount=avgcount,
            rattlers=rattlers,
            k=k,
            dtol=dtol,
            eps=eps,
            hmin=hmin,
            hmax=hmax,
            hbinsize=hbinsize,
            report_steps=report_steps,
            pt_eq_niter=pt_eq_niter,
            seeds=seeds,
            use_cell_lists=use_cell_lists,
            record_histogram=record_histogram,
            distance_method=distance_method,
            use_frozen=use_frozen,
            frozen_atoms=frozen_atoms,
            rcontainer=rcontainer,
        )

    def get_pot_optimizer(self):
        # here put a flag and pick potential
        if self.interaction is Interaction.HS_WCA:
            pot_optimizer = HS_WCA(
                distance_method=self.distance_method,
                pot_kwargs=self.pot_kwargs,
                use_cell_lists=self.use_cell_lists,
                use_frozen=self.use_frozen,
                eps=self.eps,
                sca=self.sca,
                radii=self.hs_radii,
                boxvec=self.boxv,
                reference_coords=self.origin,
                ndim=self.bdim,
                ncellx_scale=self.ncellx_scale,
                frozen_atoms=self.frozen_atoms,
            )
        elif self.interaction is Interaction.INVERSE_POWER_STILLINGER:
            pow = self.pot_kwargs["pow"]
            rcut = self.pot_kwargs["rcut"]
            pot_optimizer = InversePowerStillingerCut(
                pow,
                self.stillinger_a_radii,
                ndim=self.bdim,
                boxvec=self.boxv,
                rcut=rcut,
                use_cell_lists=True,
            )
        elif self.interaction is Interaction.INVERSE_POWER:
            power = self.pot_kwargs["power"]
            eps = self.pot_kwargs["eps"]
            if len(self.hs_radii) * self.bdim > 100:
                use_cell_lists = True
            else:
                use_cell_lists = False
            pot_optimizer = InversePower(
                power,
                eps,
                ndim=self.bdim,
                boxvec=self.boxv,
                radii=self.hs_radii,
                use_cell_lists=use_cell_lists,
            )
        else:
            raise NotImplementedError
        return pot_optimizer

    def get_optimizer(self):
        # res = modifiedfire_cpp(self.start_coords, self.pot_optimizer, dtmax=self.opt_dtmax,
        #                        maxstep=self.opt_maxstep, tol=self.opt_tol, nsteps=self.opt_nsteps)
        # maxErise = res.energy * 1e-9
        if self.minimizer is Minimizer.LBFGS:
            optimizer = LBFGS_CPP(
                self.start_coords,
                self.pot_optimizer,
                tol=self.opt_tol,
                nsteps=self.opt_nsteps,
                maxstep=self.opt_maxstep,
                maxErise=0,
            )
        elif self.minimizer is Minimizer.CVODE:
            optimizer = CVODEBDFOptimizer(
                self.pot_optimizer,
                self.start_coords,
                tol=self.opt_tol,
                atol=INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim],
                rtol=INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim],
            )
        else:
            optimizer = ModifiedFireCPP(
                self.start_coords,
                self.pot_optimizer,
                dtmax=self.opt_dtmax,
                maxstep=self.opt_maxstep,
                tol=self.opt_tol,
                nsteps=self.opt_nsteps,
            )
        return optimizer

    def _get_check_same_minimum(self):
        use_cgd = self.minimizer is Minimizer.CG
        csm = CheckSameMinimum(
            self.pot_optimizer,
            self.red_origin,
            self.rattlers,
            self.dtol,
            opt=self.optimizer,
            opt_tol=self.opt_tol,
            opt_maxiter=self.opt_nsteps,
            bdim=self.bdim,
            eqsteps=self.equilibration_steps,
            use_cgd=use_cgd,
            perform_convergence_test=self.perform_convergence_test,
            collect_minima_list=self.collect_minima_list,
        )
        return csm

    def _set_conf_tests(self):
        if self.use_frozen:
            self.conftest0 = CheckSphericalContainer(
                self.rcontainer, self.bdim
            )
            self.add_conf_test(self.conftest0)
        if self.interaction is Interaction.HS_WCA:
            if self.distance_method is Distance.PERIODIC:
                if (
                    self.checkoverlap_cell_lists is None
                    and self.use_cell_lists
                ) or self.checkoverlap_cell_lists:
                    self.conftest1 = CheckOverlapPeriodicCellLists(
                        self.hs_radii,
                        self.boxv,
                        ncellx_scale=self.ncellx_scale,
                        use_frozen=self.use_frozen,
                        frozen_atoms=self.frozen_atoms,
                        reference_coords=self.origin,
                    )
                else:
                    self.conftest1 = CheckOverlapPeriodic(
                        self.hs_radii,
                        self.boxv,
                        use_frozen=self.use_frozen,
                        reference_coords=self.origin,
                        frozen_atoms=self.frozen_atoms,
                    )
            elif self.distance_method is Distance.CARTESIAN:
                if (
                    self.checkoverlap_cell_lists is None
                    and self.use_cell_lists
                ) or self.checkoverlap_cell_lists:
                    self.conftest1 = CheckOverlapCartesianCellLists(
                        self.hs_radii,
                        self.boxv,
                        ncellx_scale=self.ncellx_scale,
                        use_frozen=self.use_frozen,
                        frozen_atoms=self.frozen_atoms,
                        reference_coords=self.origin,
                    )
                else:
                    self.conftest1 = CheckOverlapCartesian(
                        self.hs_radii,
                        self.bdim,
                        use_frozen=self.use_frozen,
                        reference_coords=self.origin,
                        frozen_atoms=self.frozen_atoms,
                    )
            elif self.distance_method is Distance.LEES_EDWARDS:
                if (
                    self.checkoverlap_cell_lists is None
                    and self.use_cell_lists
                ) or self.checkoverlap_cell_lists:
                    self.conftest1 = CheckOverlapLeesEdwardsCellLists(
                        self.hs_radii,
                        self.boxv,
                        shear=self.pot_kwargs["shear"],
                        ncellx_scale=self.ncellx_scale,
                        use_frozen=self.use_frozen,
                        frozen_atoms=self.frozen_atoms,
                        reference_coords=self.origin,
                    )
                else:
                    self.conftest1 = CheckOverlapLeesEdwards(
                        self.hs_radii,
                        self.boxv,
                        shear=self.pot_kwargs["shear"],
                        use_frozen=self.use_frozen,
                        reference_coords=self.origin,
                        frozen_atoms=self.frozen_atoms,
                    )
            else:
                raise NotImplementedError(
                    "Specified distance method " "not implemented."
                )
            self.add_late_conf_test(self.conftest1)
        else:
            warnings.warn(
                "not setting an excluded volume conf_test because using other potential than hs_wca"
            )
        self.conftest2 = self._get_check_same_minimum()
        self.add_late_conf_test(self.conftest2)

    def dump_minima_list(self, fname):
        """write minima list to pele database"""
        if self.interaction is Interaction.HS_WCA:
            system = HSWCASystem(
                self.eps,
                self.sca,
                self.hs_radii,
                self.boxv,
                bdim=self.bdim,
                dtol=self.dtol,
                etol=1,
            )
            db = system.create_database(fname)
            minima_dicts = []
            # add origin to database, with _id == 0, to make post processing
            # possible
            # for origin: set count to zero, but it does not have meaning, since we are only recording minima when quench took us to neighbor
            # distance should be zero because it is distance to itself
            mindict0 = dict(
                energy=self.pot_optimizer.getEnergy(self.red_origin),
                coords=self.origin,
                user_data=dict(count=0, distance=0),
            )
            minima_dicts.append(mindict0)
            # add neighboring minima to database
            self.conftest2.dump_minima(minima_dicts)
            # add spring constant to user_data
            for m in minima_dicts:
                m["user_data"].update(k=self.k)
                if self.use_frozen:
                    redcoords = m["coords"]
                    m["coords"] = full_coordinates(
                        redcoords, self.origin, self.frozen_atoms, self.bdim
                    )
            assert (
                len(minima_dicts) == self.conftest2.ml_nr_distinct_minima() + 1
            )
            logging.info("Number of minima: %i" % len(minima_dicts))
            size_estimate = len(minima_dicts) * len(self.origin)
            if size_estimate > 1e8:
                logging.warning(
                    "The size of the minima dictionary is extremely "
                    "long. This can cause the program to run out of "
                    "memory and crash (indicated by MPI noticing that "
                    "a process has exited on signal 9 (Killed)). You "
                    "should probably deactivate minima collection "
                    "(--nocollectminima)."
                )
            db.engine.execute(Minimum.__table__.insert(), minima_dicts)
            db.session.commit()
        else:
            warnings.warn(
                "dump_minima_list is not implemented for potentials other than hs_wca"
            )


class BV_MCRunner_State(object):
    """
    This class saves the state of an BV_MCrunner in a NumPy array
    """

    def __init__(
        self,
        state=None,
        coords=None,
        energy=0.0,
        k=0.0,
        stepsize=0.0,
        counters=None,
        takestep_count=0,
        step_adaptation_counters=None,
    ):
        if state is None:
            self.coords = coords
            self.energy = energy
            self.k = k
            self.stepsize = stepsize
            self.counters = counters
            self.takestep_count = takestep_count
            self.step_adaptation_counters = step_adaptation_counters
        else:
            self._set_state(state)

    def _set_state(self, state):
        self.coords = state.coords
        self.energy = state.energy
        self.k = state.k
        self.stepsize = state.stepsize
        self.counters = state.counters
        self.takestep_count = state.takestep_count
        self.step_adaptation_counters = state.step_adaptation_counters


class BV_MCrunner(SpheresMCRunner):
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
    record_steps_timeseries : bool
        record steps timeseries
    record_steps_timeseries_every : list
        array of intervals over which to record step distances

    .. Note:
    must set_control in the contructor!
    """

    def __init__(
        self,
        potential,
        full_coords,
        temperature,
        stepsize,
        niter,
        origin,
        hs_radii,
        boxv,
        sca,
        rattlers=None,
        k=1.0,
        dtol=1e-3,
        eps=1.0,
        hmin=0,
        hmax=1,
        hbinsize=0.001,
        acceptance=0.2,
        adjustf=0.9,
        adjustf_niter=1e4,
        adjustf_navg=100,
        pt_eq_niter=0,
        ts_niter=None,
        ts_freq=1,
        opt_dtmax=1,
        opt_maxstep=0.5,
        opt_tol=1e-5,
        opt_nsteps=1e5,
        perform_convergence_test=False,
        collect_minima_list=False,
        seeds=None,
        use_cell_lists=True,
        checkoverlap_cell_lists=None,
        record_histogram=False,
        record_steps_timeseries=False,
        record_steps_timeseries_every=[1],
        record_trajectory=False,
        record_trajectory_npoints=1e4,
        single=False,
        distance_method=Distance.PERIODIC,
        use_frozen=False,
        frozen_atoms=None,
        rcontainer=None,
        minimizer=Minimizer.FIRE,
        interaction=Interaction.HS_WCA,
        pot_kwargs={},
    ):
        # actions parameters
        if ts_niter is None:
            ts_niter = niter
        self.ts_niter = ts_niter
        self.ts_freq = ts_freq
        self.record_trajectory = record_trajectory
        self.record_trajectory_npoints = record_trajectory_npoints
        self.record_steps_timeseries = record_steps_timeseries
        self.record_steps_timeseries_every = record_steps_timeseries_every
        # takestep paramters
        self.adjustf_navg = adjustf_navg
        self.adjustf = adjustf
        self.acceptance = acceptance
        self.single = single
        super(BV_MCrunner, self).__init__(
            potential,
            full_coords,
            temperature,
            stepsize,
            niter,
            origin,
            hs_radii,
            boxv,
            sca,
            rattlers=rattlers,
            k=k,
            dtol=dtol,
            eps=eps,
            hmin=hmin,
            hmax=hmax,
            hbinsize=hbinsize,
            report_steps=adjustf_niter,
            pt_eq_niter=pt_eq_niter,
            opt_dtmax=opt_dtmax,
            opt_maxstep=opt_maxstep,
            opt_tol=opt_tol,
            opt_nsteps=opt_nsteps,
            perform_convergence_test=perform_convergence_test,
            collect_minima_list=collect_minima_list,
            seeds=seeds,
            use_cell_lists=use_cell_lists,
            checkoverlap_cell_lists=checkoverlap_cell_lists,
            record_histogram=record_histogram,
            distance_method=distance_method,
            use_frozen=use_frozen,
            frozen_atoms=frozen_atoms,
            rcontainer=rcontainer,
            minimizer=minimizer,
            interaction=interaction,
            pot_kwargs=pot_kwargs,
        )
        # set control
        self.set_control(k)

    def _set_takestep(self, stepsize):
        self.takestep = RandomCoordsDisplacement(
            self.seeds["seed_takestep"],
            stepsize,
            report_interval=self.adjustf_navg,
            factor=self.adjustf,
            min_acc_ratio=self.acceptance,
            max_acc_ratio=self.acceptance,
            single=self.single,
            nparticles=self.nparticles,
            bdim=self.bdim,
        )
        self.set_takestep(self.takestep)

    def _set_accept_tests(self):
        self.metropolis = MetropolisTest(self.seeds["seed_metropolis"])
        self.add_accept_test(
            self.metropolis
        )  # metropolis uses the harmonic potential

    def _set_actions(self):
        self.time_series = RecordDisplacementTimeseries(
            self.red_origin, self.bdim, self.ts_niter, self.ts_freq
        )
        self.add_action(self.time_series)
        if self.record_trajectory:
            rte = max(
                int(
                    old_div(
                        (self.niter - self.equilibration_steps),
                        self.record_trajectory_npoints,
                    )
                ),
                1,
            )
            self.record_trajectory = RecordCoordsTimeseries(
                self.ndim, record_every=rte, eqsteps=self.equilibration_steps
            )
            self.add_action(self.record_trajectory)
        if self.record_steps_timeseries:
            self.steps_timeseries_list = []
            for freq in self.record_steps_timeseries_every:
                self.steps_timeseries_list.append(
                    RecordStepsTimeseries(
                        self.red_origin,
                        self.rattlers,
                        self.bdim,
                        self.ts_niter,
                        freq,
                    )
                )
            for action in self.steps_timeseries_list:
                self.add_action(action)

    def set_control(self, c, reset=True):
        """set temperature, canonical control parameter"""
        self.k = c
        self.potential.set_k(c)
        if reset:
            self.reset_energy()

    def dump_histogram(self, fname):
        """write histogram to fname"""
        Emin, Emax = self.histogram.get_bounds_val()
        histl = self.histogram.get_histogram()
        hist = np.array(histl)
        Energies, step = np.linspace(
            Emin, Emax, num=len(hist), endpoint=False, retstep=True
        )
        Energies += 0.5 * step
        assert abs(step - self.binsize) < old_div(self.binsize, 100)
        np.savetxt(fname, np.column_stack((Energies, hist)), delimiter="\t")
        mean, variance = self.histogram.get_mean_variance()
        return mean, variance

    def dump_timeseries(
        self, fname, clear=True
    ):  # Writes ABSOLUTE DISPLACEMENTS from the origin
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.time_series.get_time_series())
        np.savetxt(fname, timeseries)
        if clear:
            self.time_series.clear()
        return timeseries

    def dump_steps_timeseries(
        self, fname, clear=True
    ):  # Writes RELATIVE DISPLACEMENTS within the walk
        """write time series to fname, returns the timeseries"""
        for i, action in enumerate(self.steps_timeseries_list):
            timeseries = np.array(action.get_time_series())
            np.savetxt(
                fname
                + ".every{}".format(self.record_steps_timeseries_every[i]),
                timeseries,
            )
            if clear:
                action.clear()

    def get_timeseries(self, clear=False):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.time_series.get_time_series())
        if clear:
            self.time_series.clear()
        return timeseries

    def check_convergence(
        self, nr_steps_to_check=10000, rel_std_threshold=0.05
    ):
        return self.time_series.check_convergence(
            nr_steps_to_check=nr_steps_to_check,
            rel_std_threshold=rel_std_threshold,
        )

    def show_histogram(self):
        hist = self.histogram.get_histogram()
        val = (
            np.array([i * self.binsize for i in range(len(hist))])
            + 0.5 * self.binsize
        )
        plt.hist(val, weights=hist, bins=len(hist))
        plt.show()

    def show_histogram_kmax(self):
        """
        shows the histogram against the analytical curve when k=kmax
        this function is useful for testing
        """
        hist = self.histogram.get_histogram()
        val = (
            np.array([i * self.binsize for i in range(len(hist))])
            + 0.5 * self.binsize
        )
        n, bins, patches = plt.hist(
            val,
            weights=hist,
            bins=len(hist),
            normed=1,
            alpha=0.4,
            edgecolor=color_cycle[0],
            color=color_cycle[0],
        )
        ###analytical
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        and2 = old_div(
            vec_analytical_d2(val, self.k, self.nparticles),
            quad(
                vec_analytical_d2,
                bincenters[0],
                bincenters[-1],
                args=(self.k, self.nparticles),
            )[0],
        )
        plt.plot(
            bincenters, and2, linewidth=2.5, ls="--", color=color_cycle[-1]
        )
        # plt.xlim(0,1)
        plt.xlabel(r"$|{\bf r}-{\bf r}_0|^2$")
        plt.ylabel(r"frequency $\times 10$")
        plt.tight_layout()
        plt.savefig("kmax_histogram.eps")
        plt.show()

    def get_mean_variance_coordinate_vector(self):
        """
        returns the average coordinate vector from the sampling and the elementwise variance
        """
        (
            mean_coord,
            var_coord,
        ) = self.record_trajectory.get_mean_variance_time_series()
        return mean_coord, var_coord

    def dump_trajectory(self, fname, clear=True):
        """write time series to fname, returns the timeseries"""
        trajectory = self.get_trajectory()
        write_2d_array_to_hdf5(trajectory, "trajectory", fname)
        if clear:
            self.clear_trajectory()
        return trajectory

    def get_trajectory(self):
        trajectory = self.record_trajectory.get_time_series()
        return trajectory

    def clear_trajectory(self):
        self.record_trajectory.clear()

    def get_complete_state(self):
        return BV_MCRunner_State(
            coords=self.get_coords(),
            energy=self.get_energy(),
            k=self.k,
            stepsize=self.takestep.get_stepsize(),
            counters=self.get_counters(),
            takestep_count=self.takestep.get_count(),
            step_adaptation_counters=self.takestep.get_adaptation_counters(),
        )

    def set_complete_state(self, mcrunner_state):
        self.set_config(mcrunner_state.coords, mcrunner_state.energy)
        self.set_control(mcrunner_state.k, reset=False)
        self.set_counters(mcrunner_state.counters)
        self.takestep.set_stepsize(mcrunner_state.stepsize)
        self.takestep.set_count(mcrunner_state.takestep_count)
        self.takestep.set_adaptation_counters(
            mcrunner_state.step_adaptation_counters
        )


class Findk_MCrunner(SpheresMCRunner):
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
    *
    *stepsize
    *Etol: tolerance with which a mini mised structure is accepted
     when compared to origin energy
    *dtol: tolerance on the rms displacement of the minimised structure
     with respect to the origin coordinates
    *ktarget: target acceptance associated to kmax
    *knavg: number of steps over findk averages the acceptance
    *ktol: when acceptance-ktarget<ktol the search for k terminates
    * this class requires 1 seed
    """

    def __init__(
        self,
        potential,
        full_coords,
        temperature,  # always set to 1.0 #TODO: remove
        stepsize,
        niter,
        origin,
        hs_radii,
        boxv,
        sca,
        rattlers=None,
        dtol=1e-3,
        eps=1.0,
        ktarget=0.75,
        knavg=500,
        avgcount=int(1e4),
        ktol=0.05,
        opt_dtmax=1,
        opt_maxstep=0.6,
        opt_tol=1e-4,
        opt_nsteps=1e5,
        hmin=0,
        hmax=1,
        binsize=0.005,
        perform_convergence_test=False,
        collect_minima_list=False,
        seeds=None,
        use_cell_lists=False,
        single=False,
        distance_method=Distance.PERIODIC,
        use_frozen=False,
        frozen_atoms=None,
        rcontainer=None,
        minimizer=Minimizer.FIRE,
        interaction=Interaction.HS_WCA,
        pot_kwargs={},
    ):
        # findk parameters
        self.ktarget = ktarget
        self.knavg = knavg
        self.ktol = ktol
        super(Findk_MCrunner, self).__init__(
            potential,
            full_coords,
            temperature,
            stepsize,
            niter,
            origin,
            hs_radii,
            boxv,
            sca=0.0,
            rattlers=rattlers,
            k=1,
            dtol=dtol,
            avgcount=avgcount,
            eps=eps,
            hmin=hmin,
            hmax=hmax,
            hbinsize=binsize,
            report_steps=0,
            pt_eq_niter=0,
            opt_dtmax=opt_dtmax,
            opt_maxstep=opt_maxstep,
            opt_tol=opt_tol,
            opt_nsteps=opt_nsteps,
            perform_convergence_test=perform_convergence_test,
            collect_minima_list=collect_minima_list,
            seeds=seeds,
            use_cell_lists=use_cell_lists,
            record_histogram=False,
            distance_method=distance_method,
            use_frozen=use_frozen,
            frozen_atoms=frozen_atoms,
            rcontainer=rcontainer,
            minimizer=minimizer,
            interaction=interaction,
            pot_kwargs=pot_kwargs,
        )

    def _set_takestep(self, stepsize):
        self.takestep = SampleGaussian(
            self.seeds["seed_takestep"], stepsize, self.origin
        )
        self.set_takestep(self.takestep)

    def _set_actions(self):
        self.findk = Findk(
            self.red_origin,
            self.rattlers,
            self.bdim,
            self.avgcount,
            self.ktarget,
            self.knavg,
            self.ktol,
            self.hmin,
            self.hmax,
            self.binsize,
        )
        self.add_action(self.findk)

    def _set_accept_tests(self):
        pass

    def set_control(self, c):
        """set k"""
        print(
            "WARNING: findk set control is not defined, spring constant is set through stepsize",
            file=sys.stderr,
        )

    def get_k(self):
        """in findk, potential is pretty much fictitious, k is adjusted through the stepsize"""
        stepsize = self.get_stepsize()
        k = 1.0 / (stepsize * stepsize)
        # k = self.bdim*len(self.hs_radii)/(stepsize*stepsize)##############
        return k

    def get_entries(self):
        return self.findk.get_entries()

    def show_histogram(self):
        """shows the histogram"""
        hist = self.findk.get_histogram()
        val = (
            np.array([i * self.binsize for i in range(len(hist))])
            + 0.5 * self.binsize
        )
        n, bins, patches = plt.hist(
            val,
            weights=hist,
            bins=len(hist),
            normed=1,
            alpha=0.4,
            edgecolor=color_cycle[0],
            color=color_cycle[0],
        )
        ###analytical
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        and2 = old_div(
            vec_analytical_d2(val, self.get_k(), self.nparticles, self.bdim),
            quad(
                vec_analytical_d2,
                bincenters[0],
                bincenters[-1],
                args=(self.get_k(), self.nparticles, self.bdim),
            )[0],
        )
        plt.plot(
            bincenters, and2, linewidth=2.5, ls="--", color=color_cycle[-1]
        )
        # plt.xlim(0,1)
        plt.xlabel(r"$|{\bf r}-{\bf r}_0|^2$")
        plt.ylabel(r"frequency $\times 10$")
        plt.tight_layout()
        plt.savefig("findk_histogram.eps")
        plt.show()


if __name__ == "__main__":
    # to run harmonic potential go to tests

    from pele.utils.rotations import vector_random_uniform_hypersphere
    from pele.optimize._quench import modifiedfire_cpp
    import time

    nparticles = 1
    ndim = nparticles * 3
    origin = np.array([0, 0, 0], dtype="d")
    # build start configuration
    Emax = 0.1
    start_coords = vector_random_uniform_hypersphere(ndim) * np.sqrt(
        2 * Emax
    )  # coordinates sampled from Pow(ndim)
    # Harmonic(origin,1)
    res = modifiedfire_cpp(start_coords, Harmonic(origin, 1))
    print(res)

    # print(res.coords)

    # Parallel Tempering
    # test = BV_MCrunner(
    #     start_coords,
    #     origin,
    #     temperature=1,
    #     k=1,
    #     niter=1e5,
    #     hmin=0,
    #     hmax=100,
    #     stepsize=0.5,
    #     adjustf=0.9,
    #     adjustf_niter=5000,
    #     sca=100,
    # )

    # test = BV_MCrunner(
    #     potential,
    #     full_coords,
    #     temperature=1,
    #     stepsize=0.5,
    #     niter=1e1,
    #     origin=origin,
    #     hs_radii,
    #     boxv,
    #     sca,
    # )
    # test.set_control(1)
    start = time.time()
    # test.run()
    end = time.time()
    print(end - start)
    test.show_histogram()
