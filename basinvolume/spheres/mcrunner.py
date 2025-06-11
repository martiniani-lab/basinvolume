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
    PoweredCosineSum,
)
from pele.optimize import (
    ModifiedFireCPP,
    LBFGS_CPP,
    CVODEBDFOptimizer,
    ExtendedMixedOptimizer,
)
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
    CheckSameMinimumConfig,
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
from basinvolume.utils import INVERSE_POWER_CVODE_95_ACC, get_mxd_t
from basinvolume.base_basinvolume import BaseBVMCRunner, BV_MCRunner_State
from basinvolume.utils import analytical_d2, vec_analytical_d2, PlottingMixin

try:
    from mcpele.monte_carlo import RecordCoordsTimeseries
except Exception as e:
    print(e)

# for plotting histogram
from itertools import cycle
from scipy.integrate import quad

try:
    import matplotlib.pyplot as plt
    from basinvolume.utils import get_color_cycle, get_line_cycler
    
    # more stuff for plotting histogram and comparing to prediction
    #######################SET LATEX OPTIONS###################
    plt.rc("text", usetex=True)
    plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    plt.rcParams.update({"font.size": 20})
    plt.rcParams["xtick.major.pad"] = 8
    plt.rcParams["ytick.major.pad"] = 8

    ##########################################################
    ####SET COLOUR MAP######
    cm = plt.get_cmap("Dark2")
    ########################
    #####################LINE STYLE CYCLER####################
    lines = ["-", "--", "-."]
    linecycler = get_line_cycler()
    color_cycle = get_color_cycle()
    ##########################################################
except ImportError as err:
    print(err)

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
        bias_params=None,
        bias="harmonic",
        dtol=1e-3,
        eps=1.0,
        hmin=0,
        hmax=1,
        hbinsize=0.001,
        avgcount=int(1e4),
        report_steps=0,
        pt_eq_niter=0,
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
        opt_kwargs={},
    ):
        self.minimizer = minimizer
        # optimizer parameters
        self.opt_maxstep = opt_maxstep
        self.opt_tol = opt_tol
        self.opt_nsteps = opt_nsteps
        self.interaction = interaction
        self.fix_com = self.interaction is not Interaction.NEGATIVE_COS
        self.pot_kwargs = pot_kwargs
        self.opt_kwargs = opt_kwargs
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
            bias_params=bias_params,
            bias=bias,
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
            fix_com=self.fix_com,
        )

    def get_pot_optimizer(self):
        # here put a flag and pick potential
        if self.interaction is Interaction.HS_WCA:
            pot_optimizer = HS_WCA(
                eps=self.eps,
                sca=self.sca,
                use_cell_lists=self.use_cell_lists,
                bdim=self.bdim,
                radii=self.hs_radii,
                use_periodic=True,
                boxvec=self.boxv,
                balance_omp=False,
            )
        elif self.interaction is Interaction.INVERSE_POWER:
            pot_optimizer = InversePower(
                pow=self.pot_kwargs["pow"],
                eps=self.eps,
                use_cell_lists=self.use_cell_lists,
                ndim=self.ndim,
                radii=self.hs_radii,
                use_periodic=True,
                boxvec=self.boxv,
            )
        elif self.interaction is Interaction.INVERSE_POWER_STILLINGER:
            pot_optimizer = InversePowerStillingerCut(
                pow=self.pot_kwargs["pow"],
                eps=self.eps,
                use_cell_lists=self.use_cell_lists,
                ndim=self.ndim,
                radii=self.hs_radii,
                use_periodic=True,
                boxvec=self.boxv,
            )
        elif self.interaction is Interaction.NEGATIVE_COS:
            pot_optimizer = PoweredCosineSum(
                pow=self.pot_kwargs["pow"],
                radii=self.hs_radii,
                eps=self.eps,
            )
        else:
            raise RuntimeError("unknown interaction")
        return pot_optimizer

    def get_optimizer(self):
        # here put a flag and pick opt strategy
        pot_optimizer = self.get_pot_optimizer()
        if self.minimizer is Minimizer.FIRE:
            return ModifiedFireCPP(
                pot_optimizer,
                maxstep=self.opt_maxstep,
                dtmax=self.opt_maxstep,
                maxErise=1e-4,
                tol=self.opt_tol,
                nsteps=self.opt_nsteps,
                **self.opt_kwargs
            )
        elif self.minimizer is Minimizer.LBFGS:
            return LBFGS_CPP(
                pot_optimizer,
                maxstep=self.opt_maxstep,
                tol=self.opt_tol,
                nsteps=self.opt_nsteps,
                **self.opt_kwargs
            )
        elif self.minimizer is Minimizer.CVODE:
            gamma = self.pot_kwargs.get("gamma", INVERSE_POWER_CVODE_95_ACC)
            mxstep = self.pot_kwargs.get("mxstep", get_mxd_t(gamma))
            return CVODEBDFOptimizer(
                pot_optimizer,
                tol=self.opt_tol,
                gamma=gamma,
                mxstep=mxstep,
                rtol=1e-9,
                atol=1e-12,
                **self.opt_kwargs
            )
        elif self.minimizer is Minimizer.MIXED:
            return ExtendedMixedOptimizer(
                pot_optimizer,
                maxstep=self.opt_maxstep,
                tol=self.opt_tol,
                nsteps=self.opt_nsteps,
                **self.opt_kwargs
            )
        else:
            raise RuntimeError("unknown minimizer")

    def _get_check_same_minimum(self):
        if self.interaction is Interaction.NEGATIVE_COS:
            # modular arithmetic distance
            ma_rcut = 0.25
            return CheckSameMinimumConfig(
                self.minimizer,
                self.coords,
                self.distance_method,
                use_cell_lists=self.use_cell_lists,
                checkoverlap_cell_lists=self.checkoverlap_cell_lists,
                bdim=self.bdim,
                ndim=self.ndim,
                nparticles=self.nparticles,
                origin=self.origin,
                boxvec=self.boxv,
                radii=self.hs_radii,
                ma_rcut=ma_rcut,
            )
        else:
            return CheckSameMinimum(
                self.minimizer,
                self.coords,
                self.distance_method,
                use_cell_lists=self.use_cell_lists,
                checkoverlap_cell_lists=self.checkoverlap_cell_lists,
                bdim=self.bdim,
                ndim=self.ndim,
                nparticles=self.nparticles,
                origin=self.origin,
                boxvec=self.boxv,
                radii=self.hs_radii,
            )

    def _set_conf_tests(self):
        # configuration tests
        if self.distance_method is Distance.PERIODIC:
            if self.use_cell_lists:
                overlap_test = CheckOverlapPeriodicCellLists(
                    self.hs_radii, self.boxv, ncellx_scale=self.sca
                )
            else:
                overlap_test = CheckOverlapPeriodic(
                    self.hs_radii, self.boxv, ndim=self.bdim
                )
        elif self.distance_method is Distance.LEES_EDWARDS:
            if self.use_cell_lists:
                overlap_test = CheckOverlapLeesEdwardsCellLists(
                    self.hs_radii, self.boxv, shear=0.0, ncellx_scale=self.sca
                )
            else:
                overlap_test = CheckOverlapLeesEdwards(
                    self.hs_radii, self.boxv, shear=0.0, ndim=self.bdim
                )
        elif self.distance_method is Distance.CARTESIAN:
            if self.use_cell_lists:
                overlap_test = CheckOverlapCartesianCellLists(
                    self.hs_radii, ncellx_scale=self.sca
                )
            else:
                overlap_test = CheckOverlapCartesian(self.hs_radii)
            if self.use_frozen and self.rcontainer is not None:
                spherical_container = CheckSphericalContainer(
                    self.red_origin, self.rcontainer
                )
                self.add_conf_test(spherical_container)
        else:
            raise RuntimeError("distance method not recognized")
        self.add_conf_test(overlap_test)

        if self.perform_convergence_test:
            same_minimum = self._get_check_same_minimum()
            self.add_conf_test(same_minimum)

    def dump_minima_list(self, fname):
        """
        dump the minima list in the database
        """
        mlist = self.same_minimum.get_minima_list()
        print("number of minima visited during the simulation", len(mlist))
        for minimum in mlist:
            minimum.coords = full_coordinates(
                minimum.coords.copy(), self.nparticles, self.bdim
            )
        f = open(fname, "wb")
        system = HSWCASystem()
        system.params.database.accuracy = 1e-3
        # database = Database(accuracy=1e-3)
        for m in mlist:
            # database.addMinimum(m.energy, m.coords)
            new_minimum = system.params.database.addMinimum(m.energy, m.coords)
        system.params.database.write_minima_xyz(f)
        f.close()


class BV_MCrunner(SpheresMCRunner, BaseBVMCRunner, PlottingMixin):
    """
    Basin volume MC runner

    Parameters
    ----------
    bias_potential : pele potential
        Biasing potential used in the thermodynamic integration.
        These can be harmonic springs that tie each particle to its original
        position during the walk, or any arbitrary bias.
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
    bias_params : list of doubles
        Parameters for biasing potential. # XXX SHOULD MAKE THIS A DICT IDEALLY?
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
    opt_maxstep : double
        MaxStep parameter of modified fire optimiser.
    opt_tol : double
        Tolerance of modified fire optimiser.
    opt_nsteps : integer
        Number of iterations of modified fire optimiser.
    opt_kwargs : dict
        optimizer-specific kwargs, fed as a dict to avoid useless variables
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
        bias_potential,
        full_coords,
        temperature,
        stepsize,
        niter,
        origin,
        hs_radii,
        boxv,
        sca,
        rattlers=None,
        bias_params=None,
        bias="harmonic",
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
        opt_maxstep=0.5,
        opt_tol=1e-5,
        opt_nsteps=1e5,
        opt_kwargs={},
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
            bias_potential,
            full_coords,
            temperature,
            stepsize,
            niter,
            origin,
            hs_radii,
            boxv,
            sca,
            rattlers=rattlers,
            bias_params=bias_params,
            bias=bias,
            dtol=dtol,
            eps=eps,
            hmin=hmin,
            hmax=hmax,
            hbinsize=hbinsize,
            opt_maxstep=opt_maxstep,
            opt_tol=opt_tol,
            opt_nsteps=opt_nsteps,
            opt_kwargs=opt_kwargs,
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

        self.bias = bias
        self.bias_potential = bias_potential
        self.bias_params = bias_params

        # set up pele:MC
        self._set_takestep(stepsize)
        self._set_accept_tests()
        self._set_actions()

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
        self.add_accept_test(self.metropolis)

    def _set_actions(self):
        self.action_record_displ = RecordDisplacementTimeseries(
            self.origin, self.bdim, self.ts_niter, self.ts_freq, self.fix_com
        )
        self.add_action(self.action_record_displ)
        if self.record_steps_timeseries:
            self.steps_timeseries_list = []
            self.record_steps_timeseries_every = self.record_steps_timeseries_every
            for freq in self.record_steps_timeseries_every:
                self.steps_timeseries_list.append(
                    RecordStepsTimeseries(
                        self.origin,
                        self.rattlers,
                        self.bdim,
                        self.ts_niter,
                        freq,
                        self.equilibration_steps,
                        self.fix_com,
                    )
                )
                self.add_action(self.steps_timeseries_list[-1])
        if self.record_trajectory:
            rte = max(
                int((self.niter - self.equilibration_steps) / self.record_trajectory_npoints),
                1,
            )
            self.record_trajectory = RecordCoordsTimeseries(
                self.ndim, record_every=rte, eqsteps=self.equilibration_steps
            )
            self.add_action(self.record_trajectory)

    def set_bias_parameters(self, bias, bias_params, reset=True):  # XXX
        """set new bias parameters, generally called from parallel tempering"""
        if bias == "harmonic":
            self.bias_potential.set_k(bias_params[0])
        elif bias == "radial_gaussian":
            self.bias_potential.set_A(bias_params[0])
            self.bias_potential.set_sig(bias_params[1])
        else:
            raise NotImplementedError("bias={} not implemented".format(bias))
        self.bias_params = bias_params
        if reset:
            self.reset()

    def dump_histogram(self, fname):
        """write histogram to fname"""
        Emin, Emax = self.histogram.get_bounds_val()
        histl = self.histogram.get_histogram()
        hist = np.array(histl)
        Energies, step = np.linspace(Emin, Emax, num=len(hist), endpoint=False, retstep=True)
        Energies += 0.5 * step
        assert abs(step - self.binsize) < self.binsize / 100
        np.savetxt(fname, np.column_stack((Energies, hist)), delimiter="\t")
        mean, variance = self.histogram.get_mean_variance()
        return mean, variance

    # Common methods now inherited from BaseBVMCRunner
    # dump_timeseries, get_timeseries, check_convergence, dump_trajectory, 
    # get_trajectory, clear_trajectory, get_complete_state, set_complete_state

    def dump_steps_timeseries(self, fname, clear=True):
        """Writes RELATIVE DISPLACEMENTS within the walk"""
        for i, steps_timeseries in enumerate(self.steps_timeseries_list):
            timeseries = np.array(steps_timeseries.get_time_series())
            fname_mod = fname + "_" + str(self.record_steps_timeseries_every[i])
            np.savetxt(fname_mod, timeseries)
            if clear:
                steps_timeseries.clear()
        return

    def show_histogram_kmax(self):
        """show the histogram against theoretical expectation when K = Kmax"""
        hist = self.histogram.get_histogram()
        val = np.array([i * self.binsize for i in range(len(hist))]) + 0.5 * self.binsize
        plt.clf()
        plt.hist(val, weights=hist, bins=len(hist), density=True, stacked=True)
        ###analytical
        k = self.bias_params[0]
        and2 = vec_analytical_d2(val[:-1], k, self.nparticles, self.bdim)
        norm = and2.sum() * self.binsize
        and2 /= norm
        plt.plot(val[:-1], and2, linewidth=2.5, ls=next(linecycler), color=color_cycle[-1])
        if hasattr(self, "kmax"):
            kmax = self.kmax
            and2 = vec_analytical_d2(val[:-1], kmax, self.nparticles, self.bdim)
            norm = and2.sum() * self.binsize
            and2 /= norm
            plt.plot(
                val[:-1],
                and2,
                linewidth=2.5,
                ls=next(linecycler),
                color=color_cycle[-2],
                label="kmax analytical",
            )
        else:
            print("no kmax available")
        if hasattr(self, "prob_kmax"):
            prob_kmax = self.prob_kmax
            k = 3.0 * prob_kmax
            integral = quad(
                analytical_d2,
                0,
                200,
                args=(
                    k,
                    self.nparticles,
                    self.bdim,
                ),
            )
            print("test integral", integral)
            and2 = vec_analytical_d2(val[:-1], k, self.nparticles, self.bdim)
            norm = integral[0]
            and2 /= norm
            plt.plot(
                val[:-1],
                and2,
                linewidth=2.5,
                ls=next(linecycler),
                color=color_cycle[-3],
                label="kmax x 3 analytical",
            )
        plt.xlabel(r"$|{\bf r}-{\bf r}_0|^2$")
        plt.ylabel(r"$P(|{\bf r}-{\bf r}_0|^2)$")
        plt.legend()
        plt.tight_layout()
        # plt.ylim(0,0.05)
        # plt.xlim(0,200)
        plt.savefig("bv_histogram.eps")
        plt.show()


class Findk_MCrunner(SpheresMCRunner):
    """
    Find k

    Parameters
    ----------
    See SpheresMCRunner.

    ktarget : double
        target step acceptance ratio
    knavg : int
        number of MC steps from which to compute the acceptance ratio
    ktol : double
        tolerance on the target acceptance ratio
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
        opt_maxstep=0.6,
        opt_tol=1e-4,
        opt_nsteps=1e5,
        opt_kwargs={},
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
        fix_com=True,
    ):
        # findk parameters
        self.ktarget = ktarget
        self.knavg = knavg
        self.ktol = ktol
        self.avgcount = avgcount
        self.binsize = binsize
        super(Findk_MCrunner, self).__init__(
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
            dtol=dtol,
            eps=eps,
            hmin=hmin,
            hmax=hmax,
            hbinsize=binsize,
            avgcount=avgcount,
            opt_maxstep=opt_maxstep,
            opt_tol=opt_tol,
            opt_nsteps=opt_nsteps,
            opt_kwargs=opt_kwargs,
            perform_convergence_test=perform_convergence_test,
            collect_minima_list=collect_minima_list,
            seeds=seeds,
            use_cell_lists=use_cell_lists,
            single=single,
            distance_method=distance_method,
            use_frozen=use_frozen,
            frozen_atoms=frozen_atoms,
            rcontainer=rcontainer,
            minimizer=minimizer,
            interaction=interaction,
            pot_kwargs=pot_kwargs,
            fix_com=fix_com,
        )

        # set up pele:MC
        self._set_takestep(stepsize)
        self._set_actions()
        self._set_accept_tests()

    def _set_takestep(self, stepsize):
        self.takestep = SampleGaussian(self.seeds["seed_takestep"], stepsize, self.origin)
        self.set_takestep(self.takestep)

    def _set_actions(self):
        self.findk = Findk(
            self.origin,
            self.rattlers,
            self.bdim,
            self.avgcount,
            self.ktarget,
            self.knavg,
            self.ktol,
            self.hmin,
            self.hmax,
            self.binsize,
            self.fix_com,
        )
        self.add_action(self.findk)

    def _set_accept_tests(self):
        pass

    def set_control(self, c):
        print("WARNING: findk set control is not defined")

    def get_k(self):
        return self.findk.get_k()

    def get_entries(self):
        return self.findk.get_entries()

    def show_histogram(self):
        hist = self.findk.get_histogram()
        val = np.array([i * self.binsize for i in range(len(hist))]) + 0.5 * self.binsize
        plt.hist(val, weights=hist, bins=len(hist), density=True, stacked=True)
        ###analytical
        k = self.get_k()
        and2 = vec_analytical_d2(val[:-1], k, self.nparticles, self.bdim)
        norm = and2.sum() * self.binsize
        and2 /= norm
        plt.plot(val[:-1], and2, linewidth=2.5, ls="--", color=color_cycle[-1])
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
