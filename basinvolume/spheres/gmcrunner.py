import random
import warnings
from past.utils import old_div
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from pele.distance import Distance
from pele.optimize import CVODEBDFOptimizer, ExtendedMixedOptimizer, LBFGS_CPP, ModifiedFireCPP
from pele.potentials import PoweredCosineSum
from mcpele.monte_carlo import MetropolisTest, RecordCoordsTimeseries
from mcpele.galilean_monte_carlo import _BaseGMCRunner
from basinvolume.enums import Minimizer, Interaction
from basinvolume.monte_carlo import (CheckSameMinimumConfig, RecordDisp2Histogram, RecordDisplacementTimeseries,
                                     RecordStepsTimeseries)
from basinvolume.utils import INVERSE_POWER_CVODE_95_ACC, get_mxd_t, write_2d_array_to_hdf5
from .mcrunner import vec_analytical_d2, color_cycle, BV_MCRunner_State


class SpheresGMCRunner(_BaseGMCRunner):
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

        acceptance=0.2,
        adjustf=0.9,
        adjustf_niter=1e4,
        adjustf_navg=100,
        ts_niter=None,
        ts_freq=1,
        record_steps_timeseries=False,
        record_steps_timeseries_every=(1,),
        record_trajectory=False,
        record_trajectory_npoints=1e4,
        single=False,
    ):
        if ts_niter is None:
            ts_niter = niter
        self.ts_niter = ts_niter
        self.ts_freq = ts_freq
        self.record_trajectory = record_trajectory
        self.record_trajectory_npoints = record_trajectory_npoints
        self.record_steps_timeseries = record_steps_timeseries
        self.record_steps_timeseries_every = record_steps_timeseries_every
        # takestep paramteres
        self.adjustf_niter = adjustf_niter
        self.adjustf_navg = adjustf_navg
        self.adjustf = adjustf
        self.acceptance = acceptance
        self.single = single

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

        if use_frozen:
            raise NotImplementedError("use_frozen not implemented")
            # assert distance_method is Distance.CARTESIAN and frozen_atoms is not None
            # red_coords = reduce_coordinates(full_coords, frozen_atoms, len(boxv))
        else:
            red_coords = full_coords

        # TODO CHECK THAT THE FOLLOWING WORKS!
        super().__init__(potential, red_coords, temperature, niter)

        self.boxv = boxv
        self.bdim = len(boxv)
        self.origin = np.array(origin)
        self.red_origin = np.array(origin)
        self.hs_radii = np.array(hs_radii)
        self.red_radii = np.array(hs_radii)
        if use_frozen:
            raise NotImplementedError("use_frozen not implemented")
            # self.red_radii = np.delete(self.red_radii, frozen_atoms)
            # self.red_origin = reduce_coordinates(self.red_origin, frozen_atoms, self.bdim)
            # assert len(self.red_radii) == (len(self.hs_radii) - len(frozen_atoms))
            # assert len(self.red_origin) == self.ndim
            # assert rcontainer is not None
        self.sca = sca
        self.dtol = dtol
        self.eps = eps
        self.nparticles = len(self.red_radii)
        self.use_cell_lists = use_cell_lists
        self.use_frozen = use_frozen
        self.frozen_atoms = frozen_atoms
        self.distance_method = distance_method
        self.rcontainer = rcontainer
        self.equilibration_steps = report_steps + pt_eq_niter
        self.hmin = hmin
        self.hmax = hmax
        self.binsize = hbinsize
        self.avgcount = avgcount

        # compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(
                seed_takestep=random.randint(0, i32max),
                seed_metropolis=random.randint(0, i32max),
            )
        self.seeds = seeds

        self.resample_velocity_steps = niter  # TODO: THIS SHOULD REALLY BECOME SOMETHING ELSE
        self.reflect_boundary = True
        self.reflect_potential = False
        if self.nparticles != 1:
            raise NotImplementedError("nparticles != 1 not implemented")
        super().__init__(potential, red_coords, temperature, niter, stepsize, self.nparticles,
                         self.bdim, self.seeds["seed_takestep"], self.resample_velocity_steps, 0.0, False,
                         self.adjustf_navg, self.adjustf, self.acceptance, self.acceptance, self.reflect_boundary,
                         self.reflect_potential)

        # manage array of rattlers, if not rattler: 1 -> jammed dof
        #                                          0 -> rattler dof
        if rattlers is None:
            self.rattlers = np.array([1.0 for _ in range(self.ndim)], dtype="d")
        else:
            self.rattlers = np.array(rattlers, dtype="d")
        if self.use_frozen:
            raise NotImplementedError("use_frozen not implemented")
            # self.rattlers = reduce_coordinates(self.rattlers, frozen_atoms, self.bdim)
        assert len(self.rattlers) == self.ndim
        assert self.rattlers.all() >= 0 and self.rattlers.all() <= 1

        # rcut set to largest particle diameter
        self.rcut = np.amax(self.hs_radii) * 2.0 * (1.0 + self.sca)
        if self.use_cell_lists:
            if np.amin(self.boxv) // self.rcut <= 3:
                print("warning: use_cell_lists flag was set, rcut is too large though")
                print("setting use_cell_lists to False")
                self.use_cell_lists = False
        self.ncellx_scale = 1.0



        # get potential and minimizer
        self.pot_optimizer = self.get_pot_optimizer()
        self.optimizer = self.get_optimizer()

        # construct base test/action classes
        if record_histogram:
            self._set_record_histogram(self.hmin, self.hmax, self.binsize)

        # construct custom test/action/takestep classes
        self.set_report_steps(report_steps)
        self._set_conf_tests()
        self._set_accept_tests()
        self._set_actions()

        self.bias_potential = potential
        self.bias = bias
        self.full_coords = full_coords
        self.set_bias_parameters(bias, bias_params)

    def _set_accept_tests(self):
        self.metropolis = MetropolisTest(self.seeds["seed_metropolis"])
        self.add_accept_test(self.metropolis)

    def _set_record_histogram(self, hmin, hmax, binsize):
        self.histogram = RecordDisp2Histogram(
            self.red_origin,
            self.rattlers,
            self.bdim,
            hmin,
            hmax,
            binsize,
            self.equilibration_steps,
            fix_com=self.fix_com,
        )
        self.add_action(self.histogram)

    def get_stepsize(self):
        return self.get_timestep()

    def get_status(self):
        """
        overloading the base class method to include stepsize
        """
        status = super().get_status()
        status.stepsize = self.get_stepsize()
        return status

    def _set_actions(self):
        self.time_series = RecordDisplacementTimeseries(
            self.red_origin, self.bdim, self.ts_niter, self.ts_freq, fix_com=self.fix_com
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

    def set_bias_parameters(self, bias, bias_params, reset=True):  # XXX
        """set temperature, canonical control parameter"""
        self.bias_params = bias_params
        if bias == "harmonic":
            self.bias_potential.set_k(bias_params[0])
        elif bias == "radial_gaussian":
            self.bias_potential.set_k(bias_params[0])
            self.bias_potential.set_l0(bias_params[1])
            self.bias_potential.set_r_cutoff(bias_params[2])
            if bias_params[0] == 0.0:
                # remove the log part for k= 0 run
                self.bias_potential.set_log_prefactor(0.0)
            else:
                self.bias_potential.set_log_prefactor(
                    1.0
                )  # If temperature != 1.0, this should be 1/beta so that exp(- beta log_term ) = r^(1-d)
        else:
            raise NotImplementedError
        if reset:
            self.reset_energy()

    def get_pot_optimizer(self):
        # here put a flag and pick potential
        if self.interaction is Interaction.HS_WCA:
            raise NotImplementedError("HS_WCA not implemented")
            # pot_optimizer = HS_WCA(
            #     distance_method=self.distance_method,
            #     pot_kwargs=self.pot_kwargs,
            #     use_cell_lists=self.use_cell_lists,
            #     use_frozen=self.use_frozen,
            #     eps=self.eps,
            #     sca=self.sca,
            #     radii=self.hs_radii,
            #     boxvec=self.boxv,
            #     reference_coords=self.origin,
            #     ndim=self.bdim,
            #     ncellx_scale=self.ncellx_scale,
            #     frozen_atoms=self.frozen_atoms,
            # )
        elif self.interaction is Interaction.INVERSE_POWER_STILLINGER:
            raise NotImplementedError("INVERSE_POWER_STILLINGER not implemented")
            # pow = self.pot_kwargs["pow"]
            # rcut = self.pot_kwargs["rcut"]
            # pot_optimizer = InversePowerStillingerCut(
            #     pow,
            #     self.stillinger_a_radii,
            #     ndim=self.bdim,
            #     boxvec=self.boxv,
            #     rcut=rcut,
            #     use_cell_lists=True,
            # )
        elif self.interaction is Interaction.INVERSE_POWER:
            raise NotImplementedError("INVERSE_POWER not implemented")
            # power = self.pot_kwargs["power"]
            # eps = self.pot_kwargs["eps"]
            # if len(self.hs_radii) * self.bdim > 100:
            #     use_cell_lists = True
            # else:
            #     use_cell_lists = False
            # pot_optimizer = InversePower(
            #     power,
            #     eps,
            #     ndim=self.bdim,
            #     boxvec=self.boxv,
            #     radii=self.hs_radii,
            #     use_cell_lists=use_cell_lists,
            # )
        elif self.interaction is Interaction.NEGATIVE_COS:
            pot_optimizer = PoweredCosineSum(
                dim=self.ndim,
                period=self.pot_kwargs["period"],
                power=0.5,
                offset=float(self.ndim),
            )

        else:
            raise NotImplementedError
        return pot_optimizer

    def get_optimizer(self):
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
            try:
                atol = self.opt_kwargs["atol_values"][str(len(self.start_coords) // self.bdim)]
                rtol = self.opt_kwargs["rtol_values"][str(len(self.start_coords) // self.bdim)]
            except KeyError:
                atol = 0.1 * INVERSE_POWER_CVODE_95_ACC[len(self.start_coords) // self.bdim]
                rtol = 0.1 * INVERSE_POWER_CVODE_95_ACC[len(self.start_coords) // self.bdim]
            optimizer = CVODEBDFOptimizer(
                self.pot_optimizer,
                self.start_coords,
                tol=self.opt_tol,
                atol=atol,
                rtol=rtol,
            )
        elif self.minimizer is Minimizer.MXD:
            try:
                atol = self.opt_kwargs["atol_values"][str(len(self.start_coords) // self.bdim)]
                rtol = self.opt_kwargs["rtol_values"][str(len(self.start_coords) // self.bdim)]
            except KeyError:
                atol = 0.1 * INVERSE_POWER_CVODE_95_ACC[len(self.start_coords) // self.bdim]
                rtol = 0.1 * INVERSE_POWER_CVODE_95_ACC[len(self.start_coords) // self.bdim]
            if self.interaction is Interaction.NEGATIVE_COS:
                global_symmetry_offset = np.zeros((len(self.start_coords), len(self.start_coords)))
            else:
                global_symmetry_offset = []
            optimizer = ExtendedMixedOptimizer(
                self.pot_optimizer,
                self.start_coords,
                tol=self.opt_tol,
                nsteps=1e7,
                atol=atol,
                rtol=rtol,
                T=get_mxd_t(self.nparticles),
                global_symmetry_offset=global_symmetry_offset,
            )
        elif self.minimizer is Minimizer.FIRE:
            try:
                opt_dtmax = self.opt_kwargs["opt_dtmax"]
            except KeyError:
                opt_dtmax = 1.0
            optimizer = ModifiedFireCPP(
                self.start_coords,
                self.pot_optimizer,
                dtmax=opt_dtmax,
                maxstep=self.opt_maxstep,
                tol=self.opt_tol,
                nsteps=self.opt_nsteps,
            )
        else:
            raise NotImplementedError("minimizer={} not implemented".format(self.minimizer))
        return optimizer

    def _get_check_same_minimum(self):
        use_cgd = self.minimizer is Minimizer.CG
        if self.interaction is Interaction.NEGATIVE_COS:
            csm = CheckSameMinimumConfig(
                self.pot_optimizer,
                self.red_origin,
                self.dtol,
                opt=self.optimizer,
                opt_tol=self.opt_tol,
                opt_maxiter=self.opt_nsteps,
            )
        else:
            raise NotImplementedError("interaction={} not implemented".format(self.interaction))
            # csm = CheckSameMinimum(
            #     self.pot_optimizer,
            #     self.red_origin,
            #     self.rattlers,
            #     self.dtol,
            #     opt=self.optimizer,
            #     opt_tol=self.opt_tol,
            #     opt_maxiter=self.opt_nsteps,
            #     bdim=self.bdim,
            #     eqsteps=self.equilibration_steps,
            #     use_cgd=use_cgd,
            #     perform_convergence_test=self.perform_convergence_test,
            #     collect_minima_list=self.collect_minima_list,
            # )
        return csm

    def _set_conf_tests(self):
        if self.use_frozen:
            raise NotImplementedError("use_frozen not implemented")
            # self.conftest0 = CheckSphericalContainer(self.rcontainer, self.bdim)
            # self.add_conf_test(self.conftest0)
        if self.interaction is Interaction.HS_WCA:
            raise NotImplementedError("HS_WCA not implemented")
            # if self.distance_method is Distance.PERIODIC:
            #     if (
            #         self.checkoverlap_cell_lists is None and self.use_cell_lists
            #     ) or self.checkoverlap_cell_lists:
            #         self.conftest1 = CheckOverlapPeriodicCellLists(
            #             self.hs_radii,
            #             self.boxv,
            #             ncellx_scale=self.ncellx_scale,
            #             use_frozen=self.use_frozen,
            #             frozen_atoms=self.frozen_atoms,
            #             reference_coords=self.origin,
            #         )
            #     else:
            #         self.conftest1 = CheckOverlapPeriodic(
            #             self.hs_radii,
            #             self.boxv,
            #             use_frozen=self.use_frozen,
            #             reference_coords=self.origin,
            #             frozen_atoms=self.frozen_atoms,
            #         )
            # elif self.distance_method is Distance.CARTESIAN:
            #     if (
            #         self.checkoverlap_cell_lists is None and self.use_cell_lists
            #     ) or self.checkoverlap_cell_lists:
            #         self.conftest1 = CheckOverlapCartesianCellLists(
            #             self.hs_radii,
            #             self.boxv,
            #             ncellx_scale=self.ncellx_scale,
            #             use_frozen=self.use_frozen,
            #             frozen_atoms=self.frozen_atoms,
            #             reference_coords=self.origin,
            #         )
            #     else:
            #         self.conftest1 = CheckOverlapCartesian(
            #             self.hs_radii,
            #             self.bdim,
            #             use_frozen=self.use_frozen,
            #             reference_coords=self.origin,
            #             frozen_atoms=self.frozen_atoms,
            #         )
            # elif self.distance_method is Distance.LEES_EDWARDS:
            #     if (
            #         self.checkoverlap_cell_lists is None and self.use_cell_lists
            #     ) or self.checkoverlap_cell_lists:
            #         self.conftest1 = CheckOverlapLeesEdwardsCellLists(
            #             self.hs_radii,
            #             self.boxv,
            #             shear=self.pot_kwargs["shear"],
            #             ncellx_scale=self.ncellx_scale,
            #             use_frozen=self.use_frozen,
            #             frozen_atoms=self.frozen_atoms,
            #             reference_coords=self.origin,
            #         )
            #     else:
            #         self.conftest1 = CheckOverlapLeesEdwards(
            #             self.hs_radii,
            #             self.boxv,
            #             shear=self.pot_kwargs["shear"],
            #             use_frozen=self.use_frozen,
            #             reference_coords=self.origin,
            #             frozen_atoms=self.frozen_atoms,
            #         )
            # else:
            #     raise NotImplementedError("Specified distance method " "not implemented.")
            # self.add_late_conf_test(self.conftest1)
        else:
            warnings.warn(
                "not setting an excluded volume conf_test because using other potential than hs_wca"
            )
        self.conftest2 = self._get_check_same_minimum()
        self.add_late_conf_test(self.conftest2)

    def dump_minima_list(self, fname):
        """write minima list to pele database"""
        if self.interaction is Interaction.HS_WCA:
            raise NotImplementedError("HS_WCA not implemented")
            # system = HSWCASystem(
            #     self.eps,
            #     self.sca,
            #     self.hs_radii,
            #     self.boxv,
            #     bdim=self.bdim,
            #     dtol=self.dtol,
            #     etol=1,
            # )
            # db = system.create_database(fname)
            # minima_dicts = []
            # add origin to database, with _id == 0, to make post processing
            # possible
            # for origin: set count to zero, but it does not have meaning, since we are only recording minima when quench took us to neighbor
            # distance should be zero because it is distance to itself
            # mindict0 = dict(
            #     energy=self.pot_optimizer.getEnergy(self.red_origin),
            #     coords=self.origin,
            #     user_data=dict(count=0, distance=0),
            # )
            # minima_dicts.append(mindict0)
            # add neighboring minima to database
            # self.conftest2.dump_minima(minima_dicts)
            # add spring constant to user_data
            # for m in minima_dicts:
                # XXX Is this good?
                # m["user_data"].update(k=self.bias_params[0])
                # if self.use_frozen:
                #     redcoords = m["coords"]
                #     m["coords"] = full_coordinates(
                #         redcoords, self.origin, self.frozen_atoms, self.bdim
                #     )
            # assert len(minima_dicts) == self.conftest2.ml_nr_distinct_minima() + 1
            # logging.info("Number of minima: %i" % len(minima_dicts))
            # size_estimate = len(minima_dicts) * len(self.origin)
            # if size_estimate > 1e8:
            #     logging.warning(
            #         "The size of the minima dictionary is extremely "
            #         "long. This can cause the program to run out of "
            #         "memory and crash (indicated by MPI noticing that "
            #         "a process has exited on signal 9 (Killed)). You "
            #         "should probably deactivate minima collection "
            #         "(--nocollectminima)."
            #     )
            # db.engine.execute(Minimum.__table__.insert(), minima_dicts)
            # db.session.commit()
        else:
            warnings.warn("dump_minima_list is not implemented for potentials other than hs_wca")

    def dump_histogram(self, fname):
        """write histogram to fname"""
        Emin, Emax = self.histogram.get_bounds_val()
        histl = self.histogram.get_histogram()
        hist = np.array(histl)
        Energies, step = np.linspace(Emin, Emax, num=len(hist), endpoint=False, retstep=True)
        Energies += 0.5 * step
        assert abs(step - self.binsize) < old_div(self.binsize, 100)
        np.savetxt(fname, np.column_stack((Energies, hist)), delimiter="\t")
        mean, variance = self.histogram.get_mean_variance()
        return mean, variance

    def dump_timeseries(self, fname, clear=True):  # Writes ABSOLUTE DISPLACEMENTS from the origin
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
                fname + ".every{}".format(self.record_steps_timeseries_every[i]),
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

    def check_convergence(self, nr_steps_to_check=10000, rel_std_threshold=0.05):
        return self.time_series.check_convergence(
            nr_steps_to_check=nr_steps_to_check,
            rel_std_threshold=rel_std_threshold,
        )

    def show_histogram(self):
        hist = self.histogram.get_histogram()
        val = np.array([i * self.binsize for i in range(len(hist))]) + 0.5 * self.binsize
        plt.hist(val, weights=hist, bins=len(hist))
        plt.show()

    def show_histogram_kmax(self):
        """
        shows the histogram against the analytical curve when k=kmax
        this function is useful for testing
        """
        hist = self.histogram.get_histogram()
        val = np.array([i * self.binsize for i in range(len(hist))]) + 0.5 * self.binsize
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
            vec_analytical_d2(val, self.bias_params[0], self.nparticles),
            quad(
                vec_analytical_d2,
                bincenters[0],
                bincenters[-1],
                args=(self.bias_params[0], self.nparticles),
            )[0],
        )
        plt.plot(bincenters, and2, linewidth=2.5, ls="--", color=color_cycle[-1])
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
            bias_params=self.bias_params,
            stepsize=self.get_timestep(),
            counters=self.get_counters(),
            takestep_count=self.get_count(),
            step_adaptation_counters=self.get_adaptation_counters(),
        )

    def set_complete_state(self, mcrunner_state):
        self.set_config(mcrunner_state.coords, mcrunner_state.energy)
        self.set_bias_parameters(self.bias, mcrunner_state.bias_params, reset=False)
        self.set_counters(mcrunner_state.counters)
        self.set_timestep(mcrunner_state.stepsize)
        self.set_count(mcrunner_state.takestep_count)
        self.set_adaptation_counters(mcrunner_state.step_adaptation_counters)
