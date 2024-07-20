import numpy as np
from mcpele.monte_carlo import RecordCoordsTimeseries
from mcpele.guided_monte_carlo import _BaseGuidedMCRunner
from basinvolume.monte_carlo import (
    RecordDisplacementTimeseries,
    CheckHyperCubicContainerGMC,
    RecordStepsTimeseries,
    RecordDisp2Histogram,
)
from .mcrunner import BV_MCRunner_State


class HypercubeGuidedMCRunner(_BaseGuidedMCRunner):
    def __init__(
            self,
            bias_potential,
            full_coords,
            temperature,
            stepsize,
            standard_deviation,
            niter,
            origin,
            sidelength=1,
            bias="harmonic",
            bias_params=(1.0,),
            acceptance=0.2,
            adjustf=0.9,
            hmin=0,
            hmax=1,
            hbinsize=0.001,
            adjustf_niter=1e4,
            adjustf_navg=100,
            pt_eq_niter=0,
            ts_niter=None,
            ts_freq=1,
            seeds=None,
            record_steps_timeseries=False,
            record_steps_timeseries_every=(1,),
            record_trajectory=False,
            record_trajectory_npoints=1e4,
            single=False,
            record_histogram=False,
            hyperball=False):
        print("Using Guided MC.")
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(
                seed_takestep=np.random.randint(i32max)
            )
        super().__init__(bias_potential, full_coords, temperature, niter, stepsize, standard_deviation,
                         seeds["seed_takestep"], True, 0.0, adjustf_navg, adjustf, acceptance, acceptance)

        # Necessary variables for PT.
        self.niter = niter
        self.bdim = len(full_coords)
        self.nparticles = 1
        self.equilibration_steps = adjustf_niter + pt_eq_niter
        self.red_origin = np.array(origin)
        self.bias = bias
        self.bias_potential = bias_potential
        self.set_bias_parameters(self.bias, bias_params)

        if ts_niter is None:
            ts_niter = niter

        self.set_report_steps(adjustf_niter)  # set number of iterations for which steps are adapted

        if not hyperball:
            self.conftest = CheckHyperCubicContainerGMC(np.zeros(self.bdim), sidelength, self.bdim, True)
        else:
            raise RuntimeError("Hyperball not implemented for GMC")
        self.add_late_conf_test(self.conftest)

        self.action_record_displ = RecordDisplacementTimeseries(
            np.array(origin), self.bdim, ts_niter, ts_freq, fix_com=False)
        self.add_action(self.action_record_displ)
        if record_histogram:
            self.hist_action = RecordDisp2Histogram(
                np.array(origin),
                np.ones(self.bdim),
                self.bdim,
                hmin,
                hmax,
                hbinsize,
                self.equilibration_steps,
                fix_com=False)
            self.add_action(self.hist_action)
        if record_trajectory:
            rte = max(
                int((self.niter - self.equilibration_steps) / record_trajectory_npoints),
                1)
            self.record_trajectory = RecordCoordsTimeseries(
                self.bdim, record_every=rte, eqsteps=self.equilibration_steps)
            self.add_action(self.record_trajectory)
        if record_steps_timeseries:
            self.steps_timeseries_list = []
            for freq in record_steps_timeseries_every:
                self.steps_timeseries_list.append(RecordStepsTimeseries(
                    np.array(origin), np.ones(self.bdim), self.bdim, ts_niter, freq))
            for action in self.steps_timeseries_list:
                self.add_action(action)

    def set_control(self, c):
        raise NotImplementedError

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
        self.set_bias_parameters(
            self.bias, mcrunner_state.bias_params, reset=False
        )
        self.set_counters(mcrunner_state.counters)
        self.set_timestep(mcrunner_state.stepsize)
        self.set_count(mcrunner_state.takestep_count)
        self.set_adaptation_counters(mcrunner_state.step_adaptation_counters)

    def get_timeseries(self, clear=False):
        timeseries = np.array(self.action_record_displ.get_time_series())
        if clear:
            self.action_record_displ.clear()
        return timeseries
