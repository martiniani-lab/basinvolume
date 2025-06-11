from __future__ import division, print_function
import numpy as np
from mcpele.monte_carlo import _BaseMCRunner
from basinvolume.utils import write_2d_array_to_hdf5
from basinvolume.base_basinvolume.mcrunner_state import BV_MCRunner_State

try:
    from mcpele.monte_carlo import RecordCoordsTimeseries
except Exception as e:
    print(e)


class BaseBVMCRunner(_BaseMCRunner):
    """
    Base class for Basin Volume MC runners with common functionality
    """

    def dump_timeseries(self, fname, clear=True):
        """
        Write time series to fname, returns the timeseries
        Writes ABSOLUTE DISPLACEMENTS from the origin
        """
        timeseries = np.array(self.action_record_displ.get_time_series())
        np.savetxt(fname, timeseries)
        if clear:
            self.action_record_displ.clear()
        return timeseries

    def get_timeseries(self, clear=False):
        """
        Get time series data, optionally clearing it
        """
        timeseries = np.array(self.action_record_displ.get_time_series())
        if clear:
            self.action_record_displ.clear()
        return timeseries

    def check_convergence(self, nr_steps_to_check=10000, rel_std_threshold=0.05):
        """
        Check convergence of the timeseries
        """
        return self.action_record_displ.check_convergence(
            nr_steps_to_check=nr_steps_to_check,
            rel_std_threshold=rel_std_threshold,
        )

    def get_mean_variance_coordinate_vector(self):
        """
        Get mean and variance of coordinate vector from timeseries
        """
        timeseries = self.get_timeseries()
        mean = np.mean(timeseries, axis=0)
        variance = np.var(timeseries, axis=0)
        return mean, variance

    def dump_trajectory(self, fname, clear=True):
        """
        Write trajectory to fname if recording is enabled
        """
        if hasattr(self, 'record_trajectory') and self.record_trajectory:
            trajectory = self.get_trajectory()
            write_2d_array_to_hdf5(fname, trajectory)
            if clear:
                self.clear_trajectory()
            return trajectory
        else:
            raise RuntimeError("trajectory recording is not enabled")

    def get_trajectory(self):
        """
        Get trajectory data if recording is enabled
        """
        if hasattr(self, 'record_trajectory') and self.record_trajectory:
            return np.array(self.record_trajectory.get_trajectory())
        else:
            raise RuntimeError("trajectory recording is not enabled")

    def clear_trajectory(self):
        """
        Clear trajectory data if recording is enabled
        """
        if hasattr(self, 'record_trajectory') and self.record_trajectory:
            self.record_trajectory.clear()

    def get_complete_state(self):
        """
        Get complete state of the MCRunner for checkpointing
        """
        state = self.get_status()
        counters = getattr(self, 'counters', None)
        takestep_count = getattr(self.takestep, 'get_count', lambda: 0)()
        step_adaptation_counters = getattr(self.takestep, 'get_adaptation_counters', lambda: None)()
        
        return BV_MCRunner_State(
            coords=state.coords,
            energy=state.energy,
            bias_params=getattr(self, 'bias_params', [0.0]),
            stepsize=getattr(state, 'stepsize', self.get_stepsize() if hasattr(self, 'get_stepsize') else 0.0),
            counters=counters,
            takestep_count=takestep_count,
            step_adaptation_counters=step_adaptation_counters,
        )

    def set_complete_state(self, mcrunner_state):
        """
        Set complete state of the MCRunner from a saved state
        """
        self.set_coords(mcrunner_state.coords)
        self.set_energy(mcrunner_state.energy)
        if hasattr(self, 'set_bias_parameters'):
            self.set_bias_parameters(getattr(self, 'bias', 'harmonic'), mcrunner_state.bias_params)
        if hasattr(self.takestep, 'set_stepsize'):
            self.takestep.set_stepsize(mcrunner_state.stepsize)
        if hasattr(self.takestep, 'set_count') and mcrunner_state.takestep_count:
            self.takestep.set_count(mcrunner_state.takestep_count)
        if hasattr(self.takestep, 'set_adaptation_counters') and mcrunner_state.step_adaptation_counters:
            self.takestep.set_adaptation_counters(mcrunner_state.step_adaptation_counters) 