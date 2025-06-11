from builtins import object


class BV_MCRunner_State(object):
    """
    This class saves the state of an BV_MCrunner in a NumPy array
    """

    def __init__(
        self,
        state=None,
        coords=None,
        energy=0.0,
        bias_params=None,
        stepsize=0.0,
        counters=None,
        takestep_count=0,
        step_adaptation_counters=None,
    ):
        if state is None:
            self.coords = coords
            self.energy = energy
            self.bias_params = bias_params if bias_params is not None else [0.0]
            self.stepsize = stepsize
            self.counters = counters
            self.takestep_count = takestep_count
            self.step_adaptation_counters = step_adaptation_counters
        else:
            self._set_state(state)

    def _set_state(self, state):
        self.coords = state.coords
        self.energy = state.energy
        self.bias_params = state.bias_params
        self.stepsize = state.stepsize
        self.counters = state.counters
        self.takestep_count = state.takestep_count
        self.step_adaptation_counters = state.step_adaptation_counters 