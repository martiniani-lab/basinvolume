from __future__ import division
import logging
import numpy as np
from mpi4py import MPI
from basinvolume.utils import get_dist_com
from basinvolume.spheres import ReplicaState

class PT_Worker(object):

    def __init__(self, mcrunner, fix_com=True):
        self.comm = MPI.COMM_WORLD
        self.mcrunner = mcrunner
        self.fix_com = fix_com

    def run(self):
        state = ReplicaState(0, self.mcrunner.get_complete_state())
        data_buffer = np.empty(state.size())
        self.comm.Recv(data_buffer, source=0)
        state.deserialize(data_buffer)
        while state.id >= 0:
            # Sending an id (first element of the data array) of -1 is the signal to stop working
            timeseries = self._one_iteration(state)
            self.comm.Send(np.append(state.serialize(), timeseries), dest=0)
            self.comm.Recv(data_buffer, source=0)
            state.deserialize(data_buffer)
        logging.info("Worker finished")

    def _one_iteration(self, state):
        self.mcrunner.set_complete_state(state)
        if np.isnan(state.energy):
            # Setting state.energy to NaN is the signal for necessary energy recalculation
            state.energy = self.mcrunner.potential.getEnergy(state.coords)
            self.mcrunner.set_config(state.coords, state.energy)
        self.mcrunner.run()

        #collect the results
        state.set_mc_state(self.mcrunner.get_complete_state())
        if self.fix_com:
            state.dx = get_dist_com(state.coords, self.mcrunner.red_origin,
                                    self.mcrunner.bdim)
        else:
            state.dx = np.linalg.norm(state.coords - self.mcrunner.red_origin)
        return self.mcrunner.get_timeseries(clear=True)
