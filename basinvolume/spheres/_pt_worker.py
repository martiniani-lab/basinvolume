from __future__ import division
import logging
import numpy as np
from mpi4py import MPI
from basinvolume.utils import get_dist_com
from basinvolume.spheres import RunnerConfig

class PT_Worker:

    def __init__(self, mcrunner, fix_com=True):
        self.comm = MPI.COMM_WORLD
        self.mcrunner = mcrunner
        self.fix_com = fix_com

    def run(self):
        config = RunnerConfig(0, 0, 0, self.mcrunner.red_origin)
        self.comm.Recv(config.data, source=0)
        # Sending an id (first element of the data array) of -1 is the signal to stop working
        while config.get_id() >= 0:
            timeseries = self.__one_iteration(config)
            self.comm.Send(np.append(config.data, timeseries), dest=0)
            self.comm.Recv(config.data, source=0)
        logging.info("Worker finished")

    def __one_iteration(self, config):
        self.mcrunner.set_control(config.get_k(), reset=False)
        self.mcrunner.set_config(config.get_coords(), config.get_energy())
        self.mcrunner.run()
        #collect the results
        result = self.mcrunner.get_results()
        config.set_energy(result.energy)
        config.set_coords(result.coords)
        if self.fix_com:
            config.set_dx(get_dist_com(config.get_coords(), self.mcrunner.red_origin,
                                       self.mcrunner.bdim))
        else:
            config.set_dx(np.linalg.norm(config.get_coords() - self.mcrunner.red_origin))
        return self.mcrunner.get_timeseries(clear=True)
