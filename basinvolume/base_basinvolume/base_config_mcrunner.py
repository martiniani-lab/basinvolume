from __future__ import division, print_function
from future import standard_library

standard_library.install_aliases()
from builtins import str
import numpy as np
import os
from basinvolume.utils import trymakedir
import configparser
import time
import warnings


class BaseConfigBVMCRunner(object):
    """
    Base class for configuration MC runners with common functionality
    """

    def __init__(self, rank, nprocs):
        self.rank = rank
        self.nprocs = nprocs

    def _get_histogram_bin(self, k):
        """automatically estimate size of histogram"""
        hmax = self.displ_k_min * k
        hbinsize = hmax * 0.0001
        return hbinsize

    def _initialise(self):
        """initialisation function"""
        # change directory only at the end of initialise
        self._print_initialise()
        base_dir = getattr(self, 'base_dir', getattr(self, 'base_directory', None))
        if base_dir:
            os.chdir(base_dir)

    def _print_initialise(self):
        """print initialisation information"""
        base_dir = getattr(self, 'base_dir', getattr(self, 'base_directory', None))
        if base_dir:
            trymakedir(base_dir)
            if self.rank == 0:
                self._print_parameters()

    def _print_parameters(self):
        """
        print parameters to config file
        """
        configfile = getattr(self, 'configfile', None)
        if configfile:
            f = open(configfile, "w")
            self._write_sim_params(f)
            f.close()

    def _write_sim_params(self, f):
        """
        write simulation parameters - should be implemented by subclasses
        """
        raise NotImplementedError("_write_sim_params must be implemented by subclasses")

    def print_success_all(self, success):
        """
        print whether calculation has completed successfully
        """
        configfile = getattr(self, 'configfile', None)
        if configfile and self.rank == 0:
            configf = configparser.ConfigParser()
            configf.read(str(configfile))
            for i in range(self.nprocs):
                configf.set("STATUS", "success_rank{}".format(str(i)), success)
            configf.write(open(str(configfile), "w")) 