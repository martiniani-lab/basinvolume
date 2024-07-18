from __future__ import division
from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import str
from builtins import range
import numpy as np
import os
from pele.potentials import Harmonic, RadialGaussian
from basinvolume.spheres import ConfigMCRunner
from basinvolume.hypercube import HypercubeMCrunner, HypercubeGMCRunner
from basinvolume.utils import trymakedir
import configparser
import time
import warnings


class _hypercube_bv_mcrunner(ConfigMCRunner):
    """ """

    def __init__(self, rank, nprocs):
        self.rank = rank
        self.nprocs = nprocs

    def __call__(
        self,
        base_dir,
        bias = "harmonic",
        bias_params=[1.0],
        stepsize=1e-3,
        niter=2e4,
        hmin=0,
        hmax=100,
        hbinsize=1,
        acceptance=0.2,
        adjustf=0.9,
        adjustf_niter=5e3,
        adjustf_navg=100,
        pt_eq_niter=0,
        ts_niter=None,
        ts_freq=1,
        single=False,
        seeds=None,
        record_trajectory=False,
        record_trajectory_npoints=1e4,
        record_histogram=False,
        verbose=False,
        runner="metropolis"
    ):

        self.temperature = 1.0
        self._set_paths(base_dir)
        self._import_packing_config_files()
        self.coords = np.zeros(int(self.ndof))
        self.bias = bias
        hbinsize = self._get_histogram_bin(bias_params[0])

        # set parameters
        kwargs = dict(
            sidelength=self.sidelength,
            bias = self.bias,
            bias_params=bias_params,
            acceptance=acceptance,
            adjustf=adjustf,
            adjustf_niter=adjustf_niter,
            adjustf_navg=adjustf_navg,
            pt_eq_niter=pt_eq_niter,
            ts_niter=ts_niter,
            ts_freq=ts_freq,
            hmin=hmin,
            hmax=hmax,
            hbinsize=hbinsize,
            record_trajectory=record_trajectory,
            record_trajectory_npoints=record_trajectory_npoints,
            seeds=seeds,
            single=single,
            record_histogram=record_histogram,
        )

        self.mc_params = dict(temperature=self.temperature, niter=niter, stepsize=stepsize)
        self.mc_params.update(kwargs)
        if seeds is None:
            warnings.warn("seeds not passed")

        self._initialise()

        # construct mcrunner
        # self.coords is origin, set initial configuration and origin to be the same
        # harmonic bias_potential withOUT fixed centre of mass
        # XXX BIAS 
        ndim = self.coords.shape[0]
        if bias == "harmonic":
            bias_params = [1.0]
            bias_potential = Harmonic(self.coords, bias_params[0], bdim = ndim, com=False)
        elif bias == "radial_gaussian":
            bias_params = [1.0, 1.0, 1.0]
            bias_potential = RadialGaussian(self.coords, bias_params[0], bias_params[1], bdim = ndim, com=False)
        else:
            raise NotImplementedError(
                "bias={} not implemented".format(bias)
            )
        
        kwargs["bias_params"] = bias_params # Needed to properly initialise with the right lengths in each list of parameters

        if runner == "metropolis":
            mcrunner = HypercubeMCrunner(
                bias_potential, self.coords, self.temperature, stepsize, niter, self.coords, **kwargs
            )
        else:
            if runner != "galilean":
                raise ValueError("runner={} not implemented".format(runner))
            mcrunner = HypercubeGMCRunner(
                bias_potential, self.coords, self.temperature, stepsize, niter, self.coords, **kwargs)
        return mcrunner

    def _set_paths(self, base_dir):
        """
        set base_directory, packings_directory and configpaths, configfile
        """
        dlist = base_dir.split("_")
        assert dlist[2] == "hypercube"
        if not os.path.isabs(base_dir):
            base_directory = os.path.join(os.getcwd(), base_dir)
            assert os.path.exists(base_directory)
        self.base_directory = base_directory

        dname = dlist[2] + "_" + dlist[3] + "_" + dlist[4]
        self.findk_configpath = os.path.join(self.base_directory, "findk_" + dname + ".config")
        self.kmin_configpath = os.path.join(self.base_directory, "kmin_" + dname + ".config")
        self.configfile = "{}/explore_{}.config".format(self.base_directory, dname)

    def _get_histogram_bin(self, k):
        """automatically estimate size of histogram"""
        hmax = self.displ_k_min * k  # self.displ_k_max*self.kmax
        hbinsize = hmax * 0.0001
        return hbinsize

    def _initialise(self):
        """initialisation function"""
        # change directory only at the end of initialise
        self._print_initialise()
        os.chdir(self.base_directory)

    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        if self.rank == 0:
            self._print_parameters()

    def _write_sim_params(self, f):
        """
        write simulation parameters
        """
        f.write("#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n")
        f.write("#Explore_Jammed_Packings wrapper class input parameters\n")
        f.write("[BV_HYPERCUBE]\n")
        f.write("ndof: {}\n".format(self.ndof))
        f.write("sidelength: {}\n".format(self.sidelength))
        f.write("[MCRUNNER]\n")
        for key, value in list(self.mc_params.items()):
            f.write("{}: {}\n".format(key, value))
        f.write("[STATUS]\n")
        for i in range(self.nprocs):
            f.write("success_rank{}: {}\n".format(str(i), "False"))

    def _import_packing_config_files(self):
        configf = configparser.ConfigParser()
        configf.read(str(self.findk_configpath))
        kmax_ndof = configf.getfloat("FINDK_HYPERCUBE", "ndof")
        kmax_sidelength = configf.getfloat("FINDK_HYPERCUBE", "sidelength")
        self.kmax = configf.getfloat("FINDK", "kmax")
        self.prob_kmax = configf.getfloat("FINDK", "prob")
        self.displ_k_max = configf.getfloat("FINDK", "displ_k_max")
        self.var_displ_k_max = configf.getfloat("FINDK", "var_displ_k_max")
        configf.read(str(self.kmin_configpath))
        self.displ_k_min = configf.getfloat("KMIN", "displ_k_min")
        self.var_displ_k_min = configf.getfloat("KMIN", "var_displ_k_min")
        kmin_ndof = configf.getfloat("KMIN_HYPERCUBE", "ndof")
        kmin_sidelength = configf.getfloat("KMIN_HYPERCUBE", "sidelength")
        assert kmax_ndof == kmin_ndof
        assert kmax_sidelength == kmin_sidelength
        self.ndof = kmin_ndof
        self.sidelength = kmin_sidelength

    def print_success_all(self, success):
        """
        print whether calculation has completed successfully
        """
        assert hasattr(self, "configfile")
        if self.rank == 0:
            configf = configparser.ConfigParser()
            configf.read(str(self.configfile))
            for i in range(self.nprocs):
                configf.set("STATUS", "success_rank{}".format(str(i)), success)
            configf.write(open(str(self.configfile), "w"))


if __name__ == "__main__":

    # first 7 primary pseudo perfect numbers
    pppn = [2, 6, 42, 1806, 47058, 2214502422, 52495396602]
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])

    sim = _hypercube_bv_mcrunner(0, 1)
    mcrunner = sim("explore_bv_hypercube_n100_l1", seeds=seeds, verbose=True, niter=1e6)
    print("simulation started")
    start = time.time()
    mcrunner.run()
    end = time.time()
    print("time elapsed", end - start)
    status = mcrunner.get_status()
    print(status)
