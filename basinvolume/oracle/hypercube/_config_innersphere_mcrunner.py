from __future__ import division
from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import str
import numpy as np
import os
from mcpele.monte_carlo import NullPotential
from basinvolume.spheres import _configure_mcrunner
from basinvolume.utils import trymakedir, view_traceback
from basinvolume.hypercube import HypercubeInnerSphereMCrunner
import configparser
import time
import warnings


class _hypercube_innersphere_mcrunner(_configure_mcrunner):
    """this is a class that implements a mcrunner that samples the inner sphere of a basin

    when niter=None, niter is set equal to exact number of PT niter
    """

    def __init__(
        self,
        base_dir,
        niter=None,
        hmin=0,
        hmax=0.01,
        hbinsize=0.0005,
        seeds=None,
        record_histogram=False,
        verbose=False,
    ):

        self.temperature = 1.0

        self._set_paths(base_dir)
        self._import_packing_config_files()
        self.k = 1.0 / self.u2_k0
        self.stepsize = 1.0 / np.sqrt(self.k)
        self.coords = np.ones(
            self.ndof
        )  # *0.32 #CHANGE THIS: I have shifted the centre to see the effect
        if niter is not None:
            self.niter = niter

        # self.mc_params = dict(k=k, temperature=temperature, )
        kwargs = dict(
            hmin=hmin,
            hmax=hmax,
            hbinsize=hbinsize,
            seeds=seeds,
            record_histogram=record_histogram,
            sidelength=self.sidelength,
        )

        self.mc_params = dict(
            k=self.k,
            temperature=self.temperature,
            niter=self.niter,
            stepsize=self.stepsize,
        )
        self.mc_params.update(kwargs)
        # add seeds dictionary to mc_params
        if seeds is None:
            warnings.warn("seeds not passed")

        # construct mcrunner
        potential = NullPotential()
        self.mcrunner = HypercubeInnerSphereMCrunner(
            potential,
            self.coords,
            self.temperature,
            self.stepsize,
            self.niter,
            self.coords,
            **kwargs
        )

        self._initialise()

    def run(self):
        try:
            self.mcrunner.run()
            self._print_results()
            self._print_success(True)
        except:
            view_traceback()
            self._print_success(False)

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
        self.findk_configpath = os.path.join(
            self.base_directory, "findk_" + dname + ".config"
        )
        configfile = "innersphere_" + dname
        self.configfile = "{}/{}.config".format(self.base_directory, configfile)

    def _import_packing_config_files(self):
        configf = configparser.ConfigParser()
        configf.read(str(self.findk_configpath))
        self.ndof = configf.getfloat("FINDK_HYPERCUBE", "ndof")
        self.sidelength = configf.getfloat("FINDK_HYPERCUBE", "sidelength")
        self.kmax = configf.getfloat("FINDK", "kmax")
        self.prob_kmax = configf.getfloat("FINDK", "prob")
        self.displ_k_max = configf.getfloat("FINDK", "displ_k_max")
        self.var_displ_k_max = configf.getfloat("FINDK", "var_displ_k_max")
        # import mean displacement of replica with second largest k
        path = os.path.join(self.base_directory, "0/hist_mean")
        fileHandle = open(path, "r")
        lineList = fileHandle.readlines()
        fileHandle.close()
        niter, u2, var, std_err = lineList[-1].split()
        self.u2_k0 = float(u2)
        self.var_k0 = float(var)
        self.niter = int(niter) + 1

    def _initialise(self):
        self._print_initialise()

    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        self._print_parameters()

    def _write_sim_params(self, f):
        """
        write simulation parameters
        """
        f.write("#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n")
        f.write("#Explore_Jammed_Packings wrapper class input parameters\n")
        f.write("[INNERSPHERE_HYPERCUBE]\n")
        f.write("ndof: {}\n".format(self.ndof))
        f.write("sidelength: {}\n".format(self.sidelength))
        f.write("[INNERSPHERE_MCRUNNER]\n")
        for key, value in list(self.mc_params.items()):
            f.write("{}: {}\n".format(key, value))

    def _print_results(self):
        """
        note that self.displ_k_min *= 1.5 to account for the limited computation time,
        this is just an approximation
        """
        fname = self.configfile
        f = open(fname, "a")
        f.write("[INNERSPHERE_MCRUNNER_STATUS]\n")
        status = self.mcrunner.get_status()
        for key, value in list(status.items()):
            f.write("{}: {}\n".format(key, value))
        f.close()
        path = os.path.join(self.base_directory, "inner_sphere.timeseries")
        self.mcrunner.dump_timeseries(path, clear=False)


if __name__ == "__main__":

    pppn = [2, 6, 42, 1806, 47058, 2214502422, 52495396602]
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])

    sim = _hypercube_innersphere_mcrunner(
        "explore_bv_hypercube_n93_l1", niter=1e5, seeds=seeds, verbose=False
    )
    print("simulation started")
    start = time.time()
    sim.run()
    end = time.time()
    print("time elapsed", end - start)
    status = sim.mcrunner.get_status()
    print(status)
    print("stepsize: ", sim.mcrunner.get_stepsize())
    sim.mcrunner.show_histogram_analytical()
