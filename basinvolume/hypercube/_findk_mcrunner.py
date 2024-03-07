from __future__ import division
from __future__ import print_function
from builtins import str
import numpy as np
import os
from basinvolume.spheres import ConfigMCRunner
from basinvolume.utils import trymakedir
from basinvolume.utils import view_traceback
from mcpele.monte_carlo import NullPotential
from basinvolume.hypercube import HypercubeFindkMCrunner
from basinvolume.monte_carlo import Findk
import time
import warnings


class _hypercube_findk_mcrunner(ConfigMCRunner):
    """ """

    def __init__(
        self,
        ndof,
        sidelength=1,
        k=100.0,
        niter=5e4,
        avgcount=1e4,
        target_acceptance=0.9,
        knavg=1000,
        ktol=0.025,
        hmin=0,
        hmax=0.01,
        hbinsize=0.0005,
        seeds=None,
        verbose=False,
        workspace=None,
        hyperball=False
    ):

        self.temperature = 1.0
        self.ndof = ndof
        self.sidelength = sidelength
        self.coords = np.zeros(
            self.ndof
        )  # np.ones(self.ndof)*0.32 #CHANGE THIS: I have shifted the centre to see the effect
        if workspace is None:
            self.workspace = os.getcwd()
        else:
            self.workspace = os.path.abspath(workspace)

        self.hyperball = hyperball
        self._set_paths()
        stepsize = np.sqrt(1.0 / k)  # stepsize plays the role of the standard deviation

        # self.mc_params = dict(k=k, temperature=temperature, )
        kwargs = dict(
            target_acceptance=target_acceptance,
            knavg=knavg,
            ktol=ktol,
            avgcount=avgcount,
            hmin=hmin,
            hmax=hmax,
            hbinsize=hbinsize,
            sidelength=self.sidelength,
            seeds=seeds,
            hyperball=hyperball
        )

        self.mc_params = dict(k=k, temperature=self.temperature, niter=niter, stepsize=stepsize)
        self.mc_params.update(kwargs)
        if seeds is None:
            warnings.warn("seeds not passed")

        # self.coords is origin, set initial configuration and origin to be the same
        potential = NullPotential()
        #####
        self.mcrunner = HypercubeFindkMCrunner(
            potential, self.coords, self.temperature, stepsize, niter, self.coords, **kwargs
        )

        self._initialise()

    def run(self):
        try:
            self.mcrunner.run()
            self.kmax = self.mcrunner.get_k()
            self.prob = self.mcrunner.findk.get_prob()
            (
                self.displ_k_max,
                self.var_displ_k_max,
            ) = self.mcrunner.findk.get_mean_variance()
            self._print_results()
            self._print_success(True)
        except:
            view_traceback()
            self._print_success(False)

    def _set_paths(self):
        if not self.hyperball:
            dname = "hypercube_n" + str(self.ndof) + "_l" + str(self.sidelength)
        else:
            dname = "hyperball_n" + str(self.ndof) + "_l" + str(self.sidelength)
        self.base_directory = os.path.join(self.workspace, "explore_bv_" + dname)
        configfile = "findk_" + dname
        self.configfile = "{}/{}.config".format(self.base_directory, configfile)

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
        f.write("[FINDK_HYPERCUBE]\n")
        f.write("ndof: {}\n".format(self.ndof))
        f.write("sidelength: {}\n".format(self.sidelength))
        f.write("[FINDK_MCRUNNER]\n")
        for key, value in list(self.mc_params.items()):
            f.write("{}: {}\n".format(key, value))

    def _print_results(self):
        fname = self.configfile
        f = open(fname, "a")
        f.write("[FINDK_MCRUNNER_STATUS]\n")
        status = self.mcrunner.get_status()
        for key, value in list(status.items()):
            f.write("{}: {}\n".format(key, value))
        f.write("[FINDK]\n")
        f.write("kmax: {:.16f}\n".format(self.kmax))
        f.write("prob: {:.16f}\n".format(self.prob))
        f.write("displ_k_max: {:.16f}\n".format(self.displ_k_max))
        f.write("var_displ_k_max: {:.16f}\n".format(self.var_displ_k_max))
        f.close()

    def _import_packing_config_files(self):
        """pure virtual, must overload"""
        pass


if __name__ == "__main__":

    # sim = _findk_mcrunner('jammed_packing0.xydr')
    pppn = [2, 6, 42, 1806, 47058, 2214502422, 52495396602]
    seeds = dict(seed_takestep=1158925890)
    ndof = 1000
    kguess = 100
    sim = _hypercube_findk_mcrunner(
        ndof,
        sidelength=1,
        avgcount=1e6,
        k=kguess,
        target_acceptance=1.0,
        ktol=0.0025 / ndof,
        knavg=1e5,
        niter=1e8,
        seeds=seeds,
        verbose=True,
    )
    print("simulation started")
    start = time.time()
    sim.run()
    end = time.time()
    print("time elapsed", end - start)
    status = sim.mcrunner.get_status()
    print(status)
    print("self.kmax:", sim.kmax)
    print("self.prob:", sim.prob)
    print("self.displ_k_max:", sim.displ_k_max)
    print("self.var_displ_k_max:", sim.var_displ_k_max)
    # print "Nd/k: ", sim.nparticles * sim.bdim / sim.kmax
    print("(N-1)d/k", sim.ndof / sim.kmax)
    sim.mcrunner.show_histogram()
    print("entries in histogram:", sim.mcrunner.get_entries())
