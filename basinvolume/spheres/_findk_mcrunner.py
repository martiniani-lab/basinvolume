from __future__ import division
from __future__ import print_function
import logging
from future import standard_library

standard_library.install_aliases()
from builtins import str
import numpy as np
import os
from pele.potentials import Harmonic
from basinvolume.spheres import Findk_MCrunner, ConfigMCRunner
from basinvolume.utils import trymakedir, view_traceback
from basinvolume.spheres import read_jammed_packing_config
from basinvolume.enums import Minimizer
import configparser
import time
import warnings


class _findk_mcrunner(ConfigMCRunner):
    """
    this is a class that implements configure_findk_mcrunner class,
    *k: harmonic spring constant
    *ktarget: target acceptance associated to kmax
    *knavg: number of steps over findk averages the acceptance
    *ktol: when acceptance-ktarget<ktol the search for k terminates
    """

    def __init__(
        self,
        fname,
        k=150,
        niter=1e8,
        avgcount=1e4,
        dtol=1e-4,
        eps=1.0,
        ktarget=0.9,
        knavg=1000,
        ktol=0.025,
        opt_dtmax=1,
        opt_maxstep=None,
        opt_tol=1e-5,
        opt_nsteps=1e5,
        perform_convergence_test=False,
        collect_minima_list=False,
        seeds=None,
        use_cell_lists=False,
        minimizer=Minimizer.FIRE,
        packings_dir="jammed_packings",
        explore_dir="explore_bv_jammed_packing",
        verbose=False,
    ):

        self.temperature = 1.0
        self.eps = eps
        self.fname = fname
        self.minimizer = minimizer
        self.opt_tol = opt_tol

        self._set_paths(packings_dir, explore_dir)
        imp_packing = read_jammed_packing_config(str(self.configpath))
        self.nparticles = imp_packing["nparticles"]
        self.packing_frac = imp_packing["packing_frac"]
        self.bdim = imp_packing["bdim"]
        self.ndim = imp_packing["ndim"]
        self.boxv = imp_packing["boxv"].copy()
        self.vcavity = imp_packing["vcavity"]
        self.sca = imp_packing["sca"]
        self.distance_method = imp_packing["distance_method"]
        self.interaction = imp_packing["interaction"]
        if hasattr(self, "pot_kwargs") and self.pot_kwargs is not None:
            self.pot_kwargs.update(imp_packing["pot_kwargs"])
        else:
            self.pot_kwargs = imp_packing["pot_kwargs"].copy()
        self.opt_maxstep_factor = imp_packing["maxstep_factor"]
        if not imp_packing["sorted"]:
            print(
                "WARNING: The jammed packing has not been sorted, "
                "which can negatively impact performance."
            )
        else:
            if imp_packing["pot_kwargs"]["balance_omp"] and imp_packing[
                "sorted_nsubdoms"
            ] != int(os.environ["OMP_NUM_THREADS"]):
                print(
                    "WARNING: The jammed packing has been sorted with a different number "
                    "of subdomains (OpenMP threads), which changes the number of cells "
                    "and can negatively impact performance."
                )
        self._import_packing_configuration()
        opt_maxstep = self._get_opt_maxstep(opt_maxstep)
        # self.coords is origin, set initial configuration and origin to be the same
        potential = Harmonic(
            self.coords, 0, bdim=self.bdim, com=False
        )  # set the potential to 0, the potential is completely fictitious here (there's no energy test),
        # k is entirely controlled by the stepsize
        stepsize = np.sqrt(
            1.0 / k
        )  # stepsize plays the role of the standard deviation
        # stepsize = np.sqrt(self.ndim/k)  #####################
        #####

        kwargs = dict(
            dtol=dtol,
            eps=eps,
            ktarget=ktarget,
            knavg=knavg,
            ktol=ktol,
            avgcount=avgcount,
            opt_dtmax=opt_dtmax,
            opt_maxstep=opt_maxstep,
            opt_tol=self.opt_tol,
            opt_nsteps=opt_nsteps,
            perform_convergence_test=perform_convergence_test,
            collect_minima_list=collect_minima_list,
            seeds=seeds,
            use_cell_lists=use_cell_lists,
            minimizer=self.minimizer,
            distance_method=self.distance_method,
            use_frozen=False,
            interaction=self.interaction,
            pot_kwargs=self.pot_kwargs,
        )

        self.mc_params = dict(
            temperature=self.temperature, niter=niter, stepsize=stepsize
        )
        self.mc_params.update(kwargs)

        if seeds is None:
            warnings.warn("seeds not passed")

        self._requench_coords(dtol, opt_maxstep, verbose, gtol=self.opt_tol)

        self.mcrunner = Findk_MCrunner(
            potential,
            self.coords,
            self.temperature,
            stepsize,
            niter,
            self.coords,
            self.hs_radii,
            self.boxv,
            self.sca,
            rattlers=self.rattlers,
            **kwargs
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

    def _set_paths(self, packings_dir, explore_dir):
        dname = os.path.splitext(self.fname)[0]
        packing_nr = dname[len("jammed_packing") :]
        self.base_directory = os.path.join(
            os.getcwd(), explore_dir + packing_nr
        )
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(), packings_dir)
        self.packings_dir = packings_dir
        self.configpath = os.path.join(packings_dir, "{}.config".format(dname))
        configfile = "findk_" + dname
        self.configfile = "{}/{}.config".format(
            self.base_directory, configfile
        )

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
        f.write("[FINDK_IMPORTED_JAMMED_PACKING]\n")
        f.write("nparticles: {}\n".format(self.nparticles))
        f.write("packing_fraction: {}\n".format(self.packing_frac))
        f.write("boxdim: {}\n".format(self.bdim))
        f.write("ndim: {}\n".format(self.ndim))
        f.write("boxv: ")
        for val in self.boxv:
            f.write("{:.16f} ".format(val))
        f.write("\n")
        assert self.sca >= 0.0, "sca must be positive"
        f.write("sca: {:.16f}\n".format(self.sca))
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


if __name__ == "__main__":

    # sim = _findk_mcrunner('jammed_packing0.xydr')
    pppn = [2, 6, 42, 1806, 47058, 2214502422, 52495396602]
    seeds = dict(seed_takestep=1158925890)
    sim = _findk_mcrunner(
        "jammed_packing0.xyzdr",
        avgcount=1e4,
        k=759,
        ktarget=0.9,
        knavg=1e3,
        seeds=seeds,
        use_cell_lists=True,
        verbose=True,
    )
    # print("simulation started")
    start = time.time()
    sim.run()
    frac = sim.mcrunner.conftest2.get_failed_quench_frac()
    print("failed quench frac", frac)
    end = time.time()
    print("time elapsed", end - start)
    status = sim.mcrunner.get_status()
    print(status)
    print("self.kmax:", sim.kmax)
    print("self.prob:", sim.prob)
    # print "Nd/k: ", sim.nparticles * sim.bdim / sim.kmax
    print("(N-1)d/k", (sim.nparticles - 1) * sim.bdim / sim.kmax)
    sim.mcrunner.show_histogram()
    print("entries in histogram:", sim.mcrunner.get_entries())
