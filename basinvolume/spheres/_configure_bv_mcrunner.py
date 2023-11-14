from __future__ import division
from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import str
from builtins import range
import numpy as np
import os
import logging
from pele.potentials import Harmonic
from basinvolume.spheres import BV_MCrunner, ConfigMCRunner
from basinvolume.utils import trymakedir, conf_get_default
from basinvolume.spheres import read_jammed_packing_config
from basinvolume.enums import Minimizer
import configparser
import time
import warnings


class ConfigBVMCRunner(ConfigMCRunner):
    """
    this is an abstract class that implements the basic components of a configure bv_mcrunner class,
    and declares a number of abstract methods which should be implemented in all inheriting classes
    *nparticles: number of particles
    *bdim: dimensionality of the box
    *ndim: dimensionality of the problem (i.e. size of the coordinates array)
    *target_packing_frac: target jammed packing fraction
    *boxv: an array of size bdim that contains the vectors defining the box
    *dtol: tolerance on the rms displacement of the minimised structure with respect to the origin coordinates
    """

    def __init__(self, rank, nprocs):
        self.rank = rank
        self.nprocs = nprocs

    def __call__(
        self,
        fname,
        k=1.0,
        temperature=1.0,
        stepsize=1e-1,
        niter=2e4,
        eps=1.0,
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
        opt_maxstep=None,
        opt_nsteps=1e5,
        perform_convergence_test=False,
        collect_minima_list=False,
        single=False,
        seeds=None,
        use_cell_lists=False,
        checkoverlap_cell_lists=None,
        record_histogram=False,
        packings_dir="jammed_packings",
        base_dir=None,
        verbose=False,
    ):

        self.fname = fname
        self._set_paths(base_dir, packings_dir)
        self._import_packing_config_files()
        self._import_packing_configuration()
        hbinsize = self._get_histogram_bin(k)
        opt_maxstep = self._get_opt_maxstep(opt_maxstep)
        self.eps = eps

        # set parameters
        # self.mc_params = dict(k=k, temperature=temperature, )
        kwargs = dict(
            k=k,
            dtol=self.dtol,
            eps=eps,
            hmin=hmin,
            hmax=hmax,
            hbinsize=hbinsize,
            acceptance=acceptance,
            adjustf=adjustf,
            adjustf_niter=adjustf_niter,
            adjustf_navg=adjustf_navg,
            pt_eq_niter=pt_eq_niter,
            ts_niter=ts_niter,
            ts_freq=ts_freq,
            opt_maxstep=opt_maxstep,
            opt_tol=self.opt_tol,
            opt_nsteps=opt_nsteps,
            perform_convergence_test=perform_convergence_test,
            record_histogram=record_histogram,
            collect_minima_list=collect_minima_list,
            seeds=seeds,
            use_cell_lists=use_cell_lists,
            checkoverlap_cell_lists=checkoverlap_cell_lists,
            single=single,
            distance_method=self.distance_method,
            use_frozen=False,
            minimizer=self.minimizer,
            record_trajectory=False,
            interaction=self.interaction,
            pot_kwargs=self.pot_kwargs,
        )

        self.mc_params = dict(
            temperature=temperature, niter=niter, stepsize=stepsize
        )
        self.mc_params.update(kwargs)
        if seeds is None:
            warnings.warn("seeds not passed")

        self._initialise()
        self._requench_coords(self.dtol, opt_maxstep, verbose, gtol = self.opt_tol)

        # construct mcrunner
        # self.coords is origin, set initial configuration and origin to be the same
        # harmonic potential with fixed centre of mass
        potential = Harmonic(self.coords, k, bdim=self.bdim, com=True)
        mcrunner = BV_MCrunner(
            potential,
            self.coords,
            temperature,
            stepsize,
            niter,
            self.coords,
            self.hs_radii,
            self.boxv,
            self.sca,
            rattlers=self.rattlers,
            **kwargs
        )

        return mcrunner

    def _set_paths(self, base_dir, packings_dir):
        """
        set base_dir, packings_dir and configpaths, configfile
        """
        dname = os.path.splitext(self.fname)[0]

        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), "explore_bv_" + str(dname))
            assert os.path.exists(base_dir)
        else:
            if not os.path.isabs(base_dir):
                base_dir = os.path.join(os.getcwd(), base_dir)
        self.base_dir = base_dir

        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(), packings_dir)
        self.packings_dir = packings_dir

        self.packing_configpath = os.path.join(
            packings_dir, "{}.config".format(dname)
        )
        self.findk_configpath = os.path.join(
            self.base_dir, "findk_" + dname + ".config"
        )
        self.kmin_configpath = os.path.join(
            self.base_dir, "kmin_" + dname + ".config"
        )
        self.configfile = "{}/explore_{}.config".format(self.base_dir, dname)

    def _get_histogram_bin(self, k):
        """automatically estimate size of histogram"""
        hmax = self.displ_k_min * k  # self.displ_k_max*self.kmax
        hbinsize = hmax * 0.0001
        return hbinsize

    def _initialise(self):
        """initialisation function"""
        # change directory only at the end of initialise
        self._print_initialise()
        os.chdir(self.base_dir)

    def _print_initialise(self):
        trymakedir(self.base_dir)
        if self.rank == 0:
            self._print_parameters()

    def _write_sim_params(self, f):
        """
        write simulation parameters
        """
        f.write("#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n")
        f.write("#Explore_Jammed_Packings wrapper class input parameters\n")
        f.write("[IMPORTED_JAMMED_PACKING]\n")
        f.write("nparticles: {}\n".format(self.nparticles))
        f.write("packing_fraction: {}\n".format(self.packing_frac))
        f.write("boxdim: {}\n".format(self.bdim))
        f.write("ndim: {}\n".format(self.ndim))
        f.write("boxv: ")
        for val in self.boxv:
            f.write("{:.16f} ".format(val))
        f.write("\n")
        assert self.sca >= 0
        f.write("sca: {:.16f}\n".format(self.sca))
        f.write("[MCRUNNER]\n")
        for key, value in list(self.mc_params.items()):
            f.write("{}: {}\n".format(key, value))
        f.write("[STATUS]\n")
        for i in range(self.nprocs):
            f.write("success_rank{}: {}\n".format(str(i), "False"))

    def _import_packing_config_files(self):
        imp_packing = read_jammed_packing_config(str(self.packing_configpath))
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
            logging.warning(
                "The jammed packing has not been sorted, "
                "which can negatively impact performance."
            )
        else:
            if imp_packing["pot_kwargs"]["balance_omp"] and imp_packing[
                "sorted_nsubdoms"
            ] != int(os.environ["OMP_NUM_THREADS"]):
                logging.warning(
                    "The jammed packing has been sorted with a "
                    "different number of subdomains (OpenMP threads), "
                    "which changes the number of cells and can "
                    "negatively impact performance."
                )
        configf = configparser.ConfigParser()
        print(self.findk_configpath)
        configf.read(str(self.findk_configpath))
        # fails if Success is false
        self.kmax = configf.getfloat("FINDK", "kmax")
        self.prob_kmax = configf.getfloat("FINDK", "prob")
        self.dtol = configf.getfloat("FINDK_MCRUNNER", "dtol")
        self.minimizer = Minimizer[conf_get_default(configf, "FINDK_MCRUNNER", "minimizer", "FIRE")]
        self.opt_tol = configf.getfloat("FINDK_MCRUNNER","opt_tol")
        self.opt_dtmax = configf.getfloat("FINDK_MCRUNNER","opt_dtmax")
        configf.read(str(self.kmin_configpath))
        self.displ_k_min = configf.getfloat("KMIN", "displ_k_min")
        self.var_displ_k_min = configf.getfloat("KMIN", "var_displ_k_min")

    def print_success_all(self, success):
        """
        print whether calculation has completed successfully
        """
        assert hasattr(self, "configfile")
        if self.rank == 0:
            configf = configparser.ConfigParser()
            configf.read(str(self.configfile))
            for i in range(self.nprocs):
                configf.set(
                    "STATUS", "success_rank{}".format(str(i)), str(success)
                )
            configf.write(open(str(self.configfile), "w"))


if __name__ == "__main__":

    # first 7 primary pseudo perfect numbers
    pppn = [2, 6, 42, 1806, 47058, 2214502422, 52495396602]
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])

    sim = ConfigBVMCRunner(0, 1)
    mcrunner = sim(
        "jammed_packing0.xydr", seeds=seeds, use_cell_lists=True, verbose=True
    )
    print("simulation started")
    start = time.time()
    mcrunner.run()
    end = time.time()
    print("time elapsed", end - start)
    status = mcrunner.get_status()
    print(status)
    mcrunner.dump_minima_list("minima_list.db")
    mcrunner.show_histogram()
