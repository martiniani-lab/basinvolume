"""
Contains the base class for finding kmax
"""
import os
from basinvolume.spheres.mcrunner import SpheresMCRunner
from basinvolume.spheres import ConfigMCRunner


class BaseFindKMCRunner(ConfigMCRunner):
    """
    This class optimises the k to achieve the target acceptance
    *k: harmonic spring constant
    *target_acceptance: target acceptance associated to kmax
    *knavg: number of steps over findk averages the acceptance
    *ktol: when acceptance-ktarget<ktol the search for k terminates
    """

    def __init__(
        self,
        fpath,
        outpath,
        k=150,
        niter=1e8,
        avgcount=1e4,
        dtol=1e-4,
        eps=1.0,
        target_acceptance=0.9,
        knavg=1000,
        ktol=0.025,
        opt_dtmax=1,
        opt_tol=1e-5,
        opt_nsteps=1e5,
        perform_convergence_test=False,
        collect_minima_list=False,
        seeds=None,
        use_cell_lists=False,
        minimizer=Minimizer.FIRE,
        verbose=False,
    ):
        # self.coords is origin, set initial configuration and origin to be the same
        potential = Harmonic(
            self.coords, 0, bdim=self.bdim, com=False
        )  # set the potential to 0, the potential is completely fictitious here (there's no energy test),
        # k is entirely controlled by the stepsize
        stepsize = np.sqrt(1.0 / k)  # stepsize plays the role of the standard deviation
        # stepsize = np.sqrt(self.ndim/k)  #####################
        #####

        # self.mc_params = dict(k=k, temperature=temperature, )
        kwargs = dict(
            dtol=dtol,
            eps=eps,
            ktarget=target_acceptance,
            knavg=knavg,
            ktol=ktol,
            avgcount=avgcount,
            opt_dtmax=opt_dtmax,
            opt_maxstep=opt_maxstep,
            opt_tol=opt_tol,
            opt_nsteps=opt_nsteps,
            perform_convergence_test=perform_convergence_test,
            collect_minima_list=collect_minima_list,
            seeds=seeds,
            minimizer=minimizer,
            interaction=self.interaction,
            pot_kwargs=self.pot_kwargs,
        )

        self.mc_params = dict(temperature=self.temperature, niter=niter, stepsize=stepsize)
        self.mc_params.update(kwargs)

        if seeds is None:
            warnings.warn("seeds not passed")

        self._requench_coords(dtol, opt_maxstep, verbose)

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
            self._print_results()
            self._print_success(True)
        except:
            view_traceback()
            self._print_success(False)

    def _set_paths(self, packings_dir, explore_dir):
        dname = os.path.splitext(self.fname)[0]
        packing_nr = dname[len("jammed_packing") :]
        self.base_directory = os.path.join(os.getcwd(), explore_dir + packing_nr)
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(), packings_dir)
        self.packings_dir = packings_dir
        self.configpath = os.path.join(packings_dir, "{}.config".format(dname))
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
        f.write("[FINDK_IMPORTED_JAMMED_PACKING]\n")
        f.write("nparticles: {}\n".format(self.nparticles))
        f.write("packing_fraction: {}\n".format(self.packing_frac))
        f.write("boxdim: {}\n".format(self.bdim))
        f.write("ndim: {}\n".format(self.ndim))
        f.write("boxv: ")
        for val in self.boxv:
            f.write("{:.16f} ".format(val))
        f.write("\n")
        assert self.sca > 0
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
        f.close()


class Findk_MCrunner(SpheresMCRunner):
    """Findk MCrunner
    *coords: initial coordinates, can be the same as origin
    *origin: jammed minimised structure
    *hs_radii: array of the radii of the particles
    *boxv: array with the box size lengths
    *rattlers: array of rattlers, if not rattler: 1 -> jammed dof
                                                  0 -> rattler dof
    *k: spring constant
    *temperature
    *niter: number of MC takesteps to perform
    *
    *stepsize
    *Etol: tolerance with which a mini mised structure is accepted
     when compared to origin energy
    *dtol: tolerance on the rms displacement of the minimised structure
     with respect to the origin coordinates
    *ktarget: target acceptance associated to kmax
    *knavg: number of steps over findk averages the acceptance
    *ktol: when acceptance-ktarget<ktol the search for k terminates
    * this class requires 1 seed
    """

    def __init__(
        self,
        potential,
        full_coords,
        temperature,
        stepsize,
        niter,
        origin,
        hs_radii,
        boxv,
        sca,
        rattlers=None,
        dtol=1e-3,
        eps=1.0,
        ktarget=0.75,
        knavg=500,
        avgcount=int(1e4),
        ktol=0.05,
        opt_dtmax=1,
        opt_maxstep=0.6,
        opt_tol=1e-4,
        opt_nsteps=1e5,
        hmin=0,
        hmax=1,
        binsize=0.005,
        perform_convergence_test=False,
        collect_minima_list=False,
        seeds=None,
        use_cell_lists=False,
        single=False,
        distance_method=Distance.PERIODIC,
        use_frozen=False,
        frozen_atoms=None,
        rcontainer=None,
        minimizer=Minimizer.FIRE,
        interaction=Interaction.HS_WCA,
        pot_kwargs={},
    ):
        # findk parameters
        self.ktarget = ktarget
        self.knavg = knavg
        self.ktol = ktol
        super(Findk_MCrunner, self).__init__(
            potential,
            full_coords,
            temperature,
            stepsize,
            niter,
            origin,
            hs_radii,
            boxv,
            sca,
            rattlers=rattlers,
            k=1,
            dtol=dtol,
            avgcount=avgcount,
            eps=eps,
            hmin=hmin,
            hmax=hmax,
            hbinsize=binsize,
            report_steps=0,
            pt_eq_niter=0,
            opt_dtmax=opt_dtmax,
            opt_maxstep=opt_maxstep,
            opt_tol=opt_tol,
            opt_nsteps=opt_nsteps,
            perform_convergence_test=perform_convergence_test,
            collect_minima_list=collect_minima_list,
            seeds=seeds,
            use_cell_lists=use_cell_lists,
            record_histogram=False,
            distance_method=distance_method,
            use_frozen=use_frozen,
            frozen_atoms=frozen_atoms,
            rcontainer=rcontainer,
            minimizer=minimizer,
            interaction=interaction,
            pot_kwargs=pot_kwargs,
        )

    def _set_takestep(self, stepsize):
        self.takestep = SampleGaussian(self.seeds["seed_takestep"], stepsize, self.origin)
        self.set_takestep(self.takestep)

    def _set_actions(self):
        self.findk = Findk(
            self.red_origin,
            self.rattlers,
            self.bdim,
            self.avgcount,
            self.ktarget,
            self.knavg,
            self.ktol,
            self.hmin,
            self.hmax,
            self.binsize,
        )
        self.add_action(self.findk)

    def _set_accept_tests(self):
        pass

    def set_control(self, c):
        """set k"""
        print(
            "WARNING: findk set control is not defined, spring constant is set through stepsize",
            file=sys.stderr,
        )

    def get_k(self):
        """in findk, potential is pretty much fictitious, k is adjusted through the stepsize"""
        stepsize = self.get_stepsize()
        k = 1.0 / (stepsize * stepsize)
        # k = self.bdim*len(self.hs_radii)/(stepsize*stepsize)##############
        return k

    def get_entries(self):
        return self.findk.get_entries()

    def show_histogram(self):
        """shows the histogram"""
        hist = self.findk.get_histogram()
        val = np.array([i * self.binsize for i in range(len(hist))]) + 0.5 * self.binsize
        n, bins, patches = plt.hist(
            val,
            weights=hist,
            bins=len(hist),
            normed=1,
            alpha=0.4,
            edgecolor=color_cycle[0],
            color=color_cycle[0],
        )
        ### analytical
        bincenters = 0.5 * (bins[1:] + bins[:-1])
        and2 = old_div(
            vec_analytical_d2(val, self.get_k(), self.nparticles, self.bdim),
            quad(
                vec_analytical_d2,
                bincenters[0],
                bincenters[-1],
                args=(self.get_k(), self.nparticles, self.bdim),
            )[0],
        )
        plt.plot(bincenters, and2, linewidth=2.5, ls="--", color=color_cycle[-1])
        # plt.xlim(0,1)
        plt.xlabel(r"$|{\bf r}-{\bf r}_0|^2$")
        plt.ylabel(r"frequency $\times 10$")
        plt.tight_layout()
        plt.savefig("findk_histogram.eps")
        plt.show()
