from __future__ import division
from future import standard_library

standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object
import numpy as np
import abc
import os
from pele.distance import put_in_box, Distance
from pele.potentials import HS_WCA, InversePowerStillingerCut
from pele.optimize._quench import modifiedfire_cpp, lbfgs_cpp
from PyCG_DESCENT import CGDescent
from basinvolume.utils import (
    trymakedir,
    get_git_version,
    get_python_version,
    volume_nball,
    import_packing,
    calc_distance,
    get_cython_version,
    cround,
    in_hull,
    origin_in_hull_2d,
    conf_get_default,
    conf_getboolean_default,
    conf_getint_default,
    conf_getfloat_default,
)
from basinvolume.spheres import read_packing_config
from basinvolume.enums import Minimizer, Interaction
import configparser
import re
import argparse
import subprocess
import shlex
import glob
import ast
import logging
from future.utils import with_metaclass
from basinvolume.utils import INVERSE_POWER_CVODE_95_ACC, get_mxd_t

# try:
#     import pylab
# except:
#     pass

from basinvolume.inverse_power_soft.soft_sphere_ensemble import (
    setup_bidisperse,
    BINARY_SOFT_SPHERE_DEFAULTS,
)


def cartesian_to_polar2d(vector):
    vector = np.array(vector)
    r = np.linalg.norm(vector)
    theta = np.arctan2(vector[1], vector[0]) + np.pi
    return r, theta


def sum_neighbor_angles2d(neigh_vec):
    sum_ = 0.0
    for idx in range(len(neigh_vec) - 1):
        sum_ += np.arccos(
            np.dot(neigh_vec[idx], neigh_vec[idx + 1])
            / (
                np.linalg.norm(neigh_vec[idx])
                * np.linalg.norm(neigh_vec[idx + 1])
            )
        )
    sum_ += np.arccos(
        np.dot(neigh_vec[-1], neigh_vec[0])
        / (np.linalg.norm(neigh_vec[-1]) * np.linalg.norm(neigh_vec[0]))
    )
    return sum_


def read_jammed_packing_config(configpath, frozen=False):
    if not os.path.isfile(str(configpath)):
        raise IOError("Config file does not exist: {}".format(str(configpath)))
    configf = configparser.ConfigParser()
    configf.read(str(configpath))
    parameters = {}
    parameters["nparticles"] = configf.getint("JAMMED_PACKING", "nparticles")
    parameters["packing_frac"] = configf.getfloat(
        "JAMMED_PACKING", "packing_fraction"
    )
    parameters["bdim"] = configf.getint("JAMMED_PACKING", "boxdim")
    assert (
        parameters["bdim"] == 2 or parameters["bdim"] == 3
    ), "bdim={} not implemented".format(parameters["bdim"])
    parameters["ndim"] = parameters["nparticles"] * parameters["bdim"]
    boxv = configf.get("JAMMED_PACKING", "boxv")
    parameters["boxv"] = np.array([float(x) for x in boxv.split()])
    if frozen:
        parameters["vcavity"] = configf.getfloat("JAMMED_PACKING", "vcavity")
    else:
        parameters["vcavity"] = np.prod(parameters["boxv"])
    parameters["distance_method"] = Distance[
        conf_get_default(
            configf, "JAMMED_PACKING", "distance_method", "PERIODIC"
        )
    ]
    parameters["interaction"] = Interaction[
        conf_get_default(configf, "JAMMED_PACKING", "interaction", "HS_WCA")
    ]
    parameters["minimizer"] = Minimizer[
        conf_get_default(configf, "JAMMED_PACKING", "minimizer", "FIRE")
    ]
    parameters["opt_tol"] = configf.getfloat(
        "JAMMED_PACKING", "opt_tol"
    )
    parameters["opt_maxstep"] = configf.getfloat(
        "JAMMED_PACKING", "opt_maxstep"
    )
    parameters["pot_kwargs"] = ast.literal_eval(
        conf_get_default(configf, "JAMMED_PACKING", "pot_kwargs", "{}")
    )
    parameters["sca"] = configf.getfloat("JAMMED_PACKING", "sca")
    parameters["sorted"] = conf_getboolean_default(
        configf, "JAMMED_PACKING", "sorted", False
    )
    parameters["sorted_nsubdoms"] = conf_getint_default(
        configf, "JAMMED_PACKING", "sorted_nsubdoms", 1
    )
    parameters["maxstep_factor"] = conf_getfloat_default(
        configf, "JAMMED_PACKING", "maxstep_factor", 1.0
    )
    return parameters


class _Generate_Jammed_Packing(with_metaclass(abc.ABCMeta, object)):
    """
    this is an abstract class that implements the basic components of a generate packing class,
    and declares a number of abstract methods which should be implemented in all inheriting classes
    *method to generate packing, this could be for example direct sampling,
    sequential sampling,quench or LSA
    *nparticles: number of particles
    *bdim: dimensionality of the box
    *ndim: dimensionality of the problem (i.e. size of the coordinates array)
    *target_packing_frac: target jammed packing fraction
    *boxv: an array of size bdim that contains the vectors defining the box
    """

    def __init__(
        self,
        target_packing_frac=0.65,
        packings_dir="packings",
        packing_nrs=None,
        import_jammed=False,
        outdir="jammed_packings",
        override_pot_kwargs=None,
        minimizer=Minimizer.FIRE,
        maxstep_factor=1.0,
        logging_tag="",
        write_opengl=False,
        sort_atoms=False,
    ):
        self.target_packing_frac = target_packing_frac
        self.base_directory = os.path.join(os.getcwd(), outdir)
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(), packings_dir)
        self.packings_dir = packings_dir
        self.packing_nrs = packing_nrs
        self.import_jammed = import_jammed
        self.override_pot_kwargs = override_pot_kwargs
        self.iteration = 0
        self.sca = -1
        self.eps = 1.0
        self.minimizer = minimizer
        self.maxstep_factor = maxstep_factor
        self.logging_tag = logging_tag
        self.write_opengl = write_opengl
        self.sort_atoms = sort_atoms

    def _import_single_config_file(self, fname):
        dname = os.path.splitext(fname)[0]
        self.configpath = os.path.join(self.packings_dir, dname + ".config")
        if self.import_jammed:
            imp_packing = read_jammed_packing_config(str(self.configpath))
            self.nparticles = imp_packing["nparticles"]
            self.bdim = imp_packing["bdim"]
            self.ndim = imp_packing["ndim"]
            self.boxv = imp_packing["boxv"].copy()
            self.vcavity = imp_packing["vcavity"]
            self.distance_method = imp_packing["distance_method"]
            if hasattr(self, "pot_kwargs") and self.pot_kwargs is not None:
                self.pot_kwargs.update(imp_packing["pot_kwargs"])
            else:
                self.pot_kwargs = imp_packing["pot_kwargs"].copy()
            if self.override_pot_kwargs is not None:
                self.pot_kwargs.update(self.override_pot_kwargs)
            self.sca = imp_packing["sca"]
            self.packing_frac = (
                self.target_packing_frac / (1 + self.sca) ** self.bdim
            )
        else:
            self._import_packing_config_file(str(self.configpath))

    @abc.abstractmethod
    def _initialise(self):
        """initialisation function"""
        self.configpath = os.path.join(self.packings_dir, "packings.config")
        assert os.path.isfile(self.configpath)
        self._import_packing_config_file(str(self.configpath))

    def _import_packing_config_file(self, configpath):
        imp_packing = read_packing_config(configpath)
        self.nparticles = imp_packing["nparticles"]
        self.packing_frac = imp_packing["packing_frac"]
        self.bdim = imp_packing["boxdim"]
        self.ndim = imp_packing["ndim"]
        self.hs_mean = imp_packing["radii_mean"]
        self.hs_stddev = imp_packing["radii_stddev"]
        self.boxv = imp_packing["boxv"].copy()
        self.vcavity = imp_packing["vcavity"]
        self.distance_method = imp_packing["distance_method"]
        if hasattr(self, "pot_kwargs") and self.pot_kwargs is not None:
            self.pot_kwargs.update(imp_packing["pot_kwargs"])
        else:
            self.pot_kwargs = imp_packing["pot_kwargs"].copy()
        if self.override_pot_kwargs is not None:
            self.pot_kwargs.update(self.override_pot_kwargs)

    @abc.abstractmethod
    def _import_packing_configuration(self, fname):
        """imports the coordinates and data relative to the shape of the particles
        this should be run in initialise()
        """

    @abc.abstractmethod
    def _generate_packing_coords(self):
        """function that generates the packing"""

    @abc.abstractmethod
    def _sort_atoms(self):
        """sorts the atoms according to the potential"""

    @abc.abstractmethod
    def _write_opengl_input(self, n):
        """writes a opengl input file, n is the unique identifier of the structure"""

    @abc.abstractmethod
    def _dump_configuration(self, n):
        """writes a configuration file, e.g .xyzd, n is the unique identifier of the structure"""

    def _print_initialise(self):
        trymakedir(self.base_directory)

    def _print_parameters(self, n):
        """writes the simulation parameters"""
        fname = "{}/jammed_packing{}.config".format(self.base_directory, n)
        f = open(fname, "w")
        f.write("#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n")
        f.write("#Generate_Jammed_Packings base class input parameters\n")
        f.write("[JAMMED_PACKING]\n")
        f.write("nparticles: {}\n".format(self.nparticles))
        f.write("packing_fraction: {:.16f}\n".format(self.target_packing_frac))
        f.write("boxdim: {}\n".format(self.bdim))
        f.write("ndim: {}\n".format(self.ndim))
        f.write("boxv: ")
        for val in self.boxv:
            f.write("{:.16f} ".format(val))
        f.write("\n")
        f.write("distance_method: {}\n".format(self.distance_method.name))
        f.write("interaction: {}\n".format(self.interaction.name))
        f.write("minimizer: {}\n".format(self.minimizer.name))
        f.write("opt_tol: {}\n".format(self.tol))
        f.write("opt_maxstep: {}\n".format(self.opt_maxstep))
        f.write("pot_kwargs: {}\n".format(self.pot_kwargs))
        if self.sca < 0:
            logging.warning(
                "WARNING: sca is < 0, this does not make sense for hard sphere configurations"
            )
        f.write("sca: {:.16f}\n".format(self.sca))
        f.write("sorted: {}\n".format(self.sort_atoms))
        f.write("sorted_nsubdoms: {}\n".format(os.environ["OMP_NUM_THREADS"]))
        f.write("maxstep_factor: {}\n".format(self.maxstep_factor))
        f.write("\n")
        # print software version
        f.write("[CODEVERSION]\n")
        f.write(
            "basinvolume_version: {}\n".format(get_git_version("basinvolume"))
        )
        f.write("mcpele_version: {}\n".format(get_git_version("mcpele")))
        f.write("pele_version: {}\n".format(get_git_version("pele")))
        f.write("python_version: {}\n".format(get_python_version()))
        f.write("cython_version: {}\n".format(get_cython_version()))
        f.close()

    def _print(self, n):
        """dump configuration and opengl input to packings directory
        n is the unique identifier of the structure
        """
        self._print_parameters(n)
        if self.sort_atoms:
            self._sort_atoms(n)
        self._dump_configuration(n)
        if self.write_opengl:
            self._write_opengl_input(n)

    def _log(self, message):
        if self.logging_tag is None or self.logging_tag == "":
            return "{}".format(message)
        else:
            return "{}: {}".format(self.logging_tag, message)

    # @abc.abstractmethod
    # def _histogram_eigenvalues(self):
    #     """ method to plot eigenvalues histograms
    #     """

    @abc.abstractmethod
    def one_iteration(self, fname):
        """perform one iteration"""

    def run(self):
        """run generate packings"""
        self._initialise()
        successes = []
        # list of lists of strings. One string out of each list of strings must
        # be in the filename
        filter_stringss = []
        if self.import_jammed:
            filter_stringss.append(["xyzdr", "xydr"])
        else:
            filter_stringss.append(["xyzd", "xyd"])
        if self.packing_nrs is not None:
            filter_stringss.append([str(nr) + "." for nr in self.packing_nrs])
        for fname in os.listdir(self.packings_dir):
            if all(
                [
                    any([filter_str in fname for filter_str in filter_strings])
                    for filter_strings in filter_stringss
                ]
            ):
                logging.debug("")
                logging.info(self._log(fname))
                success = self.one_iteration(fname)
                successes.append((fname, success))
        return successes
        # self._histogram_eigenvalues()


class HS_Generate_Jammed_Packing(_Generate_Jammed_Packing):
    """
    This class generates packings and identifies rattlers by computing the
    hessian eigenvalues for each particle in the equilibrium jammed structure.
    A .xyzdr file is produced that contains the 3 system coordinates, the particle
    diameter and if not it's a rattler (0 if a rattler, 1 otherwise)
    PARAMETERS
    hs_radii: array with the radii of the particles,
              if none sample particle sizes from a normal distribution
    mu: average particle size, passable to normal distribution
    sig: standard deviaton of normal distribution from which to sample particles
    sca: determines % by which the hs is inflated
    eps: LJ interaction energy of WCA part of the HS potential
    tol: rms tolerance for the minimizer
    """

    def __init__(
        self,
        target_packing_frac=0.7,
        tol=1e-9,
        maxstep_factor=1.0,
        packings_dir="packings",
        packing_nrs=None,
        import_jammed=False,
        outdir="jammed_packings",
        use_cell_lists=False,
        show=False,
        interaction=Interaction.HS_WCA,
        override_pot_kwargs=None,
        minimizer=Minimizer.FIRE,
        logging_tag="",
        write_opengl=False,
        check_packing=True,
        sort_atoms=False,
    ):
        super(HS_Generate_Jammed_Packing, self).__init__(
            target_packing_frac=target_packing_frac,
            packings_dir=packings_dir,
            packing_nrs=packing_nrs,
            import_jammed=import_jammed,
            outdir=outdir,
            minimizer=minimizer,
            maxstep_factor=maxstep_factor,
            override_pot_kwargs=override_pot_kwargs,
            logging_tag=logging_tag,
            write_opengl=write_opengl,
            sort_atoms=sort_atoms,
        )
        self.interaction = interaction
        self.use_cell_lists = use_cell_lists
        self.tol = tol
        self.check_packing = check_packing

    def _initialise(self):
        self._print_initialise()

    def one_iteration(self, fname):
        """perform one iteration"""
        self._import_single_config_file(fname)
        self.rattlers = np.empty(self.nparticles, dtype="d")
        self.rattlers_draw = np.empty(self.nparticles, dtype="d")

        self._import_packing_configuration(fname)
        self.max_nrattlers = int(self.nparticles * 0.5)

        # assert that largest soft particle is not > 1/2 of smallest box size
        if (
            np.amax(self.hs_radii) * 2 * (1 + self.sca)
            >= np.amin(self.boxv) / 2
        ):
            logging.warning(self._log("Max soft diameter >= 1/2 box side!"))
        if np.amax(self.hs_radii) * 2 * (1 + self.sca) >= np.amin(self.boxv):
            raise Exception("WARNING: particle does not fit the box")

        # potential needs to be called because self.coords is an input argument
        # of HS_WCAPeriodicCellLists
        # rcut set to largest particle diameter
        rcut = np.amax(self.hs_radii) * 2.0 * (1.0 + self.sca)
        if self.use_cell_lists:
            if np.amin(self.boxv) // rcut <= 3:
                self.use_cell_lists = False
        if self.interaction is Interaction.HS_WCA:
            if self.use_cell_lists:
                self.potential = HS_WCA(
                    use_cell_lists=True,
                    eps=self.eps,
                    sca=self.sca,
                    radii=self.hs_radii,
                    boxvec=self.boxv,
                    reference_coords=self.coords,
                    ndim=self.bdim,
                    ncellx_scale=1.0,
                    distance_method=self.distance_method,
                    pot_kwargs=self.pot_kwargs,
                )
            else:
                self.potential = HS_WCA(
                    eps=self.eps,
                    sca=self.sca,
                    radii=self.hs_radii,
                    boxvec=self.boxv,
                    ndim=self.bdim,
                    distance_method=self.distance_method,
                    pot_kwargs=self.pot_kwargs,
                )
        elif self.interaction is Interaction.INVERSE_POWER_STILLINGER:
            self.stillinger_a_radii = self.hs_radii * (1 + self.sca)
            pow = self.pot_kwargs["pow"]
            rcut = self.pot_kwargs["rcut"]
            self.potential = InversePowerStillingerCut(
                pow,
                self.stillinger_a_radii,
                ndim=self.bdim,
                boxvec=self.boxv,
                rcut=rcut,
                use_cell_lists=True,
            )

        else:
            raise NotImplementedError

        success = self._generate_packing_coords()  # returns false if saddle

        n = int(re.search(r"\d+", fname).group())
        if success:
            self._print(n)
        else:
            path_list = glob.glob(
                "{0}/jammed_packing{1}.*".format(self.base_directory, n)
            )
            if len(path_list) > 0:
                with open(
                    "{0}/mismatching_rattlers.txt".format(self.base_directory),
                    "a",
                ) as f:
                    f.write("jammed_packing{}\n".format(n))
            for path_ in path_list:
                p = subprocess.call(shlex.split("rm {}".format(path_)))
        self.iteration += 1
        return success

    def _find_rattlers(self):
        """
        finish this, I need to remove the rattler and break.
        Also need to get compare to existing jammed_packing option
        :return:
        """
        if self.bdim < 4:
            zmin = self.bdim + 1
        else:
            raise NotImplementedError

        check_again = list(range(len(self.hs_radii)))
        self.ss_radii = self.hs_radii * (1.0 + self.sca)
        neighbor_indicess, neighbor_distancess = self.potential.getNeighbors(
            self.coords
        )
        nrattlers = 0
        if self.bdim != 2:
            origin = np.zeros(self.bdim)
        while len(check_again) > 0:
            check_inds = check_again
            check_again = set()
            if nrattlers > self.max_nrattlers:
                logging.warning(
                    self._log("Too many rattlers. Discarding packing.")
                )
                return False
            for atomi in check_inds:
                found_rattler = False
                no_neighbors = len(neighbor_indicess[atomi])
                if no_neighbors < zmin:
                    found_rattler = True
                    logging.debug(
                        self._log(
                            "Particle {} is not isostatic.".format(atomi)
                        )
                    )
                else:
                    if self.bdim == 2:
                        found_rattler = not origin_in_hull_2d(
                            neighbor_distancess[atomi]
                        )
                    else:
                        points = np.asarray(
                            neighbor_distancess[atomi]
                        ).reshape((-1, self.bdim))
                        found_rattler = not in_hull(origin, points)
                    if found_rattler:
                        logging.debug(
                            self._log(
                                "Particle {} is not in "
                                "contacts' convex hull.".format(atomi)
                            )
                        )
                self.rattlers[atomi] = 0 if found_rattler else 1000
                self.rattlers_draw[atomi] = float(not found_rattler)
                if found_rattler:
                    if atomi in check_again:
                        check_again.remove(atomi)
                    for atomj in neighbor_indicess[atomi]:
                        check_again.add(atomj)
                        i_in_j = neighbor_indicess[atomj].index(atomi)
                        del neighbor_indicess[atomj][i_in_j]
                        del neighbor_distancess[atomj][i_in_j]
                    nrattlers += 1

        # test that number of contacts is sufficient for bulk modulus to be positive,
        # see eq 4 in http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.095704
        # see eq 19 in arXiv:1406.1529
        no_stable = self.nparticles - nrattlers
        total_contacts = sum(
            [len(neighbor_indices) for neighbor_indices in neighbor_indicess]
        )
        N_min = int(2 * (self.bdim * (no_stable - 1) + 1))
        logging.debug(
            self._log(
                "N_min: {} total_contacts: {}".format(N_min, total_contacts)
            )
        )
        logging.debug(self._log("Number of rattlers: {}".format(nrattlers)))
        if nrattlers > self.max_nrattlers:
            logging.warning(
                self._log("Too many rattlers. Discarding packing.")
            )
            return False
        if total_contacts >= N_min:
            return True
        else:
            logging.warning(
                self._log(
                    "Packing is not globally stable, N_min: {} "
                    "total_contacts: {}".format(N_min, total_contacts)
                )
            )
            return False

    def _generate_packing_coords(self):
        """
        perform quench and run tests
        """
        success = self._generate_packing_coords_iteration(tol=self.tol)
        return success

    def _generate_packing_coords_iteration(self, tol=1e-9, iprint=-1):
        """quenches the imported structure"""

        # asserts that none of the hard spheres is overlapping before quenching
        if __debug__ and self.check_packing:
            no_overlap = self._check_no_overlaps()
            if not no_overlap:
                logging.warning(self._log("Overlap found before quenching"))
                return False

        self.opt_maxstep = (
            self.sca * np.amin(self.hs_radii) * 0.5 * self.maxstep_factor
        )
        if self.minimizer is Minimizer.FIRE:
            res = modifiedfire_cpp(
                self.coords,
                self.potential,
                maxstep=self.opt_maxstep,
                nsteps=1e6,
                tol=tol,
                iprint=iprint,
            )
        elif self.minimizer is Minimizer.CG:
            optimizer = CGDescent(
                self.coords,
                self.potential,
                tol=tol,
                nsteps=1e6,
                print_level=iprint,
            )
            res = optimizer.run()
        elif self.minimizer is Minimizer.CVODE:
            from pele.optimize import CVODEBDFOptimizer

            self.optimizer = CVODEBDFOptimizer(
                self.potential,
                self.coords,
                tol=tol,
                atol=INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim],
                rtol=INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim],
            )
            res = self.optimizer.run(int(1e6))
        elif self.minimizer is Minimizer.MXD:
            from pele.optimize import ExtendedMixedOptimizer

            ratol = INVERSE_POWER_CVODE_95_ACC[self.nparticles] * 1e-1
            self.optimizer = ExtendedMixedOptimizer(
                self.potential,
                self.coords,
                tol=tol,
                nsteps=1e7,
                atol=ratol,
                rtol=ratol,
                T=get_mxd_t(self.nparticles),
            )
            res = self.optimizer.run(int(1e6))
        elif self.minimizer is Minimizer.LBFGS:
            res = lbfgs_cpp(
                self.coords,
                self.potential,
                maxstep=self.opt_maxstep,
                tol=tol,
                nsteps=1e6,
                maxErise=0,
                iprint=iprint,
            )
        else:
            raise NotImplementedError

        if not res.success:
            print(res)
            logging.warning(self._log("Quench failed"))
            return False

        self.coords = res.coords
        self.energy = res.energy

        # test that on re-minimisation the structure does not change
        if __debug__ and self.check_packing:
            if self.minimizer is Minimizer.FIRE:
                res2 = modifiedfire_cpp(
                    self.coords,
                    self.potential,
                    maxstep=self.opt_maxstep,
                    nsteps=1e6,
                    tol=tol,
                )
            elif self.minimizer is Minimizer.CG:
                optimizer = CGDescent(
                    self.coords,
                    self.potential,
                    tol=tol,
                    nsteps=1e6,
                    print_level=iprint,
                )
                res2 = optimizer.run()
            elif self.minimizer is Minimizer.CVODE:
                from pele.optimize import CVODEBDFOptimizer

                self.optimizer = CVODEBDFOptimizer(
                    self.potential,
                    self.coords,
                    tol=tol,
                    atol=INVERSE_POWER_CVODE_95_ACC[
                        len(self.coords) // self.bdim
                    ],
                    rtol=INVERSE_POWER_CVODE_95_ACC[
                        len(self.coords) // self.bdim
                    ],
                )

                res2 = self.optimizer.run(int(1e6))
            elif self.minimizer is Minimizer.MXD:
                from pele.optimize import ExtendedMixedOptimizer

                ratol = (
                    INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim]
                    * 1e-1
                )
                self.optimizer = ExtendedMixedOptimizer(
                    self.potential,
                    self.coords,
                    tol=tol,
                    nsteps=1e7,
                    atol=ratol,
                    rtol=ratol,
                    T=get_mxd_t(self.nparticles),
                )

                res2 = self.optimizer.run()
            elif self.minimizer is Minimizer.LBFGS:
                res2 = lbfgs_cpp(
                    self.coords,
                    self.potential,
                    maxstep=self.opt_maxstep,
                    tol=tol,
                    nsteps=1e6,
                    maxErise=0,
                    iprint=iprint,
                )
            else:
                raise NotImplementedError
            if res2.nfev > 1:
                logging.warning(
                    self._log(
                        "Quench failed (structure changed at "
                        "second minimisation)"
                    )
                )
                return False

        # asserts that none of the hard sphere is overlapping
        if __debug__ and self.check_packing:
            no_overlap = self._check_no_overlaps()
            if not no_overlap:
                logging.warning(self._log("Overlap found after quenching"))
                return False

        return self._find_rattlers()

    def _get_particles_volume(self):
        """returns volume of n=self.bdim dimensional sphere"""
        volumes = volume_nball(self.hs_radii, self.bdim)
        vtot = np.sum(volumes)
        return vtot

    def _import_packing_configuration(self, fname):
        path = os.path.join(self.packings_dir, fname)
        packing = import_packing(path, self.import_jammed, self.bdim)
        self.coords = packing["coords"]
        self.hs_radii = packing["hs_radii"]
        self._compute_sca()

    def _compute_sca(self):
        # TEST
        vol_part = self._get_particles_volume()
        vol_box = np.prod(self.boxv)
        phi = vol_part / vol_box  # instanteneous pack frac
        assert phi - self.packing_frac < 1e-4
        # END TEST
        # r_soft = r_hs*(1+sca)
        self.sca = (
            np.power(
                self.target_packing_frac / self.packing_frac, 1.0 / self.bdim
            )
            - 1
        )

    def _check_no_overlaps(self):
        """check that no two particles are overlapping
        (using nearest image convention)"""
        return len(self.potential.getOverlaps(self.coords)) == 0

    def _correct_coords(self):
        """this function returns the nearest images in the central box,
        useful for dumping the configurations"""
        if self.distance_method is Distance.LEES_EDWARDS:
            return put_in_box(
                self.coords,
                self.bdim,
                self.distance_method,
                self.boxv,
                self.pot_kwargs["shear"],
            )
        else:
            return put_in_box(
                self.coords, self.bdim, self.distance_method, self.boxv
            )

    def _sort_atoms(self, n):
        """sorts the atoms according to the potential"""
        new_order = self.potential.getAtomOrder(self.coords)
        if new_order is not None:
            new_radii = np.empty(len(self.hs_radii))
            new_rattlers = np.empty(len(self.rattlers))
            new_coords = np.empty(len(self.coords))
            for i in range(len(self.hs_radii)):
                new_radii[i] = self.hs_radii[new_order[i]]
                new_rattlers[i] = self.rattlers[new_order[i]]
                for j in range(self.bdim):
                    new_coords[i * self.bdim + j] = self.coords[
                        new_order[i] * self.bdim + j
                    ]
            self.hs_radii = new_radii
            self.rattlers = new_rattlers
            self.coords = new_coords
            self._dump_permutations(new_order, n)

    def _dump_permutations(self, old_ind, n):
        fname = "{0}/jammed_packing{1}.perm".format(self.base_directory, n)
        perms = []
        for new_ind in range(len(old_ind)):
            perms.append((old_ind[new_ind], new_ind))
        perms = sorted(perms, key=lambda perm: perm[0])
        with open(fname, "w") as forder:
            forder.write("old index, new index\n")
            for perm in perms:
                forder.write("{}, {}\n".format(perm[0], perm[1]))

    def _dump_configuration(self, n):
        """write coordinates to file .xyzdr"""
        directory = self.base_directory
        if self.bdim == 2:
            fname = "{0}/jammed_packing{1}.xydr".format(directory, n)
        elif self.bdim == 3:
            fname = "{0}/jammed_packing{1}.xyzdr".format(directory, n)

        # dump configuration
        coords = self._correct_coords()
        if self.bdim == 2:
            f = open(fname, "w")
            for i in range(self.nparticles):
                f.write(
                    "{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n".format(
                        coords[i * self.bdim],
                        coords[i * self.bdim + 1],
                        self.hs_radii[i] * 2,
                        self.rattlers[i],
                    )
                )
        elif self.bdim == 3:
            f = open(fname, "w")
            for i in range(self.nparticles):
                f.write(
                    "{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\n".format(
                        coords[i * self.bdim],
                        coords[i * self.bdim + 1],
                        coords[i * self.bdim + 2],
                        self.hs_radii[i] * 2,
                        self.rattlers[i],
                    )
                )
        else:
            raise NotImplementedError(
                "bdim={} not implemented".format(self.bdim)
            )
        f.close()

    def _write_opengl_input(self, n):
        """write opengl input file"""
        coords = self._correct_coords()
        boxv = self.boxv
        colour = 14
        directory = self.base_directory
        fname = "{0}/jammed_packing{1}.dat".format(directory, n)
        f = open(fname, "w")
        f.write("{}\n".format(self.nparticles))
        if self.bdim == 2:
            f.write(
                "{} {} {}\n".format(
                    -boxv[0] / 2, -boxv[1] / 2, -np.amax(self.hs_radii)
                )
            )
            f.write("{} \t 0.0 \t 0.0\n".format(boxv[0]))
            f.write("0.0 \t {} \t 0.0\n".format(boxv[1]))
            f.write("0.0 \t 0.0 \t {}\n".format(np.amax(self.hs_radii) * 2))
            for i in range(self.nparticles):
                for j in range(self.bdim):
                    f.write("{}\t".format(coords[i * self.bdim + j]))
                f.write("{}\t".format(0.0))
                f.write("{}\t".format(self.hs_radii[i] * 2 * (1.0 + self.sca)))
                f.write("{}\n".format(colour - int(self.rattlers_draw[i])))
        elif self.bdim == 3:
            f.write(
                "{} {} {}\n".format(-boxv[0] / 2, -boxv[1] / 2, -boxv[2] / 2)
            )
            f.write("{} \t 0.0 \t 0.0\n".format(boxv[0]))
            f.write("0.0 \t {} \t 0.0\n".format(boxv[1]))
            f.write("0.0 \t 0.0 \t {}\n".format(boxv[2]))
            for i in range(self.nparticles):
                for j in range(self.bdim):
                    f.write("{}\t".format(coords[i * self.bdim + j]))
                f.write("{}\t".format(self.hs_radii[i] * 2 * (1.0 + self.sca)))
                f.write("{}\n".format(colour - int(self.rattlers_draw[i])))
        else:
            raise NotImplementedError(
                "bdim={} not implemented".format(self.bdim)
            )
        f.close()

    # def _histogram_eigenvalues(self):
    #     #self.eigenvalues = np.array(self.eigenvalues,dtype='d')
    #     self.block_evalues = np.real(self.block_evalues)
    #     pylab.figure()
    #     self.block_histogram, bins = np.histogram(self.block_evalues ,bins=self.nbins)
    #     width = bins[1] - bins[0]
    #     center = (bins[:-1] + bins[1:]) / 2
    #     pylab.bar(center, self.block_histogram, align='center', width=width)
    #     pylab.savefig(os.path.join(self.base_directory,'blocks_histogram.eps'))
    #     if self.show:
    #         pylab.show()
    #     pylab.figure()
    #     self.block_histogram_low, bins = np.histogram(self.block_evalues,
    # bins=self.nbins_low, range=self.low_range)
    #     width = bins[1] - bins[0]
    #     center = (bins[:-1] + bins[1:]) / 2
    #     pylab.bar(center, self.block_histogram_low, align='center', width=width)
    #     pylab.savefig(os.path.join(self.base_directory,
    # 'blocks_histogram_low{}.eps'.format(self.low_range[1])) )
    #     if self.show:
    #         pylab.show()
    #
    #     self.whole_evalues = np.real(self.whole_evalues)
    #     pylab.figure()
    #     self.whole_histogram, bins = np.histogram(self.whole_evalues ,bins=self.nbins)
    #     width = bins[1] - bins[0]
    #     center = (bins[:-1] + bins[1:]) / 2
    #     pylab.bar(center, self.whole_histogram, align='center', width=width)
    #     pylab.savefig(os.path.join(self.base_directory,'whole_histogram.eps'))
    #     if self.show:
    #         pylab.show()
    #     pylab.figure()
    #     self.whole_histogram_low, bins = np.histogram(self.whole_evalues,
    # bins=self.nbins_low, range=self.low_range)
    #     width = bins[1] - bins[0]
    #     center = (bins[:-1] + bins[1:]) / 2
    #     pylab.bar(center, self.whole_histogram_low, align='center', width=width)
    #     pylab.savefig(os.path.join(self.base_directory,'whole_histogram_low{}.eps'.format(self.low_range[1])))
    #     if self.show:
    #         pylab.show()


class InversePowerGeneratePackings(HS_Generate_Jammed_Packing):
    """Generate soft sphere jammed packings for potential

    Note that this class is a hacky way of getting the basin volume workflow running for
    inverse power potentials. There are no hard sphere packings in the workflow
    This workflow should be refactored to better reflect the generality of the method
    """

    def __init__(
        self,
        target_packing_frac=0.7,
        tol=1e-9,
        maxstep_factor=1.0,
        packings_dir="packings",  # not necessary: exists for compatibility reasons
        packing_nrs=None,
        import_jammed=False,
        outdir="jammed_packings",
        use_cell_lists=False,
        show=False,
        interaction=Interaction.HS_WCA,
        override_pot_kwargs=None,
        minimizer=Minimizer.FIRE,
        logging_tag="",
        write_opengl=False,
        check_packing=True,
        sort_atoms=False,
        eps=1.0,
        power=2.5,  # Hertzian exponent
        r1=1.0,
        r2=1.4,
        rstd1=0.05,
        rstd2=0.05 * 1.4,
        seed=0,
    ):
        super().__init__(
            target_packing_frac=target_packing_frac,
            tol=tol,
            maxstep_factor=maxstep_factor,
            packings_dir=packings_dir,
            packing_nrs=packing_nrs,
            import_jammed=import_jammed,
            outdir=outdir,
            use_cell_lists=use_cell_lists,
            show=show,
            interaction=interaction,
            override_pot_kwargs=override_pot_kwargs,
            minimizer=minimizer,
            logging_tag=logging_tag,
            write_opengl=write_opengl,
            check_packing=check_packing,
            sort_atoms=sort_atoms,
        )
        self.parameters = BINARY_SOFT_SPHERE_DEFAULTS.copy()

        self.parameters["eps"] = eps
        self.parameters["power"] = power
        self.parameters["phi"] = target_packing_frac
        self.parameters["r1"] = r1
        self.parameters["r2"] = r2
        self.parameters["rstd1"] = rstd1
        self.parameters["rstd2"] = rstd2
        self.seed = seed

    def one_iteration(self, fname):
        """perform one iteration"""
        self._import_single_config_file(fname)
        self.parameters["n_part"] = self.nparticles
        self.parameters["ndim"] = self.bdim
        self.rattlers = np.empty(self.nparticles, dtype="d")
        self.rattlers_draw = np.empty(self.nparticles, dtype="d")
        print("ndim", self.ndim)
        result_dict = setup_bidisperse(self.parameters, seed=self.seed)
        radii = result_dict["radii"]
        box_length = result_dict["box_length"]
        self._import_packing_configuration(fname)
        self.max_nrattlers = int(self.nparticles * 0.5)
        self.hs_radii = radii
        self.sca = 0.0  # no sca for inverse power
        self.boxv = np.array([box_length] * self.bdim)
        self.boxl = box_length
        self.pot_kwargs = self.parameters.copy()
        # assert that largest soft particle is not > 1/2 of smallest box size
        if np.amax(self.hs_radii) * 2 >= np.amin(self.boxv) / 2:
            logging.warning(self._log("Max soft diameter >= 1/2 box side!"))
        if np.amax(self.hs_radii) * 2 * (1 + self.sca) >= np.amin(self.boxv):
            raise Exception("WARNING: particle does not fit the box")

        # potential needs to be called because self.coords is an input argument
        # of HS_WCAPeriodicCellLists
        # rut set to largest particle diameter

        self.parameters["radii"] = self.hs_radii
        self.parameters["box_length"] = self.boxl
        self.potential = setup_bidisperse(self.parameters, seed=self.seed)["potential"]
        self.parameters.pop("radii")
        self.parameters.pop("box_length")

        success = self._generate_packing_coords()  # returns false if saddle

        n = int(re.search(r"\d+", fname).group())
        if success:
            self._print(n)
        else:
            path_list = glob.glob(
                "{0}/jammed_packing{1}.*".format(self.base_directory, n)
            )
            if len(path_list) > 0:
                with open(
                    "{0}/mismatching_rattlers.txt".format(self.base_directory),
                    "a",
                ) as f:
                    f.write("jammed_packing{}\n".format(n))
            for path_ in path_list:
                p = subprocess.call(shlex.split("rm {}".format(path_)))
        self.iteration += 1
        return success

    def _generate_packing_coords_iteration(self, tol=1e-9, iprint=-1):
        """quenches the imported structure"""

        opt_maxstep = np.amin(self.hs_radii) * 0.5 * self.maxstep_factor
        if self.minimizer is Minimizer.FIRE:
            res = modifiedfire_cpp(
                self.coords,
                self.potential,
                maxstep=opt_maxstep,
                nsteps=1e6,
                tol=tol,
                iprint=iprint,
            )
        elif self.minimizer is Minimizer.CG:
            optimizer = CGDescent(
                self.coords,
                self.potential,
                tol=tol,
                nsteps=1e6,
                print_level=iprint,
            )
            res = optimizer.run()
        elif self.minimizer is Minimizer.CVODE:
            from pele.optimize import CVODEBDFOptimizer

            print("this should be what I'm searching for")
            self.optimizer = CVODEBDFOptimizer(
                self.potential,
                self.coords,
                tol=tol,
                nsteps=int(1e7),
                atol=INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim],
                rtol=INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim],
            )
            res = self.optimizer.run(int(1e6))
        elif self.minimizer is Minimizer.MXD:
            from pele.optimize import ExtendedMixedOptimizer

            print("here")
            ratol = (
                INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim]
                * 1e-1
            )
            self.optimizer = ExtendedMixedOptimizer(
                self.potential,
                self.coords,
                tol=tol,
                nsteps=1e7,
                atol=ratol,
                rtol=ratol,
                T=get_mxd_t(self.nparticles),
            )
            res = self.optimizer.run(int(1e6))
        elif self.minimizer is Minimizer.LBFGS:
            res = lbfgs_cpp(
                self.coords,
                self.potential,
                maxstep=opt_maxstep,
                tol=tol,
                nsteps=1e6,
                maxErise=0,
                iprint=iprint,
            )
        else:
            raise NotImplementedError

        if not res.success:
            print(res)
            logging.warning(self._log("Quench failed"))
            return False

        self.coords = res.coords
        self.energy = res.energy

        # test that on re-minimisation the structure does not change
        if __debug__ and self.check_packing:
            if self.minimizer is Minimizer.FIRE:
                res2 = modifiedfire_cpp(
                    self.coords,
                    self.potential,
                    maxstep=opt_maxstep,
                    nsteps=1e6,
                    tol=tol,
                )
            elif self.minimizer is Minimizer.CG:
                optimizer = CGDescent(
                    self.coords,
                    self.potential,
                    tol=tol,
                    nsteps=1e6,
                    print_level=iprint,
                )
                res2 = optimizer.run()
            elif self.minimizer is Minimizer.CVODE:
                from pele.optimize import CVODEBDFOptimizer

                self.optimizer = CVODEBDFOptimizer(
                    self.potential,
                    self.coords,
                    tol=tol,
                    nsteps=1e7,
                    atol=INVERSE_POWER_CVODE_95_ACC[
                        len(self.coords) // self.bdim
                    ],
                    rtol=INVERSE_POWER_CVODE_95_ACC[
                        len(self.coords) // self.bdim
                    ],
                )
                res2 = self.optimizer.run(int(1e6))
            elif self.minimizer is Minimizer.MXD:
                from pele.optimize import ExtendedMixedOptimizer

                ratol = (
                    INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim]
                    * 1e-1
                )
                self.optimizer = ExtendedMixedOptimizer(
                    self.potential,
                    self.coords,
                    tol=tol,
                    atol=ratol,
                    rtol=ratol,
                    T=get_mxd_t(self.nparticles),
                )
                res2 = self.optimizer.run(int(1e6))
            elif self.minimizer is Minimizer.LBFGS:
                res2 = lbfgs_cpp(
                    self.coords,
                    self.potential,
                    maxstep=opt_maxstep,
                    tol=tol,
                    nsteps=1e6,
                    maxErise=0,
                    iprint=iprint,
                )
            else:
                raise NotImplementedError
            if res2.nfev > 1:
                logging.warning(
                    self._log(
                        "Quench failed (structure changed at "
                        "second minimisation)"
                    )
                )
                return False

        return self._find_rattlers()


class NegativeCosGeneratePackings(HS_Generate_Jammed_Packing):
    """Hacky way to keep the interface the same while providing
    the correct initial condition for negative cosine position
    Generates [0]*ndim and saves it as coords
    All the arguments don't matter. just keeps the interface the same
    """

    def __init__(
        self,
        target_packing_frac=0.7,
        tol=1e-9,
        maxstep_factor=1,
        packings_dir="packings",
        packing_nrs=None,
        import_jammed=False,
        outdir="jammed_packings",
        use_cell_lists=False,
        show=False,
        interaction=Interaction.HS_WCA,
        override_pot_kwargs=None,
        minimizer=Minimizer.FIRE,
        logging_tag="",
        write_opengl=False,
        check_packing=True,
        sort_atoms=False,
    ):
        super().__init__(
            target_packing_frac,
            tol,
            maxstep_factor,
            packings_dir,
            packing_nrs,
            import_jammed,
            outdir,
            use_cell_lists,
            show,
            interaction,
            override_pot_kwargs,
            minimizer,
            logging_tag,
            write_opengl,
            check_packing,
            sort_atoms,
        )

    def one_iteration(self, fname):
        """perform one iteration"""
        self._import_single_config_file(fname)
        self.parameters["n_part"] = self.nparticles
        self.parameters["ndim"] = self.bdim
        self.rattlers = np.empty(self.nparticles, dtype="d")
        self.rattlers_draw = np.empty(self.nparticles, dtype="d")
        print("ndim", self.ndim)
        radii = np.array([1.0] * self.nparticles)  # no radiii
        # periodic anyway
        box_length = 4.0

        self._import_packing_configuration(fname)
        self.max_nrattlers = int(self.nparticles)
        self.hs_radii = radii
        self.sca = 0.0  # no sca for inverse power
        self.boxv = np.array([box_length] * self.bdim)
        self.boxl = box_length
        self.pot_kwargs = dict(dim=1.0, period=1.0)

        self.parameters["radii"] = self.hs_radii
        self.parameters["box_length"] = self.boxl
        self.potential = setup_bidisperse(self.parameters, seed=self.seed)["potential"]

        success = self._generate_packing_coords()  # returns false if saddle

        n = int(re.search(r"\d+", fname).group())
        if success:
            self._print(n)
        else:
            path_list = glob.glob(
                "{0}/jammed_packing{1}.*".format(self.base_directory, n)
            )
            if len(path_list) > 0:
                with open(
                    "{0}/mismatching_rattlers.txt".format(self.base_directory),
                    "a",
                ) as f:
                    f.write("jammed_packing{}\n".format(n))
            for path_ in path_list:
                p = subprocess.call(shlex.split("rm {}".format(path_)))
        self.iteration += 1
        return success

    def _generate_packing_coords_iteration(self, tol=1e-9, iprint=-1):
        """quenches the imported structure"""

        opt_maxstep = np.amin(self.hs_radii) * 0.5 * self.maxstep_factor
        if self.minimizer is Minimizer.FIRE:
            res = modifiedfire_cpp(
                self.coords,
                self.potential,
                maxstep=opt_maxstep,
                nsteps=1e6,
                tol=tol,
                iprint=iprint,
            )
        elif self.minimizer is Minimizer.CG:
            optimizer = CGDescent(
                self.coords,
                self.potential,
                tol=tol,
                nsteps=1e6,
                print_level=iprint,
            )
            res = optimizer.run()
        elif self.minimizer is Minimizer.CVODE:
            from pele.optimize import CVODEBDFOptimizer

            self.optimizer = CVODEBDFOptimizer(
                self.potential,
                self.coords,
                tol=tol,
                nsteps=1e7,
                atol=INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim],
                rtol=INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim],
            )
            res = self.optimizer.run(int(1e6))
        elif self.minimizer is Minimizer.MXD:
            from pele.optimize import ExtendedMixedOptimizer

            ratol = (
                INVERSE_POWER_CVODE_95_ACC[len(self.coords) // self.bdim]
                * 1e-1
            )
            self.optimizer = ExtendedMixedOptimizer(
                self.potential,
                self.coords,
                tol=tol,
                nsteps=1e7,
                atol=ratol,
                rtol=ratol,
                T=get_mxd_t(self.nparticles),
            )
            res = self.optimizer.run(int(1e6))
        elif self.minimizer is Minimizer.LBFGS:
            res = lbfgs_cpp(
                self.coords,
                self.potential,
                maxstep=opt_maxstep,
                tol=tol,
                nsteps=1e6,
                maxErise=0,
                iprint=iprint,
            )
        else:
            raise NotImplementedError
        print("after")
        if not res.success:
            print(res)
            logging.warning(self._log("Quench failed"))
            return False

        self.coords = np.array([0.0] * self.ndim)
        self.energy = -1.0
        return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="generate 2/3-D hard disks/spheres packings"
    )
    parser.add_argument(
        "-p",
        "--density",
        type=float,
        help="target packing fraction",
        default=0.7,
    )
    parser.add_argument(
        "--nocell",
        action="store_true",
        help="don't use cell lists, " "default: False",
        default=False,
    )
    parser.add_argument(
        "--packingsdir",
        type=str,
        help="name of directory with packings, " "must be in cwd",
        default="packings",
    )
    parser.add_argument(
        "--packing-nrs",
        type=int,
        nargs="*",
        help="Restrict the " "packings to jam by a list of packing numbers.",
        default=None,
    )
    parser.add_argument(
        "--import-jammed",
        action="store_true",
        help="Take a jammed packing as input " "instead of an unjammed one.",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Directory to save jammed packings in. "
        "Default: 'jammed_packings'",
        default="jammed_packings",
    )
    parser.add_argument(
        "--show", action="store_true", help="show histograms", default=False
    )
    parser.add_argument(
        "--write-opengl",
        action="store_true",
        help="Write input for OpenGL.",
        default=False,
    )
    parser.add_argument(
        "-t",
        "--tol",
        type=float,
        help="rms tolerance of the minimizer",
        default=1e-9,
    )
    parser.add_argument(
        "--maxstep",
        type=float,
        help="Factor by which the maximum step size of the "
        "minimizer is corrected.",
        default=1.0,
    )
    parser.add_argument(
        "--minimizer",
        type=str,
        help="Energy minimization algorithm "
        "used for quenching. Options: 'CG', 'FIRE', 'LBFGS'. "
        "Default: 'FIRE'",
        default="FIRE",
    )
    # potential arguments
    parser.add_argument(
        "--interaction",
        type=str,
        help="Particle interaction potential. "
        "Options: 'HS_WCA', 'INVERSE_POWER_STILLINGER', INVERSE_POWER, \
            'NEGATIVE_COS'. "
        "Default: 'HS_WCA'",
        default="HS_WCA",
    )
    parser.add_argument(
        "--wca-exp",
        type=int,
        help="Exponent of the WCA potential (if applicable). "
        "Options: 1, 2, 6. Default: 6 (Lennard-Jones-like)",
        default=6,
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Use the potential to sort the atoms before saving. "
        "Default: False",
        default=False,
    )
    parser.add_argument(
        "--balance-omp",
        type=bool,
        help="Balance subdomains when using multi-threaded "
        "cell lists. Default: Use setting from packing config",
        default=None,
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info(args)

    if args.minimizer.upper() in Minimizer.__members__:
        minimizer = Minimizer[args.minimizer.upper()]
    else:
        raise ValueError("Unknown minimizer: {}".format(args.minimizer))

    # potential type
    if args.interaction.upper() in Interaction.__members__:
        interaction = Interaction[args.interaction.upper()]
    else:
        raise ValueError("Unknown interaction: {}".format(args.interaction))

    override_pot_kwargs = dict()
    if interaction is Interaction.HS_WCA:
        override_pot_kwargs["exp"] = args.wca_exp
    elif interaction is Interaction.INVERSE_POWER_STILLINGER:
        override_pot_kwargs.update(pow=8, rcut=4.5)
        logging.info(
            "Setting inverse_power_stillinger parameters: {}".format(
                override_pot_kwargs
            )
        )
    elif interaction is Interaction.INVERSE_POWER:
        override_pot_kwargs.update(pow=2.5)
    else:
        raise NotImplementedError
    if args.balance_omp is not None:
        override_pot_kwargs["balance_omp"] = args.balance_omp

    if interaction is Interaction.INVERSE_POWER:
        sim = InversePowerGeneratePackings(
            target_packing_frac=args.density,
            packings_dir=args.packingsdir,
            packing_nrs=args.packing_nrs,
            import_jammed=args.import_jammed,
            outdir=args.outdir,
            tol=args.tol,
            maxstep_factor=args.maxstep,
            use_cell_lists=not args.nocell,
            show=args.show,
            interaction=interaction,
            minimizer=minimizer,
            override_pot_kwargs=override_pot_kwargs,
            write_opengl=args.write_opengl,
            sort_atoms=args.sort,
        )
    elif interaction is Interaction.NEGATIVE_COS:
        sim = NegativeCosGeneratePackings(
            packings_dir=args.packingsdir,
            packing_nrs=args.packing_nrs,
            import_jammed=args.import_jammed,
            outdir=args.outdir,
            tol=args.tol,
            maxstep_factor=args.maxstep,
            use_cell_lists=not args.nocell,
            show=args.show,
            interaction=interaction,
            minimizer=minimizer,
            override_pot_kwargs=override_pot_kwargs,
            write_opengl=args.write_opengl,
            sort_atoms=args.sort,
        )

    else:
        sim = HS_Generate_Jammed_Packing(
            target_packing_frac=args.density,
            packings_dir=args.packingsdir,
            packing_nrs=args.packing_nrs,
            import_jammed=args.import_jammed,
            outdir=args.outdir,
            tol=args.tol,
            maxstep_factor=args.maxstep,
            use_cell_lists=not args.nocell,
            show=args.show,
            interaction=interaction,
            minimizer=minimizer,
            override_pot_kwargs=override_pot_kwargs,
            write_opengl=args.write_opengl,
            sort_atoms=args.sort,
        )
    sim.run()
