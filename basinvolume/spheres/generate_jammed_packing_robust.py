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
from pele.utils._pressure_tensor import pressure_tensor
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
from basinvolume.spheres.generate_packing import HS_Generate_Packing
from basinvolume.spheres.generate_jammed_packing import HS_Generate_Jammed_Packing, read_jammed_packing_config


class HS_Generate_Jammed_Packing_Robust(HS_Generate_Jammed_Packing):
    """
    Robust version of HS_Generate_Jammed_Packing that handles _find_rattlers failures
    by regenerating initial configurations until a valid jammed packing is found.
    
    This class extends the original HS_Generate_Jammed_Packing and adds retry logic
    that calls HS_Generate_Packing to generate new initial configurations when
    the rattler detection fails.
    """

    def __init__(
        self,
        target_packing_frac=0.7,
        opt_tol=1e-9,
        opt_maxstep_factor=1.0,
        opt_dtmax=1.0,
        opt_nsteps=1e5,
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
        max_retries=1000,  
        regeneration_method="direct",  # Method for generating new configurations: "direct" or "quench"
    ):
        super().__init__(
            target_packing_frac=target_packing_frac,
            opt_tol=opt_tol,
            opt_maxstep_factor=opt_maxstep_factor,
            opt_dtmax=opt_dtmax,
            opt_nsteps=opt_nsteps,
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
        self.max_retries = max_retries
        self.regeneration_method = regeneration_method
        self.packing_generator = None
        self.current_attempt = 0
        
    def _initialize_packing_generator(self, fname):
        """Initialize the HS_Generate_Packing instance for regenerating configurations"""
        if self.packing_generator is None:
            # Import configuration to get parameters
            self._import_single_config_file(fname)
            
            # Create HS_Generate_Packing instance with matching parameters
            self.packing_generator = HS_Generate_Packing(
                nparticles=self.nparticles,
                output_dir=self.packings_dir,
                method=self.regeneration_method,
                bdim=self.bdim,
                boxv=self.boxv.copy(),
                packing_frac=self.packing_frac,
                hs_radii=None,  # Will be set from imported configuration
                mu=self.hs_mean if hasattr(self, 'hs_mean') else 1.0,
                sig=self.hs_stddev / self.hs_mean if hasattr(self, 'hs_stddev') and hasattr(self, 'hs_mean') else 0.1,
                use_cell_lists=self.use_cell_lists,
                distance_method=self.distance_method,
                pot_kwargs=self.pot_kwargs.copy() if hasattr(self, 'pot_kwargs') else {},
            )
            
            # Initialize the generator
            self.packing_generator._initialise()
            
    def _generate_new_initial_configuration(self):
        """Generate a new initial configuration using HS_Generate_Packing"""
        if self.packing_generator is None:
            raise RuntimeError("Packing generator not initialized")
            
        self.packing_generator.hs_radii = self.hs_radii.copy()
        
        success = False
        attempts = 0
        max_generation_attempts = 100
        
        while not success and attempts < max_generation_attempts:
            if self.regeneration_method == "direct":
                self.packing_generator._generate_packing_coords_direct()
            elif self.regeneration_method == "quench":
                if attempts == 0:
                    self.packing_generator._initialise_coords_quench()
                self.packing_generator._generate_packing_coords_quench()
            else:
                raise ValueError(f"Unknown regeneration method: {self.regeneration_method}")
            self.packing_generator.coords = self.packing_generator.coords if hasattr(self.packing_generator, 'coords') else self.packing_generator._sample_random_coords()
            success = self.packing_generator._check_no_overlaps()
            attempts += 1
            
            if not success:
                logging.debug(self._log(f"Failed to generate valid initial configuration, attempt {attempts}/{max_generation_attempts}"))
                
        if success:
            self.coords = self.packing_generator.coords.copy()
            self.initial_coords = self.coords.copy()
            return True
        else:
            logging.warning(self._log("Failed to generate valid initial configuration after maximum attempts"))
            return False

    def _generate_packing_coords(self):
        """
        Robust version that retries with new initial configurations when _find_rattlers fails.
        """
        for attempt in range(self.max_retries):
            self.current_attempt = attempt + 1
            

            if attempt > 0:
                logging.info(self._log(f"Attempt {attempt + 1}/{self.max_retries}: Generating new initial configuration"))
                if not self._generate_new_initial_configuration():
                    logging.warning(self._log(f"Failed to generate new initial configuration on attempt {attempt + 1}"))
                    continue
                    

            success = self._generate_packing_coords_iteration(opt_tol=self.opt_tol)
            
            if success:
                logging.info(self._log(f"Successfully generated jammed packing on attempt {attempt + 1}"))
                return True
            else:
                logging.info(self._log(f"Failed to generate valid jammed packing on attempt {attempt + 1}"))
                
        logging.warning(self._log(f"Failed to generate jammed packing after {self.max_retries} attempts"))
        return False

    def one_iteration(self, fname):
        """
        Override one_iteration to initialize the packing generator and use robust generation.
        """
        self._initialize_packing_generator(fname)
        return super().one_iteration(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate robust 2/3-D hard disks/spheres jammed packings")
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
        help="don't use cell lists, default: False",
        default=False,
    )
    parser.add_argument(
        "--packingsdir",
        type=str,
        help="name of directory with packings, must be in cwd",
        default="packings",
    )
    parser.add_argument(
        "--packing-nrs",
        type=int,
        nargs="*",
        help="Restrict the packings to jam by a list of packing numbers.",
        default=None,
    )
    parser.add_argument(
        "--import-jammed",
        action="store_true",
        help="Take a jammed packing as input instead of an unjammed one.",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Directory to save jammed packings in. Default: 'jammed_packings'",
        default="jammed_packings",
    )
    parser.add_argument("--show", action="store_true", help="show histograms", default=False)
    parser.add_argument(
        "--write-opengl",
        action="store_true",
        help="Write input for OpenGL.",
        default=False,
    )
    parser.add_argument(
        "-t",
        "--opt_tol",
        type=float,
        help="rms tolerance of the minimizer",
        default=1e-9,
    )
    parser.add_argument(
        "--opt_maxstep_factor",
        type=float,
        help="Factor by which the maximum step size of the minimizer is corrected.",
        default=1.0,
    )
    parser.add_argument("--opt_dtmax", type=float, default=1, help="For FIRE, max time step")
    parser.add_argument(
        "--opt_nsteps",
        type=float,
        default=1e7,
        help="number of steps for optimizer",
    )
    parser.add_argument(
        "--minimizer",
        type=str,
        help="Energy minimization algorithm used for quenching. Options: 'CG', 'FIRE', 'LBFGS'. Default: 'FIRE'",
        default="FIRE",
    )
    # potential arguments
    parser.add_argument(
        "--interaction",
        type=str,
        help="Particle interaction potential. Options: 'HS_WCA', 'INVERSE_POWER_STILLINGER'. Default: 'HS_WCA'",
        default="HS_WCA",
    )
    parser.add_argument(
        "--wca-exp",
        type=int,
        help="Exponent of the WCA potential (if applicable). Options: 1, 2, 6. Default: 6 (Lennard-Jones-like)",
        default=6,
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Use the potential to sort the atoms before saving. Default: False",
        default=False,
    )
    parser.add_argument(
        "--balance-omp",
        type=bool,
        help="Balance subdomains when using multi-threaded cell lists. Default: Use setting from packing config",
        default=None,
    )
    # Robust generation parameters
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum number of attempts to generate a valid jammed packing. Default: 1000",
        default=10000,
    )
    parser.add_argument(
        "--regeneration-method",
        type=str,
        help="Method for generating new initial configurations. Options: 'direct', 'quench'. Default: 'direct'",
        default="direct",
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
        logging.info("Setting inverse_power_stillinger parameters: {}".format(override_pot_kwargs))
    else:
        raise NotImplementedError
        
    if args.balance_omp is not None:
        override_pot_kwargs["balance_omp"] = args.balance_omp
        
    sim = HS_Generate_Jammed_Packing_Robust(
        target_packing_frac=args.density,
        packings_dir=args.packingsdir,
        packing_nrs=args.packing_nrs,
        import_jammed=args.import_jammed,
        outdir=args.outdir,
        opt_tol=args.opt_tol,
        opt_maxstep_factor=args.opt_maxstep_factor,
        opt_dtmax=args.opt_dtmax,
        opt_nsteps=args.opt_nsteps,
        use_cell_lists=not args.nocell,
        show=args.show,
        interaction=interaction,
        minimizer=minimizer,
        override_pot_kwargs=override_pot_kwargs,
        write_opengl=args.write_opengl,
        sort_atoms=args.sort,
        max_retries=args.max_retries,
        regeneration_method=args.regeneration_method,
    )
    
    sim.run()