from __future__ import division
from __future__ import absolute_import
from future import standard_library

standard_library.install_aliases()
from builtins import str
import numpy as np
import os
import traceback
import configparser
import logging
import argparse
from pele.utils._pressure_tensor import pressure_tensor
from pele.potentials import InversePowerStillingerCut
from pele.optimize._quench import modifiedfire_cpp
from basinvolume.utils import trymakedir
from basinvolume.enums import Interaction
from ._structural_analysis import StructuralAnalysis


class PressureTensor(StructuralAnalysis):
    def __init__(
        self,
        workspace,
        jammed_packings_dir="jammed_packings",
        analysis_dir="analysis",
        force=False,
        existing_only=True,
        prefix="explore_bv_",
        verbose=True,
        use_cell_lists=True,
    ):
        super(PressureTensor, self).__init__(
            workspace,
            jammed_packings_dir=jammed_packings_dir,
            analysis_dir=analysis_dir,
            force=force,
            existing_only=existing_only,
            prefix=prefix,
            verbose=verbose,
            use_cell_lists=use_cell_lists,
        )
        self.analysis_name = "pressure_data"

    @staticmethod
    def read(pressure_fname):
        configf = configparser.ConfigParser()
        configf.read(pressure_fname)
        pressure_dict = {}
        pressure_dict["P"] = configf.getfloat("PRESSURE", "P")
        pressure_dict["maxshear_xyplane"] = configf.getfloat(
            "PRESSURE", "maxshear_xyplane"
        )
        pressure_dict["Ptensor"] = np.array(
            [float(x) for x in configf.get("PRESSURE", "Ptensor").split()]
        )
        pressure_dict["E"] = configf.getfloat("ENERGY", "E")
        return pressure_dict

    def _calculate(self, pressure_fname, packing_name, input_fname):
        """compute the pressure tensor for packings"""
        if self.verbose:
            logging.info(
                "Calculating pressure: {}".format(self.prefix + str(packing_name))
            )
        (
            self.coords,
            self.hs_radii,
            self.ss_radii,
            _,
        ) = self._import_packing_configuration(input_fname)
        self.init_pressure_potential()
        # refine structure (does not make a difference if tol was small enough to start with)
        # if self.packing_frac < 0.835:
        #     fire_maxstep = np.amin(self.hs_radii) * self.sca
        #     res = modifiedfire_cpp(self.coords, self.potential, maxstep=fire_maxstep,
        #                            nsteps=1e6, tol=1e-11, iprint=-1)
        #     self.coords = res.coords
        p, ptensor = pressure_tensor(
            self.potential, self.coords, self.vcavity, self.bdim
        )
        max_shear_xyplane = np.sqrt(
            ((ptensor[0] - ptensor[3]) / 2.0) ** 2 + ptensor[1] ** 2
        )
        energy = self.potential.getEnergy(self.coords)
        with open(pressure_fname, "w") as f:
            f.write("#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND \n")
            f.write("[PRESSURE]\n")
            f.write("P: {:.16f}\n".format(p))
            f.write("maxshear_xyplane: {:.16f}\n".format(max_shear_xyplane))
            f.write("Ptensor: ")
            for val in ptensor:
                f.write("{:.16f} ".format(val))
            f.write("\n")
            f.write("[ENERGY]\n")
            f.write("E: {:.16f}\n".format(energy))

    def init_pressure_potential(self):
        # here put a flag and pick potential
        if self.interaction is Interaction.HS_WCA:
            self._initialise_potential()
        elif self.interaction is Interaction.INVERSE_POWER_STILLINGER:
            pow = self.pot_kwargs["pow"]
            rcut = self.pot_kwargs["rcut"]
            pot_optimizer = InversePowerStillingerCut(
                pow,
                self.stillinger_a_radii,
                ndim=self.bdim,
                boxvec=self.boxv,
                rcut=rcut,
                use_cell_lists=True,
            )
        else:
            raise NotImplementedError


def worker_pressure(workspace, kwargs):
    try:
        pressure = PressureTensor(workspace, **kwargs)
        pressure.run()
    except Exception:
        logging.error("worker_pressure worker: %s" % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the pressure tensor for " "jammed packings."
    )
    parser.add_argument(
        "-d",
        "--workspace-dir",
        type=str,
        help="Top-level dir containing " "the packings, e.g. 'n32_phi88_2D'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force to run on all packings.",
        default=False,
    )
    parser.add_argument(
        "--nonex",
        action="store_false",
        help="Run also for packings "
        "for which there are no work folders ('explore_bv_[...]', "
        "created e.g. by parallel tempering).",
        default=True,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix for the work directory. " "Default: 'explore_bv_'",
        default="explore_bv_",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing the " "jammed packings. Default: 'jammed_packings'",
        default="jammed_packings",
    )
    parser.add_argument(
        "--nocell",
        action="store_true",
        help="Don't use cell lists. " "Default: False",
        default=False,
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    kwargs = dict(
        force=args.force,
        existing_only=args.nonex,
        jammed_packings_dir=args.input_dir,
        prefix=args.prefix,
        use_cell_lists=not args.nocell,
    )

    if not args.workspace_dir:
        workspace_dir = os.getcwd()
    else:
        workspace_dir = os.path.abspath(args.workspace_dir)

    pts = PressureTensor(workspace_dir, **kwargs)
    pts.run()
