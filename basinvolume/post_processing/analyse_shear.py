from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
import numpy as np
import argparse
import os
import traceback
import sys
import shutil
import pandas as pd
import multiprocessing as mp
import logging
from basinvolume.post_processing.structural_properties import (
    BondOrientationalOrder,
    PressureTensor,
    Neighbors,
    InversionSymmetry,
    Displacement,
    worker_boo,
    worker_disp,
    worker_invsym,
    worker_neighbors,
    worker_pressure,
)


def worker_lasting_neighbors(workspace_dir, kwargs):
    try:
        for subshear in np.arange(
            kwargs["shear"] - kwargs["step"],
            kwargs["shear"] - 0.5 * kwargs["substep"],
            kwargs["substep"],
        ):
            restrict_prefix = os.path.join("shear_{}".format(subshear), "explore_bv_")
            subshear_dname = "shear_{}".format(subshear + kwargs["substep"])
            subshear_prefix = os.path.join(subshear_dname, "explore_bv_")
            last_substep = np.isclose(subshear, kwargs["shear"] - kwargs["substep"])
            kwargs["neighbors_dyn_kwargs"].update(
                jammed_packings_dir=os.path.join(subshear_dname, "jammed_packings"),
                prefix=subshear_prefix,
                restrict_neighbors=restrict_prefix,
                write_analysis=last_substep,
            )
            worker_neighbors(workspace_dir, kwargs["neighbors_dyn_kwargs"])
    except Exception:
        logging.error("worker_lasting_neighbors worker: %s" % (traceback.format_exc()))


class AnalyseShear(object):
    def __init__(
        self,
        input_dir=".",
        output_dir="shear_analysis",
        force=False,
        force_rel=False,
        start=0.0,
        step=0.01,
        substep=0.001,
        stop=1.0,
        njobs=1,
        use_cell_lists=True,
        calc_neighbors=False,
        calc_neighbors_dyn=False,
        calc_neighbors_orig=False,
        calc_boo=False,
        calc_invsym=False,
        calc_pressure=False,
        calc_displacement=False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.force = force
        self.force_rel = force_rel
        self.start = start
        self.step = step
        self.substep = substep
        self.stop = stop
        self.njobs = njobs
        self.use_cell_lists = use_cell_lists
        self.calc_neighbors = calc_neighbors
        self.calc_neighbors_dyn = calc_neighbors_dyn
        self.calc_neighbors_orig = calc_neighbors_orig
        self.calc_boo = calc_boo
        self.calc_invsym = calc_invsym
        self.calc_pressure = calc_pressure
        self.calc_displacement = calc_displacement

    def run(self):
        self.make_output_dirs()
        if self.njobs > 1:
            self.mypool = mp.Pool(self.njobs)
        for shear in np.arange(self.start, self.stop + 0.5 * self.step, self.step):
            self.results = []
            shear_dir = "shear_{}".format(shear)
            if not os.path.isdir(os.path.join(self.input_dir, shear_dir)):
                logging.error(
                    "The shear directory {} does not exist. Stopping analysis.".format(shear_dir)
                )
                sys.exit(1)
            self.calc_parameters(shear, shear_dir)
            if self.njobs > 1:
                for result in self.results:
                    result.get()
            self.collect_files(shear, shear_dir)
        if self.njobs > 1:
            self.mypool.close()
            self.mypool.join()
        self.collect_parameters()

    def make_output_dirs(self):
        if os.path.exists(self.output_dir):
            logging.info("Old output directory '{}' found.".format(self.output_dir))
            if self.force:
                logging.info("Removing old output directory '{}'.".format(self.output_dir))
                shutil.rmtree(self.output_dir)
                os.mkdir(self.output_dir)
        else:
            os.mkdir(self.output_dir)

        # Get parameter directories to create
        all_param_dirs = [
            "neighbors",
            "neighbors_dyn",
            "neighbors_orig",
            "boo",
            "inversion_symmetry",
            "pressure_tensor",
            "displacement",
        ]
        params = [
            self.calc_neighbors,
            self.calc_neighbors_dyn,
            self.calc_neighbors_orig,
            self.calc_boo,
            self.calc_invsym,
            self.calc_pressure,
            self.calc_displacement,
        ]
        param_dirs = [all_param_dirs[i] for i in range(len(params)) if params[i]]

        # Get packings to create
        input_files = os.listdir(
            os.path.join(
                self.input_dir,
                "shear_{}".format(self.start),
                "jammed_packings",
            )
        )
        packing_files = [
            pname
            for pname in input_files
            if "jammed_packing" in pname and ("xyzdr" in pname or "xydr" in pname)
        ]
        packings = [packing.split("_")[1].split(".")[0] for packing in packing_files]

        # Create directory structure
        for packing in packings:
            packing_path = os.path.join(self.output_dir, packing)
            if not os.path.exists(packing_path):
                os.mkdir(packing_path)
            for param_dir in param_dirs:
                param_path = os.path.join(packing_path, param_dir)
                if not os.path.exists(param_path):
                    os.mkdir(param_path)

    def calc_parameters(self, shear, input_relpath):
        logging.info("Calculating parameters for shear={}".format(shear))

        if os.path.isabs(self.input_dir):
            workspace_dir = self.input_dir
        else:
            workspace_dir = os.path.abspath(self.input_dir)
        kwargs = dict(
            verbose=False,
            force=self.force,
            existing_only=False,
            jammed_packings_dir=os.path.join(input_relpath, "jammed_packings"),
            prefix=os.path.join(input_relpath, "explore_bv_"),
            use_cell_lists=self.use_cell_lists,
        )
        structural_props = []

        # Bond orientational order
        if self.calc_boo:
            boo_kwargs = dict(kwargs, solid_angle_weighted=False)
            structural_props.append((worker_boo, boo_kwargs))

        # Displacement from previous packing
        if self.calc_displacement:
            disp_kwargs = dict(kwargs)
            disp_kwargs["force"] = disp_kwargs["force"] or self.force_rel
            if shear == self.start:
                disp_kwargs["shear"] = 0.0
                disp_kwargs["packings_old"] = os.path.join(input_relpath, "jammed_packings")
            else:
                disp_kwargs["shear"] = self.step
                disp_kwargs["packings_old"] = os.path.join(
                    "shear_{}".format(shear - self.step), "jammed_packings"
                )
            structural_props.append((worker_disp, disp_kwargs))

        # Local inversion symmetry
        if self.calc_invsym:
            invsym_kwargs = dict(kwargs)
            structural_props.append((worker_invsym, invsym_kwargs))

        # Neighbors (coordination number)
        if self.calc_neighbors:
            neighbors_kwargs = dict(kwargs, cutoff=1.0)
            structural_props.append((worker_neighbors, neighbors_kwargs))

        # Lasting neighbors
        if self.calc_neighbors_dyn:
            neighbors_dyn_kwargs = dict(kwargs, cutoff=1.0, analysis_name="neighbors_dyn")
            if shear == self.start:
                structural_props.append((worker_neighbors, neighbors_dyn_kwargs))
            else:
                neighbors_dyn_worker_kwargs = dict(
                    neighbors_dyn_kwargs=neighbors_dyn_kwargs,
                    shear=shear,
                    step=self.step,
                    substep=self.substep,
                )
                structural_props.append((worker_lasting_neighbors, neighbors_dyn_worker_kwargs))

        # Original neighbors
        if self.calc_neighbors_orig:
            neighbors_orig_kwargs = dict(kwargs, cutoff=1.0, analysis_name="neighbors_orig")
            if shear == self.start:
                structural_props.append((worker_neighbors, neighbors_orig_kwargs))
            else:
                restrict_prefix = os.path.join("shear_{}".format(self.start), "explore_bv_")
                neighbors_orig_kwargs.update(restrict_neighbors=restrict_prefix)
                structural_props.append((worker_neighbors, neighbors_orig_kwargs))

        # Pressure tensor
        if self.calc_pressure:
            pressure_kwargs = dict(kwargs)
            structural_props.append((worker_pressure, pressure_kwargs))

        # Start parallel calculations
        if self.njobs > 1:
            for prop in structural_props:
                self.results.append(
                    self.mypool.apply_async(
                        prop[0],
                        args=(
                            workspace_dir,
                            prop[1],
                        ),
                    )
                )
        else:
            for prop in structural_props:
                prop[0](workspace_dir, prop[1])

    def get_subdir_paths(self, directory, filter_str):
        files = os.listdir(directory)
        packings = [
            pname
            for pname in files
            if filter_str in pname and os.path.isdir(os.path.join(directory, pname))
        ]
        return sorted([os.path.join(directory, packing) for packing in packings])

    def collect_files(self, shear, input_relpath):
        logging.info("Collecting files for shear={}".format(shear))

        analysis_paths = self.get_subdir_paths(
            os.path.join(self.input_dir, input_relpath),
            "explore_bv_jammed_packing",
        )
        analysis_paths = [os.path.join(path, "analysis") for path in analysis_paths]
        output_paths = self.get_subdir_paths(self.output_dir, "packing")

        # Get parameter directory and file names
        params_from_to_all = [
            ("neighbors", "neighbors"),
            ("neighbors_dyn", "neighbors_dyn"),
            ("neighbors_orig", "neighbors_orig"),
            ("glob_boo", "boo"),
            ("inversion_symmetry", "inversion_symmetry"),
            ("pressure_data", "pressure_tensor"),
            ("displacement", "displacement"),
        ]
        params_choice = [
            self.calc_neighbors,
            self.calc_neighbors_dyn,
            self.calc_neighbors_orig,
            self.calc_boo,
            self.calc_invsym,
            self.calc_pressure,
            self.calc_displacement,
        ]
        params_from_to = [
            params_from_to_all[i] for i in range(len(params_choice)) if params_choice[i]
        ]

        # Iterate over packings and parameters
        for analysis_path, output_path in zip(analysis_paths, output_paths):
            for param_from_to in params_from_to:
                if os.path.isfile(os.path.join(analysis_path, param_from_to[0])):
                    shutil.copyfile(
                        os.path.join(analysis_path, param_from_to[0]),
                        os.path.join(
                            output_path,
                            param_from_to[1],
                            "shear_{}".format(shear),
                        ),
                    )

    def collect_boo_data(self, path, data):
        boo_path = os.path.join(path, "boo")
        boo_entry = pd.Series()
        for shear_file in sorted(os.listdir(boo_path)):
            boo_dict = BondOrientationalOrder.read(os.path.join(boo_path, shear_file))
            shear = float(shear_file.split("_")[1])
            boo_entry[shear] = boo_dict["BOO"][1]
        data["Bond-orientational order {}".format(boo_dict["BOO"][0])] = boo_entry

    def collect_displacement_data(self, path, data):
        displacement_path = os.path.join(path, "displacement")
        avg_abs_displacement_norm = pd.Series()
        avg_abs_nonaff_displacement_norm = pd.Series()
        dims = []
        avg_abs_displacement = []
        avg_abs_nonaff_displacement = []
        for shear_file in sorted(os.listdir(displacement_path)):
            displ_dict = Displacement.read(os.path.join(displacement_path, shear_file))
            shear = float(shear_file.split("_")[1])
            avg_abs_displacement_norm[shear] = displ_dict["avg_abs_displacement_norm"]
            avg_abs_nonaff_displacement_norm[shear] = displ_dict[
                "avg_abs_nonaff_displacement_norm"
            ]
            if len(dims) == 0:
                dims = ["x", "y", "z"]
                dims = dims[: len(displ_dict["avg_displacement"])]
                for i in range(len(dims)):
                    avg_abs_displacement.append(pd.Series())
                    avg_abs_nonaff_displacement.append(pd.Series())
            for i in range(len(dims)):
                avg_abs_displacement[i][shear] = displ_dict["avg_abs_displacement"][i]
                avg_abs_nonaff_displacement[i][shear] = displ_dict["avg_abs_nonaff_displacement"][
                    i
                ]
        data["Average absolute displacement"] = avg_abs_displacement_norm
        data["Average absolute non-affine displacement"] = avg_abs_nonaff_displacement_norm
        for i in range(len(dims)):
            data["Average absolute displacement {}".format(dims[i])] = avg_abs_displacement[i]
            data[
                "Average absolute non-affine displacement {}".format(dims[i])
            ] = avg_abs_nonaff_displacement[i]

    def collect_invsym_data(self, path, data):
        invsym_path = os.path.join(path, "inversion_symmetry")
        invsym_entry = pd.Series()
        for shear_file in sorted(os.listdir(invsym_path)):
            invsym_dict = InversionSymmetry.read(os.path.join(invsym_path, shear_file))
            shear = float(shear_file.split("_")[1])
            invsym_entry[shear] = invsym_dict["inversion_symmetry"]
        data["Local inversion symmetry"] = invsym_entry

    def collect_neighbors_data(self, neighbors_path, data, label):
        neighbors_entry = pd.Series()
        for shear_file in sorted(os.listdir(neighbors_path)):
            neighbors_dict = Neighbors.read(os.path.join(neighbors_path, shear_file))
            shear = float(shear_file.split("_")[1])
            neighbors_entry[shear] = neighbors_dict["avg_neighbors"]
        data[label] = neighbors_entry

    def collect_pressure_data(self, path, data):
        pressure_path = os.path.join(path, "pressure_tensor")
        energy_entry = pd.Series()
        pressure_entry = pd.Series()
        shear_entry = pd.Series()
        shears = ["xy"]
        shear_tensor = [pd.Series()]
        for shear_file in sorted(os.listdir(pressure_path)):
            pressure_dict = PressureTensor.read(os.path.join(pressure_path, shear_file))
            shear = float(shear_file.split("_")[1])
            energy_entry[shear] = pressure_dict["E"]
            pressure_entry[shear] = pressure_dict["P"]
            shear_entry[shear] = pressure_dict["maxshear_xyplane"]
            shear_tensor[0][shear] = pressure_dict["Ptensor"][1]
            if len(pressure_dict["Ptensor"]) == 9:
                if len(shears) == 1:
                    shears += ["xz", "yz"]
                    shear_tensor.append(pd.Series())
                    shear_tensor.append(pd.Series())
                shear_tensor[1][shear] = pressure_dict["Ptensor"][2]
                shear_tensor[2][shear] = pressure_dict["Ptensor"][5]
            elif not len(pressure_dict["Ptensor"]) == 4:
                raise NotImplementedError
        data["Energy"] = energy_entry
        data["Pressure"] = pressure_entry
        data["Shear stress"] = shear_entry
        for i in range(len(shears)):
            data["Shear stress {}".format(shears[i])] = shear_tensor[i]

    def collect_parameters(self):
        output_paths = self.get_subdir_paths(self.output_dir, "packing")

        # Iterate over packings
        for path in output_paths:
            logging.info("Collecting parameters for {}".format(path))
            data = pd.DataFrame()

            if self.calc_boo:
                self.collect_boo_data(path, data)
            if self.calc_displacement:
                self.collect_displacement_data(path, data)
            if self.calc_invsym:
                self.collect_invsym_data(path, data)
            if self.calc_neighbors:
                self.collect_neighbors_data(
                    os.path.join(path, "neighbors"), data, "Average neighbors"
                )
            if self.calc_neighbors_dyn:
                self.collect_neighbors_data(
                    os.path.join(path, "neighbors_dyn"),
                    data,
                    "Average lasting neighbors",
                )
            if self.calc_neighbors_orig:
                self.collect_neighbors_data(
                    os.path.join(path, "neighbors_orig"),
                    data,
                    "Average original neighbors",
                )
            if self.calc_pressure:
                self.collect_pressure_data(path, data)

            data.index.name = "Shear"
            data.to_csv(os.path.join(path, "data.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a shearing process.")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing the " "shearing process. Default: Current directory",
        default=".",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for saving the analyses. " "Default: 'shear_analysis'",
        default="shear_analysis",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force to run on all packings.",
        default=False,
    )
    parser.add_argument(
        "--force-rel",
        action="store_true",
        help="Force to recalculate relative measures (e.g. displacement) "
        "for all packings. This does not include lasting neighbors.",
        default=False,
    )
    parser.add_argument(
        "--start",
        type=float,
        help="Lowest shear to analyse. Default: 0.0",
        default=0.0,
    )
    parser.add_argument(
        "--step",
        type=float,
        help="Shear step size for structural properties. " "Default: 0.01",
        default=0.01,
    )
    parser.add_argument(
        "--substep",
        type=float,
        help="Actual shear step size. Default: step size",
        default=None,
    )
    parser.add_argument(
        "--stop",
        type=float,
        help="Highest shear to analyse. Default: 1.0",
        default=1.0,
    )
    parser.add_argument(
        "-j",
        "--njobs",
        type=int,
        help="Number of jobs to run in parallel. " "Default: 1 (serial)",
        default=1,
    )
    parser.add_argument(
        "--nocell",
        action="store_true",
        help="Don't use cell lists. " "Default: False",
        default=False,
    )
    parser.add_argument(
        "-z",
        "--neighbors",
        action="store_true",
        help="Calculate the static coordination numbers. Default: False",
        default=False,
    )
    parser.add_argument(
        "-zd",
        "--neighbors-dynamic",
        action="store_true",
        help="Calculate the dynamic coordination numbers (counts "
        "lasting neighbors). Default: False",
        default=False,
    )
    parser.add_argument(
        "-zo",
        "--neighbors-original",
        action="store_true",
        help="Calculate the number of original neighbors. Default: False",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--bond-orientation-order",
        action="store_true",
        help="Calculate the bond orientational order. Default: False",
        default=False,
    )
    parser.add_argument(
        "-i",
        "--inversion-symmetry",
        action="store_true",
        help="Calculate the local inversion symmetry. Default: False",
        default=False,
    )
    parser.add_argument(
        "-p",
        "--pressure-tensor",
        action="store_true",
        help="Calculate the pressure tensor (shear stress). Default: False",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--displacement",
        action="store_true",
        help="Calculate the displacement of the particles. Default: False",
        default=False,
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.substep is None:
        substep = args.step
    else:
        substep = args.substep

    analyse_shear = AnalyseShear(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force=args.force,
        force_rel=args.force_rel,
        start=args.start,
        step=args.step,
        substep=substep,
        stop=args.stop,
        njobs=args.njobs,
        use_cell_lists=not args.nocell,
        calc_neighbors=args.neighbors,
        calc_neighbors_dyn=args.neighbors_dynamic,
        calc_neighbors_orig=args.neighbors_original,
        calc_boo=args.bond_orientation_order,
        calc_invsym=args.inversion_symmetry,
        calc_pressure=args.pressure_tensor,
        calc_displacement=args.displacement,
    )
    analyse_shear.run()
