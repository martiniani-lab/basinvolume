import numpy as np
import argparse
import os
import sys
import shutil
import ConfigParser
import pandas as pd
from compute_neighbours import Neighbours
from compute_inversion_symmetry import InversionSymmetry
from compute_movement import Movement
from compute_structural_properties import BondOrientationalOrder, PressureTensor


class AnalyseShear:
    def __init__(self, input_dir=".", output_dir="shear_analysis", force=False,
                 start=0., step=0.01, substep=0.001, stop=1., calc_neighbours=False,
                 calc_neighbours_dyn=False, calc_boo=False, calc_invsym=False,
                 calc_pressure=False, calc_movement=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.force = force
        self.start = start
        self.step = step
        self.substep = substep
        self.stop = stop
        self.calc_neighbours = calc_neighbours
        self.calc_neighbours_dyn = calc_neighbours_dyn
        self.calc_boo = calc_boo
        self.calc_invsym = calc_invsym
        self.calc_pressure = calc_pressure
        self.calc_movement = calc_movement


    def make_output_dirs(self):
        # Create output dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # Get parameter directories to create
        all_param_dirs = ["neighbours", "neighbours_dyn", "boo", "inversion_symmetry",
                          "pressure_tensor", "movements"]
        params = [self.calc_neighbours, self.calc_neighbours_dyn, self.calc_boo,
                  self.calc_invsym, self.calc_pressure, self.calc_movement]
        param_dirs = [all_param_dirs[i] for i in range(len(params)) if params[i]]

        # Get packings to create
        input_files = os.listdir(os.path.join(self.input_dir, "shear_{}".format(self.start)))
        packing_files = filter(lambda pname: "jammed_packing" in pname
                               and ('xyzdr' in pname or 'xydr' in pname), input_files)
        packings = [packing.split('_')[1].split('.')[0] for packing in packing_files]

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
        print("Calculating parameters for shear={}".format(shear))
        if not os.path.isdir(os.path.join(self.input_dir, input_relpath)):
            print("The shear directory {} does not exist. Stopping analysis.".format(input_relpath))
            sys.exit(1)
        if os.path.isabs(self.input_dir):
            workspace_dir = self.input_dir
        else:
            workspace_dir = os.path.abspath(self.input_dir)
        kwargs = dict(verbose=False, force=self.force, existing_only=False,
                      jammed_packings_dir=input_relpath,
                      prefix=os.path.join(input_relpath, "explore_bv_"))

        # Calculate neighbours (coordination number)
        if self.calc_neighbours:
            neighbours_kwargs = dict(kwargs, cutoff=1.)
            neighbours = Neighbours(workspace_dir, **neighbours_kwargs)
            neighbours.run()

        # Calculate lasting neighbours
        if self.calc_neighbours_dyn:
            neighbours_dyn_kwargs = dict(kwargs, cutoff=1., analysis_fname="neighbours_dyn")
            if shear == self.start:
                neighbours_dyn = Neighbours(workspace_dir, **neighbours_dyn_kwargs)
                neighbours_dyn.run()
            else:
                for subshear in np.arange(shear - self.step, shear - 0.5 * self.substep, self.substep):
                    restrict_prefix = os.path.join("shear_{}".format(subshear), "explore_bv_")
                    subshear_dname = "shear_{}".format(subshear + self.substep)
                    subshear_prefix = os.path.join(subshear_dname, "explore_bv_")
                    neighbours_dyn_kwargs.update(jammed_packings_dir=subshear_dname,
                                                 prefix=subshear_prefix,
                                                 restrict_neighbours=restrict_prefix)
                    neighbours_dyn = Neighbours(workspace_dir, **neighbours_dyn_kwargs)
                    neighbours_dyn.run()

        # Calculate bond orientational order
        if self.calc_boo:
            boo_kwargs = dict(kwargs, solid_angle_weighted=False)
            boo = BondOrientationalOrder(workspace_dir, **boo_kwargs)
            boo.run_all()

        # Calculate local inversion symmetry
        if self.calc_invsym:
            invsym_kwargs = dict(kwargs)
            invsym = InversionSymmetry(workspace_dir, **invsym_kwargs)
            invsym.run()

        # Calculate pressure tensor
        if self.calc_pressure:
            pressure_kwargs = dict(kwargs, opt_pot_str='hs_wca')
            pressure = PressureTensor(workspace_dir, **pressure_kwargs)
            pressure.run()

        # Calculate movement away from start packing
        if self.calc_movement:
            movement_kwargs = dict(kwargs)
            del movement_kwargs['jammed_packings_dir']
            orig_path = "shear_{}".format(self.start)
            movement = Movement(workspace_dir, orig_path, input_relpath, **movement_kwargs)
            movement.run()


    def collect_files(self, shear, input_relpath):

        print("Collecting files for shear={}".format(shear))

        # Get the input directory paths
        input_files = os.listdir(os.path.join(self.input_dir, input_relpath))
        explore_dirs = filter(lambda expname: "explore_bv_jammed_packing" in expname
                              and os.path.isdir(os.path.join(self.input_dir, input_relpath, expname)),
                              input_files)
        analysis_paths = [os.path.join(self.input_dir, input_relpath, expdir, "analysis")
                          for expdir in explore_dirs]

        # Get the output directory paths
        output_files = os.listdir(self.output_dir)
        output_packings = filter(lambda pname: "packing" in pname
                                 and os.path.isdir(os.path.join(self.output_dir, pname)),
                                 output_files)
        output_paths = [os.path.join(self.output_dir, packing) for packing in output_packings]

        # Get parameter directory and file names
        params_from_to_all = [("neighbours", "neighbours"), ("neighbours_dyn", "neighbours_dyn"),
                              ("glob_boo", "boo"), ("inversion_symmetry", "inversion_symmetry"),
                              ("pressure_data", "pressure_tensor"), ("movements", "movements")]
        params_choice = [self.calc_neighbours, self.calc_neighbours_dyn, self.calc_boo,
                         self.calc_invsym, self.calc_pressure, self.calc_movement]
        params_from_to = [params_from_to_all[i] for i in range(len(params_choice)) if params_choice[i]]

        # Iterate over packings and parameters
        for analysis_path, output_path in zip(analysis_paths, output_paths):
            for param_from_to in params_from_to:
                if os.path.isfile(os.path.join(analysis_path, param_from_to[0])):
                    shutil.copyfile(os.path.join(analysis_path, param_from_to[0]),
                                    os.path.join(output_path, param_from_to[1],
                                                 "shear_{}".format(shear)))


    def collect_parameters(self):

        print("Collecting parameters")

        # Get the output directory paths
        output_files = os.listdir(self.output_dir)
        output_packings = filter(lambda pname: "packing" in pname
                                 and os.path.isdir(os.path.join(self.output_dir, pname)),
                                 output_files)
        output_paths = [os.path.join(self.output_dir, packing) for packing in output_packings]

        # Iterate over packings
        configf = ConfigParser.ConfigParser()
        for path in output_paths:
            data = pd.DataFrame()
            data.index.name = "Shear"

            # Neighbours (coordination number)
            if self.calc_neighbours:
                neighbours_path = os.path.join(path, "neighbours")
                for shear_file in os.listdir(neighbours_path):
                    neighbours_entry = pd.Series()
                    neighbours_entry.name = float(shear_file.split('_')[1])
                    configf.read(os.path.join(neighbours_path, shear_file))
                    neighbours_entry['Average neighbours'] = configf.getfloat("NEIGHBOURS",
                                                                              "avg_neighbours")
                    data = data.append(neighbours_entry)

            # Lasting neighbours
            if self.calc_neighbours_dyn:
                neighbours_dyn_path = os.path.join(path, "neighbours_dyn")
                for shear_file in os.listdir(neighbours_dyn_path):
                    neighbours_dyn_entry = pd.Series()
                    neighbours_dyn_entry.name = float(shear_file.split('_')[1])
                    configf.read(os.path.join(neighbours_dyn_path, shear_file))
                    neighbours_dyn_entry['Average lasting neighbours'] = configf.getfloat("NEIGHBOURS",
                                                                                          "avg_neighbours")
                    data = data.append(neighbours_dyn_entry)

            # Bond orientational order
            if self.calc_boo:
                boo_path = os.path.join(path, "boo")
                for shear_file in os.listdir(boo_path):
                    boo_entry = pd.Series()
                    boo_entry.name = float(shear_file.split('_')[1])
                    configf.read(os.path.join(boo_path, shear_file))
                    boo_config = configf.items("BOO")
                    for item in boo_config:
                        boo_entry["Bond-orientational order {}".format(item[0].upper())] = float(item[1])
                    data = data.append(boo_entry)

            # Local inversion symmetry
            if self.calc_invsym:
                invsym_path = os.path.join(path, "inversion_symmetry")
                for shear_file in os.listdir(invsym_path):
                    invsym_entry = pd.Series()
                    invsym_entry.name = float(shear_file.split('_')[1])
                    configf.read(os.path.join(invsym_path, shear_file))
                    invsym_entry['Local inversion symmetry'] = configf.getfloat("INVERSION_SYMMETRY",
                                                                                "inversion_symmetry")
                    data = data.append(invsym_entry)

            # Pressure tensor
            if self.calc_pressure:
                pressure_path = os.path.join(path, "pressure_tensor")
                for shear_file in os.listdir(pressure_path):
                    pressure_entry = pd.Series()
                    pressure_entry.name = float(shear_file.split('_')[1])
                    configf.read(os.path.join(pressure_path, shear_file))
                    pressure_entry['Energy'] = configf.getfloat("ENERGY", "E")
                    pressure_entry['Pressure'] = configf.getfloat("PRESSURE", "P")
                    pressure_entry['Shear stress'] = configf.getfloat("PRESSURE", "maxshear_xyplane")
                    p_tensor = [float(p) for p in configf.get("PRESSURE", "Ptensor").split(' ')]
                    if len(p_tensor) == 4:
                        pressure_entry['Shear stress xy'] = p_tensor[1]
                    elif len(p_tensor) == 9:
                        pressure_entry['Shear stress xy'] = p_tensor[1]
                        pressure_entry['Shear stress xz'] = p_tensor[2]
                        pressure_entry['Shear stress yz'] = p_tensor[5]
                    else:
                        raise NotImplementedError
                    data = data.append(pressure_entry)

            # Movement away from start packing
            if self.calc_movement:
                movement_path = os.path.join(path, "movements")
                for shear_file in os.listdir(movement_path):
                    movement_entry = pd.Series()
                    movement_entry.name = float(shear_file.split('_')[1])
                    configf.read(os.path.join(movement_path, shear_file))
                    movement_entry['Average movement'] = configf.getfloat("MOVEMENTS",
                                                                          "avg_distance")
                    data = data.append(movement_entry)

            # Merge on indices
            data = data.groupby(data.index).sum()

            # Save data to csv
            data.to_csv(os.path.join(path, "data.csv"))


    def run(self):
        self.make_output_dirs()
        for shear in np.arange(self.start, self.stop + 0.5 * self.step, self.step):
            input_dir = "shear_{}".format(shear)
            self.calc_parameters(shear, input_dir)
            self.collect_files(shear, input_dir)
        self.collect_parameters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a shearing process.")
    parser.add_argument("--input_dir", type=str, help="Directory containing the "
                        "shearing process. Default: Current directory", default=".")
    parser.add_argument("--output_dir", type=str, help="Directory for saving the analyses. "
                        "Default: 'shear_analysis'", default='shear_analysis')
    parser.add_argument("--force", action='store_true', help="Force to run on all packings.",
                        default=False)
    parser.add_argument("--start", type=float, help="Lowest shear to analyse. Default: 0.0", default=0.)
    parser.add_argument("--step", type=float, help="Shear step size for structural properties. "
                        "Default: 0.01", default=0.01)
    parser.add_argument("--substep", type=float, help="Actual shear step size. Default: step size",
                        default=None)
    parser.add_argument("--stop", type=float, help="Highest shear to analyse. Default: 1.0", default=1.)
    parser.add_argument("-z", "--neighbours", action='store_true',
                        help="Calculate the static coordination numbers. Default: False",
                        default=False)
    parser.add_argument("-zd", "--neighbours_dynamic", action='store_true',
                        help="Calculate the dynamic coordination numbers (counts "
                        "lasting neighbours). Default: False",
                        default=False)
    parser.add_argument("-b", "--bond_orientation_order", action='store_true',
                        help="Calculate the bond orientational order. Default: False", default=False)
    parser.add_argument("-i", "--inversion_symmetry", action='store_true',
                        help="Calculate the local inversion symmetry. Default: False", default=False)
    parser.add_argument("-p", "--pressure_tensor", action='store_true',
                        help="Calculate the pressure tensor (shear stress). Default: False",
                        default=False)
    parser.add_argument("-m", "--movement", action='store_true',
                        help="Calculate the movement of the particles. Default: False", default=False)
    args = parser.parse_args()

    if args.substep is None:
        substep = args.step
    else:
        substep = args.substep

    analyse_shear = AnalyseShear(input_dir=args.input_dir, output_dir=args.output_dir,
                                 force=args.force, start=args.start, step=args.step,
                                 substep=substep, stop=args.stop,
                                 calc_neighbours=args.neighbours,
                                 calc_neighbours_dyn=args.neighbours_dynamic,
                                 calc_boo=args.bond_orientation_order,
                                 calc_invsym=args.inversion_symmetry,
                                 calc_pressure=args.pressure_tensor,
                                 calc_movement=args.movement)
    analyse_shear.run()
