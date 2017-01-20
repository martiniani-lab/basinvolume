import numpy as np
import argparse
import os
import sys
import shutil
import pandas as pd
import multiprocessing as mp
from basinvolume.post_processing.structural_properties \
    import BondOrientationalOrder,  PressureTensor, Neighbours, InversionSymmetry, \
           Displacement, worker_boo, worker_disp, worker_invsym, worker_neighbours, \
           worker_pressure


class AnalyseShear:
    def __init__(self, input_dir=".", output_dir="shear_analysis", force=False,
                 start=0., step=0.01, substep=0.001, stop=1., njobs=1,
                 calc_neighbours=False, calc_neighbours_dyn=False, calc_boo=False,
                 calc_invsym=False, calc_pressure=False, calc_displacement=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.force = force
        self.start = start
        self.step = step
        self.substep = substep
        self.stop = stop
        self.njobs = njobs
        self.calc_neighbours = calc_neighbours
        self.calc_neighbours_dyn = calc_neighbours_dyn
        self.calc_boo = calc_boo
        self.calc_invsym = calc_invsym
        self.calc_pressure = calc_pressure
        self.calc_displacement = calc_displacement

    def run(self):
        self.make_output_dirs()
        if self.njobs > 1:
            self.mypool = mp.Pool(self.njobs)
        for shear in np.arange(self.start, self.stop + 0.5 * self.step, self.step):
            shear_dir = "shear_{}".format(shear)
            if not os.path.isdir(os.path.join(self.input_dir, shear_dir)):
                print("The shear directory {} does not exist. Stopping analysis."
                      .format(shear_dir))
                sys.exit(1)
            self.calc_parameters(shear, shear_dir)
            self.collect_files(shear, shear_dir)
        if self.njobs > 1:
            self.mypool.terminate()
            self.mypool.join()
        self.collect_parameters()

    def make_output_dirs(self):
        # Create output dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # Get parameter directories to create
        all_param_dirs = ["neighbours", "neighbours_dyn", "boo", "inversion_symmetry",
                          "pressure_tensor", "displacement"]
        params = [self.calc_neighbours, self.calc_neighbours_dyn, self.calc_boo,
                  self.calc_invsym, self.calc_pressure, self.calc_displacement]
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

        if os.path.isabs(self.input_dir):
            workspace_dir = self.input_dir
        else:
            workspace_dir = os.path.abspath(self.input_dir)
        kwargs = dict(verbose=False, force=self.force, existing_only=False,
                      jammed_packings_dir=input_relpath,
                      prefix=os.path.join(input_relpath, "explore_bv_"))
        structural_props = []

        # Bond orientational order
        if self.calc_boo:
            boo_kwargs = dict(kwargs, solid_angle_weighted=False)
            structural_props.append((worker_boo, boo_kwargs))

        # Displacement from previous packing
        if self.calc_displacement:
            disp_kwargs = dict(kwargs)
            if shear == self.start:
                disp_kwargs['shear'] = 0.
                disp_kwargs['packings_old'] = input_relpath
            else:
                disp_kwargs['shear'] = self.step
                disp_kwargs['packings_old'] = "shear_{}".format(shear - self.step)
            structural_props.append((worker_disp, disp_kwargs))

        # Local inversion symmetry
        if self.calc_invsym:
            invsym_kwargs = dict(kwargs)
            structural_props.append((worker_invsym, invsym_kwargs))

        # Neighbours (coordination number)
        if self.calc_neighbours:
            neighbours_kwargs = dict(kwargs, cutoff=1.)
            structural_props.append((worker_neighbours, neighbours_kwargs))

        # Lasting neighbours
        if self.calc_neighbours_dyn:
            neighbours_dyn_kwargs = dict(kwargs, cutoff=1., analysis_fname="neighbours_dyn")
            if shear == self.start:
                worker_neighbours(workspace_dir, neighbours_dyn_kwargs)
            else:
                for subshear in np.arange(shear - self.step, shear - 0.5 * self.substep, self.substep):
                    restrict_prefix = os.path.join("shear_{}".format(subshear), "explore_bv_")
                    subshear_dname = "shear_{}".format(subshear + self.substep)
                    subshear_prefix = os.path.join(subshear_dname, "explore_bv_")
                    neighbours_dyn_kwargs.update(jammed_packings_dir=subshear_dname,
                                                 prefix=subshear_prefix,
                                                 restrict_neighbours=restrict_prefix)
                    worker_neighbours(workspace_dir, neighbours_dyn_kwargs)

        # Pressure tensor
        if self.calc_pressure:
            pressure_kwargs = dict(kwargs, opt_pot_str='hs_wca')
            structural_props.append((worker_pressure, pressure_kwargs))

        # Start parallel calculations
        if self.njobs > 1:
            for prop in structural_props:
                self.mypool.apply_async(prop[0], args=(workspace_dir, prop[1],))
        else:
            for prop in structural_props:
                prop[0](workspace_dir, prop[1])

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
                              ("pressure_data", "pressure_tensor"), ("displacement", "displacement")]
        params_choice = [self.calc_neighbours, self.calc_neighbours_dyn, self.calc_boo,
                         self.calc_invsym, self.calc_pressure, self.calc_displacement]
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
        for path in output_paths:
            data = pd.DataFrame()
            data.index.name = "Shear"

            # Bond orientational order
            if self.calc_boo:
                boo_path = os.path.join(path, "boo")
                for shear_file in os.listdir(boo_path):
                    boo_dict = BondOrientationalOrder.read(os.path.join(boo_path,
                                                                        shear_file))
                    boo_entry = pd.Series()
                    boo_entry.name = float(shear_file.split('_')[1])
                    boo_entry['Bond-orientational order {}'
                              .format(boo_dict['BOO'][0])] = boo_dict['BOO'][1]
                    data = data.append(boo_entry)

            # Displacement from previous packing
            if self.calc_displacement:
                displacement_path = os.path.join(path, "displacement")
                for shear_file in os.listdir(displacement_path):
                    displ_dict = Displacement.read(os.path.join(displacement_path,
                                                                shear_file))
                    displacement_entry = pd.Series()
                    displacement_entry.name = float(shear_file.split('_')[1])
                    displacement_entry['Average absolute displacement'] \
                        = displ_dict['avg_abs_displacement_norm']
                    displacement_entry['Average absolute non-affine displacement'] \
                        = displ_dict['avg_abs_nonaff_displacement_norm']
                    dims = ['x', 'y', 'z']
                    for i in xrange(len(displ_dict['avg_displacement'])):
                        displacement_entry['Average absolute displacement {}'.format(dims[i])] \
                            = displ_dict['avg_abs_displacement'][i]
                        displacement_entry['Average absolute non-affine displacement {}'
                                           .format(dims[i])] \
                            = displ_dict['avg_abs_nonaff_displacement'][i]
                    data = data.append(displacement_entry)

            # Local inversion symmetry
            if self.calc_invsym:
                invsym_path = os.path.join(path, "inversion_symmetry")
                for shear_file in os.listdir(invsym_path):
                    invsym_dict = InversionSymmetry.read(os.path.join(invsym_path,
                                                                      shear_file))
                    invsym_entry = pd.Series()
                    invsym_entry.name = float(shear_file.split('_')[1])
                    invsym_entry['Local inversion symmetry'] = invsym_dict['inversion_symmetry']
                    data = data.append(invsym_entry)

            # Neighbours (coordination number)
            if self.calc_neighbours:
                neighbours_path = os.path.join(path, "neighbours")
                for shear_file in os.listdir(neighbours_path):
                    neighbours_dict = Neighbours.read(os.path.join(neighbours_path,
                                                                   shear_file))
                    neighbours_entry = pd.Series()
                    neighbours_entry.name = float(shear_file.split('_')[1])
                    neighbours_entry['Average neighbours'] = \
                        neighbours_dict['avg_neighbours']
                    data = data.append(neighbours_entry)

            # Lasting neighbours
            if self.calc_neighbours_dyn:
                neighbours_dyn_path = os.path.join(path, "neighbours_dyn")
                for shear_file in os.listdir(neighbours_dyn_path):
                    neighbours_dyn_dict = Neighbours.read(os.path.join(neighbours_dyn_path,
                                                                       shear_file))
                    neighbours_dyn_entry = pd.Series()
                    neighbours_dyn_entry.name = float(shear_file.split('_')[1])
                    neighbours_dyn_entry['Average lasting neighbours'] = \
                        neighbours_dyn_dict['avg_neighbours']
                    data = data.append(neighbours_dyn_entry)

            # Pressure tensor
            if self.calc_pressure:
                pressure_path = os.path.join(path, "pressure_tensor")
                for shear_file in os.listdir(pressure_path):
                    pressure_dict = PressureTensor.read(os.path.join(pressure_path,
                                                                     shear_file))
                    pressure_entry = pd.Series()
                    pressure_entry.name = float(shear_file.split('_')[1])
                    pressure_entry['Energy'] = pressure_dict['E']
                    pressure_entry['Pressure'] = pressure_dict['P']
                    pressure_entry['Shear stress'] = pressure_dict['maxshear_xyplane']
                    if len(pressure_dict['Ptensor']) == 4:
                        pressure_entry['Shear stress xy'] = pressure_dict['Ptensor'][1]
                    elif len(pressure_dict['Ptensor']) == 9:
                        pressure_entry['Shear stress xy'] = pressure_dict['Ptensor'][1]
                        pressure_entry['Shear stress xz'] = pressure_dict['Ptensor'][2]
                        pressure_entry['Shear stress yz'] = pressure_dict['Ptensor'][5]
                    else:
                        raise NotImplementedError
                    data = data.append(pressure_entry)

            # Merge on indices
            data = data.groupby(data.index).sum()

            # Save data to csv
            data.to_csv(os.path.join(path, "data.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a shearing process.")
    parser.add_argument("--input_dir", type=str, help="Directory containing the "
                        "shearing process. Default: Current directory", default=".")
    parser.add_argument("--output_dir", type=str, help="Directory for saving the analyses. "
                        "Default: 'shear_analysis'", default='shear_analysis')
    parser.add_argument("--force", action='store_true', help="Force to run on all packings.",
                        default=False)
    parser.add_argument("--start", type=float, help="Lowest shear to analyse. Default: 0.0",
                        default=0.)
    parser.add_argument("--step", type=float, help="Shear step size for structural properties. "
                        "Default: 0.01", default=0.01)
    parser.add_argument("--substep", type=float, help="Actual shear step size. Default: step size",
                        default=None)
    parser.add_argument("--stop", type=float, help="Highest shear to analyse. Default: 1.0",
                        default=1.)
    parser.add_argument("-j", "--njobs", type=int, help="Number of jobs to run in parallel. "
                        "Default: 1 (serial)", default=1)
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
    parser.add_argument("-d", "--displacement", action='store_true',
                        help="Calculate the displacement of the particles. Default: False", default=False)
    args = parser.parse_args()

    if args.substep is None:
        substep = args.step
    else:
        substep = args.substep

    analyse_shear = AnalyseShear(input_dir=args.input_dir, output_dir=args.output_dir,
                                 force=args.force, start=args.start, step=args.step,
                                 substep=substep, stop=args.stop, njobs=args.njobs,
                                 calc_neighbours=args.neighbours,
                                 calc_neighbours_dyn=args.neighbours_dynamic,
                                 calc_boo=args.bond_orientation_order,
                                 calc_invsym=args.inversion_symmetry,
                                 calc_pressure=args.pressure_tensor,
                                 calc_displacement=args.displacement)
    analyse_shear.run()
