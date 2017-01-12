import numpy as np
import argparse
import os
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


    def calc_parameters(self):
        for shear in np.arange(self.start, self.stop + self.step, self.step):
            dname = "shear_{}".format(shear)
            workspace_dir = os.getcwd()
            kwargs = dict(force=self.force, existing_only=False,
                          jammed_packings_dir=dname, prefix="{}/explore_bv_".format(dname))

            # Calculate neighbours (coordination number)
            if(self.calc_neighbours):
                neighbours_kwargs = dict(kwargs, cutoff=1.)
                neighbours = Neighbours(workspace_dir, **neighbours_kwargs)
                neighbours.run()

            # Calculate lasting neighbours
            if(self.calc_neighbours_dyn):
                neighbours_dyn_kwargs = dict(kwargs, cutoff=1., analysis_fname="neighbours_dyn")
                if shear == self.start:
                    neighbours_dyn = Neighbours(workspace_dir, **neighbours_dyn_kwargs)
                    neighbours_dyn.run()
                else:
                    for subshear in np.arange(shear - self.step, shear, self.substep):
                        restrict_prefix = "shear_{}/explore_bv_".format(subshear)
                        subshear_dname = "shear_{}".format(subshear + self.substep)
                        subshear_prefix = "{}/explore_bv_".format(subshear_dname)
                        neighbours_dyn_kwargs.update(jammed_packings_dir=subshear_dname,
                                                     prefix=subshear_prefix,
                                                     restrict_neighbours=restrict_prefix)
                        neighbours_dyn = Neighbours(workspace_dir, **neighbours_dyn_kwargs)
                        neighbours_dyn.run()

            # Calculate bond orientational order
            if(self.calc_boo):
                boo_kwargs = dict(kwargs, solid_angle_weighted=False)
                boo = BondOrientationalOrder(workspace_dir, **boo_kwargs)
                boo.run_all()

            # Calculate local inversion symmetry
            if(self.calc_invsym):
                invsym_kwargs = dict(kwargs)
                invsym = InversionSymmetry(workspace_dir, **invsym_kwargs)
                invsym.run()

            # Calculate pressure tensor
            if(self.calc_pressure):
                pressure_kwargs = dict(kwargs, opt_pot_str='hs_wca')
                pressure = PressureTensor(workspace_dir, **pressure_kwargs)
                pressure.run()

            # Calculate movement away from start packing
            if(self.calc_movement):
                movement_kwargs = dict(kwargs)
                del movement_kwargs['jammed_packings_dir']
                movement = Movement(workspace_dir, "shear_{}".format(self.start), dname, **movement_kwargs)
                movement.run()

    def collect_parameters(self):
        pass


    def run(self):
        self.calc_parameters()
        self.collect_parameters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse a shearing process.")
    parser.add_argument("--input_dir", type=str, help="Directory containing the "
                        "shearing process. Default: Current directory", default='.')
    parser.add_argument("--output_dir", type=str, help="Directory for saving the analyses. "
                        "Default: 'shear_analysis'", default='shear_analysis')
    parser.add_argument("--force", action='store_true', help="Force to run on all packings.",
                        default=False)
    parser.add_argument("--start", type=float, help="Lowest shear to analyse. Default: 0.0", default=0.)
    parser.add_argument("--step", type=float, help="Shear step size for structural properties. "
                        "Default: 0.01", default=0.01)
    parser.add_argument("--substep", type=float, help="Actual shear step size. Default: 0.001",
                        default=0.001)
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

    analyse_shear = AnalyseShear(input_dir=args.input_dir, output_dir=args.output_dir, force=args.force,
                                 start=args.start, step=args.step, substep=args.substep, stop=args.stop,
                                 calc_neighbours=args.neighbours, calc_neighbours_dyn=args.neighbours_dynamic,
                                 calc_boo=args.bond_orientation_order, calc_invsym=args.inversion_symmetry,
                                 calc_pressure=args.pressure_tensor, calc_movement=args.movement)
    analyse_shear.run()
