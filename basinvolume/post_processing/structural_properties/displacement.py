from __future__ import division
import ConfigParser
import os
import ast
import traceback
import numpy as np
import argparse
import logging
from basinvolume.utils import trymakedir, calc_distance
from _structural_analysis import StructuralAnalysis


class Displacement(StructuralAnalysis):
    def __init__(self, workspace, packings_old, jammed_packings_dir,
                 analysis_dir='analysis', force=False, existing_only=True,
                 prefix='explore_bv_', verbose=True, shear=None, sub_centre_mass=True,
                 use_cell_lists=True):
        super(Displacement, self).__init__(workspace, jammed_packings_dir=jammed_packings_dir,
                                                analysis_dir=analysis_dir, force=force,
                                                existing_only=existing_only, prefix=prefix,
                                                verbose=verbose, use_cell_lists=use_cell_lists)
        if not os.path.isabs(packings_old):
            packings_old = os.path.join(self.workspace, packings_old)
        self.packings_old = packings_old
        self.shear = shear
        self.sub_centre_mass = sub_centre_mass

    @staticmethod
    def read(displacement_fname):
        configf = ConfigParser.ConfigParser()
        configf.read(displacement_fname)
        disp_dict = {}
        disp_dict['avg_abs_displacement_norm'] \
            = configf.getfloat('DISPLACEMENT', 'avg_abs_displacement_norm')
        disp_dict['avg_abs_displacement'] \
            = ast.literal_eval(configf.get('DISPLACEMENT', 'avg_abs_displacement'))
        disp_dict['avg_displacement'] \
            = ast.literal_eval(configf.get('DISPLACEMENT', 'avg_displacement'))
        disp_dict['displacements'] \
            = ast.literal_eval(configf.get('DISPLACEMENT', 'displacements'))
        if configf.has_section('NONAFFINE_DISPLACEMENT'):
            disp_dict['avg_abs_nonaff_displacement_norm'] \
                = configf.getfloat('NONAFFINE_DISPLACEMENT', 'avg_abs_displacement_norm')
            disp_dict['avg_abs_nonaff_displacement'] \
                = ast.literal_eval(configf.get('NONAFFINE_DISPLACEMENT', 'avg_abs_displacement'))
            disp_dict['avg_nonaff_displacement'] \
                = ast.literal_eval(configf.get('NONAFFINE_DISPLACEMENT', 'avg_displacement'))
            disp_dict['nonaff_displacements'] \
                = ast.literal_eval(configf.get('NONAFFINE_DISPLACEMENT', 'displacements'))
        return disp_dict

    def _averages(self, displacements):
        avg_displacement = [np.mean(displacements_1d)
                            for displacements_1d in zip(*displacements)]
        # This should be zero when the centre of mass is subtracted
        if self.sub_centre_mass:
            check_avg_displacement = list(avg_displacement)
            if self.distance_method == 'lees-edwards':
                del check_avg_displacement[1]
            assert np.linalg.norm(check_avg_displacement) < 10**(-12),\
                "The mean displacement should be zero when accounting for "\
                "a shifted centre of mass. It isn't: {}".format(avg_displacement)

        avg_abs_displacement = [np.mean(np.abs(displacements_1d))
                                for displacements_1d in zip(*displacements)]
        avg_abs_displacement_norm = np.linalg.norm(avg_abs_displacement)
        return avg_displacement, avg_abs_displacement, avg_abs_displacement_norm

    def run(self):
        for fname in os.listdir(self.jammed_packings_dir):
            if 'xyzdr' in fname or 'xydr' in fname:
                dname = os.path.splitext(fname)[0]

                # Get configuration
                configpath = os.path.join(self.jammed_packings_dir, dname + '.config')
                self._import_packing_config_file(configpath)

                # Check if the work directory exists
                base_directory_path = os.path.join(self.workspace, self.prefix + str(dname))
                if os.path.isdir(base_directory_path) or not self.existing_only:
                    trymakedir(base_directory_path)

                    # Check if this packing has already been analysed
                    analysis_dir_path = os.path.join(base_directory_path, self.analysis_dir)
                    displacement_fname = os.path.join(analysis_dir_path,'displacement')
                    compute = False
                    try:
                        self.read(displacement_fname)
                    except Exception:
                        compute = True

                    if compute or self.force:
                        if self.verbose:
                            logging.info("Calculating displacements: {}"
                                         .format(self.prefix + str(dname)))
                        trymakedir(analysis_dir_path)

                        # Read coordinates
                        path_new = os.path.join(self.jammed_packings_dir, fname)
                        coords_new, _, _, _ = self._import_packing_configuration(path_new)
                        path_old = os.path.join(self.packings_old, fname)
                        coords_old, _, _, _ = self._import_packing_configuration(path_old)

                        # Calculate displacements
                        # The displacement is measured with the boundary conditions of the new packing
                        displacements = []
                        if self.shear is not None:
                            nonaff_displacements = []
                        for i in xrange(self.nparticles):
                            displacement = calc_distance(
                                coords_new[i * self.bdim : (i + 1) * self.bdim],
                                coords_old[i * self.bdim : (i + 1) * self.bdim],
                                self.bdim, self.distance_method, self.boxv, self.pot_kwargs)
                            displacements.append(displacement)
                            if self.shear is not None:
                                nonaff_displacement = displacement.copy()
                                nonaff_displacement[0] -= coords_old[i * self.bdim + 1] * self.shear
                                nonaff_displacements.append(nonaff_displacement)

                        # Subtract centre of mass displacement
                        if self.sub_centre_mass:
                            centre_of_mass_displacement = np.mean(displacements, 0)
                            # No freedom in y-direction for Lees-Edwards
                            if self.distance_method == 'lees-edwards':
                                centre_of_mass_displacement[1] = 0
                            displacements -= centre_of_mass_displacement
                            if self.shear is not None:
                                nonaff_displacements -= centre_of_mass_displacement
                                # Centre of mass displacement by affine shear component
                                affine_com_displacement = np.mean(coords_old[1::self.bdim] * self.shear)
                                for i in xrange(self.nparticles):
                                    nonaff_displacements[i][0] += affine_com_displacement

                        # Calculate averages
                        (avg_displacement, avg_abs_displacement,
                         avg_abs_displacement_norm) = self._averages(displacements)
                        if self.shear is not None:
                            (avg_nonaff_displacement, avg_abs_nonaff_displacement,
                             avg_abs_nonaff_displacement_norm) = self._averages(nonaff_displacements)

                        # Output displacements to file
                        with open(displacement_fname, 'w') as f:
                            f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
                            f.write('[DISPLACEMENT]\n')
                            f.write('avg_abs_displacement_norm: {}\n'
                                    .format(avg_abs_displacement_norm))
                            f.write('avg_abs_displacement: {}\n'
                                    .format(avg_abs_displacement))
                            f.write('avg_displacement: {}\n'
                                    .format(avg_displacement))
                            f.write('displacements: {}\n'
                                    .format([disp.tolist() for disp in displacements]))
                            if self.shear is not None:
                                f.write('[NONAFFINE_DISPLACEMENT]\n')
                                f.write('avg_abs_displacement_norm: {}\n'
                                        .format(avg_abs_nonaff_displacement_norm))
                                f.write('avg_abs_displacement: {}\n'
                                        .format(avg_abs_nonaff_displacement))
                                f.write('avg_displacement: {}\n'
                                        .format(avg_nonaff_displacement))
                                f.write('displacements: {}\n'
                                        .format([disp.tolist() for disp in nonaff_displacements]))


def worker_disp(workspace, kwargs):
    try:
        disp = Displacement(workspace, **kwargs)
        disp.run()
    except:
        logging.error('worker_disp worker: %s' % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compares the particle positions "
                                     "in different sets of jammed packings.")
    parser.add_argument("packings-old", type=str, help="Directory containing the "
                        "jammed packings with the old particle positions.")
    parser.add_argument("packings-new", type=str, help="Directory containing the "
                        "changed jammed packings.")
    parser.add_argument("-d", "--workspace-dir", type=str, help="Top-level dir containing "
                        "the packings, e.g. 'n32_phi88_2D'.")
    parser.add_argument("--force", action='store_true', help="Force to run on all packings.",
                        default=False)
    parser.add_argument("--nonex", action='store_false', help="Run also for packings "
                        "for which there are no work folders ('explore_bv_[...]', "
                        "created e.g. by parallel tempering).", default=True)
    parser.add_argument("--prefix", type=str, help="Prefix for the work directory. "
                        "Default: 'explore_bv_'", default='explore_bv_')
    parser.add_argument("--drift", action='store_true', help="Don't subtract the "
                        "centre of mass displacement.", default=False)
    parser.add_argument("--shear", type=float, help="Difference in shear between "
                        "the two packings. Setting this triggers the additional "
                        "calculation of non-affine displacements.", default=None)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set up arguments
    kwargs = dict(packings_old=args.packings_old, jammed_packings_dir=args.packings_new,
                  force=args.force, existing_only=args.nonex, prefix=args.prefix,
                  shear=args.shear, sub_centre_mass=not args.drift)

    # Create workspace directory name
    if not args.workspace_dir:
        workspace_dir = os.getcwd()
    else:
        workspace_dir = os.path.abspath(args.workspace_dir)

    # Start displacement computation
    displacement = Displacement(workspace_dir, **kwargs)
    displacement.run()
