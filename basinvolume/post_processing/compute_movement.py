import ConfigParser
import os
import ast
import numpy as np
import argparse
from basinvolume.utils import trymakedir, read_xydr, read_xyzdr, calc_distance


class Movement:
    def __init__(self, workspace, packings_orig, packings_new,
                 analysis_dir='analysis', force=False, existing_only=True,
                 prefix='explore_bv_'):
        if not os.path.isabs(workspace):
            workspace = os.path.abspath(workspace)
        self.workspace = workspace
        if not os.path.isabs(packings_orig):
            packings_orig = os.path.join(self.workspace, packings_orig)
        if not os.path.isabs(packings_new):
            packings_new = os.path.join(self.workspace, packings_new)
        self.packings_orig = packings_orig
        self.packings_new = packings_new
        self.analysis_dir = analysis_dir
        self.eps = 1.
        self.frozen = False
        self.force = force
        self.existing_only = existing_only
        self.prefix = prefix


    def _import_packing_config_file(self, configpath):
        configf = ConfigParser.ConfigParser()
        configf.read(str(configpath))
        self.nparticles = configf.getint('JAMMED_PACKING','nparticles')
        self.bdim = configf.getint('JAMMED_PACKING','boxdim')
        assert self.bdim==2 or self.bdim==3, "bdim={} not implemented".format(self.bdim)
        self.ndof = self.nparticles * self.bdim
        boxv = configf.get('JAMMED_PACKING','boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        if self.frozen:
            self.vcavity = configf.getfloat('JAMMED_PACKING', 'vcavity')
        else:
            self.vcavity = np.prod(self.boxv)
        self.packing_frac = configf.getfloat('JAMMED_PACKING','packing_fraction')
        self.sca = configf.getfloat('JAMMED_PACKING','sca')
        self.distance_method = configf.get('JAMMED_PACKING', 'distance_method')
        if hasattr(self, 'pot_kwargs') and self.pot_kwargs is not None:
            self.pot_kwargs.update(ast.literal_eval(configf.get('JAMMED_PACKING', 'pot_kwargs')))
        else:
            self.pot_kwargs = ast.literal_eval(configf.get('JAMMED_PACKING', 'pot_kwargs'))


    def _import_packing_configuration(self, path):
        if self.bdim == 2:
            coords, hs_diameters, rattlers = read_xydr(path)
        elif self.bdim == 3:
            coords, hs_diameters, rattlers = read_xyzdr(path)
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        hs_radii = hs_diameters/2
        ss_radii = hs_radii * (1 + self.sca)
        return coords, hs_radii, ss_radii, rattlers


    def _get_dname(self, dname):
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        return dname


    def run(self):
        for fname in os.listdir(self.packings_new):
            if 'xyzdr' in fname or 'xydr' in fname:
                dname = self._get_dname(fname)

                # Get configuration
                configpath = os.path.join(self.packings_new, dname + '.config')
                self._import_packing_config_file(configpath)

                # Check if the work directory exists
                base_directory_path = os.path.join(self.workspace, self.prefix + str(dname))
                if os.path.isdir(base_directory_path) or not self.existing_only:
                    trymakedir(base_directory_path)

                    # Check if this packing has already been analysed
                    analysis_dir_path = os.path.join(base_directory_path, self.analysis_dir)
                    movements_fname = os.path.join(analysis_dir_path,'movements')
                    compute = False
                    try:
                        configf = ConfigParser.ConfigParser()
                        configf.read(movements_fname)
                        ast.literal_eval(configf.getfloat('MOVEMENTS', 'avg_distance'))
                        ast.literal_eval(configf.get('MOVEMENTS', 'distances'))
                    except Exception:
                        compute = True

                    if compute or self.force:
                        print("Calculating movements: {}".format(self.prefix + str(dname)))
                        trymakedir(analysis_dir_path)

                        # Read coordinates
                        path_new = os.path.join(self.packings_new, fname)
                        coords_new, _, _, _ = self._import_packing_configuration(path_new)
                        path_orig = os.path.join(self.packings_orig, fname)
                        coords_orig, _, _, _ = self._import_packing_configuration(path_orig)

                        # Calculate distances
                        # The distance is measure with the boundary conditions of the new packing
                        distances = []
                        for i in xrange(self.nparticles):
                            distance = calc_distance(
                                coords_new[i * self.bdim : (i + 1) * self.bdim],
                                coords_orig[i * self.bdim : (i + 1) * self.bdim],
                                self.bdim, self.distance_method, self.boxv, self.pot_kwargs)
                            distances.append(np.linalg.norm(distance))
                        avg_distance = np.mean(distances)

                        # Output distances to file
                        with open(movements_fname, 'w') as f:
                            f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
                            f.write('[MOVEMENTS]\n')
                            f.write('avg_distance: {}\n'.format(avg_distance))
                            f.write('distances: {}\n'.format(distances))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compares the particle positions "
                                     "in different sets of jammed packings.")
    parser.add_argument("packings_orig", type=str, help="Directory containing the "
                        "jammed packings with the original particle positions.")
    parser.add_argument("packings_new", type=str, help="Directory containing the "
                        "changed jammed packings.")
    parser.add_argument("-d", "--workspace_dir", type=str, help="Top-level dir containing "
                        "the packings, e.g. 'n32_phi88_2D'.")
    parser.add_argument("--force", action='store_true', help="Force to run on all packings.",
                        default=False)
    parser.add_argument("--nonex", action='store_false', help="Run also for packings "
                        "for which there are no work folders ('explore_bv_[...]', "
                        "created e.g. by parallel tempering).", default=True)
    parser.add_argument("--prefix", type=str, help="Prefix for the work directory. "
                        "Default: 'explore_bv_'", default='explore_bv_')
    args = parser.parse_args()

    # Set up arguments
    kwargs = dict(packings_orig=args.packings_orig, packings_new=args.packings_new,
                  force=args.force, existing_only=args.nonex, prefix=args.prefix)

    # Create workspace directory name
    if not args.workspace_dir:
        workspace_dir = os.getcwd()
    else:
        workspace_dir = os.path.abspath(args.workspace_dir)

    # Start movement computation
    movement = Movement(workspace_dir, **kwargs)
    movement.run()
