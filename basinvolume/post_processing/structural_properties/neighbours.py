from __future__ import division
import ConfigParser
import os
import ast
import traceback
import numpy as np
import argparse
import logging
from basinvolume.utils import trymakedir, find_neighbours
from _structural_analysis import StructuralAnalysis


class Neighbours(StructuralAnalysis):

    def __init__(self, workspace, jammed_packings_dir='jammed_packings',
                 analysis_dir='analysis', analysis_fname='neighbours', force=False,
                 existing_only=True, prefix='explore_bv_', verbose=True,
                 restrict_neighbours=None, cutoff=1.):
        super(Neighbours, self).__init__(workspace, jammed_packings_dir=jammed_packings_dir,
                                                analysis_dir=analysis_dir, force=force,
                                                existing_only=existing_only, prefix=prefix,
                                                verbose=verbose)
        self.cutoff = cutoff
        self.restrict_neighbours = restrict_neighbours
        self.analysis_fname = analysis_fname

    @staticmethod
    def read(neighbours_fname):
        configf = ConfigParser.ConfigParser()
        configf.read(neighbours_fname)
        neighbours_dict = {}
        neighbours_dict['avg_neighbours'] = \
            configf.getfloat('NEIGHBOURS', 'avg_neighbours')
        neighbours_dict['neighbour_counts'] = \
            ast.literal_eval(configf.get('NEIGHBOURS', 'neighbour_counts'))
        neighbours_dict['neighbour_lists'] = \
            ast.literal_eval(configf.get('NEIGHBOURS', 'neighbour_lists'))
        return neighbours_dict

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
                    neighbours_fname = os.path.join(analysis_dir_path, self.analysis_fname)
                    compute = False
                    try:
                        self.read(neighbours_fname)
                    except Exception:
                        compute = True

                    if compute or self.force:
                        if self.verbose:
                            if self.restrict_neighbours is None:
                                logging.info("Calculating neighbours: {}"
                                             .format(self.prefix + str(dname)))
                            else:
                                logging.info("Calculating restricted neighbours: {}"
                                             .format(self.prefix + str(dname)))
                        trymakedir(analysis_dir_path)

                        # Read coordinates and compute neighbours
                        self.coords, _, self.ss_radii, _ = self._import_packing_configuration(fname)
                        _, neighbour_lists = find_neighbours(self.coords, self.ss_radii, self.bdim,
                                                              self.boxv, self.distance_method,
                                                              self.pot_kwargs, cutoff_factor=self.cutoff)

                        # Filter neighbours
                        if self.restrict_neighbours is not None:
                            # Get conditional neighbour lists
                            base_restrict_path = os.path.join(self.workspace,
                                                              self.restrict_neighbours + str(dname))
                            restrict_dir = os.path.join(base_restrict_path, self.analysis_dir)
                            restrict_path = os.path.join(restrict_dir, self.analysis_fname)
                            if not os.path.isfile(restrict_path):
                                raise IOError("The restrict neighbours file {} does "
                                              "not exist.".format(restrict_path))
                            configf = ConfigParser.ConfigParser()
                            configf.read(restrict_path)
                            restrict_neighbour_lists = ast.literal_eval(configf.get('NEIGHBOURS',
                                                                                    'neighbour_lists'))

                            neighbour_lists = [filter(lambda particle: particle in
                                                      restrict_neighbour_lists[i], neighbour_lists[i])
                                               for i in range(self.nparticles)]

                        # Output neighbour lists to file
                        with open(neighbours_fname, 'w') as f:
                            f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
                            f.write('[NEIGHBOURS]\n')
                            f.write('avg_neighbours: {}\n'.format(np.mean([len(neighbours) for neighbours
                                                                     in neighbour_lists])))
                            f.write('neighbour_counts: {}\n'.format([len(neighbours) for neighbours
                                                                     in neighbour_lists]))
                            f.write('neighbour_lists: {}\n'.format(neighbour_lists))


def worker_neighbours(workspace, kwargs):
    try:
        neighbours = Neighbours(workspace, **kwargs)
        neighbours.run()
    except:
        logging.error('worker_neighbours worker: %s' % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute neighbour lists for jammed packings.")
    parser.add_argument("-d", "--workspace-dir", type=str, help="Top-level dir containing "
                        "the packings, e.g. 'n32_phi88_2D'.")
    parser.add_argument("--force", action='store_true', help="Force to run on all packings.",
                        default=False)
    parser.add_argument("--nonex", action='store_false', help="Run also for packings "
                        "for which there are no work folders ('explore_bv_[...]', "
                        "created e.g. by parallel tempering).", default=True)
    parser.add_argument("--prefix", type=str, help="Prefix for the work directory. "
                        "Default: 'explore_bv_'", default='explore_bv_')
    parser.add_argument("--input-dir", type=str, help="Directory containing the "
                        "jammed packings. Default: 'jammed_packings'", default='jammed_packings')
    parser.add_argument("--restrict-neighbours", type=str, help="Prefix leading to "
                        "a neighbour lists file. This string is analogous to the normal prefix. "
                        "Only neighbours in these lists are considered.",
                        default=None)
    parser.add_argument("--cutoff", type=float, help="Multiple of particle radii "
                        "defining the maximum neighbour distance. Default: 1", default=1.)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set up arguments
    kwargs = dict(force=args.force, existing_only=args.nonex,
                  jammed_packings_dir=args.input_dir, prefix=args.prefix, cutoff=args.cutoff,
                  restrict_neighbours=args.restrict_neighbours)

    # Create workspace directory name
    if not args.workspace_dir:
        workspace_dir = os.getcwd()
    else:
        workspace_dir = os.path.abspath(args.workspace_dir)

    # Start neighbours computation
    neighbours = Neighbours(workspace_dir, **kwargs)
    neighbours.run()
