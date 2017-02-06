from __future__ import division
import ConfigParser
import os
import ast
import traceback
import numpy as np
import argparse
import logging
import cPickle
from basinvolume.utils import trymakedir
from _structural_analysis import StructuralAnalysis


class Neighbors(StructuralAnalysis):

    def __init__(self, workspace, jammed_packings_dir='jammed_packings',
                 analysis_dir='analysis', analysis_fname='neighbors', force=False,
                 existing_only=True, prefix='explore_bv_', verbose=True,
                 restrict_neighbors=None, cutoff=1., use_cell_lists=True,
                 import_config_once=False, write_analysis=True):
        super(Neighbors, self).__init__(workspace, jammed_packings_dir=jammed_packings_dir,
                                                analysis_dir=analysis_dir, force=force,
                                                existing_only=existing_only, prefix=prefix,
                                                verbose=verbose, use_cell_lists=use_cell_lists,
                                                import_config_once=import_config_once)
        self.cutoff = cutoff
        self.restrict_neighbors = restrict_neighbors
        self.analysis_fname = analysis_fname
        self.write_analysis = write_analysis

    @staticmethod
    def read(neighbors_fname):
        configf = ConfigParser.ConfigParser()
        configf.read(neighbors_fname)
        neighbors_dict = {}
        neighbors_dict['avg_neighbors'] = \
            configf.getfloat('NEIGHBORS', 'avg_neighbors')
        neighbors_dict['neighbor_counts'] = \
            ast.literal_eval(configf.get('NEIGHBORS', 'neighbor_counts'))
        return neighbors_dict

    def run(self):
        for fname in os.listdir(self.jammed_packings_dir):
            if 'xyzdr' in fname or 'xydr' in fname:
                packing_name = os.path.splitext(fname)[0]

                # Get configuration
                configpath = os.path.join(self.jammed_packings_dir, packing_name + '.config')
                self._import_packing_config_file(configpath)

                # Check if the work directory exists
                base_directory_path = os.path.join(self.workspace, self.prefix + str(packing_name))
                if os.path.isdir(base_directory_path) or not self.existing_only:
                    trymakedir(base_directory_path)

                    # Check if this packing has already been analysed
                    analysis_dir_path = os.path.join(base_directory_path, self.analysis_dir)
                    neighbors_fname = os.path.join(analysis_dir_path, self.analysis_fname)
                    neighbors_dumpname = os.path.join(analysis_dir_path, self.analysis_fname + '_dump.p')
                    compute = False
                    try:
                        self.read(neighbors_fname)
                    except Exception:
                        compute = True

                    if compute or self.force:
                        if self.verbose:
                            if self.restrict_neighbors is None:
                                logging.info("Calculating neighbors: {}"
                                             .format(self.prefix + str(packing_name)))
                            else:
                                logging.info("Calculating restricted neighbors: {}"
                                             .format(self.prefix + str(packing_name)))
                        trymakedir(analysis_dir_path)

                        # Read coordinates and compute neighbors
                        self.coords, self.hs_radii, _, _ = \
                            self._import_packing_configuration(fname)

                        # Create potential
                        if not hasattr(self, 'potential') or not self.import_config_once:
                            self._initialise_potential()

                        # Compute neighbors
                        neighbor_lists, _ = self.potential.getNeighbors(
                            self.coords, cutoff_factor=self.cutoff)

                        # Filter neighbors
                        if self.restrict_neighbors is not None:
                            self._filter_neighbors(neighbor_lists, packing_name)


                        # Output neighbor lists to file
                        self._dump_neighbors(neighbors_dumpname, neighbor_lists)
                        if self.write_analysis:
                            self._write_output(neighbors_fname, neighbor_lists)

    def _filter_neighbors(self, neighbor_lists, packing_name):
        # Get conditional neighbor lists
        base_restrict_path = os.path.join(self.workspace,
                                          self.restrict_neighbors + str(packing_name))
        restrict_dir = os.path.join(base_restrict_path, self.analysis_dir)
        restrict_path = os.path.join(restrict_dir, self.analysis_fname + '_dump.p')
        if not os.path.isfile(restrict_path):
            raise IOError("The restrict neighbors file {} does "
                          "not exist.".format(restrict_path))
        restrict_neighbor_lists = cPickle.load(open(restrict_path, 'r'))

        # Filter neighbors
        neighbor_lists = [filter(lambda particle: particle in
                                  restrict_neighbor_lists[i], neighbor_lists[i])
                           for i in xrange(self.nparticles)]

    def _write_output(self, neighbors_fname, neighbor_lists):
        with open(neighbors_fname, 'w') as f:
            f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
            f.write('[NEIGHBORS]\n')
            neighbor_counts = [len(neighbors) for neighbors in neighbor_lists]
            f.write('avg_neighbors: {}\n'.format(np.mean(neighbor_counts)))
            f.write('neighbor_counts: {}\n'.format(neighbor_counts))

    def _dump_neighbors(self, neighbors_dumpname, neighbor_lists):
        cPickle.dump(neighbor_lists, open(neighbors_dumpname, 'w'))


def worker_neighbors(workspace, kwargs):
    try:
        neighbors = Neighbors(workspace, **kwargs)
        neighbors.run()
    except:
        logging.error('worker_neighbors worker: %s' % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute neighbor lists for jammed packings.")
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
    parser.add_argument("--restrict-neighbors", type=str, help="Prefix leading to "
                        "a neighbor lists file. This string is analogous to the normal prefix. "
                        "Only neighbors in these lists are considered.",
                        default=None)
    parser.add_argument("--cutoff", type=float, help="Multiple of particle radii "
                        "defining the maximum neighbor distance. Default: 1", default=1.)
    parser.add_argument("--nocell", action='store_true', help="Don't use cell lists. "
                        "Default: False", default=False)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set up arguments
    kwargs = dict(force=args.force, existing_only=args.nonex,
                  jammed_packings_dir=args.input_dir, prefix=args.prefix,
                  cutoff=args.cutoff, restrict_neighbors=args.restrict_neighbors,
                  use_cell_lists=not args.nocell)

    # Create workspace directory name
    if not args.workspace_dir:
        workspace_dir = os.getcwd()
    else:
        workspace_dir = os.path.abspath(args.workspace_dir)

    # Start neighbors computation
    neighbors = Neighbors(workspace_dir, **kwargs)
    neighbors.run()
