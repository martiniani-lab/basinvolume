import ConfigParser
import os
import ast
import numpy as np
import argparse
from pele.distance import get_distance
from compute_structural_properties import StructuralAnalysis
from basinvolume.utils import trymakedir, read_xydr, read_xyzdr


class Neighbours(StructuralAnalysis):
    def __init__(self, workspace, jammed_packings_dir='jammed_packings',
                 analysis_dir='analysis', force=False, existing_only=True,
                 prefix='explore_bv_', include_neighbourss = None, cutoff = 1.):
        super(Neighbours, self).__init__(workspace, jammed_packings_dir=jammed_packings_dir,
                                                analysis_dir=analysis_dir, force=force,
                                                existing_only=existing_only, prefix=prefix)
        self.cutoff = cutoff
        self.include_neighbourss = include_neighbourss


    def _distance (self, coord1, coord2):
        if self.distance_method == 'lees-edwards':
            return get_distance(coord1, coord2, self.bdim, self.distance_method,
                                box=self.boxv, shear=self.pot_kwargs['shear'])
        else:
            return get_distance(coord1, coord2, self.bdim, self.distance_method, box=self.boxv)


    def _find_neighbours(self, coords, radii):
        nparticles = radii.size
        neighbour_distancess = [[] for _ in xrange(nparticles)]
        neighbour_indicess = [[] for _ in xrange(nparticles)]

        # Loop over all unique pairs of different particles
        for i in xrange(nparticles - 1):
            for j in xrange(i + 1, nparticles):

                # Calculate distance
                dij = self._distance(coords[i * self.bdim : (i + 1) * self.bdim],
                                         coords[j * self.bdim : (j + 1) * self.bdim])
                dijnorm = np.linalg.norm(dij)

                # Check if this particle lies within neighbour range
                dmax = self.cutoff * (radii[i] + radii[j])
                if dijnorm <= dmax:
                    neighbour_distancess[i].append(dij)
                    neighbour_distancess[j].append(-dij)
                    neighbour_indicess[i].append(j)
                    neighbour_indicess[j].append(i)

        return neighbour_distancess, neighbour_indicess


    def run(self):
        for fname in os.listdir(self.jammed_packings_dir):
            if 'xyzdr' in fname or 'xydr' in fname:
                dname = self._get_dname(fname)

                # Get configuration
                configpath = os.path.join(self.jammed_packings_dir, dname + '.config')
                self._import_packing_config_file(configpath)

                # Check if the work directory exists
                base_directory_path = os.path.join(self.workspace, self.prefix + str(dname))
                if os.path.isdir(base_directory_path) or not self.existing_only:
                    trymakedir(base_directory_path)

                    # Check if this packing has already been analysed
                    analysis_dir_path = os.path.join(base_directory_path, self.analysis_dir)
                    neighbours_fname = os.path.join(analysis_dir_path,'neighbours')
                    compute = False
                    try:
                        configf = ConfigParser.ConfigParser()
                        configf.read(neighbours_fname)
                        ast.literal_eval(configf.get('NEIGHBOURS', 'neighbour_listss'))
                        ast.literal_eval(configf.get('NEIGHBOURS', 'neighbour_counts'))
                    except Exception:
                        compute = True

                    if compute or self.force:
                        print "neighbours ", dname
                        trymakedir(analysis_dir_path)

                        # Read coordinates and compute neighbours
                        self.coords, _, self.ss_radii, _ = self._import_packing_configuration(fname)
                        _, neighbour_listss = self._find_neighbours(self.coords, self.ss_radii)

                        # Filter neighbours
                        if self.include_neighbourss is not None:
                            neighbour_listss = [filter(lambda particle: particle in
                                                      include_neighbourss[i], neighbour_listss[i])
                                               for i in range(self.nparticles)]

                        # Output neighbour lists to file
                        with open(neighbours_fname, 'w') as f:
                            f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
                            f.write('[NEIGHBOURS]\n')
                            f.write('neighbour_counts: {}\n'.format([len(neighbours) for neighbours
                                                                     in neighbour_listss]))
                            f.write('neighbour_listss: {}\n'.format(neighbour_listss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute neighbour lists for jammed packings.")
    parser.add_argument("-d", "--workspace_dir", type=str, help="Top-level dir containing "
                        "the packings, e.g. 'n32_phi88_2D'.")
    parser.add_argument("--force", action='store_true', help="Force to run on all packings.",
                        default=False)
    parser.add_argument("--nonex", action='store_false', help="Run also for packings "
                        "for which there are no work folders ('explore_bv_[...]', "
                        "created e.g. by parallel tempering).", default=True)
    parser.add_argument("--prefix", type=str, help="Prefix for the work directory. "
                        "Default: 'explore_bv_'", default='explore_bv_')
    parser.add_argument("--input_dir", type=str, help="Directory containing the "
                        "jammed packings. Default: 'jammed_packings'", default='jammed_packings')
    parser.add_argument("--restrict_neighbours", type=str, help="File containing "
                        "neighbour lists. Only neighbours in these lists will be considered.",
                        default=None)
    parser.add_argument("--cutoff", type=float, help="Multiple of particle radii "
                        "defining the maximum neighbour distance. Default: 1", default=1.)
    args = parser.parse_args()

    # Set up arguments
    kwargs = dict(force=args.force, existing_only=args.nonex,
                  jammed_packings_dir=args.input_dir, prefix=args.prefix, cutoff=args.cutoff)

    # Get conditional neighbour lists
    if args.restrict_neighbours is not None:
        configf = ConfigParser.ConfigParser()
        configf.read(args.restrict_neighbours)
        include_neighbourss = ast.literal_eval(configf.get('NEIGHBOURS', 'neighbour_listss'))
        kwargs.update(include_neighbourss=include_neighbourss)

    # Create workspace directory name
    if not args.workspace_dir:
        workspace_dir = os.getcwd()
    else:
        workspace_dir = os.path.abspath(args.workspace_dir)

    # Start neighbours computation
    neighbours = Neighbours(workspace_dir, **kwargs)
    neighbours.run()
