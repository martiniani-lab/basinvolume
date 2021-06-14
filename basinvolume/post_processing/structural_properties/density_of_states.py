from __future__ import division
from __future__ import absolute_import
<<<<<<< Updated upstream
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
import configparser
=======
import ConfigParser
>>>>>>> Stashed changes
import os
import ast
import traceback
import numpy as np
import argparse
import logging
from basinvolume.utils import trymakedir
from ._structural_analysis import StructuralAnalysis


class DensityOfStates(StructuralAnalysis):

    def __init__(self, workspace, jammed_packings_dir='jammed_packings',
                 analysis_dir='analysis', force=False, existing_only=True,
                 prefix='explore_bv_', verbose=True, use_cell_lists=True):
        super(DensityOfStates, self).__init__(workspace, jammed_packings_dir=jammed_packings_dir,
                                                analysis_dir=analysis_dir, force=force,
                                                existing_only=existing_only, prefix=prefix,
                                                verbose=verbose, use_cell_lists=use_cell_lists)
        self.analysis_name = 'density_of_states'

    @staticmethod
    def read(dos_fname):
        configf = configparser.ConfigParser()
        configf.read(dos_fname)
        dos_dict = {}
        dos_dict['neg_eigenvalues'] \
            = ast.literal_eval(configf.get('DENSITY_OF_STATES', 'neg_eigenvalues'))
        dos_dict['eigenmodes'] \
            = ast.literal_eval(configf.get('DENSITY_OF_STATES', 'eigenmodes'))
        dos_dict['participation_ratio'] \
            = ast.literal_eval(configf.get('DENSITY_OF_STATES', 'participation_ratio'))
        return dos_dict

    def calc_participation(self, eigenvectors):
        eigvec_norm = np.empty(self.nparticles)
        participation = np.empty((eigenvectors.shape[1]))
        for i in range(eigenvectors.shape[1]):
            eigvec_norm = np.sqrt(sum((eigenvectors[dim::self.bdim, i] ** 2
                                       for dim in range(self.bdim))))
            participation[i] = (sum(eigvec_norm ** 2) ** 2
                                / (self.nparticles * sum(eigvec_norm ** 4)))
        return participation

    def filter_eigenmodes(self, eigenvalues):
        eigvalues = np.array([0. if np.isclose(eigmode, 0.) else eigmode
                              for eigmode in eigenvalues])
        neg_eigvalues = eigvalues[eigvalues < 0]
        if len(neg_eigvalues) > 0:
            logging.warning("There are negative eigenvalues!")
        return np.sqrt(eigvalues[eigvalues >= 0]), neg_eigvalues

    def _calculate(self, dos_fname, packing_name, input_fname):
        if self.verbose:
            logging.info("Calculating density of states: {}"
                         .format(self.prefix + str(packing_name)))

        # Read coordinates
        self.coords, self.hs_radii, self.ss_radii, _ \
            = self._import_packing_configuration(input_fname)

        # Create potential
        self._initialise_potential()

        hessian = self.potential.getHessian(self.coords)
        eigs = np.linalg.eigh(hessian)
        eigmodes, neg_eigvalues = self.filter_eigenmodes(eigs[0])
        participation_ratio = self.calc_participation(eigs[1])

        # Output density of states to file
        with open(dos_fname, 'w') as f:
            f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
            f.write('[DENSITY_OF_STATES]\n')
            f.write('neg_eigenvalues: {}\n'.format(list(neg_eigvalues)))
            f.write('eigenmodes: {}\n'.format(list(eigmodes)))
            f.write('participation_ratio: {}\n'.format(list(participation_ratio)))


def worker_dos(workspace, kwargs):
    try:
        dos = DensityOfStates(workspace, **kwargs)
        dos.run()
    except Exception:
        logging.error('worker_dos worker: %s' % (traceback.format_exc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the density of states of the frequency modes "
                    "for jammed packings.")
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
    parser.add_argument("--nocell", action='store_true', help="Don't use cell lists. "
                        "Default: False", default=False)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set up arguments
    kwargs = dict(force=args.force, existing_only=args.nonex,
                  jammed_packings_dir=args.input_dir, prefix=args.prefix,
                  use_cell_lists=not args.nocell)

    # Create workspace directory name
    if not args.workspace_dir:
        workspace_dir = os.getcwd()
    else:
        workspace_dir = os.path.abspath(args.workspace_dir)

    # Start density of states computation
    dos = DensityOfStates(workspace_dir, **kwargs)
    dos.run()
