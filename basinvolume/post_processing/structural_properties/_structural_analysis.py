from __future__ import division
import numpy as np
import os
import abc
from basinvolume.utils import import_packing
from basinvolume.spheres.generate_jammed_packing import read_jammed_packing_config


class StructuralAnalysis(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, workspace, packings_dir='packings',
                 jammed_packings_dir='jammed_packings', analysis_dir='analysis',
                 force=False, existing_only=True, prefix='explore_bv_', verbose=True):
        if not os.path.isabs(workspace):
            workspace = os.path.abspath(workspace)
        self.workspace = workspace
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(self.workspace, packings_dir)
        if not os.path.isabs(jammed_packings_dir):
            jammed_packings_dir = os.path.join(self.workspace, jammed_packings_dir)
        self.packings_dir = packings_dir
        self.jammed_packings_dir = jammed_packings_dir
        self.analysis_dir = analysis_dir
        self.iteration = 0
        self.eps = 1.
        self.frozen = False
        self.force = force
        self.existing_only = existing_only
        self.prefix = prefix
        self.verbose = verbose

    def _import_packing_config_file(self, configpath):
        imp_packing = read_jammed_packing_config(configpath, self.frozen)
        self.nparticles = imp_packing['nparticles']
        self.packing_frac = imp_packing['packing_frac']
        self.bdim = imp_packing['bdim']
        self.boxv = imp_packing['boxv'].copy()
        self.vcavity = imp_packing['vcavity']
        self.distance_method = imp_packing['distance_method']
        if hasattr(self, 'pot_kwargs') and self.pot_kwargs is not None:
            self.pot_kwargs.update(imp_packing['pot_kwargs'])
        else:
            self.pot_kwargs = imp_packing['pot_kwargs'].copy()
        self.sca = imp_packing['sca']

    def _import_packing_configuration(self, fname):
        path = os.path.join(self.jammed_packings_dir, fname)
        packing = import_packing(path, True, self.bdim, self.sca)
        return packing['coords'], packing['hs_radii'], packing['ss_radii'], packing['stable_atoms']
