from __future__ import division
import numpy as np
import os
import abc
from basinvolume.utils import import_packing


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

    def _import_packing_configuration(self, fname):
        path = os.path.join(self.jammed_packings_dir, fname)
        packing = import_packing(path, True, self.bdim, self.sca)
        return packing['coords'], packing['hs_radii'], packing['ss_radii'], packing['rattlers']
