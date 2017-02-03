from __future__ import division
import numpy as np
import os
import abc
from basinvolume.utils import import_packing
from basinvolume.spheres.generate_jammed_packing import read_jammed_packing_config
from pele.potentials import HS_WCA


class StructuralAnalysis(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, workspace, packings_dir='packings',
                 jammed_packings_dir='jammed_packings', analysis_dir='analysis',
                 force=False, existing_only=True, prefix='explore_bv_', verbose=True,
                 use_cell_lists=True, import_config_once=False):
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
        self.use_cell_lists = use_cell_lists
        self.import_config_once = import_config_once
        self.cancel_import_config = False

    def _import_packing_config_file(self, configpath):
        if not self.cancel_import_config:
            config = read_jammed_packing_config(configpath, self.frozen)
            self.nparticles = config['nparticles']
            self.packing_frac = config['packing_frac']
            self.bdim = config['bdim']
            self.boxv = config['boxv'].copy()
            self.vcavity = config['vcavity']
            self.distance_method = config['distance_method']
            if hasattr(self, 'pot_kwargs') and self.pot_kwargs is not None:
                self.pot_kwargs.update(config['pot_kwargs'])
            else:
                self.pot_kwargs = config['pot_kwargs'].copy()
            self.sca = config['sca']
            self.cancel_import_config = self.import_config_once

    def _import_packing_configuration(self, fname):
        path = os.path.join(self.jammed_packings_dir, fname)
        packing = import_packing(path, True, self.bdim, self.sca)
        return packing['coords'], packing['hs_radii'], packing['ss_radii'], packing['stable_atoms']



    def _initialise_potential(self):
        if self.use_cell_lists:
            self.potential = HS_WCA(use_cell_lists=True, eps=self.eps, sca=self.sca,
                                    radii=self.hs_radii, boxvec=self.boxv,
                                    reference_coords=self.coords, ndim=self.bdim,
                                    ncellx_scale=1.0, distance_method=self.distance_method,
                                    pot_kwargs=self.pot_kwargs)
        else:
            self.potential = HS_WCA(eps=self.eps, sca=self.sca, radii=self.hs_radii,
                                    boxvec=self.boxv, ndim=self.bdim,
                                    distance_method=self.distance_method,
                                    pot_kwargs=self.pot_kwargs)
