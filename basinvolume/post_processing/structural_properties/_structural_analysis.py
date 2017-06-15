from __future__ import division
import numpy as np
import os
import abc
from basinvolume.utils import import_packing, trymakedir
from basinvolume.spheres import read_jammed_packing_config
from pele.potentials import HS_WCA


class StructuralAnalysis(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, workspace, jammed_packings_dir='jammed_packings',
                 analysis_dir='analysis', force=False, existing_only=True,
                 prefix='explore_bv_', verbose=True, use_cell_lists=True):
        if not os.path.isabs(workspace):
            workspace = os.path.abspath(workspace)
        self.workspace = workspace
        if not os.path.isabs(jammed_packings_dir):
            jammed_packings_dir = os.path.join(self.workspace,
                                               jammed_packings_dir)
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

    def _import_packing_config_file(self, configpath):
        config = read_jammed_packing_config(configpath, self.frozen)
        self.nparticles = config['nparticles']
        self.packing_frac = config['packing_frac']
        self.bdim = config['bdim']
        self.boxv = config['boxv'].copy()
        self.vcavity = config['vcavity']
        self.distance_method = config['distance_method']
        self.interaction = config['interaction']
        if hasattr(self, 'pot_kwargs') and self.pot_kwargs is not None:
            self.pot_kwargs.update(config['pot_kwargs'])
        else:
            self.pot_kwargs = config['pot_kwargs'].copy()
        self.sca = config['sca']

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

    def run(self):
        for fname in os.listdir(self.jammed_packings_dir):
            if 'xyzdr' in fname or 'xydr' in fname:
                packing_name = os.path.splitext(fname)[0]

                # Get configuration
                configpath = os.path.join(self.jammed_packings_dir,
                                          packing_name + '.config')
                self._import_packing_config_file(configpath)

                # Check if the work directory exists
                base_directory_path = os.path.join(self.workspace,
                                                   self.prefix + str(packing_name))
                if os.path.isdir(base_directory_path) or not self.existing_only:
                    trymakedir(base_directory_path)

                    # Check if this packing has already been analysed
                    self.analysis_dir_path = os.path.join(base_directory_path,
                                                          self.analysis_dir)
                    analysis_fname = os.path.join(self.analysis_dir_path,
                                                  self.analysis_name)
                    already_computed = True
                    if not self.force:
                        try:
                            self.read(analysis_fname)
                        except Exception:
                            already_computed = False

                    if self.force or not already_computed:
                        trymakedir(self.analysis_dir_path)
                        self._calculate(analysis_fname, packing_name, fname)

    @abc.abstractmethod
    def read(self):
        """Read structural property from a file"""

    @abc.abstractmethod
    def _calculate(self, analysis_fname, input_packing, input_fname):
        """Calculate the structural property"""
