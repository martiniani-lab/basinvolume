from __future__ import division
from basinvolume.utils._utils import _sort_pair
import numpy as np
import argparse
import ConfigParser
import os
import re
from itertools import cycle
from basinvolume.utils import *
import scipy
from scipy.stats import t
from scipy.interpolate import spline
from scipy.integrate import simps
import glob
from itertools import chain
import cPickle as pickle
from basinvolume.post_processing import PackingData, PackingDataSet
try:
    
    import matplotlib.pyplot as plt
except ImportError as err:
    print err

class BasinAnalysis(object):
    """
    to use mbar change
    set_dir is the directo
    volume_file = "mbar_volume"
    volume_title = "MBAR_VOLUME"
    """
    def __init__(self, workspace=None, packings_dir='packings', jammed_packings_dir='jammed_packings', 
                 analysis_dir='analysis', volume_file="volume_data", pressure_file="pressure_data", 
                 zboo_file="glob_boo", volume_title = "VOLUME_FULL_PT"):
        if workspace is None:
            workspace = os.getcwd()
        if not os.path.isabs(workspace):
            workspace = os.path.abspath(workspace)
        self.workspace = workspace
        self.packings_dir = packings_dir
        self.jammed_packings_dir = jammed_packings_dir
        self.volume_file = volume_file
        self.pressure_file = pressure_file
        self.analysis_dir = analysis_dir
        self.volume_title = volume_title
        self.zboo_file = zboo_file
        self.iteration = 0
        self.packing_datasets = []
    
    def _get_dname(self, dname):
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        return dname    
        
    def _get_dname_packing(self, inp):
        jammed_dname = self._get_dname(inp)
        return jammed_dname.split("_")[1]
    
    def collect_data_every_set_all(self, data_name="basin_analysis.pickle", no_pickle=False, 
                                   dir_signature='n*_phi*_phi*_*D'):
        listdir = glob.glob(os.path.join(self.workspace, dir_signature))
        #print listdir
        data_pickle = os.path.join(self.workspace, data_name)
        if os.path.isfile(data_pickle) and not no_pickle:
            self.packing_datasets = pickle.load( open(data_pickle, "rb") )
        else: 
            for set_path in listdir:
                print("set_path", set_path)
                print "collecting data from ", os.path.split(set_path)[1]
                self.collect_data_single_all(set_path=set_path)
            self.packing_datasets = sorted(self.packing_datasets, key=lambda data: data.nparticles)
            pickle.dump(self.packing_datasets, open( data_pickle, "wb" ) )
        
    def collect_data_single_all(self, set_path=None):
        if set_path is None:
            set_path = self.workspace
        packing_dataset = self._collect_data_single_all(set_path)
        self.packing_datasets.append(packing_dataset)
    
    def _collect_data_single_all(self, set_path):
        """compute boo for packings
        """
        pd_list = []
        packing_dataset = PackingDataSet(set_path)
        for fname in os.listdir(os.path.join(set_path, self.jammed_packings_dir)):
            if 'xyzd' in fname or 'xyd' in fname:
                dname = self._get_dname(fname)
                dname_packing = self._get_dname_packing(fname)
                base_directory_path = os.path.join(set_path, 'explore_bv_' + str(dname))
                if os.path.isdir(base_directory_path):
                    configpath = os.path.join(set_path, self.jammed_packings_dir, dname + '.config')
                    configpath_packing = os.path.join(set_path, self.packings_dir, dname_packing + ".config")
                    pd = PackingData(str(dname), configpath, configpath_packing)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.volume_file)
                    pd.import_volume_data(path)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.pressure_file)
                    pd.import_pressure_data(path)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.zboo_file)
                    pd.import_structural_data(path)
                    pd_list.append(pd)
        packing_dataset.add_data_all(pd_list)
        return packing_dataset
    
    def collect_data_every_set_structure(self, data_name="basin_analysis.pickle", no_pickle=False,
                                         dir_signature='n*_phi*_phi*_*D'):
        """
        collect only structural data from all sets
        """
        listdir = glob.glob(os.path.join(self.workspace, dir_signature))
        data_pickle = os.path.join(self.workspace, data_name)
        if os.path.isfile(data_pickle) and not no_pickle:
            self.packing_datasets = pickle.load( open(data_pickle, "rb") )
        else: 
            for set_path in listdir:
                print("set_path", set_path)
                print "collecting data from ", os.path.split(set_path)[1]
                self.collect_data_single_structure(set_path=set_path)
            self.packing_datasets = sorted(self.packing_datasets, key=lambda data: data.nparticles)
            pickle.dump(self.packing_datasets, open( data_pickle, "wb" ) )
        
    def collect_data_single_structure(self, set_path=None):
        if set_path is None:
            set_path = self.workspace
        packing_dataset = self._collect_data_single_structure(set_path)
        self.packing_datasets.append(packing_dataset)
    
    def _collect_data_single_structure(self, set_path):
        """
        """
        pd_list = []
        packing_dataset = PackingDataSet(set_path)
        for fname in os.listdir(os.path.join(set_path, self.jammed_packings_dir)):
            if 'xyzd' in fname or 'xyd' in fname:
                dname = self._get_dname(fname)
                dname_packing = self._get_dname_packing(fname)
                base_directory_path = os.path.join(set_path, 'explore_bv_' + str(dname))
                if os.path.isdir(base_directory_path):
                    configpath = os.path.join(set_path, self.jammed_packings_dir, dname + '.config')
                    configpath_packing = os.path.join(set_path, self.packings_dir, dname_packing + ".config")
                    pd = PackingData(str(dname), configpath, configpath_packing)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.pressure_file)
                    pd.import_pressure_data(path)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.zboo_file)
                    pd.import_structural_data(path)
                    pd_list.append(pd)
        packing_dataset.add_data_structure(pd_list)
        return packing_dataset

if __name__ == "__main__":
    pts = BasinAnalysis()
    pts.collect_data_every_set_all()
    pts.collect_data_every_set_strcture()
    
