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
try:
    
    import matplotlib.pyplot as plt
except ImportError as err:
    print err

class PackingDataSet(object):
    """
    this assumes naming convention n24_phi50_phi70_3D
    
    Parameters
    ----------
    set_path : string
        path to set of explore_bv folders, for instance /path/to/n24_phi50_phi70_3D
    extras: list
        extras is any arbitrary list of data that one might want to add on to this class
        during the analysis for convenience
    """
    def __init__(self, set_path):
        self.set_path = set_path
        self.set_name = os.path.split(self.set_path)[1]
        str_values = re.findall('\d+', self.set_name)
        self.nparticles, self.hs_phi = float(str_values[0]), float('0.'+str_values[1])
        self.ss_phi, self.bdim = float('0.'+str_values[2]), float(str_values[3]) 
        self.packing_data = []
        self.free_energies = []
        self.pressures = []
        self.contacts = []
        self.boos = []
        self.extras = []
    
    def add_data(self, packing_data):
        """
        packing data is a list of PackingData objects
        """
        self.packing_data.extend(packing_data)
        for data in packing_data:
            if data.F is not None and data.P is not None:
                self.free_energies.append(data.F)
                self.pressures.append(data.P)
            if data.Z is not None and data.boo is not None:
                self.contacts.append(data.Z)
                self.boos.append(data.boo)
    
    def add_extras(self, extra):
        self.extras.append(np.array(extra))
    
class PackingData(object):
    def __init__(self, name, configpath, packing_path=None):
        self.eps = 1.
        self.frozen = False
        self.name = name
        self.configpath = configpath
        self._import_packing_config_file(self.configpath)
        if packing_path is not None:
            self._import_packing_configuration(packing_path)
        self.F = None 
        self.Ferr = None
        self.P = None
        self.Ptensor = None
        self.Facc = None
        self.Z = None
        self.boo = None
        
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
    
    def _import_packing_configuration(self, path):
        #path = os.path.join(self.packings_dir, fname)
        if self.bdim == 2:
            coords, hs_diameters, rattlers = read_xydr(path)
        elif self.bdim == 3:
            coords, hs_diameters, rattlers = read_xyzdr(path)
        else:
            raise NotImplementedError("bdim={} not implemented".format(self.bdim))
        hs_radii = hs_diameters/2
        ss_radii = hs_radii * (1+self.sca)
        self.coords = coords
        self.hs_radii = hs_radii
        self.ss_radii = ss_radii
        self.rattlers = rattlers
            
    def import_volume_data(self, path, title="VOLUME_FULL_PT", vfluid_title="VOLUME_HS_FLUID"):
        if os.path.isfile(path):
            configf = ConfigParser.ConfigParser()
            configf.read(path)
            self.F, self.Ferr = configf.getfloat(title, 'F0'), configf.getfloat(title, 'sigF0')
            try:
                self.Facc = configf.getfloat(vfluid_title, 'F0_acc')
            except Exception,e:
                pass
            
    def import_pressure_data(self, path, title="PRESSURE"):
        if os.path.isfile(path):
            configf = ConfigParser.ConfigParser()
            configf.read(path)
            self.P = configf.getfloat(title, 'P')
            Ptensor = configf.get(title, 'Ptensor')
            self.Ptensor = np.array([float(x) for x in Ptensor.split()])
    
    def import_structural_data(self, path, title_boo="BOO", title_z="Z"):
        """
        import average contact number and bond orientational order parameters
        """
        if os.path.isfile(path):
            configf = ConfigParser.ConfigParser()
            configf.read(path)
            Q4, Q6 = configf.getfloat(title_boo,'Q4'), configf.getfloat(title_boo,'Q6')
            Q8, Q10 = configf.getfloat(title_boo,'Q8'), configf.getfloat(title_boo,'Q10')
            Q12 = configf.getfloat(title_boo,'Q12')
            self.boo = Bunch(Q4=Q4, Q6=Q6, Q8=Q8, Q10=Q10, Q12=Q12)
            z = configf.getfloat(title_z,'Z')
            self.Z = z
