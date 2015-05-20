from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import re
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from itertools import cycle
    from basinvolume.utils import *
    import scipy
    from scipy.stats import t
    from scipy.interpolate import spline
    from scipy.integrate import simps
    import glob
    from itertools import chain
except ImportError as err:
    print err
#######################SET LATEX OPTIONS###################
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
#rc('text.latex',preamble=r'\usepackage{times}')
plt.rcParams.update({'font.size': 16})
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
##########################################################
####SET COLOUR MAP######                                                               
cm = plt.get_cmap('Dark2')
########################
#####################LINE STYLE CYCLER####################                             
lines = ["-","--","-."]
linecycler = cycle(lines)
color_cycle=cycle([cm(1. * i / 6) for i in xrange(6)])
##########################################################
"""
for plotting a linear fit with intervals of confidence see http://nbviewer.ipython.org/url/bagrow.com/dsv/LEC10_notes_2014-02-13.ipynb
"""
def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


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
            
class BasinAnalysis(object):
    """
    to use mbar change
    volume_file = "mbar_volume"
    volume_title = "MBAR_VOLUME"
    """
    def __init__(self, packings_dir='packings', jammed_packings_dir='jammed_packings', analysis_dir='analysis', 
                 volume_file="volume_data", pressure_file="pressure_data", volume_title = "VOLUME_FULL_PT"):
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        if not os.path.isabs(jammed_packings_dir):
            packings_dir = os.path.join(os.getcwd(), jammed_packings_dir)
        self.packings_dir = packings_dir
        self.jammed_packings_dir = jammed_packings_dir
        self.volume_file = volume_file
        self.pressure_file = pressure_file
        self.analysis_dir = analysis_dir
        self.volume_title = volume_title
        self.iteration = 0
        self.packing_data = []
    
    def _get_dname(self, dname):
        if dname.endswith('.xyzdr'):
            dname = dname[:-6]
        elif dname.endswith('.xydr'):
            dname = dname[:-5]
        return dname    
    
    def _collect_data(self):
        """compute boo for packings
        """
        for fname in os.listdir(self.packings_dir):
            if 'xyzd' in fname or 'xyd' in fname:
                dname = self._get_dname(fname)
                base_directory_path = os.path.join(os.getcwd(),'explore_bv_'+str(dname))
                if os.path.isdir(base_directory_path):
                    configpath = os.path.join(self.jammed_packings_dir, dname + '.config')
                    pd = PackingData(str(dname), configpath)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.volume_file)
                    pd.import_volume_data(path)
                    path = os.path.join(base_directory_path, self.analysis_dir, self.pressure_file)
                    pd.import_pressure_data(path)
                    self.packing_data.append(pd)
        self.free_energies = np.array([data.F for data in self.packing_data])
        self.pressures = np.array([data.P for data in self.packing_data])
        
        #THESE ARE JUST QUICK PLOTS, CLEAN THIS UP
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.free_energies, np.log(self.pressures))
        plt.xlabel("F")
        plt.ylabel("lnP")
        plt.show()

        from basinvolume.experiment_2d.cross_validation_bandwidth_selection import get_bandwidth_estimate, get_pdf
#        kde pressures
#        bw = get_bandwidth_estimate(np.array(self.pressures), kernel="gaussian", method="cross_validation")
#        edges = np.linspace(np.amin(self.pressures), np.amax(self.pressures), 1000)
#        hist = get_pdf(self.pressures, edges, bandwidth=bw, kernel="gaussian")
#        plt.plot(edges, hist)
#        plt.show()

#        kde free energies
#        free_energies = np.array([f for f in self.free_energies if f is not None])
#        bw = get_bandwidth_estimate(np.array(free_energies), kernel="gaussian", method="cross_validation")
#        edges = np.linspace(np.amin(free_energies), np.amax(free_energies), 100)
#        hist = get_pdf(free_energies, edges, bandwidth=bw, kernel="gaussian")
#        plt.plot(edges, hist)
#        plt.show()
        
        

if __name__ == "__main__":
#    boo = BondOrientationalOrder()
#    #boo.run(deg=12)
#    boo.run_all()
    pts = BasinAnalysis()
    pts._collect_data()