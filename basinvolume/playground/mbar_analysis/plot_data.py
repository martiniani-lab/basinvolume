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
    from basinvolume.utils import log_factorial, read_csv_xy
    import scipy
    from scipy.stats import t
    from scipy.interpolate import spline
    from scipy.integrate import simps, trapz
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
"""
for plotting a linear fit with intervals of confidence see http://nbviewer.ipython.org/url/bagrow.com/dsv/LEC10_notes_2014-02-13.ipynb
"""
def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

def _read_nparticles(folder):
    nparticles = ""
    for char in folder[1:]:
        if char == "_":
            break
        nparticles+=char
    nparticles = int(nparticles)
    return nparticles

def _sort_pair(x,y):
    """
    sorts x and moves elements of y accordingly
    """
    xc = np.array(x)
    points = zip(xc,y)
    sorted_points = sorted(points)
    new_x = np.array([point[0] for point in sorted_points])
    new_y = np.array([point[1] for point in sorted_points])
    return new_x, new_y

class mbar_data(object):
    def __init__(self, label, analysis_folder = "analysis", log_gr_file = "log_gr.csv", 
                 log_gr_ratio_file = "log_gr_ratio.csv", gr_ratio_file = "gr_ratio.csv",
                 dos_file = "dos.csv"):
        self.label = label
        self.analysis_folder = analysis_folder
        self.log_gr_file = log_gr_file
        self.log_gr_ratio_file = log_gr_ratio_file
        self.gr_ratio_file = gr_ratio_file
        self.dos_file = dos_file
        self.log_gr = []
        self.log_gr_ratio = []
        self.gr_ratio = []
        self.dos = []
        self.log_gr_mean = []
        self.log_gr_ratio_mean = []
        self.gr_ratio_mean = []
        self.dos_mean = []
    
    def compute_mean(self):
        self.log_gr_mean = np.mean(self.log_gr_mean, axis=0).tolist()
        x, xerr, y, yerr, fit = self.log_gr[0]
        self.log_gr_mean = [(x, xerr, self.log_gr_mean, yerr, fit)]
        
        self.log_gr_ratio_mean = np.mean(self.log_gr_ratio_mean, axis=0).tolist()
        x, xerr, y, yerr, fit = self.log_gr_ratio[0]
        self.log_gr_ratio_mean = [(x, xerr, self.log_gr_ratio_mean, yerr, fit)]
        
        self.gr_ratio_mean = np.mean(self.gr_ratio_mean, axis=0).tolist()
        x, xerr, y, yerr, fit = self.gr_ratio[0]
        self.gr_ratio_mean = [(x, xerr, self.gr_ratio_mean, yerr, fit)]
        
        self.dos_mean = np.mean(self.dos_mean, axis=0).tolist()
        x, xerr, y, yerr, fit = self.dos[0]
        self.dos_mean = [(x, xerr, self.dos_mean, yerr, fit)]
        
class plot_mbar_data(object):
    def __init__(self, workdir=None, explore_dir='explore_bv_jammed_packing', Nrange=(0,1000)):
        if not workdir:
            workdir = os.getcwd() 
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
        self.workdir = workdir
        self.explore_dir = explore_dir
        self.Nrange = Nrange
        self.fcc_data = mbar_data("fcc")
        self.fcc_mono_data = mbar_data("fcc_mono")
        self.disordered_data = mbar_data("disordered")
        markers = ["bo", "r^", "gs", "kx", "c+"]
        self.markercycler = cycle(markers)
        self._collect_data(self.fcc_data)
        self._collect_data(self.fcc_mono_data)
        self._collect_data(self.disordered_data)
    
    def _collect_data(self, mbar_data):
        subdirs = get_immediate_subdirectories(os.path.join(self.workdir, mbar_data.label))
        for folder in subdirs:
            if self.explore_dir in folder:
                path = os.path.join(self.workdir, mbar_data.label, folder, mbar_data.analysis_folder)
                if os.path.isdir(path):
                    npack = int(re.findall('\d+', folder)[0])
                    if self.Nrange[0] <= npack <= self.Nrange[1]:
                        #log_gr
                        fpath = os.path.join(path, mbar_data.log_gr_file)
                        if os.path.isfile(fpath):
                            x, xerr, y, yerr, fit = read_csv_xy(fpath)
                            mbar_data.log_gr.append((x,xerr,y,yerr,fit))
                            mbar_data.log_gr_mean.append(y)
                        #log_gr_ratio
                        fpath = os.path.join(path, mbar_data.log_gr_ratio_file)
                        if os.path.isfile(fpath):
                            x, xerr, y, yerr, fit = read_csv_xy(fpath)
                            y -= np.amax(y)
                            mbar_data.log_gr_ratio.append((x,xerr,y,yerr,fit))
                            mbar_data.log_gr_ratio_mean.append(y)
                        #gr_ratio
                        fpath = os.path.join(path, mbar_data.gr_ratio_file)
                        if os.path.isfile(fpath):
                            x, xerr, y, yerr, fit = read_csv_xy(fpath)
                            mbar_data.gr_ratio.append((x,xerr,y,yerr,fit))
                            mbar_data.gr_ratio_mean.append(y)
                        #dos
                        fpath = os.path.join(path, mbar_data.dos_file)
                        if os.path.isfile(fpath):
                            x, xerr, y, yerr, fit = read_csv_xy(fpath)
                            mbar_data.dos.append((x,xerr,y,yerr,fit))
                            mbar_data.dos_mean.append(y)
        mbar_data.compute_mean()  
    
    def _plot(self, ax, csv_tuple, label=None, plot_err=False, plot_fit=False, normalize=False):
        (x, xerr, y, yerr, fit) = csv_tuple
        if normalize:
            area = trapz(y, dx=x[2]-x[1])
            y = np.array(y) / area
        if plot_err:
            ax.errorbar(x, y, yerr=yerr, fmt='bo', ms=9, label=label)
        else:
            ax.plot(x, y, label=label, linewidth=2.5)
        if plot_fit:
            ax.plot(x,fit,'b--', linewidth=2)
        return ax
    
    def _plot_all(self, ax, mbar_data, plot_type="log_gr", label=None, average=False):
        if plot_type == "log_gr":
            if average:
                log_gr = mbar_data.log_gr_mean
            else:
                log_gr = mbar_data.log_gr
            for csv_tuple in log_gr:
                ax = self._plot(ax, csv_tuple, label=label, plot_err=False, plot_fit=False)
            xlabel=r'r'
            ylabel=r'$\log(g(r))$'
        if plot_type == "log_gr_ratio":
            if average:
                log_gr_ratio = mbar_data.log_gr_ratio_mean
            else:
                log_gr_ratio = mbar_data.log_gr_ratio
            for csv_tuple in log_gr_ratio:
                ax = self._plot(ax, csv_tuple, label=label, plot_err=False, plot_fit=False)
            xlabel=r'r'
            ylabel=r'$\log(g(r)/r^{N-1})$'
        if plot_type == "gr_ratio":
            if average:
                gr_ratio = mbar_data.gr_ratio_mean
            else:
                gr_ratio = mbar_data.gr_ratio
            for csv_tuple in gr_ratio:
                ax = self._plot(ax, csv_tuple, label=label, plot_err=False, plot_fit=False)
            xlabel=r'r'
            ylabel=r'$g(r)/r^{N-1}$'
            ax.set_xlim((0,1))
        if plot_type == "dos":
            if average:
                dos = mbar_data.dos_mean
            else:
                dos = mbar_data.dos
            for csv_tuple in dos:
                ax = self._plot(ax, csv_tuple, label=label, plot_err=False, plot_fit=False, normalize=True)
            xlabel=r'r'
            ylabel=r'$g(r)$'
            ax.set_xlim((0.5,4))
        
        return ax, xlabel, ylabel
    
    def plot_all(self, plot_type="log_gr", figname=None, title=None, show=False, savefig=False, average=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax, xlabel, ylabel = self._plot_all(ax, self.fcc_data, plot_type=plot_type, average=average, label=r"fcc")
        ax, xlabel, ylabel = self._plot_all(ax, self.fcc_mono_data, plot_type=plot_type, average=average, label=r"fcc mono")
        ax, xlabel, ylabel = self._plot_all(ax, self.disordered_data, plot_type=plot_type, average=average, label=None)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        try:
            ax.legend(frameon=False, loc="best")
        except Exception, e:
            print e
            
        if title:
            plt.title(title)
        if figname is None:
            figname = plot_type + "_all.eps"
        if savefig:
            plt.savefig(figname)
        if show:
            plt.show()
     
if __name__ == "__main__":
    pe = plot_mbar_data(Nrange=(0,1000))
    pe.plot_all(plot_type="log_gr_ratio", show=True, savefig=True)
    pe.plot_all(plot_type="log_gr", show=True, savefig=True)
    pe.plot_all(plot_type="gr_ratio", show=True, savefig=True)
    pe.plot_all(plot_type="dos", show=True, savefig=True)


