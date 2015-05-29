from __future__ import division
from basinvolume.utils._utils import _sort_pair
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
    import cPickle as pickle
    from basinvolume.post_processing import PackingData, PackingDataSet, BasinAnalysis
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
cm = plt.get_cmap('Set2')
########################
#####################LINE STYLE CYCLER####################                             
lines = ["-","--","-."]
linecycler = cycle(lines)
color_cycle=cycle([cm(1. * i / 7) for i in xrange(7)])
##########################################################
"""
for plotting a linear fit with intervals of confidence see http://nbviewer.ipython.org/url/bagrow.com/dsv/LEC10_notes_2014-02-13.ipynb
"""

def plot(packing_datasets):
    from scipy.optimize import curve_fit
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i,dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
        if len(dataset.free_energies) > 0:
            nparticles = dataset.nparticles
            outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
            x = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
            y = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
            x = np.log(x)
            ax.scatter(x, y, label=int(nparticles), color=color_cycle.next())
            fit = np.polyfit(x, y,1)
            fit_fn = np.poly1d(fit)
            ax.plot(x, fit_fn(x), color='k')
            dataset.add_extras(fit)
    ax.legend(frameon=False, loc="best")
    plt.ylabel(r"$F$")
    plt.xlabel(r"$\log P$")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i,dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
        if len(dataset.free_energies) > 0:
            nparticles = dataset.nparticles
            outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
            x = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
            y = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
            x = np.log(x)
            x -= np.mean(x)
            y -= np.mean(y)
            ax.scatter(x, y, label=int(nparticles), color=color_cycle.next())
            fit = np.polyfit(x, y, 1)
            fit_fn = np.poly1d(fit)
            ax.plot(x, fit_fn(x), color='k')
            dataset.add_extras(fit)
    ax.legend(frameon=False, loc="best")
    plt.ylabel(r"$F - \langle F \rangle$")
    plt.xlabel(r"$\log P - \langle \log P \rangle$")
    
    def ff(x, a):
        return a * x
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y, y2 = [], [], []
    for dataset in sorted(packing_datasets, key=lambda data: data.nparticles):
        if len(dataset.free_energies) > 0:
            y.append(dataset.extras[0][0])
            y2.append(dataset.extras[0][1])
            x.append(dataset.nparticles)
    x, y, y2 = np.array(x), np.array(y), np.array(y2)
    ax.plot(x,y, marker='o', ms=9)
    popt, pcov = curve_fit(ff, x, y)
    print popt
    ax.plot(x, ff(x, popt[0]), label="F/log P = N/{:.3f}".format(1./popt[0]))
    ax.legend(frameon=False, loc="best")
    plt.xlabel(r"$ N $")
    plt.ylabel(r"$F/\log P$ (power law exponent)")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y2, marker='o', ms=9)
    popt, pcov = curve_fit(ff, x, y2)
    print popt
    ax.plot(x, ff(x, popt[0]), label="q = {:.3f}N".format(popt[0]))
    ax.legend(frameon=False, loc="best")
    plt.xlabel(r"$ N $")
    plt.ylabel(r"q (power law constant)")
    
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    for i,dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
#        nparticles = dataset.nparticles
#        if len(dataset.free_energies) > 0:
#            outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
#            x = np.array(dataset.contacts)[np.array(outliers.non_outliers_indexes, dtype="i")]
#            y = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
#            x = np.log(x)
#            ax.scatter(x, y, label=int(nparticles), color=color_cycle.next())
#            fit = np.polyfit(x, y,1)
#            print fit
#            fit_fn = np.poly1d(fit)
#            ax.plot(x, fit_fn(x), color='k')
#            #dataset.add_extras(fit)
#    ax.legend(frameon=False, loc="best")
#    plt.xlabel(r"$\log(z-z_{iso})$")
#    plt.ylabel(r"$F$")
    
    plt.show()        
        
#        #THESE ARE JUST QUICK PLOTS, CLEAN THIS UP AND PUTH EVERYTHING IN APPROPRIATE FUNCTIONS
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.scatter(self.free_energies, np.log(self.pressures))
#        plt.xlabel("F")
#        plt.ylabel("lnP")
#        plt.show()
#
#        from basinvolume.experiment_2d.cross_validation_bandwidth_selection import get_bandwidth_estimate, get_pdf
#        #kde pressures
#        bw = get_bandwidth_estimate(np.array(self.pressures), kernel="gaussian", method="cross_validation")
#        edges = np.linspace(np.amin(self.pressures), np.amax(self.pressures), 1000)
#        hist = get_pdf(self.pressures, edges, bandwidth=bw, kernel="gaussian")
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.plot(edges, hist)
#        plt.xlabel("P")
#        plt.show()
#
#        #kde free energies
#        free_energies = np.array([f for f in self.free_energies if f is not None])
#        bw = get_bandwidth_estimate(np.array(free_energies), kernel="gaussian", method="cross_validation")
#        edges = np.linspace(np.amin(free_energies), np.amax(free_energies), 100)
#        hist = get_pdf(free_energies, edges, bandwidth=bw, kernel="gaussian")
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.plot(edges, hist)
#        plt.xlabel("F")
#        plt.show()
        
        

if __name__ == "__main__":
    pts = BasinAnalysis()
    pts.collect_data_all_set()
    plot(pts.packing_datasets)