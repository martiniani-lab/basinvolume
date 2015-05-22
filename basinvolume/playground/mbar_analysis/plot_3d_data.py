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
color_cycle=cycle([cm(1. * i / 6) for i in xrange(6)])
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
            x = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
            y = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
            ax.scatter(x, np.log(y), label=int(nparticles), color=color_cycle.next())
            fit = np.polyfit(x, np.log(y),1)
            fit_fn = np.poly1d(fit)
            ax.plot(x, fit_fn(x), color='k')
            dataset.add_extras(fit)
    ax.legend(frameon=False, loc="best")
    plt.xlabel(r"$F$")
    plt.ylabel(r"$\log P$")
    
    def ff(x, a):
        return a / x
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = [], []
    for dataset in sorted(packing_datasets, key=lambda data: data.nparticles):
        if len(dataset.free_energies) > 0:
            y.append(dataset.extras[0][0])
            x.append(dataset.nparticles)
    x, y = np.array(x), np.array(y)
    ax.plot(x,y, marker='o', ms=9)
    #w = 1/np.array([data.nparticles for data in sorted(packing_datasets, key=lambda data: data.nparticles) if len(data.free_energies) > 0])
    popt, pcov = curve_fit(ff, x, y)
    print popt
    ax.plot(x, ff(x, popt[0]), label="y = {:.3f} / N".format(popt[0]))
    ax.legend(frameon=False, loc="best")
    plt.xlabel(r"$ N $")
    plt.ylabel("F vs logP power law exponent")
    
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