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
    
    #plot free energy vs PV for all packings
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i,dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
        if len(dataset.free_energies) > 0:
            nparticles = dataset.nparticles
            print nparticles
            outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
            x = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
            y = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
            print len(x), len(y)
            x = np.log(x)
            ax.scatter(x, y, label=int(nparticles), color=color_cycle.next())
            fit = np.polyfit(x, y,1)
            fit_fn = np.poly1d(fit)
            ax.plot(x, fit_fn(x), color='k')
            dataset.add_extras(fit)
    ax.legend(frameon=False, loc="best")
    plt.ylabel(r"$F$")
    plt.xlabel(r"$\log PV$")
    
    #plot F-<F> vs PV-<PV> for all packings
    
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
    plt.xlabel(r"$\log PV - \langle \log PV \rangle$")
    
    def ff(x, a):
        return a * x
    
    #plot power law exponent
    
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
    ax.plot(x, ff(x, popt[0]), label="F/log PV = N/{:.3f}".format(1./popt[0]))
    ax.legend(frameon=False, loc="best")
    plt.xlabel(r"$ N $")
    plt.ylabel(r"$F/\log PV$ (power law exponent)")
    
    #plot power law intercept
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y2, marker='o', ms=9)
    popt, pcov = curve_fit(ff, x, y2)
    print popt
    ax.plot(x, ff(x, popt[0]), label="q = {:.3f}N".format(popt[0]))
    ax.legend(frameon=False, loc="best")
    plt.xlabel(r"$ N $")
    plt.ylabel(r"q (power law constant)")
    
        
    from basinvolume.experiment_2d.cross_validation_bandwidth_selection import get_bandwidth_estimate, get_pdf
    if True:
        #kde gammas
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i,dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
            if len(dataset.free_energies) > 0:
                nparticles = dataset.nparticles
                outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
                x = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
                bw = get_bandwidth_estimate(np.array(x), kernel="gaussian", method="cross_validation")
                edges = np.linspace(np.amin(x)*0.5, np.amax(x)*1.5, 1000)
                hist = get_pdf(x, edges, bandwidth=bw, kernel="gaussian")
                ax.plot(edges, hist, label=int(nparticles), color=color_cycle.next(), linewidth=3)
        ax.legend(frameon=False, loc="best")
        plt.ylabel(r"$p(\Gamma)$")
        plt.xlabel(r"$PV$")
        
        #kde unbiased gammas
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i,dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
            if len(dataset.free_energies) > 0:
                nparticles = dataset.nparticles
                outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
                x = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
                bw = get_bandwidth_estimate(np.array(x), kernel="gaussian", method="cross_validation")
                edges = np.linspace(0, 1.7e4, 5000)
                #unbias 
                hist = get_pdf(x, edges, bandwidth=bw, kernel="gaussian")
                hist *= (nparticles/5.0) * np.power(edges,(nparticles/5.0 - 1.0))
                c = simps(hist, x=edges) 
                ax.plot(edges, hist/c, label=int(nparticles), color=color_cycle.next(), linewidth=3)
        ax.legend(frameon=False, loc="best")
        plt.ylabel(r"$p(\Gamma)\frac{N}{\kappa}\Gamma^{(N-\kappa)/\kappa}$")
        plt.xlabel(r"$PV$")
        
    #kde free energies predicted vs numerical
    #note the x2 = (nparticles*0.18878315)*np.log(x2)+0.9807471*nparticles (should be just +N)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i,dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
        if len(dataset.free_energies) > 0:
            nparticles = dataset.nparticles
            outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
            x = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
            bw = get_bandwidth_estimate(np.array(x), kernel="gaussian", method="cross_validation")
            edges = np.linspace(np.amin(x)*0.5, np.amax(x)*1.5, 1000)
            hist = get_pdf(x, edges, bandwidth=bw, kernel="gaussian")
            ax.plot(edges, hist, label=int(nparticles), color=color_cycle.next(), linewidth=3)
            
            x2 = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
            x2 = (nparticles*0.18878315)*np.log(x2) + 0.9807471*nparticles
            bw2 = get_bandwidth_estimate(np.array(x2), kernel="gaussian", method="cross_validation")
            edges2 = np.linspace(np.amin(x2)*0.5, np.amax(x2)*1.5, 1000)
            hist2 = get_pdf(x2, edges2, bandwidth=bw2, kernel="gaussian")
            #hist2 *= edges2/(nparticles*0.18878315)
            #hist2 /= simps(hist2, x=edges2)
            ax.plot(edges2, hist2, '--', label=int(nparticles), color=color_cycle.next(), linewidth=3)
    
    ax.legend(frameon=False, loc="best")
    plt.ylabel(r"$p(F)$")
    plt.xlabel(r"$F$")
    
    if False:
        #fit the (N/k)\log(PV)+N model to each and compute the average of the kappas
          
        fig = plt.figure()
        ax = fig.add_subplot(111)
        avg = 0
        for i,dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
            if len(dataset.free_energies) > 0:
                nparticles = dataset.nparticles
                print nparticles
                outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
                x = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
                y = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
                print len(x), len(y)
                x = np.log(x)
                ax.scatter(x, y, color=color_cycle.next())
                def ff_msf(x, a):
                    return nparticles * (a * x + 1) 
                popt, pcov = curve_fit(ff_msf, x, y)
                ax.plot(x, ff_msf(x, popt[0]), label="F = N/{:.3f}log(PV) + N".format(1./popt[0]))
                avg += popt[0]
        avg /= (i+1)
        print "avg kappa = ", 1/avg
        ax.legend(frameon=False, loc="best")
        plt.ylabel(r"$F$")
        plt.xlabel(r"$\log PV$")
    
        
    plt.show()    

if __name__ == "__main__":
    pts = BasinAnalysis()
    pts.collect_data_all_set()
    plot(pts.packing_datasets)