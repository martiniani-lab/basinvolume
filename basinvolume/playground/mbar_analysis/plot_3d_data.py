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
    from scipy.integrate import simps, quad
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
    
    if True:
        #plot free energy vs P for all packings
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
                fit = np.polyfit(x, y, 1)
                fit_fn = np.poly1d(fit)
                ax.plot(x, fit_fn(x), color='k')
                dataset.add_extras(fit)
        ax.legend(frameon=False, loc="best")
        plt.ylabel(r"$F$")
        plt.xlabel(r"$\log P$")
        
        #plot F-<F> vs P-<P> for all packings
        
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
        ax.plot(x, ff(x, popt[0]), label="F/log P = N/{:.3f}".format(1./popt[0]))
        ax.legend(frameon=False, loc="best")
        plt.xlabel(r"$ N $")
        plt.ylabel(r"$F/\log P$ (power law exponent)")
        
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
    from basinvolume.post_processing import GeneralisedLogNormal, OutlierRemovalUnbiasingEntropyLogOmega
    if True:
        #kde gammas
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        for i,dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
            nparticles = dataset.nparticles
            if len(dataset.free_energies) > 0:
                print "n:", nparticles
                outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
                F0 = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
                x = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
                #kde histogram
                bw = get_bandwidth_estimate(np.array(x), kernel="gaussian", method="cross_validation")
                edges = np.linspace(np.amin(x)*0.5, np.amax(x)*1.5, 1000)
                hist = get_pdf(x, edges, bandwidth=bw/2, kernel="gaussian")
                color = color_cycle.next()
                ax.plot(edges, hist, label=int(nparticles), color=color, linewidth=3)
                #fit log normal
                generalised_lognormal = GeneralisedLogNormal(alpha_min=0.0001, zeta_min=1.2)
                cdf = CDFAccumulator()
                cdf.add_array(x)
                x, cdf_x = cdf.get_vecdata()
                generalised_lognormal.fit_cdf(x, cdf_x)
                ax.plot(edges, [generalised_lognormal.get_fitted(xpi) for xpi in edges], "--", color=color, linewidth=2)
                print "mu, alpha, zeta:", [generalised_lognormal.mu_fit, generalised_lognormal.alpha_fit, generalised_lognormal.zeta_fit]
                #plot cdf
                ax2.plot(x, 1-np.array(cdf_x), label=int(nparticles), color=color, linewidth=3)
                ax2.plot(edges, 1-generalised_lognormal.get_cdf(edges, generalised_lognormal.mu_fit, 
                                                              generalised_lognormal.alpha_offset, 
                                                              generalised_lognormal.zeta_offset), '--', color=color, linewidth=2)
                #unbiased pdf
#                log_omega = OutlierRemovalUnbiasingEntropyLogOmega(F0, "/scratch/sm958/Results/basinvolume_tests", write=False)
#                F0acc = Bunch(F0_acc=dataset.packing_data[0].Facc, nr_particles=nparticles)
#                log_omega.compute_log_omega_entropy(F0acc)
#                avgv, avgv_error = log_omega.integral_no_jack, log_omega.integral_error
#                print "avgv, avgerror ", avgv, avgv_error
                xp = np.linspace(1, np.amax(x) * 100, 1e5)
                fit = np.array([generalised_lognormal.get_fitted_times_expx(xpi, 5.297, nparticles) for xpi in xp])
                c, cerr = quad(generalised_lognormal.get_fitted_times_expx, 1e-5, np.amax(x) * 100, args=(5.297, nparticles), 
                         points=[np.amin(x), np.amax(x), xp[np.argmax(fit)]]) 
                c *= np.exp(nparticles)*nparticles/5.297
                print "avgs2, avg2_err", c, cerr
                ax3.plot(xp, fit/c, label=int(nparticles), color=color, linewidth=3)
                
        ax.legend(frameon=False, loc="best")
        ax2.legend(frameon=False, loc="best")
        ax3.legend(frameon=False, loc="best")
        ax.set_ylabel(r"$p(P)$")
        ax.set_xlabel(r"$P$")
        ax2.set_ylabel(r"$cdf(P)$")
        ax2.set_xlabel(r"$P$")
        ax2.set_xscale('log')
        ax3.set_ylabel(r"unbiased $p(P)$")
        ax3.set_xlabel(r"$P$")
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
    if False:
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
                color = color_cycle.next()
                ax.plot(edges, hist, label=int(nparticles), color=color, linewidth=3)
                
                x2 = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
                x2 = (nparticles*0.18878315)*np.log(x2) + 0.9807471*nparticles
                bw2 = get_bandwidth_estimate(np.array(x2), kernel="gaussian", method="cross_validation")
                edges2 = np.linspace(np.amin(x2)*0.5, np.amax(x2)*1.5, 1000)
                hist2 = get_pdf(x2, edges2, bandwidth=bw2, kernel="gaussian")
                #hist2 *= edges2/(nparticles*0.18878315)
                #hist2 /= simps(hist2, x=edges2)
                ax.plot(edges2, hist2, '--', color=color, linewidth=3)
        
        ax.legend(frameon=False, loc="best")
        plt.ylabel(r"$p(F)$")
        plt.xlabel(r"$F$")
    
    if False:
        #fit the (N/k)\log(P)+N model to each and compute the average of the kappas
          
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
                ax.plot(x, ff_msf(x, popt[0]), label="F = N/{:.3f}log(P) + N".format(1./popt[0]))
                avg += popt[0]
        avg /= (i+1)
        print "avg kappa = ", 1/avg
        ax.legend(frameon=False, loc="best")
        plt.ylabel(r"$F$")
        plt.xlabel(r"$\log P$")
    
        
    plt.show()    

if __name__ == "__main__":
    pts = BasinAnalysis()
    pts.collect_data_all_set()
    plot(pts.packing_datasets)
    plt.close()