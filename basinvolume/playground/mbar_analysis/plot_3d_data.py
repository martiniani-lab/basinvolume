from __future__ import division
from basinvolume.utils._utils import _sort_pair
from scipy.misc import factorial
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

def plot(packing_datasets, figdir="figures"):
    from scipy.optimize import curve_fit
    if not os.path.isabs(figdir):
        figdir = os.path.join(os.getcwd(), figdir)
    trymakedir(figdir)
    
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
                weights = 1./np.array(dataset.free_energies_err)[np.array(outliers.non_outliers_indexes, dtype="i")]
                print len(x), len(y)
                x = np.log(x)
                ax.scatter(x, y, label=int(nparticles), color=color_cycle.next())
                fit, cov = np.polyfit(x, y, 1, w=weights, cov=True)
                fit_err = np.sqrt(np.diag(cov))
                fit_fn = np.poly1d(fit)
                ax.plot(x, fit_fn(x), color='k')
                dataset.add_extras((fit, fit_err))
                print dataset.extras
        ax.legend(frameon=False, loc="best")
        plt.ylabel(r"$F$")
        plt.xlabel(r"$\log P$")
        fig.savefig('{0}/plot_{1}.pdf'.format(figdir, "f_logp"))
        print "extras", dataset.extras
        
        #plot F-<F> vs P-<P> for all packings
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i,dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
            if len(dataset.free_energies) > 0:
                nparticles = dataset.nparticles
                outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
                x = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
                y = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
                weights = 1./np.array(dataset.free_energies_err)[np.array(outliers.non_outliers_indexes, dtype="i")]
                x = np.log(x)
                x -= np.mean(x)
                y -= np.mean(y)
                ax.scatter(x, y, label=int(nparticles), color=color_cycle.next())
                fit = np.polyfit(x, y, 1, w=weights)
                fit_fn = np.poly1d(fit)
                ax.plot(x, fit_fn(x), color='k')
        ax.legend(frameon=False, loc="best")
        plt.ylabel(r"$F - \langle F \rangle$")
        plt.xlabel(r"$\log P - \langle \log P \rangle$")
        
        def ff(x, a):
            return a * x
        
        #plot power law exponent
        
        color_marker = color_cycle.next()
        color_fit = color_cycle.next()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y, yerr, y2, y2err = [], [], [], [], []
        for dataset in sorted(packing_datasets, key=lambda data: data.nparticles):
            if len(dataset.free_energies) > 0:
                y.append(dataset.extras[0][0])
                yerr.append(dataset.extras[1][0])
                y2.append(dataset.extras[0][1])
                y2err.append(dataset.extras[1][1])
                x.append(dataset.nparticles)
        x, y, yerr, y2, y2err = np.array(x), np.array(y), np.array(yerr), np.array(y2), np.array(y2err)
        
        ax.errorbar(x,y, yerr, marker='o', linestyle='', ms=9, color=color_marker)
        popt, pcov = curve_fit(ff, x, y, sigma=yerr, absolute_sigma=True)
        print "1/kappa {:.16f}".format(popt[0])
        glob_kappa = 1./popt[0]
        ax.plot(x, ff(x, popt[0]), label="exponent = N/({:.3f} +/- {:.3f})".format(1./popt[0], np.sqrt(float(pcov[0]))), color=color_fit)
        ax.legend(frameon=False, loc="best")
        plt.xlabel(r"$ N $")
        plt.ylabel(r"power law exponent")
        fig.savefig('{0}/plot_{1}.pdf'.format(figdir, "exponent"))
        
        #plot power law intercept
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(x,y2, y2err, marker='o', linestyle='', ms=9, color=color_marker)
        popt, pcov = curve_fit(ff, x, y2, sigma=y2err, absolute_sigma=True)
        print "intercept {:.16f}".format(popt[0])
        glob_interc = popt[0]
        ax.plot(x, ff(x, popt[0]), label="intercept = ({:.3f} +/- {:.3f})N".format(popt[0], np.sqrt(float(pcov[0]))), color=color_fit)
        ax.legend(frameon=False, loc="best")
        plt.xlabel(r"$ N $")
        plt.ylabel(r"power law intercept")
        fig.savefig('{0}/plot_{1}.pdf'.format(figdir, "intercept"))
    
        
    from basinvolume.experiment_2d.cross_validation_bandwidth_selection import get_bandwidth_estimate, get_pdf
    from basinvolume.post_processing import GeneralisedLogNormal, OutlierRemovalUnbiasingEntropyLogOmega
    if True:
        #kde pressure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        fit_parameters = []
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
                #assume that zeta_min=2
                generalised_lognormal = GeneralisedLogNormal(alpha_min=0.0001, zeta_min=1.0)
                cdf = CDFAccumulator()
                cdf.add_array(x)
                x, cdf_x = cdf.get_vecdata()
                generalised_lognormal.fit_cdf(x, cdf_x, initial_zeta=2)
                ax.plot(edges, [generalised_lognormal.get_fitted(xpi) for xpi in edges], "--", color=color, linewidth=2)
                parameters = [nparticles, generalised_lognormal.mu_fit, generalised_lognormal.alpha_fit, generalised_lognormal.zeta_fit]
                print "mu, alpha, zeta:", parameters
                fit_parameters.append(parameters)
                #plot cdf
                ax2.plot(x, 1-np.array(cdf_x), label=int(nparticles), color=color, linewidth=3)
                ax2.plot(edges, 1-generalised_lognormal.get_cdf(edges, generalised_lognormal.mu_fit, 
                                                              generalised_lognormal.alpha_offset, 
                                                              generalised_lognormal.zeta_offset), '--', color=color, linewidth=2)
                #unbiased pdf
                F0acc = Bunch(F0_acc=dataset.packing_data[0].Facc, nr_particles=nparticles)
#                log_omega = OutlierRemovalUnbiasingEntropyLogOmega(F0, "/scratch/sm958/Results/basinvolume_tests", write=False)
#                log_omega.compute_log_omega_entropy(F0acc)
#                avgv, avgv_error = log_omega.integral_no_jack, log_omega.integral_error
#                print "avgv, avgerror ", avgv, avgv_error
#                print "S*", log_omega.S_star
                xp = np.linspace(1, np.amax(x) * 100, 1e5)
                fit = np.array([generalised_lognormal.get_fitted_times_xpow(xpi, glob_kappa, nparticles) for xpi in xp])
                c, cerr = quad(generalised_lognormal.get_fitted_times_xpow, 1e-5, np.amax(x) * 100, args=(glob_kappa, nparticles), 
                         points=[np.amin(x), np.amax(x), xp[np.argmax(fit)]]) 
                c *= np.exp(nparticles)*nparticles/glob_kappa
                print "avgs, avg_err from pressure", c, cerr
                ax3.plot(xp, fit/c, label=int(nparticles), color=color, linewidth=3)
                #S(V, P)
                maxps = 1e7
                minps = 10
                ps = np.linspace(minps, maxps, 3e5)
                dps = (maxps-minps)/3e5
                #vcavity = dataset.packing_data[0].vcavity
                print "S", - F0acc.F0_acc + np.log(c) - log_factorial(nparticles)
                S = - F0acc.F0_acc + nparticles + np.log(nparticles/glob_kappa) + np.array([generalised_lognormal.get_log_fitted_times_xpow(p, glob_kappa, nparticles) for p in ps]) + np.log(dps)
                S  -= log_factorial(nparticles)
                #S = - F0acc.F0_acc - np.log(c) + np.array([generalised_lognormal.get_log_fitted_times_xpow(np.exp(p), glob_kappa, nparticles) + p for p in ps])
                ax4.plot(ps, S, label=int(nparticles), color=color, linewidth=3)
        fit_parameters = np.array(fit_parameters)
        ax.legend(frameon=False, loc="best")
        ax2.legend(frameon=False, loc="best")
        ax3.legend(frameon=False, loc="best")
        ax4.legend(frameon=False, loc="best")
        ax.set_ylabel(r"$p(P)$")
        ax.set_xlabel(r"$P$")
        ax2.set_ylabel(r"$cdf(P)$")
        ax2.set_xlabel(r"$P$")
        ax2.set_xscale('log')
        ax3.set_ylabel(r"unbiased $p(P)$")
        ax3.set_xlabel(r"$P$")
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_ylabel(r"$S(V,P)$")
        ax4.set_xlabel(r"$P$")
        ax2.set_xlim((100,1e5))
        fig2.savefig('{0}/plot_{1}.pdf'.format(figdir, "p_cdf_fit"))
        fig4.savefig('{0}/plot_{1}.pdf'.format(figdir, "p_S"))
        
        #plot fit parameters
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        #mu plot
        color = color_cycle.next()
        ax5.plot(1./fit_parameters[:,0], fit_parameters[:,1], marker='o', linestyle='', ms=9, color=color, label=r'$\mu$')
        fit = np.polyfit(1./fit_parameters[:,0], fit_parameters[:,1], 1)
        fit_fn = np.poly1d(fit)
        ax5.plot(1./fit_parameters[:,0], fit_fn(1./fit_parameters[:,0]), linestyle='--', color=color, linewidth=2.0, label=str(fit))
        #alpha plot
        color = color_cycle.next()
        ax5.plot(1./fit_parameters[:,0], fit_parameters[:,2], marker='o', linestyle='', ms=9, color=color, label=r'$\alpha$')
        fit = np.polyfit(1./fit_parameters[:,0], fit_parameters[:,2], 1)
        fit_fn = np.poly1d(fit)
        ax5.plot(1./fit_parameters[:,0], fit_fn(1./fit_parameters[:,0]), linestyle='--', color=color, linewidth=2.0, label=str(fit))
        #zeta plot
        color = color_cycle.next()
        ax5.plot(1./fit_parameters[:,0], fit_parameters[:,3], marker='o', linestyle='', ms=9, color=color, label=r'$\zeta$')
        fit = np.polyfit(1./fit_parameters[:,0], fit_parameters[:,3], 1)
        fit_fn = np.poly1d(fit)
        ax5.plot(1./fit_parameters[:,0], fit_fn(1./fit_parameters[:,0]), linestyle='--', color=color, linewidth=2.0, label=str(fit))
        ax5.legend(frameon=False, loc="best")
        ax5.set_xlabel(r"$N$")
        fig5.savefig('{0}/plot_{1}.pdf'.format(figdir, "p_cdf_fit_parameters"))
        
    if True:
        #kde free energies predicted vs numerical
        #note the x2 = (nparticles/glob_kappa)*np.log(x2)+glob_interc*nparticles (should be just +N)
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
                x2 = (nparticles/glob_kappa)*np.log(x2) + glob_interc*nparticles
                bw2 = get_bandwidth_estimate(np.array(x2), kernel="gaussian", method="cross_validation")
                edges2 = np.linspace(np.amin(x2)*0.5, np.amax(x2)*1.5, 1000)
                hist2 = get_pdf(x2, edges2, bandwidth=bw2, kernel="gaussian")
                #hist2 *= edges2/(nparticles/glob_kappa)
                #hist2 /= simps(hist2, x=edges2)
                ax.plot(edges2, hist2, '--', color=color, linewidth=3)
        
        ax.legend(frameon=False, loc="best")
        plt.ylabel(r"$p(F)$")
        plt.xlabel(r"$F$")
        ax.set_xlim((30,340))
        fig.savefig('{0}/plot_{1}.pdf'.format(figdir, "predicted_F_dist"))
    
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