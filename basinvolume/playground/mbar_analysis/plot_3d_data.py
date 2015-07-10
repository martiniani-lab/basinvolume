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
    from scipy.optimize import fmin
    from scipy.stats import t
    from scipy.interpolate import spline
    from scipy.integrate import romberg, simps, quad, cumtrapz, trapz
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
plt.rcParams.update({'font.size': 18})
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams.update({'figure.autolayout': True})
##########################################################
####SET COLOUR MAP######                                                               
def get_color_cycle():
    cm = plt.get_cmap('Set2')
    color_cycle=cycle([cm(1. * i / 7) for i in xrange(7)])
    return color_cycle
########################
#####################LINE STYLE CYCLER####################                             
lines = ["-","--","-.", ":", "_"]
linecycler = cycle(lines)
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
        color_cycle = get_color_cycle()
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
        ax.legend(frameon=False, loc='center', prop={'size':18}, numpoints=1, scatterpoints=1, markerscale=1, columnspacing=0.25, labelspacing=0.25,
                  handletextpad=0, bbox_to_anchor=[0.08, 0.3])
        plt.ylabel(r"$F$")
        plt.xlabel(r"$\log \mathcal{P}$")
        print "extras", dataset.extras
        
        def ff(x, a):
            return a * x
        
        #plot power law exponent
        color_cycle = get_color_cycle()
        color_marker = color_cycle.next()
        color_fit = color_cycle.next()
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        ax3 = fig.add_axes([0.1,0.695,0.27,0.27], alpha=0.5)
               
        x, y, yerr, y2, y2err = [], [], [], [], []
        for dataset in sorted(packing_datasets, key=lambda data: data.nparticles):
            if len(dataset.free_energies) > 0:
                y.append(dataset.extras[0][0])
                yerr.append(dataset.extras[1][0])
                y2.append(dataset.extras[0][1])
                y2err.append(dataset.extras[1][1])
                x.append(dataset.nparticles)
        x, y, yerr, y2, y2err = np.array(x), np.array(y), np.array(yerr), np.array(y2), np.array(y2err)
        
        ax3.errorbar(x,y, yerr, marker='o', linestyle='', ms=9, color=color_marker)
        popt, pcov = curve_fit(ff, x, y, sigma=yerr, absolute_sigma=True)
        print "1/kappa {:.16f}".format(popt[0])
        glob_kappa = 1./popt[0]
        ax3.plot(x, ff(x, popt[0]), label="exponent = N/({:.3f} +/- {:.3f})".format(1./popt[0], np.sqrt(float(pcov[0]))), color=color_fit)
        ax3.legend(frameon=False, loc="best", framealpha=0.5, prop={'size':8}, labelspacing=0.25, 
                   columnspacing=0.25, numpoints=1, markerscale=0.5, handlelength=0.4)
        ax3.locator_params(axis = 'x', nbins = 4)
        ax3.locator_params(axis = 'y', nbins = 4)
        ax3.tick_params(axis='both', which='major', labelsize=10)
        #plt.xlabel(r"$ N $")
        #plt.ylabel(r"power law exponent")
        
        #fig.savefig('{0}/plot_{1}.pdf'.format(figdir, "exponent"))
        
        #plot power law intercept
        
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        ax4 = fig.add_axes([0.71,0.695,0.27,0.27], alpha=0.5)
        ax4.errorbar(x,y2, y2err, marker='o', linestyle='', ms=9, color=color_marker)
        popt, pcov = curve_fit(ff, x, y2, sigma=y2err, absolute_sigma=True)
        print "intercept {:.16f}".format(popt[0])
        glob_interc = popt[0]
        ax4.plot(x, ff(x, popt[0]), label="intercept = ({:.3f} +/- {:.3f})N".format(popt[0], np.sqrt(float(pcov[0]))), color=color_fit)
        ax4.legend(frameon=False, loc="best", framealpha=0.5, prop={'size':8}, labelspacing=0.25, 
                   columnspacing=0.25, numpoints=1, markerscale=0.5, handlelength=0.4)
        ax4.locator_params(axis = 'x', nbins = 4)
        ax4.locator_params(axis = 'y', nbins = 4)
        ax4.tick_params(axis='both', which='major', labelsize=10)
        #plt.xlabel(r"$ N $")
        #plt.ylabel(r"power law intercept")
        #fig.savefig('{0}/plot_{1}.pdf'.format(figdir, "intercept"))
        
        xticks = ax.yaxis.get_major_ticks()
        xticks[-3].label1.set_visible(False)
        xticks[-2].label1.set_visible(False)
        xticks[-1].label1.set_visible(False)
        fig.savefig('{0}/plot_{1}.pdf'.format(figdir, "f_logp"))
    
    from basinvolume.experiment_2d.cross_validation_bandwidth_selection import get_bandwidth_estimate, get_pdf
    from basinvolume.post_processing import GeneralisedLogNormal, OutlierRemovalUnbiasingEntropyLogOmega
    if False:
        #kde pressure
        color_cycle = get_color_cycle()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        fig7 = plt.figure()
        ax7 = fig7.add_subplot(111)
        fit_parameters = []
        fit_parameters_std = []
        s_maxima = []
        s_p_maxima = []
        s_apf = []
        s_edw = []
        s_b = []
        p_ens_avg = []
        for i,dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
            nparticles = dataset.nparticles
            vcavity = dataset.packing_data[0].vcavity
            print "vcavity", vcavity
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
                fit_parameters_std.append(generalised_lognormal.fit_error.tolist())
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
#                Sedw = log_omega.S_star - log_factorial(nparticles)
#                print "old avgv, avgerror ", avgv, avgv_error
#                print "Sedw", Sedw
#                Sapf = np.mean(F0) - F0acc['F0_acc'] - log_factorial(nparticles)
#                s_edw.append([nparticles, Sedw])
#                s_apf.append([nparticles, Sapf])
#                print "Sapf ", Sapf
                xp = np.linspace(1, np.amax(x) * 100, 1e5)
                fit = np.array([generalised_lognormal.get_fitted_times_xpow(xpi, glob_kappa, nparticles) for xpi in xp])
                c, cerr = quad(generalised_lognormal.get_fitted_times_xpow, 0, np.amax(x)*1e5, args=(glob_kappa, nparticles), limit=200, 
                               points=[np.amin(x), np.amax(x), xp[np.argmax(fit)]], epsabs=1.49e-11) 
                c *= np.exp(nparticles)
                Sb = - F0acc.F0_acc + np.log(c) - log_factorial(nparticles)
                print "new avgs, avg_err from pressure", c, cerr
                print "Sb ",Sb 
                s_b.append([nparticles, Sb])
                ax3.plot(xp, fit/c, label=int(nparticles), color=color, linewidth=3)
#                test = simps(fit, xp)/c
#                print "test", test
#                #compute p ensemble average
#                pea_array = []
#                ang_array = np.linspace(-0.5,0.5,num=25)
#                for ang in ang_array:
##                    Pea, Pea_err = quad(weighted_pressure, 0, np.inf, args=(ang, 1), epsabs=1.49e-11)
##                    zPea, zPea_err = quad(weighted_pressure_norm, 0, np.inf, args=(ang, 1), epsabs=1.49e-11)
##                    Pea = np.log10(Pea)-np.log10(zPea) #normalise in log
#                    print ang
#                    Pea = generalised_lognormal.get_log_edwards_fitted_pressure_expectation(glob_kappa, nparticles, ang, vcavity)
#                    pea_array.append([Pea, 0])
#                pea_array = np.array(pea_array)
#                ax7.plot(ang_array, pea_array[:,0], label=int(nparticles), color=color, linewidth=3)
#                print "ensemble average P, Perr", nparticles, pea_array
                #S(V, P)
                maxps = 2e6
                minps = 10
                nps = 5e5
                ps = np.linspace(minps, maxps, nps)
                dps = (maxps-minps)/nps
                S = - F0acc.F0_acc + nparticles + np.array([generalised_lognormal.get_log_fitted_times_xpow(p, glob_kappa, nparticles) for p in ps]) + np.log(dps)
                S  -= log_factorial(nparticles)
                ax4.plot(ps, S, label=int(nparticles), color=color, linewidth=3)
                s_maxima.append([nparticles, np.amax(S)])
                s_p_maxima.append([nparticles, ps[np.argmax(S)]])
                
                 
        fit_parameters = np.array(fit_parameters)
        fit_parameters_std = np.array(fit_parameters_std)
        s_maxima = np.array(s_maxima)
        s_apf = np.array(s_apf)
        s_edw = np.array(s_edw)
        s_b = np.array(s_b)
        s_p_maxima = np.array(s_p_maxima)
        ax.legend(frameon=False, loc="best", numpoints=1, markerscale=0.5, columnspacing=0.25, labelspacing=0.25)
        ax2.legend(frameon=False, loc=2, prop={'size':18}, numpoints=1, markerscale=0.5, columnspacing=0.25, labelspacing=0.25, handlelength=1)
        ax3.legend(frameon=False, loc="best", numpoints=1, markerscale=0.5, columnspacing=0.25, labelspacing=0.25)
        ax4.legend(frameon=False, loc=2, prop={'size':18}, numpoints=1, markerscale=0.5, columnspacing=0.25, labelspacing=0.25, handlelength=1)
        ax7.legend(frameon=False, loc=2, prop={'size':18}, numpoints=1, markerscale=0.5, columnspacing=0.25, labelspacing=0.25, handlelength=1)
        ax.set_ylabel(r"$p(P)$")
        ax.set_xlabel(r"$\mathcal{P}$")
        ax2.set_ylabel(r"$c.d.f.(\mathcal{P})$", fontsize=18)
        ax2.set_xlabel(r"$\mathcal{P}$", fontsize=18)
        ax2.set_xscale('log')
        ax2.set_xlim((100,1e5))
        ax3.set_ylabel(r"unbiased $p(\mathcal{P})$")
        ax3.set_xlabel(r"$P$")
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        #ax4.set_ylim(ymin=0)
        ax4.set_ylabel(r"$S(V,\mathcal{P})$")
        ax4.set_xlabel(r"$\mathcal{P}$")
        ax7.set_ylabel(r"$\alpha$")
        ax7.set_ylabel(r"$ \langle p \rangle_{ens}$")
        #ax7.set_xscale('symlog')
        #ax7.set_yscale('log')
        
        #plot fit parameters
        #fig5 = plt.figure()
        ax5 = fig2.add_axes([0.65,0.25,0.27,0.27], alpha=0.5)
        color_cycle = get_color_cycle()
        #mu plot
        color = color_cycle.next()
        ax5.errorbar(1./fit_parameters[:,0], fit_parameters[:,1], fit_parameters_std[:,0], marker='o', linestyle='', ms=9, color=color, label=r'$\mu$')
        fit, cov = np.polyfit(1./fit_parameters[:,0], fit_parameters[:,1], 1, w=1./fit_parameters_std[:,0], cov=True)
        fit_fn = np.poly1d(fit)
        fit_std = np.sqrt(np.diag(cov))
        ax5.plot(1./fit_parameters[:,0], fit_fn(1./fit_parameters[:,0]), linestyle='--', color=color, linewidth=2.0, label=str(fit)+"+/-"+str(fit_std))
        #alpha plot
        color = color_cycle.next()
        ax5.errorbar(1./fit_parameters[:,0], fit_parameters[:,2], fit_parameters_std[:,1], marker='o', linestyle='', ms=9, color=color, label=r'$\sigma$')
        fit, cov = np.polyfit(1./fit_parameters[:,0], fit_parameters[:,2], 1, w=1./fit_parameters_std[:,1], cov=True)
        fit_std = np.sqrt(np.diag(cov))
        fit_fn = np.poly1d(fit)
        ax5.plot(1./fit_parameters[:,0], fit_fn(1./fit_parameters[:,0]), linestyle='--', color=color, linewidth=2.0, label=str(fit)+"+/-"+str(fit_std))
        #zeta plot
        color = color_cycle.next()
        ax5.errorbar(1./fit_parameters[:,0], fit_parameters[:,3], fit_parameters_std[:,2], marker='o', linestyle='', ms=9, color=color, label=r'$\zeta$')
        fit, cov = np.polyfit(1./fit_parameters[:,0], fit_parameters[:,3], 1, w=1./fit_parameters_std[:,2], cov=True)
        fit_fn = np.poly1d(fit)
        fit_std = np.sqrt(np.diag(cov))
        ax5.plot(1./fit_parameters[:,0], fit_fn(1./fit_parameters[:,0]), linestyle='--', color=color, linewidth=2.0, label=str(fit)+"+/-"+str(fit_std))
        ax5.legend(frameon=False, loc="best", framealpha=0.5, prop={'size':8}, labelspacing=0.25, 
                   columnspacing=0.25, numpoints=1, markerscale=0.5, handlelength=0.4)
        plt.rc('font', size=12)
        ax5.set_xlabel(r"$1/N$", fontsize=12)
        ax5.set_ylabel("\t")
        ax5.ticklabel_format(style='sci',axis='x', scilimits=(0,0))
        #fig5.savefig('{0}/plot_{1}.pdf'.format(figdir, "p_cdf_fit_parameters"))
        ax5.locator_params(axis = 'x', nbins = 4)
        ax5.locator_params(axis = 'y', nbins = 4)
        ax5.tick_params(axis='both', which='major', labelsize=10)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        fig2.savefig('{0}/plot_{1}.pdf'.format(figdir, "p_cdf_fit"))
        plt.rc('font', size=18)
        
        #plot fit parameters
        #fig6 = plt.figure()
        #ax6 = fig6.add_subplot(111)
        #inset
        color_cycle = get_color_cycle()
        ax6=fig4.add_axes([0.2,0.25,0.27,0.27], alpha=0.5)
        color = color_cycle.next()
        ax6.plot(s_maxima[:,0], s_maxima[:,1], marker='o', linestyle='', ms=9, color=color, label=r"$\max(S_B(\mathcal{P},V))$")
        fit = np.polyfit(s_maxima[:,0], s_maxima[:,1], 1)
        fit_fn = np.poly1d(fit)
        ax6.plot(np.linspace(0,np.amax(s_maxima[:,0])), fit_fn(np.linspace(0,np.amax(s_maxima[:,0]))), linestyle='--', linewidth=2.0, color=color)
        #ax6.plot(s_p_maxima[:,0], s_p_maxima[:,1], marker='o', linestyle='', ms=9, color=color, label=r"$\max_P(S_B(\mathcal{P},V))$")
        color = color_cycle.next()
        ax6.plot(s_b[:,0], s_b[:,1], marker='^', linestyle='', color=color, ms=9, label=r'$S_{B}$')
        fit = np.polyfit(s_b[:,0], s_b[:,1], 1)
        fit_fn = np.poly1d(fit)
        ax6.plot(np.linspace(0,np.amax(s_b[:,0])), fit_fn(np.linspace(0,np.amax(s_b[:,0]))), linestyle='--', linewidth=2.0, color=color)
        ax6.set_xlabel(r"$N$", fontsize=12)
        #ax6.set_ylabel(r"$\max_S(S_B(\mathcal{P},V))$")
        ax6.legend(frameon=False, loc="best", framealpha=0.5, prop={'size':12}, labelspacing=0.25, columnspacing=0.25, numpoints=1, markerscale=0.5)
        #fig6.savefig('{0}/plot_{1}.pdf'.format(figdir, "p_entropy_maxima"))
        ax6.locator_params(axis = 'x', nbins = 4)
        ax6.locator_params(axis = 'y', nbins = 4)
        ax6.tick_params(axis='both', which='major', labelsize=10)
        fig4.savefig('{0}/plot_{1}.pdf'.format(figdir, "p_S"))
    
    if False:
        #kde free energies predicted vs numerical
        #note the x2 = (nparticles/glob_kappa)*np.log(x2)+glob_interc*nparticles (should be just +N)
        color_cycle = get_color_cycle()
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
        
        ax.legend(frameon=False, loc="best", numpoints=1, markerscale=0.5, columnspacing=0.25, labelspacing=0.25, handlelength=0.4)
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
        ax.legend(frameon=False, loc="best", numpoints=1, markerscale=0.5, columnspacing=0.25, labelspacing=0.25)
        plt.ylabel(r"$F$")
        plt.xlabel(r"$\log \mathcal{P}$")    

if __name__ == "__main__":
    pts = BasinAnalysis()
    pts.collect_data_every_set_all()
    plot(pts.packing_datasets)
    plt.show()
    plt.close()