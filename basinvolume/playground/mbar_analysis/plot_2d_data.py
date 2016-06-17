from __future__ import division

import numpy as np
import ConfigParser
import os
import re
import matplotlib.pyplot as plt
from matplotlib import rc
from itertools import cycle
from basinvolume.utils import *
from sklearn.neighbors import KernelDensity
from scipy import integrate
import scipy
from scipy.optimize import fmin
from scipy.stats import t
from scipy.interpolate import spline
from scipy.integrate import romberg, simps, quad, cumtrapz, trapz
import glob
from itertools import chain
import cPickle as pickle
from basinvolume.post_processing import PackingData, PackingDataSet, BasinAnalysis
from joblib import Parallel, delayed
from basinvolume.experiment_2d.cross_validation_bandwidth_selection import get_bandwidth_estimate, get_pdf
from basinvolume.post_processing import GeneralisedLogNormal, LogNormal, OutlierRemovalUnbiasingEntropyLogOmega
import vegas
# except ImportError as err:
#     print err
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
    color_cycle=cycle([cm(1. * i / 12) for i in xrange(12)])
    return color_cycle
########################
#####################LINE STYLE CYCLER####################                             
lines = ["-","--","-.", ":", "_"]
linecycler = cycle(lines)
##########################################################
"""
for plotting a linear fit with intervals of confidence see http://nbviewer.ipython.org/url/bagrow.com/dsv/LEC10_notes_2014-02-13.ipynb
"""

class EdwardsGeneralisedLogNormal(GeneralisedLogNormal):
    def __init__(self, mu_initial = 1, alpha_initial = 1, zeta_initial = 1, alpha_min = 1e-10, zeta_min = 1e-10, verbose = False):
        super(EdwardsGeneralisedLogNormal, self).__init__(mu_initial=mu_initial, alpha_initial=alpha_initial, zeta_initial=zeta_initial, 
                                                          alpha_min=alpha_min, zeta_min=zeta_min, verbose=verbose)
        self.glob_x = 100
    
    def get_log_edwards_fitted_pressure_expectation(self, kappa, n, ang, vcavity):
        """
        get pressure expectation value
        """
        z = self.get_zeta(self.zeta_offset)
        a = self.get_alpha(self.alpha_offset)
        mu = self.mu
        ang = ang
        def log_integrand_nom(tarr, norm):
            t = tarr[0]
            if t > 0.:
                logt = np.log(t)
                logintegrand = (n/kappa)*logt - ang*vcavity*t - 0.5*np.power(np.abs((logt - mu) / a),z) - norm
                return logintegrand
            else:
                return -1e300
        def log_integrand_den(tarr, norm):
            t = tarr[0]
            if t > 0.:
                logt = np.log(t)
                logintegrand = (n/kappa-1.)*logt - ang*vcavity*t - 0.5*np.power(np.abs((logt - mu) / a),z) - norm
                return logintegrand
            else:
                return -1e300
        #import matplotlib.pyplot as plt
        if ang < 1e-6:
            x = np.linspace(0,1e7,100000)
        else:
            x = np.linspace(0,self.glob_x,100000)
        y = np.array([log_integrand_den([xi],1) for xi in x])
        zargmax = np.argmax(y)
        maxy = np.amax(y)
        y = np.array([log_integrand_den([xi], maxy) for xi in x])
        zend = next(i for i,yy in enumerate(y[zargmax:]) if yy<np.log(1e-5))
        zend += zargmax
        
        y = np.array([log_integrand_nom([xi],1) for xi in x])
        argmax = np.argmax(y)
        maxy = np.amax(y)
        print np.amax(y)
        y = np.array([log_integrand_nom([xi], maxy) for xi in x])
        end = next(i for i,yy in enumerate(y[argmax:]) if yy<np.log(1e-5))
        end += argmax
        #plt.plot(x, y)
        #plt.show()
        
        if ang > 1e-6:
            self.glob_x = x[max(zend, end)+1]
        
        zx = np.linspace(0,x[zend],100000)
        y = np.array([log_integrand_den([xi],1) for xi in zx])
        maxy = np.amax(y)
        print "zxmax", zx[-1]
        print "xmax", x[end]
        
        intnval = 10000
        integ = vegas.Integrator([[0., x[end]]])
        result = integ(lambda t : np.exp(log_integrand_nom(t, maxy)), nitn=20, neval=intnval, alpha=0.2, beta=1)
        Pea, errPea = result.mean, result.sdev
        integ = vegas.Integrator([[0., zx[-1]]])
        result = integ(lambda t : np.exp(log_integrand_den(t, maxy)), nitn=20, neval=intnval, alpha=0.2, beta=1)
        zPea, errzPea = result.mean, result.sdev
        print "{}+/-{} \n {}+/-{}".format(Pea, errPea, zPea, errzPea)
        return np.log(Pea) - np.log(zPea), 0.434*(errPea/Pea + errzPea/zPea)

# cdf = CDFAccumulator()
# cdf.add_array(x)
# x, cdf_x = cdf.get_vecdata()

def plot(packing_datasets, figdir="figures", phi_min=0.825, phi_max=0.88):
    from scipy.optimize import curve_fit
    if not os.path.isabs(figdir):
        figdir = os.path.join(os.getcwd(), figdir)
    trymakedir(figdir)
    
    if True:
        #plot free energy vs P for all packings
        import matplotlib.gridspec as gridspec
        color_cycle = get_color_cycle()
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(7, 2)
        ax = fig.add_subplot(gs[:4, :])
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        S = []
        phi = []
        pmin, pmax = 1e100, -1e100
        p_minmax_list = []
        for i, dataset in enumerate(sorted(packing_datasets, key=lambda data: data.ss_phi)):
            if len(dataset.free_energies) > 0:
                print "set name ",dataset.set_name
                nparticles = dataset.nparticles
                j = 0
                Facc = None
                vcavity = None
                while Facc is None or vcavity is None:
                    Facc = dataset.packing_data[j].Facc
                    vcavity = dataset.packing_data[j].vcavity
                    j += 1
                ############hack
                if np.abs(dataset.ss_phi - 0.865) < 1e-6:
                    Facc -= 1.2
                print Facc
                print nparticles
                if (0.855 < dataset.ss_phi < phi_max and "fire" not in dataset.set_name) or (
                                    phi_min < dataset.ss_phi < 0.86 and "fire" in dataset.set_name):
                    #should remoe both outliers in pressure and in volume
                    x_raw = dataset.pressures
                    f_outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3*np.std(dataset.free_energies))
                    x_outliers = OutlierDetection(np.log(x_raw), p=0.5, D=3*np.std(np.log(x_raw)))
                    non_outliers_indexes = list(set(f_outliers.non_outliers_indexes).intersection(x_outliers.non_outliers_indexes))
                    x = np.array(x_raw)[np.array(non_outliers_indexes, dtype="i")]
                    f = np.array(dataset.free_energies)[np.array(non_outliers_indexes, dtype="i")]
                    weights = 1./np.array(dataset.free_energies_err)[np.array(non_outliers_indexes, dtype="i")]
                    pmin, pmax = min(pmin, np.amin(x)), max(pmax, np.amax(x))
                    p_minmax_list.append([np.amin(x), np.amax(x)])
                    x = np.log(x)
                    y = Facc-f
                    if "fire" in dataset.set_name:
                        marker = '^'
                        label = 'fire {:.3f}'.format(dataset.ss_phi)
                    else:
                        marker = 'o'
                        label = 'cgd {:.3f}'.format(dataset.ss_phi)
                    ax.scatter(x, y, label=label, marker=marker, color=color_cycle.next())
                    fit, cov = np.polyfit(x, y, 1, w=weights, cov=True)
                    fit_err = np.sqrt(np.diag(cov))
                    fit_fn = np.poly1d(fit)
                    ax.plot(x, fit_fn(x), color='k')
                    S.append(np.mean(f) - Facc - log_factorial(dataset.nparticles))
                    # S.append(- Facc - log_factorial(dataset.nparticles))
                    phi.append(dataset.ss_phi)
                    # now fit the actual power laws, not the probabilities
                    fit, cov = np.polyfit(x, f, 1, w=weights, cov=True)
                    fit_err = np.sqrt(np.diag(cov))
                    fit_fn = np.poly1d(fit)
                    dataset.add_extras((fit, fit_err))
                    print dataset.extras
        print phi
        ax.legend(frameon=False, loc=2, prop={'size':18}, numpoints=1, scatterpoints=1, markerscale=1,
                  columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax.set_ylabel(r"$F_{acc}-F$")
        # ax.set_ylabel(r"$F$")
        ax.set_xlabel(r"$\log \mathcal{P}$")
        print "extras", dataset.extras
        fig.savefig('{0}/plot_{1}.pdf'.format(figdir, "f_logp"))
        ax1.scatter(phi, S, color=color_cycle.next(), s=100)
        fit = np.polyfit(phi[2:], S[2:], 1)
        fit_fn = np.poly1d(fit)
        ax1.plot(np.linspace(phi[0],1,20), fit_fn(np.linspace(phi[0],1,20)), color='k')
        ax1.plot([0.835,1],[0,0],lw=1,color='black')
        ax1.set_ylabel(r"$S_G$")
        ax1.set_xlabel(r"$\phi_{ss}$")
        ax1.set_xlim((0.835,1))
        fig1.savefig('{0}/plot_{1}.pdf'.format(figdir, "s_phi"))
        
        #subplots
        # subplots
        if True:
            def ff(x, a, b):
                return a * x + b

            # plot power law exponent
            color_cycle = get_color_cycle()
            color_marker = color_cycle.next()
            color_fit = color_cycle.next()
            ax3 = fig.add_subplot(gs[4:, 0])
            x, y, yerr, y2, y2err = [], [], [], [], []
            for i, dataset in enumerate(sorted(packing_datasets, key=lambda data: data.ss_phi)):
                if len(dataset.free_energies) > 0 and (
                            (0.855 < dataset.ss_phi < phi_max and "fire" not in dataset.set_name) or (
                                            phi_min < dataset.ss_phi < 0.86 and "fire" in dataset.set_name)):
                    y.append(dataset.extras[0][0])
                    yerr.append(dataset.extras[1][0])
                    y2.append(dataset.extras[0][1])
                    y2err.append(dataset.extras[1][1])
                    x.append(dataset.ss_phi)
            x, y, yerr, y2, y2err = np.array(x), np.array(y), np.array(yerr), np.array(y2), np.array(y2err)

            y /= nparticles
            yerr /= nparticles
            y2 /= nparticles
            y2err /= nparticles
            ax3.errorbar(x, y, yerr, marker='o', linestyle='', ms=12, color=color_marker)
            popt, pcov = curve_fit(ff, x, y, sigma=yerr, absolute_sigma=True)
            fit_err = np.sqrt(np.diag(pcov))
            ax3.plot(x, ff(x, popt[0], popt[1]), color=color_fit)
            glob_phi_j, glob_phi_j_std = -popt[1]/popt[0], np.sqrt((fit_err[0]/popt[0])**2+(fit_err[1]/popt[1])**2)
            print "1/k: phi_j: {}+/-{}, beta: {}".format(glob_phi_j, glob_phi_j_std, popt[1])
            ax3.set_title(r'$1/\kappa$', size=18)
            ax3.locator_params(axis='x', nbins=4)
            ax3.locator_params(axis='y', nbins=4)
            ax3.tick_params(axis='both', which='major', labelsize=18)

            ax4 = fig.add_subplot(gs[4:, 1])
            ax4.errorbar(x, y2, y2err, marker='o', linestyle='', ms=12, color=color_marker)
            popt, pcov = curve_fit(ff, x, y2, sigma=y2err, absolute_sigma=True)
            ax4.plot(x, ff(x, popt[0], popt[1]), color=color_fit)
            fit_err = np.sqrt(np.diag(pcov))
            print "c: phi_c1: {}+/-{}, beta: {}".format((1-popt[1])/popt[0], np.sqrt((fit_err[0]/popt[0])**2+(fit_err[1]/popt[1])**2), popt[1])
            # label = "intercept = ({:.3f} +/- {:.3f})N".format(popt[0], np.sqrt(float(pcov[0])))
            # ax4.legend(frameon=False, loc="best", framealpha=0.5, prop={'size':12}, labelspacing=0.25,
            #           columnspacing=0.25, numpoints=1, markerscale=0.5, handlelength=0.4)
            ax4.set_title(r'$\mathcal{C}$', size=18)
            ax4.locator_params(axis='x', nbins=4)
            ax4.locator_params(axis='y', nbins=4)
            ax4.tick_params(axis='both', which='major', labelsize=18)

            fig.savefig('{0}/plot_{1}.pdf'.format(figdir, "f_logp"))

        if False:
            # kde pressure
            color_cycle = get_color_cycle()
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
            fig4 = plt.figure()
            ax4 = fig4.add_subplot(111)
            fig5 = plt.figure()
            ax5 = fig5.add_subplot(111)
            fig6 = plt.figure()
            ax6 = fig6.add_subplot(111)
            fig7= plt.figure()
            ax7 = fig7.add_subplot(111)
            fit_params = []
            fit_params_std = []
            ss_phi_list = []
            unpad_log_omega_p_hist = []
            kde_list = []
            iNK_list = []
            log_omega_list = []
            maxp, max_logdos = [], [] #pressure for which kde is max
            x_integrate_list = []
            for i, dataset in enumerate(sorted(packing_datasets, key=lambda data: data.ss_phi)):
                if len(dataset.free_energies) > 0:
                    j = 0
                    Facc = None
                    vcavity = None
                    while Facc is None or vcavity is None:
                        Facc = dataset.packing_data[j].Facc
                        vcavity = dataset.packing_data[j].vcavity
                        j += 1
                    nparticles = dataset.nparticles
                    vcavity = dataset.packing_data[0].vcavity
                    if (0.855 < dataset.ss_phi < phi_max and "fire" not in dataset.set_name) or (
                           phi_min < dataset.ss_phi < 0.86 and "fire" in dataset.set_name):
                        print "n:", nparticles
                        #fit slopes to find inverse kappa and the intercept
                        x_raw = dataset.pressures
                        f_outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3 * np.std(dataset.free_energies))
                        x_outliers = OutlierDetection(np.log(x_raw), p=0.5, D=2.5 * np.std(np.log(x_raw)))
                        non_outliers_indexes = list(set(f_outliers.non_outliers_indexes).intersection(x_outliers.non_outliers_indexes))
                        x = np.array(x_raw)[np.array(non_outliers_indexes, dtype="i")]
                        f = np.array(dataset.free_energies)[np.array(non_outliers_indexes, dtype="i")]
                        weights = 1. / np.array(dataset.free_energies_err)[
                            np.array(non_outliers_indexes, dtype="i")]
                        x = np.log(x)
                        fit, cov = np.polyfit(x, f, 1, w=weights, cov=True)
                        fit_err = np.sqrt(np.diag(cov))

                        iNK, NC = fit #N/K, C(N)
                        iNKstd, NCstd = fit_err
                        iNK_list.append(iNK)
                        K = nparticles/iNK
                        C = NC/nparticles
                        fit_params.append([K, C])
                        fit_params_std.append(fit_err)
                        ss_phi_list.append(dataset.ss_phi)

                        #free energy non parametric log omega
                        bw = get_bandwidth_estimate(np.array(f), kernel="gaussian", method="cross_validation")
                        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(f[:, np.newaxis])
                        # plot histograms
                        n_integrate = 2 ** 14 + 1
                        x_integrate = np.linspace(Facc, np.amax(f) * 100, n_integrate)
                        log_pdf = kde.score_samples(x_integrate[:, np.newaxis])
                        integrand = np.exp(np.add(log_pdf, x_integrate))
                        integral = integrate.romb(integrand, dx=x_integrate[1] - x_integrate[0])
                        log_omega = - Facc + np.log(integral)
                        log_omega_list.append(log_omega)

                        # kde histogram
                        bw = get_bandwidth_estimate(np.array(x), kernel="gaussian", method="cross_validation")
                        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(x[:, np.newaxis])
                        kde_list.append(kde)
                        #plot histograms
                        n_integrate = 2 ** 14 + 1
                        x_integrate = np.linspace(np.amin(x)*0.5, min(np.amax(x)*15, pmax*5), n_integrate)
                        x_integrate_list.append(x_integrate)
                        # x_integrate = np.linspace(0, 1000, n_integrate)
                        log_pdf = kde.score_samples(x_integrate[:, np.newaxis])
                        log_hist = log_pdf + iNK * np.log(x_integrate)
                        norm = np.log(integrate.romb(np.exp(log_hist), dx=x_integrate[1]-x_integrate[0]))
                        unpad_log_omega_p_hist.append(log_hist - norm + log_omega)
                        ax2.plot(x_integrate, log_hist - norm + log_omega, label=dataset.ss_phi,
                                 color=color_cycle.next(), linewidth=3)
                        maxp.append(x_integrate[np.argmax(log_hist - norm + log_omega)])
                        max_logdos.append(np.amax(log_hist - norm + log_omega))
                        # color_cycle = get_color_cycle()

            xmin, xmax = np.amin(x_integrate_list), np.amax(x_integrate_list)
            print xmin, xmax
            pad_log_omega_p_hist = []
            for x_, x_integrate_ in zip(unpad_log_omega_p_hist, x_integrate_list):
                dx = x_integrate_[1] - x_integrate_[0]
                for j, xx_ in enumerate(x_):
                    x_[j] = 0 if xx_ < 0 else xx_
                pad_log_omega_p_hist.append(np.pad(x_, (int(np.floor((np.amin(x_integrate_) - xmin) / dx)),
                                                        int(np.floor((xmax - np.amax(x_integrate_)) / dx))),
                                                   'constant', constant_values=(0, 0)))

            maxsize = len(max(pad_log_omega_p_hist, key=len)) // 10
            from skimage.transform import resize
            x_integrate = np.linspace(xmin, xmax, maxsize)
            for i, x_ in enumerate(pad_log_omega_p_hist):
                a = np.array(x_).reshape((1, len(x_)))
                pad_log_omega_p_hist[i] = resize(a, (1, maxsize)).flatten()
                ax2.plot(x_integrate, pad_log_omega_p_hist[i], label=dataset.ss_phi,
                         color=color_cycle.next(), linewidth=1)
            ax2.set_xscale('log')

            dphi = np.array(phi) - glob_phi_j

            x = dphi
            # y = maxp
            # for i, par in enumerate(fit_params):
            #     y[i] /= par[0]
            # y = np.log(y)
            # weights = 1./np.array(fit_params_std)[:,0] #error of inverse kappa, check that it is correct
            # color = color_cycle.next()
            # y = np.log(y) - np.log(np.amax(y))
            # ax5.scatter(x, y, color=color, label=r'$\ln(p_{peak}/max(p_{peak}))$', s=100, alpha=0.5)
            # fit, cov = np.polyfit(x, y, 1, cov=True)  # if I use dphi instead of phi it's not a power law
            # fit_fn = np.poly1d(fit)
            # ax5.plot(x, fit_fn(x), color=color, linewidth=3)

            color = color_cycle.next()
            x = np.array(phi)
            y = np.array(fit_params)[:,0]
            y = np.exp(np.array(1./y))
            ax5.scatter(x, y, color=color, label=r'$\exp(\kappa)$', s=100, alpha=0.5)
            weights = 1. / np.array(fit_params_std)[:, 0]
            fit, cov = np.polyfit(x, y, 1, w=weights, cov=True)  # if I use dphi instead of phi it's not a power law
            fit_fn = np.poly1d(fit)
            ax5.plot(x, fit_fn(x), color=color, linewidth=3)
            # ax5.axhline(0, color='k')
            print "phi max(p) fit: ", fit

            # y = np.log(np.array(p_minmax_list)[:,0])
            # color = color_cycle.next()
            # ax5.scatter(x, y, color=color, label=r'$p_{min}$', s=100)
            # fit, cov = np.polyfit(x, y, 1, cov=True)  # if I use dphi instead of phi it's not a power law
            # fit_fn = np.poly1d(fit)
            # ax5.plot(x, fit_fn(x), color=color, linewidth=3)
            # print "phi p_min fit: ", fit
            #
            # y = np.log(np.array(p_minmax_list)[:, 1])
            # color = color_cycle.next()
            # ax5.scatter(x, y, color=color, label=r'$p_{max}$', s=100)
            # fit, cov = np.polyfit(x, y, 1, cov=True)  # if I use dphi instead of phi it's not a power law
            # fit_fn = np.poly1d(fit)
            # ax5.plot(x, fit_fn(x), color=color, linewidth=3)
            # ax5.set_xlabel(r'$\Delta\phi$')
            # ax5.set_ylabel(r'$\log p_x$')
            # print "phi p_max fit: ", fit

            ax5.legend(frameon=False, loc=2, prop={'size': 18}, numpoints=1, scatterpoints=1, markerscale=1,
                       columnspacing=0.25, labelspacing=0.25, handletextpad=0)

            x, y = dphi, max_logdos
            color = color_cycle.next()
            ax6.scatter(x, y, color=color, s=100)
            fit, cov = np.polyfit(x, y, 1, cov=True)
            fit_fn = np.poly1d(fit)
            ax6.plot(x, fit_fn(x), color=color, linewidth=3)
            ax6.set_xlabel(r'$\log DOS_{max}$')
            ax6.set_xlabel(r'$\Delta\phi$')
            print "phi max dos fit: ", fit

            alpha, beta = 200, 100
            color_cycle = get_color_cycle()
            for i,(kde, iNK) in enumerate(zip(kde_list, iNK_list)):
                x_integrate = np.linspace(p_minmax_list[i][0]*0.5, min(p_minmax_list[i][1]*15, pmax*5), 2**14+1)
                # x_integrate = np.linspace(p_minmax_list[i][0], p_minmax_list[i][1], 2**14+1)
                log_pdf = kde.score_samples(x_integrate[:, np.newaxis])
                log_hist = log_pdf + iNK * np.log(x_integrate)
                log_norm = np.log(integrate.romb(np.exp(log_hist), dx=x_integrate[1] - x_integrate[0]))
                logg = (log_hist - log_norm + log_omega_list[i]) + alpha * dphi[i]
                # print np.amax(logg), alpha * dphi[i]
                ax7.plot(np.log(x_integrate) - beta*dphi[i], logg, label=phi[i],
                         color=color_cycle.next(), linewidth=3)
            ax7.legend(frameon=False, loc=2, prop={'size': 18}, numpoints=1, scatterpoints=1, markerscale=1,
                       columnspacing=0.25, labelspacing=0.25, handletextpad=0)
            # ax7.set_ylim((200,230))

            # def cost_function(x):
            #     cost = 0
            #     alpha, beta = x[0], x[1]
            #     print alpha, beta
            #     for i in xrange(dphi.size):
            #         x_integrate_i = x_integrate / np.power(dphi[i], beta)
            #         log_pdf = kde_list[i].score_samples(x_integrate_i[:, np.newaxis])
            #         log_hist = log_pdf + iNK * np.log(x_integrate_i)
            #         log_norm = np.log(integrate.romb(np.exp(log_hist), dx=x_integrate_i[1] - x_integrate_i[0]))
            #         logg_i = log_hist - log_norm + log_omega_list[i]
            #         for j in xrange(i, dphi.size):
            #             if i != j:
            #                 print i, j
            #                 x_integrate_j = x_integrate / np.power(dphi[j], beta)
            #                 log_pdf = kde_list[j].score_samples(x_integrate_j[:, np.newaxis])
            #                 log_hist = log_pdf + iNK * np.log(x_integrate_j)
            #                 log_norm = np.log(integrate.romb(np.exp(log_hist), dx=x_integrate_j[1] - x_integrate_j[0]))
            #                 logg_j = log_hist - log_norm + log_omega_list[j]
            #                 cost_ = np.sum(np.power(alpha * (np.log(dphi[i])-np.log(dphi[j])) + logg_i - logg_j,2))
            #                 print cost_
            #                 cost += cost_
            #     return cost
            # from scipy.optimize import minimize
            # print minimize(cost_function, [0,0])

            if False:
                # assert y.size == pdf.size
                X, Y, = np.array(x_integrate), np.array(ss_phi_list)
                Z = np.array(pad_log_omega_p_hist)
                ax3.imshow(Z, vmin=np.abs(Z).min(), vmax=np.abs(Z).max(), origin='lower',
                           extent=[X.min(), X.max(), Y.min(), Y.max()], interpolation='gaussian',
                           aspect='auto')

                from matplotlib import cm
                norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
                cmap = cm.jet
                levels = np.arange(np.amin(pad_log_omega_p_hist), np.amax(pad_log_omega_p_hist)*1.01, 10)
                ax4.contourf(X, Y, pad_log_omega_p_hist,
                             norm=norm,
                             cmap=cmap,#cm.get_cmap(cmap, len(levels) - 1),
                             levels=levels)
                ax4.set_xscale('log')

            #         ax.plot(edges, [generalised_lognormal.get_fitted(xpi) for xpi in edges], "--", color=color,
            #                 linewidth=2)
            #
            #         # unbias kde
            #         # F0acc = Bunch(F0_acc=dataset.packing_data[0].Facc, nr_particles=nparticles)
            #         # c, cerr = quad(generalised_lognormal.get_fitted_times_xpow, 0, np.amax(x) * 1e5,
            #         #                args=(glob_kappa, nparticles), limit=200,
            #         #                points=[np.amin(x), np.amax(x), xp[np.argmax(fit)]], epsabs=1.49e-11)
            #         # c *= np.exp(nparticles)
            #
            #
            # ax2.legend(frameon=False, loc=2, prop={'size': 22}, numpoints=1, markerscale=1, columnspacing=0.25,
            #            labelspacing=0.25, handlelength=1)
            # ax3.set_xscale('log')
            # ax3.set_yscale('log')

        if False:
            # kde free energies predicted vs numerical
            # note the x2 = (nparticles/glob_kappa)*np.log(x2)+glob_interc*nparticles (should be just +N)
            color_cycle = get_color_cycle()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i, dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
                if len(dataset.free_energies) > 0:
                    nparticles = dataset.nparticles
                    outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3 * np.std(dataset.free_energies))
                    x = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
                    bw = get_bandwidth_estimate(np.array(x), kernel="gaussian", method="cross_validation")
                    edges = np.linspace(np.amin(x) * 0.5, np.amax(x) * 1.5, 1000)
                    hist = get_pdf(x, edges, bandwidth=bw, kernel="gaussian")
                    color = color_cycle.next()
                    ax.plot(edges, hist, label=int(nparticles), color=color, linewidth=3)

                    x2 = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
                    x2 = (nparticles / glob_kappa) * np.log(x2) + glob_interc * nparticles
                    bw2 = get_bandwidth_estimate(np.array(x2), kernel="gaussian", method="cross_validation")
                    edges2 = np.linspace(np.amin(x2) * 0.5, np.amax(x2) * 1.5, 1000)
                    hist2 = get_pdf(x2, edges2, bandwidth=bw2, kernel="gaussian")
                    # hist2 *= edges2/(nparticles/glob_kappa)
                    # hist2 /= simps(hist2, x=edges2)
                    ax.plot(edges2, hist2, '--', color=color, linewidth=3)

            ax.legend(frameon=False, loc="best", numpoints=1, markerscale=0.5, columnspacing=0.25, labelspacing=0.25,
                      handlelength=0.4)
            plt.ylabel(r"$p(F)$")
            plt.xlabel(r"$F$")
            ax.set_xlim((30, 340))
            fig.savefig('{0}/plot_{1}.pdf'.format(figdir, "predicted_F_dist"))

        if False:
            # fit the (N/k)\ln(P)+N model to each and compute the average of the kappas

            fig = plt.figure()
            ax = fig.add_subplot(111)
            avg = 0
            for i, dataset in enumerate(sorted(packing_datasets, key=lambda data: data.nparticles)):
                if len(dataset.free_energies) > 0:
                    nparticles = dataset.nparticles
                    print nparticles
                    outliers = OutlierDetection(dataset.free_energies, p=0.5, D=3 * np.std(dataset.free_energies))
                    x = np.array(dataset.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
                    y = np.array(dataset.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
                    print len(x), len(y)
                    x = np.log(x)
                    ax.scatter(x, y, color=color_cycle.next())

                    def ff_msf(x, a):
                        return nparticles * (a * x + 1)

                    popt, pcov = curve_fit(ff_msf, x, y)
                    ax.plot(x, ff_msf(x, popt[0]), label="F = N/{:.3f}ln(P) + N".format(1. / popt[0]))
                    avg += popt[0]
            avg /= (i + 1)
            print "avg kappa = ", 1 / avg
            ax.legend(frameon=False, loc="best", numpoints=1, markerscale=0.5, columnspacing=0.25, labelspacing=0.25)
            plt.ylabel(r"$F$")
            plt.xlabel(r"$\ln \mathcal{P}$")

if __name__ == "__main__":
    pts = BasinAnalysis()
    pts.collect_data_every_set_all()
    plot(pts.packing_datasets)
    plt.show()
    plt.close()