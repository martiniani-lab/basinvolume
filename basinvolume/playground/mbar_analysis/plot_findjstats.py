from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import rc
from basinvolume.utils import trymakedir, Result
import numpy as np
import os
import glob
import cPickle as pickle
from itertools import cycle
from cycler import cycler
from sklearn.neighbors import KernelDensity
from basinvolume.experiment_2d.cross_validation_bandwidth_selection import get_bandwidth_estimate, get_pdf
from basinvolume.spheres import SoftPackingDataset, SoftPackingData
import scikits.bootstrap as bootstrap
import argparse
from scipy.interpolate import UnivariateSpline
from scipy.stats.stats import pearsonr
from scipy.integrate import quad

#######################SET LATEX OPTIONS###################
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
#rc('text.latex',preamble=r'\usepackage{times}')
glob_fontsize=25
plt.rcParams.update({'font.size': glob_fontsize})
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams.update({'figure.autolayout': True})
##########################################################
####SET COLOUR MAP######
def get_color_cycle(ncolors=20, reverse=True):
    cm = plt.get_cmap('Paired')
    if reverse:
        color_cycle=cycle([cm(1. * (i+0.5) / float(ncolors)) for i in xrange(ncolors)][::-1])
    else:
        color_cycle = cycle([cm(1. * (i - 0.5) / float(ncolors)) for i in xrange(ncolors)])
    return color_cycle
def get_marker_cycle():
    markers = ["o","v","s","h","^","8","p","<","*","D",">",]
    markercycle = cycle(markers)
    return markercycle
def get_line_cycle():
    lines = ["--","-"]
    linecycle = cycle(lines)
    return linecycle
def get_cycler(ncolors=20, reverse=True):
    cm = plt.get_cmap('Paired')
    if reverse:
        color_cycler = cycler('color', [cm(1. * (i+0.5) / float(ncolors)) for i in xrange(ncolors)][::-1])
    else:
        color_cycler = cycler('color', [cm(1. * (i - 0.5) / float(ncolors)) for i in xrange(ncolors)])
    return color_cycler

###########################################################

def poly_fit(x, y, yerr=None, order=1):
    w = 1./np.asarray(yerr) if yerr is not None else None
    if len(x) - order - 2 < 2 and order > 1:
        # this hack was taken from
        # https://stackoverflow.com/questions/27230285/numpy-polyfit-gives-useful-fit-but-infinite-covariance-matrix
        assert order == 2, "hack is only implemented for order=2"
        w = [1 for _ in xrange(len(x))].append(0)
        x = np.append(x, x[-1])
        y = np.append(y, y[-1])
    fit_params, cov = np.polyfit(x, y, order, w=w, cov=True)
    fit_err = np.sqrt(np.diag(cov))
    fit_fn = np.poly1d(fit_params)
    rho = pearsonr(x, y)[0]
    return fit_fn, fit_params, fit_err, rho

def collect_data_every_set_all(workspace=None,
                               data_signature='jammed_packings_*D_mu*_sig*_sca*_phi*.pickle'):
    if workspace is None:
        workspace = os.getcwd()
    listdir = glob.glob(os.path.join(workspace, data_signature))
    datasets = []
    for path in listdir:
        print "collecting data from ", os.path.split(path)[1]
        datasets.append(pickle.load(open(path, "rb")))
    return datasets

def collect_data_plot(datasets, phi_max=0.871, bdim=2, lmax=2e4):
    psuccess = []
    nrattlers = []
    energy = []
    pressure = []
    contacts = []
    contacts_all = []
    phi_ss = np.unique([dataset.phi_ss for dataset in sorted(datasets, key=lambda data: data.phi_ss)])
    phi_ss = phi_ss[phi_ss < phi_max]
    bdim = bdim
    for phi_ in phi_ss:
        success_, nrattlers_, energy_ = [], [], []
        pressure_, contacts_, contacts_all_ = [], [], []
        l = 0
        for i, dataset in enumerate(sorted(datasets, key=lambda data: data.phi_ss)):
            if phi_ == dataset.phi_ss and l < lmax:
                tmp = np.asarray(dataset.success, dtype='int')
                for data in dataset.packings_data:
                    N_contacts = int(np.sum(data.Z))
                    no_stable = len(data.Z)
                    N_min = int(2 * (bdim * (no_stable - 1) + 1))
                    if N_contacts >= N_min:
                        nrattlers_.append(data.nrattlers)
                        energy_.append(data.energy)
                        pressure_.append(data.pressure)
                        contacts_.append(np.mean(data.Z))
                        contacts_all_.append(data.Z)
                        l = l + len(data.energy)
                    else:
                        tmp[np.argmax(tmp > 0)] = 0
                success_.extend(tmp)
        psuccess.append(np.mean(success_))
        nrattlers.append(np.mean(nrattlers_))
        energy.append(energy_)
        pressure.append(pressure_)
        contacts.append(contacts_)
        contacts_all.append(contacts_all_)
    return np.asarray(psuccess), np.asarray(phi_ss), np.asarray(nrattlers), np.asarray(energy), \
           np.asarray(pressure), np.asarray(contacts), np.asarray(contacts_all)

class DataPlot(object):
    def __init__(self, datasets, prob_min=0.01, bdim=2, nparticles=64):
        psuccess, phi_ss, nrattlers, energy, pressure, contacts, contacts_all = collect_data_plot(datasets)
        self.psuccess = np.asarray(psuccess)
        self.phi_ss = np.asarray(phi_ss)
        self.phi_ss_packed = self.phi_ss[self.psuccess > prob_min]
        self.nrattlers = np.asarray(nrattlers)[self.psuccess > prob_min]
        self.energy = np.asarray(energy)[self.psuccess > prob_min]
        self.pressure = np.asarray(pressure)[self.psuccess > prob_min]
        self.contacts = np.asarray(contacts)[self.psuccess > prob_min]
        self.contacts_all = np.asarray(contacts_all)[self.psuccess > prob_min]
        self.bdim = bdim
        self.nparticles = nparticles
        self.bw = []
        self.logp_mean, self.p_mean, self.logp_mode = [], [], []
        self.logp_var,  self.p_var, self.p_rel_var= [], [], []
        self.log_pdf = []
        self.log_pdf_x = []
        self.initialized = False

    def compute_stats(self, n_samples=1e4, n_integrate=None):
        for i, p in enumerate(self.pressure):
            if n_integrate is None:
                n_integrate = 2 ** 14 + 1
            else:
                n_integrate = n_integrate
            lnp = np.log(np.asarray(p))
            # x_integrate = np.linspace(np.amin(np.log(np.hstack(self.pressure))), np.amax(np.log(np.hstack(self.pressure))), n_integrate)
            x_integrate = np.linspace(np.amin(lnp), np.amax(lnp), n_integrate)
            # build kde histogram
            bw = get_bandwidth_estimate(lnp, kernel="gaussian", method="cross_validation")
            self.bw.append(bw)
            kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(lnp[:, np.newaxis])
            log_pdf = kde.score_samples(x_integrate[:, np.newaxis])
            self.log_pdf.append(log_pdf)
            self.log_pdf_x.append(x_integrate)
            # build array of relative fluctuations around the mode
            log_maxp = x_integrate[np.argmax(log_pdf)]
            # p_rel = p / np.exp(log_maxp)
            # build array of relative fluctuations around the mean
            p_rel = p / np.mean(p)
            n = int(min(n_samples, lnp.size*25))
            varCIs = bootstrap.ci(p, lambda x : np.var(x/np.mean(x)), n_samples=n, alpha=0.32)
            self.p_rel_var.append([np.var(p_rel), varCIs[0], varCIs[1]])
            #build array of logp mean, var and maxp
            meanCIs = bootstrap.ci(lnp, np.mean, n_samples=n, alpha=0.32)
            varCIs = bootstrap.ci(lnp, np.var, n_samples=n, alpha=0.32)
            self.logp_mean.append([np.mean(lnp), meanCIs[0], meanCIs[1]])
            self.logp_var.append([np.var(lnp), varCIs[0], varCIs[1]])
            self.logp_mode.append(log_maxp)
            meanCIs = bootstrap.ci(p, np.mean, n_samples=n, alpha=0.32)
            varCIs = bootstrap.ci(p, np.var, n_samples=n, alpha=0.32)
            self.p_mean.append([np.mean(p), meanCIs[0], meanCIs[1]])
            self.p_var.append([np.var(p), varCIs[0], varCIs[1]])

        self.logp_mean, self.p_mean = np.asarray(self.logp_mean), np.asarray(self.p_mean)
        self.logp_var, self.p_var, self.p_rel_var = np.asarray(self.logp_var), np.asarray(self.p_var), np.asarray(self.p_rel_var)
        self.bw = np.asarray(self.bw)
        self.log_pdf = np.asarray(self.log_pdf)
        self.log_pdf_x = np.asarray(self.log_pdf_x)
        self.initialized = True

def plot(figdir="figures", bdim=2, nparticles=64):
    figdir = os.path.join(os.getcwd(), figdir)
    trymakedir(figdir)
    path = os.path.join(os.getcwd(), "findjstats.pickle")

    def ff(x, a, b):
        return a * x + b

    def ff2(x, a, b, c):
        return a * x**2 + b*x + c

    try:
        dp = pickle.load(open(path, "rb"))
    except Exception, e:
        datasets = collect_data_every_set_all()
        dp = DataPlot(datasets, bdim=bdim, nparticles=nparticles)
        dp.compute_stats()
        pickle.dump(dp, open("findjstats.pickle", "wb"))

    psuccess, phi_ss, phi_ss_packed, nrattlers =  dp.psuccess, dp.phi_ss, dp.phi_ss_packed, dp.nrattlers
    energy, pressure, contacts_mean, contacts_all = dp.energy, dp.pressure, dp.contacts, dp.contacts_all
    assert dp.initialized == True
    log_pdf, log_pdf_x, logp_mean, logp_var = dp.log_pdf, dp.log_pdf_x, dp.logp_mean, dp.logp_var
    p_mean, p_var, p_rel_var, logp_mode = dp.p_mean, dp.p_var, dp.p_rel_var, dp.logp_mode
    bw = dp.bw
    assert nparticles == dp.nparticles

    if True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(phi_ss, np.asarray(psuccess), marker='o', linestyle='', rasterized=True)
        spl = UnivariateSpline(phi_ss, psuccess, bbox=[phi_ss[0], phi_ss[-1]], s=7.5e-4, k=3)
        pickle.dump(spl, open(os.path.join(os.getcwd(), "phi_psuccess_spline.pickle"), "wb"))
        phi_ss_spl = np.linspace(phi_ss[0], phi_ss[-1], 1000)
        ax.plot(phi_ss_spl, spl(phi_ss_spl), marker='', linestyle='-', rasterized=True)
        ax.set_xlim([phi_ss[0],phi_ss[-1]])
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$p_{pack}$")
        fig.savefig("{}/{}".format(figdir, "phi_ppack{}.pdf".format(nparticles)))

    if True:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(phi_ss_packed, np.asarray(nrattlers), marker='o', linestyle='', rasterized=True)
        spl = UnivariateSpline(phi_ss_packed, nrattlers, bbox=[phi_ss_packed[0], phi_ss_packed[-1]], s=0.05, k=3)
        pickle.dump(spl, open(os.path.join(os.getcwd(), "phi_nrattlers_spline.pickle"), "wb"))
        phi_ss_spl = np.linspace(phi_ss_packed[0], phi_ss_packed[-1], 1000)
        ax1.plot(phi_ss_spl, spl(phi_ss_spl), marker='', linestyle='-', rasterized=True)
        ax1.set_xlim([phi_ss_packed[0], phi_ss_packed[-1]])
        ax1.set_xlabel(r"$\phi$")
        ax1.set_ylabel(r"$n_{rattlers}$")
        fig1.savefig("{}/{}".format(figdir, "phi_nrattlers{}.pdf".format(nparticles)))

    if True:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        color_cycle = get_color_cycle(ncolors=len(energy))
        for e,p in zip(energy, pressure):
            assert len(e) == len(p)
            color = color_cycle.next()
            ax2.scatter(np.log(e), np.log(p), color=color_cycle.next(), rasterized=True)
        flat_energy, flat_pressure = np.log(np.hstack(energy)), np.log(np.hstack(pressure))
        # linear fit low P
        xnew = np.linspace(np.amin(flat_energy), np.amax(flat_energy), 10)

        fit_fn, fit_params, fit_err, rho = poly_fit(flat_energy[flat_energy<-5], flat_pressure[flat_energy<-5], order=1)
        ax2.plot(xnew, fit_fn(xnew), color='k', linewidth=2.5, linestyle='--',
                 label="ln(p) = {:.3f} ln(E) + {:.3f}".format(fit_params[0], fit_params[1]), rasterized=True)
        # linear fit high P
        xnew = np.linspace(2.3, 10, 10)
        fit_fn, fit_params, fit_err, rho = poly_fit(flat_energy[flat_energy > 5], flat_pressure[flat_energy > 5], order=1)
        ax2.plot(xnew, fit_fn(xnew), color='r', linewidth=2.5, linestyle='--',
                 label="ln(p) = {:.3f} ln(E) + {:.3f}".format(fit_params[0], fit_params[1]), rasterized=True)
        # quadratic fit
        xnew = np.linspace(-0.5, 10, 10)
        fit_fn, fit_params, fit_err, rho = poly_fit(flat_energy[flat_energy > -0.5], flat_pressure[flat_energy > -0.5], order=2)
        ax2.plot(xnew, fit_fn(xnew), color='b', linewidth=2.5, linestyle=':',
                 label="ln(p) = {:.3f} ln(E)**2 + {:.3f} ln(E) + {:.3f}".format(fit_params[0], fit_params[1], fit_params[2]),
                 rasterized=True)
        ax2.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                  columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax2.set_xlabel(r"$\ln(E)$")
        ax2.set_ylabel(r"$\ln(P)$")
        fig2.savefig("{}/{}".format(figdir, "lnE_lnP{}.pdf".format(nparticles)))

    if False:
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        color = get_color_cycle(ncolors=len(pressure))
        for i, (p, z) in enumerate(zip(pressure, contacts_mean)):
            assert len(p) == len(z)
            ax3.scatter(np.log(p), z, color=color.next(), rasterized=True)
        ax3.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax3.set_xlabel(r"$\ln(P)$")
        ax3.set_ylabel(r"$\ln(\langle Z \rangle )$")
        fig3.savefig("{}/{}".format(figdir, "lnP_lnZ{}.pdf".format(nparticles)))

    if True:
        import itertools
        fig31 = plt.figure()
        ax31 = fig31.add_subplot(111)
        y = [[item for sublist in x for item in sublist] for x in contacts_all]
        yy = list(itertools.chain(*y))
        color = get_color_cycle(ncolors=int(np.amax(yy)))
        for i in xrange(int(np.amin(yy)),int(np.amax(yy))):
            yy = np.asarray([np.sum(np.asarray(x) == i)/len(x) for x in y])
            ax31.plot(phi_ss_packed, yy, color=color.next(), linestyle='-', marker='o', label='Z={}'.format(i))
        # for i, x in enumerate(contacts_all):
        #     label = "{}".format(phi_ss_packed[i])
        #     xx = [item for sublist in x for item in sublist]
        #     hist, bin_edges = np.histogram(xx, bins=np.unique(xx), density=True)
        #     ax31.plot(bin_edges[:-1], hist, color=color.next(), linestyle='-', marker='o')
        ax31.set_ylim((0,0.5))
        ax31.set_ylabel(r'$N_{Z}/N$')
        ax31.set_xlabel(r'$\phi$')
        ax31.set_xlim((0.81,0.87))
        ax31.legend(frameon=False, loc=2, prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                    columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        fig31.savefig("{}/{}".format(figdir, "Zhist{}.pdf".format(nparticles)))

    if True:
        fig32 = plt.figure()
        ax32 = fig32.add_subplot(111)
        y = [[item for sublist in x for item in sublist] for x in contacts_all]
        color = get_color_cycle(ncolors=2).next()
        ax32.plot(phi_ss_packed, [np.mean(x) for x in y], color=color, linestyle='', marker='o')
        spl = UnivariateSpline(phi_ss_packed, [np.mean(x) for x in y],
                                   bbox=[phi_ss_packed[0], phi_ss_packed[-1]], s=5e-4, k=1)
        pickle.dump(spl, open(os.path.join(os.getcwd(), "phi_meanZ_spline.pickle"), "wb"))
        pickle.dump(np.asarray([phi_ss_packed, [np.mean(x) for x in y]]), open(os.path.join(os.getcwd(), "phi_meanZ_data.pickle"), "wb"))
        phi_ss_spl = np.linspace(0.81, phi_ss_packed[-1], 1000)
        ax32.plot(phi_ss_spl, spl(phi_ss_spl), marker='', linestyle='-', color=color, rasterized=True)
        ax32.set_ylabel(r'$\langle z \rangle_{\mathcal{B}}$')
        ax32.set_xlabel(r'$\phi$')
        fig32.savefig("{}/{}".format(figdir, "phi_meanZ{}.pdf".format(nparticles)))

    if True:
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        fig42 = plt.figure()
        ax42 = fig42.add_subplot(111)
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        fig52 = plt.figure()
        ax52 = fig52.add_subplot(111)
        ax522 = ax52.twinx()
        color_cycle = get_color_cycle(ncolors=phi_ss_packed.size)
        ikappa = lambda phi : 2.71431676132 * phi - 2.23776770087
        c = lambda phi: -7.25204514516 * phi + 6.92383942465
        v_mean_bias, v_mean = [], []
        v_var_bias, v_var = [], []
        for i in xrange(phi_ss_packed.size):
            print phi_ss_packed[i]
            # kde histogram
            assert log_pdf[i].size == log_pdf_x[i].size
            log_maxp = log_pdf_x[i][np.argmax(log_pdf[i])]
            color, label = color_cycle.next(), phi_ss_packed[i]
            # ax4.plot(np.log(x_integrate)-np.log(maxp), log_pdf, color=color, label=label, rasterized=True)
            ax4.plot(log_pdf_x[i]-log_maxp, np.exp(log_pdf[i]), color=color, label=label, rasterized=True)
            ax5.plot(log_pdf_x[i]-log_maxp, log_pdf[i] - np.amax(log_pdf[i]), color=color, label=label, rasterized=True)

            # bias = 1./np.exp(np.log(pressure[i]) * ikappa(phi_ss_packed[i]) + c(phi_ss_packed[i]))
            print "len pressure ", len(pressure[i])
            y = np.log(pressure[i]) * ikappa(phi_ss_packed[i]) + c(phi_ss_packed[i])
            # bias = np.exp(y)
            y = -y
            mu_B, sig_B = np.mean(y), np.var(y)
            # mu_U = np.average(y, weights=bias)
            # sig_U = np.average((y-mu_U)**2, weights=bias)
            v_mean_bias.append(mu_B)
            v_var_bias.append(sig_B)
            # v_mean.append(mu_U)
            # v_var.append(sig_U)
            ax52.plot(phi_ss_packed[i], np.mean(np.log(pressure[i]))*ikappa(phi_ss_packed[i]), color=color, marker='o', linestyle='',
                      rasterized=True)
            ax522.plot(phi_ss_packed[i], np.var(np.log(pressure[i])), color=color, marker='^',
                      linestyle='', rasterized=True)

        ax43 = ax42.twinx()
        ax42.plot(phi_ss_packed, v_mean_bias, color='b', marker='o', linestyle='',
                  rasterized=True, label=r"$E_B(v)$")
        ax43.plot(phi_ss_packed, v_var_bias, color='r', marker='o', linestyle='',
                  rasterized=True, label=r"$Var_B(v)$")
        # ax42.plot(phi_ss_packed, v_mean, color='b', marker='^', linestyle='',
        #           rasterized=True, label=r"$E_U(v)$")
        # ax43.plot(phi_ss_packed, v_var, color='r', marker='^', linestyle='',
        #           rasterized=True, label=r"$Var_U(v)$")
        ax42.legend(frameon=False, loc=2, prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax43.legend(frameon=False, loc=3, prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                    columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax4.set_xlabel(r"$\ln(P/P_{peak})$")
        ax4.set_ylabel(r"$pdf/\max(pdf)$")
        ax4.set_xlabel(r"$ \Lambda - \overline{\Lambda}$")
        ax4.set_ylabel(r"$pdf(\Lambda)$")
        ax5.set_xlabel(r"$\ln(P/P_{peak})$")
        ax5.set_ylabel(r"$\ln(pdf)-\ln(\max(pdf))$")
        ax52.set_ylabel(r"$\mu_{\Lambda}/\kappa$")
        ax522.set_ylabel(r"$\sigma_{\Lambda}^2$")
        ax52.set_xlabel(r"$\phi$")
        # ax52.set_yscale('log')
        fig4.savefig("{}/{}".format(figdir, "lnP_pdf{}.pdf".format(nparticles)))
        fig42.savefig("{}/{}".format(figdir, "muv_varv{}.pdf".format(nparticles)))
        fig5.savefig("{}/{}".format(figdir, "lnP_lnpdf{}.pdf".format(nparticles)))
        fig52.savefig("{}/{}".format(figdir, "muP_kappa{}.pdf".format(nparticles)))


    def ff(x, a, b):
        return a * x + b

    if True:
        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111)
        fig7 = plt.figure()
        ax7 = fig7.add_subplot(111)

        varlnp = [np.var(np.log(p)) for p in pressure]
        weights = [len(p) for p in pressure]
        y = np.asarray(varlnp) * nparticles
        spl = UnivariateSpline(phi_ss_packed, y, s=1e7, k=5, w=weights)
        xx = np.linspace(np.amin(phi_ss_packed), np.amax(phi_ss_packed), 1e4)
        yspl = spl(xx)
        ax7.errorbar(phi_ss_packed, y, marker='o', linestyle='', label='N={}'.format(nparticles),
                      color=color, rasterized=True)
        ax7.plot(xx, yspl, linewidth=2, color=color, rasterized=True)

        phi_c = xx[np.argmax(yspl)] #phi_ss[np.argmin(np.abs(np.asarray(psuccess) - 0.5))]
        print phi_c
        logx = np.abs(1 - phi_ss_packed / phi_c)
        color_cycle = get_color_cycle(ncolors=3)
        color = color_cycle.next()
        logy = logp_var[:, 0]
        yerr = (logp_var[:,2]-logp_var[:,1])/(2*logp_var[:, 0])
        ax6.errorbar(logx[phi_ss_packed>phi_c], logy[phi_ss_packed>phi_c], yerr=yerr[phi_ss_packed>phi_c], fmt='o', color=color, rasterized=True)
        ax6.errorbar(logx[phi_ss_packed <= phi_c], logy[phi_ss_packed <= phi_c], yerr=yerr[phi_ss_packed <= phi_c], fmt='o',
                     color=color_cycle.next(), rasterized=True)

        ax6.set_xlabel(r"$|1-\phi/\phi_c^{(N)}|$")
        ax6.set_ylabel(r"$\sigma^2_{\Lambda}$")
        fig6.savefig("{}/{}".format(figdir, "phi_varlnP{}.pdf".format(nparticles)))

def plot_all(figdir="figures", bdim=2):
    figdir = os.path.join(os.getcwd(), figdir)
    trymakedir(figdir)
    datasets = collect_data_every_set_all(data_signature='[0-9]*/findjstats.pickle')
    for dp in datasets:
        assert dp.initialized == True
    nparticles = np.asarray([dp.nparticles for dp in sorted(datasets, key=lambda data: data.nparticles)])

    def ff(x, a, b):
        return a * x + b

    if True:
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111)
        fig01 = plt.figure()
        ax01 = fig01.add_subplot(111)
        color_cycle = get_color_cycle(ncolors=len(datasets))
        for i,dp in enumerate(sorted(datasets, key=lambda data: data.nparticles)):
            color = color_cycle.next()
            phi_ss_packed, psuccess = dp.phi_ss_packed, dp.psuccess
            contacts_all = dp.contacts_all
            y = [[item for sublist in x for item in sublist] for x in contacts_all]
            ax0.plot(phi_ss_packed, [np.mean(x) for x in y], color=color, linestyle='', marker='o',
                     label=nparticles[i], rasterized=True)
            spl = UnivariateSpline(phi_ss_packed, [np.mean(x) for x in y],
                                   bbox=[phi_ss_packed[0], phi_ss_packed[-1]], s=5e-4, k=1)
            phi_ss_spl = np.linspace(0.81, 0.87, 1000)
            ax0.plot(phi_ss_spl, spl(phi_ss_spl), marker='', linestyle='-', color=color, rasterized=True)
            ax01.plot(phi_ss_packed, [np.var(x) for x in y], color=color, linestyle='-', marker='o',
                      label=nparticles[i], rasterized=True)
        ax0.set_xlabel(r'$\phi$')
        ax0.set_ylabel(r'$\langle Z \rangle_{\mathcal{B}}$')
        ax0.set_ylim((4,4.5))
        ax0.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        fig0.savefig("{}/{}".format(figdir, "phi_meanZ_all.pdf"))
        ax01.set_xlabel(r'$\phi$')
        ax01.set_ylabel(r'$\sigma^2_{\mathcal{B}}(Z)$')
        ax01.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        fig01.savefig("{}/{}".format(figdir, "phi_stdZ_all.pdf"))

    if True:
        phi_max = 0.85
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig02 = plt.figure()
        ax02 = fig02.add_subplot(111)
        fig03 = plt.figure()
        ax03 = fig03.add_subplot(111)
        yder_argavg, yder_argmid, yder_avg = [], [], []
        yspl_list, yder_list = [], []
        phi_ss_all = np.hstack([dp.phi_ss for dp in datasets])
        xspl = np.linspace(np.amin(phi_ss_all), phi_max, 300, endpoint=True)
        color_cycle = get_color_cycle(ncolors=len(datasets))
        for i, dp in enumerate(sorted(datasets, key=lambda data: data.nparticles)):
            color = color_cycle.next()
            phi_ss, psuccess = dp.phi_ss[dp.phi_ss<phi_max], dp.psuccess[dp.phi_ss<phi_max]
            phi_ss_packed, nrattlers = dp.phi_ss_packed, dp.nrattlers
            ax.plot(phi_ss, np.asarray(psuccess), marker='o', linestyle='', color=color, rasterized=True)
            # tck = splrep(phi_ss, psuccess, s=1e-3, k=3)
            # ynew = splev(xspl, tck, der=0)
            spl = UnivariateSpline(phi_ss, psuccess, bbox=[phi_ss[0], phi_ss[-1]], s=7.5e-4, k=3)
            ynew = spl(xspl)
            yspl_list.append(ynew)
            ax.plot(xspl, ynew, marker='None', linestyle='-', color=color, rasterized=True)
            # yder = splev(xspl, tck, der=1)
            yder = spl.derivative(1)(xspl)
            yder_list.append(yder)
            ax.plot(xspl, yder/np.amax(yder), marker='None', linestyle='--', color=color, rasterized=True)
            avg = quad(lambda x : x * spl.derivative(1)(x), 0.815, 0.845)[0]/quad(spl.derivative(1), 0.815, 0.845)[0]
            yder_argavg.append(avg)
            yder_avg.append(ynew[np.argmin(np.abs(xspl-avg))])
            yder_argmid.append(xspl[np.argmin(np.abs(ynew-0.5))])
            # nrattlers plot all
            ax02.plot(phi_ss_packed, np.asarray(nrattlers)/nparticles[i], marker='o', linestyle='', color=color, rasterized=True)
            spl = UnivariateSpline(phi_ss_packed, np.asarray(nrattlers)/nparticles[i], bbox=[phi_ss_packed[0], phi_ss_packed[-1]], s=0.05, k=3)
            phi_ss_spl = np.linspace(phi_ss_packed[0], phi_ss_packed[-1], 1000)
            ax02.plot(phi_ss_spl, spl(phi_ss_spl), marker='', linestyle='-', color=color, rasterized=True)
            ax02.set_xlim([phi_ss_packed[0], phi_ss_packed[-1]])
        print "yder_argavg ", yder_argavg
        ax.set_xlim([phi_ss[0], phi_ss[-1]])
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$p_{pack}$")
        fig.savefig("{}/{}".format(figdir, "phi_ppack_all.pdf"))
        ax02.set_xlabel(r"$\phi$")
        ax02.set_ylabel(r"$N_{r}/N$")
        ax02.set_xlim((0.81,0.87))
        fig02.savefig("{}/{}".format(figdir, "phi_nrattlers_all.pdf"))

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        color_cycle = get_color_cycle(ncolors=4)
        x = np.log(np.sqrt(nparticles))
        xnew = np.log(np.sqrt(np.linspace(np.amin(nparticles), np.amax(nparticles), 100)))
        color = color_cycle.next()
        y = np.log(0.845 - np.asarray(yder_argavg))
        ax2.plot(x, y, marker='o', linestyle='None', color=color, rasterized=True)
        fit_fn, fit_params, fit_err, rho = poly_fit(x, y, order=1)
        ax2.plot(xnew, fit_fn(xnew), color=color, linewidth=2,
                 label="argavg: : ln(phi) = {:.3f} ln(N)/2 {:.3f}".format(fit_params[0], fit_params[1]), rasterized=True)
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(lambda L, a, b, c: a - b*(L**(-1/c)), np.sqrt(nparticles), np.asarray(yder_argavg), p0=[0.845,1,1])
        print "ppack_argavg phi_c, c, nu ", popt, " err ", np.sqrt(np.diag(pcov))
        color = color_cycle.next()
        y = np.log(0.845 - np.asarray(yder_argmid))
        ax2.plot(x, y, marker='o', linestyle='None', color=color, rasterized=True)
        fit_fn, fit_params, fit_err, rho = poly_fit(x, y, order=1)
        ax2.plot(xnew, fit_fn(xnew), color=color, linewidth=2,
                 label="argmid: ln(phi) = {:.3f} ln(N)/2 {:.3f}".format(fit_params[0], fit_params[1]), rasterized=True)
        ax2.set_xlabel(r"$\ln(N^{1/2})$")
        ax2.legend(frameon=False, loc='best', prop={'size':glob_fontsize}, numpoints=1, scatterpoints=1,
                   markerscale=1, columnspacing=0.25, labelspacing=0.25)
        fig2.savefig("{}/{}".format(figdir, "lnphi_lnppack_scaling.pdf"))

        color = color_cycle.next()
        y = np.log(np.asarray(yder_avg))
        ax03.plot(x, y, marker='o', linestyle='None', color=color, rasterized=True)
        fit_fn, fit_params, fit_err, rho = poly_fit(x[1:], y[1:], order=1)
        ax03.plot(xnew, fit_fn(xnew), color=color, linewidth=2,
                 label="y(phiavg): ln(phi) = {:.3f} ln(N)/2 {:.3f}".format(fit_params[0], fit_params[1]),
                 rasterized=True)
        ax03.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1,
                   markerscale=1, columnspacing=0.25, labelspacing=0.25)
        ax03.set_xlabel(r"$\ln(N^{1/2})$")
        fig03.savefig("{}/{}".format(figdir, "lnphi_lnppack_beta_scaling.pdf"))
        # ax2.set_ylim([0, 1./(np.amin(nparticles)-5)])
        # ax2.set_yscale('log')
        # ax2.set_xscale('log')

    if False:
        import fssa
        l = np.sqrt(nparticles)
        rho = xspl
        # rho = rho[30:-30]
        # ycut = np.asarray([y[30:-30] for y in yspl_list])
        a = np.asarray(yspl_list)
        print a.shape
        da = np.ones(a.shape) * 1e-2
        rho_c0 = 0.845
        nu0 = 0.8
        zeta0 = -0.1
        ret = fssa.autoscale(l, rho, a, da, rho_c0, nu0, zeta0)
        print "rho: {} +/- {}".format(ret.rho, ret.drho)
        print "nu: {} +/- {}".format(ret.nu, ret.dnu)
        print "zeta: {} +/- {}".format(ret.zeta, ret.dzeta)
        print ret.fun
        auto_scaled_data = fssa.scaledata(l, rho, a, da, ret.rho, ret.nu, ret.zeta)
        # critical exponents and errors, quality of data collapse
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.set_prop_cycle(get_cycler(ncolors=len(nparticles)))
        for i, (x, y) in enumerate(zip(auto_scaled_data.x, auto_scaled_data.y)):
            ax3.plot(x, y, '.', label=nparticles[i], rasterized=True)
        ax3.set_ylabel(r'$\ln (p_{pack} (N^{1/d})^{\beta})$')
        ax3.set_xlabel(r'$\ln(\varepsilon N^{1/d})$')
        # ax3.set_ylabel(r'$\ln (p_{pack} (N^{1/d})^{\beta/\nu})$')
        # ax3.set_xlabel(r'$\ln(\varepsilon (N^{1/d})^{1/\nu})$')
        # ax3.set_title(r'$\phi_c \approx 0.847;~\nu \approx 1;~\beta \approx 1/20$')
        ax3.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        fig3.savefig("{}/{}".format(figdir, "phi_ppack_rescaling.pdf"))

        # rho = rho[15:-15]
        # ycut = np.asarray([y[15:-15] for y in yder_list])
        # a = np.asarray([y / simps(y, rho) for y in ycut])
        # # a = np.asarray(yder_list)
        # print a.shape
        # da = np.ones(a.shape) * 1e-1
        # rho_c0 = 0.84
        # nu0 = 1
        # zeta0 = 0.1
        # ret = fssa.autoscale(l, rho, a, da, rho_c0, nu0, zeta0)
        # print "rho: {} +/- {}".format(ret.rho, ret.drho)
        # print "nu: {} +/- {}".format(ret.nu, ret.dnu)
        # print "zeta: {} +/- {}".format(ret.zeta, ret.dzeta)
        # print ret.fun
        # auto_scaled_data = fssa.scaledata(l, rho, a, da, ret.rho, ret.nu, ret.zeta)
        # # critical exponents and errors, quality of data collapse
        # fig4 = plt.figure()
        # ax4 = fig4.add_subplot(111)
        # ax4.set_prop_cycle(get_cycler(len(nparticles)))
        # ax4.plot(auto_scaled_data.x.T, auto_scaled_data.y.T, '.', label=nparticles)
        # fig4.savefig("{}/{}".format(figdir, "phi_ppack_der_rescaling.pdf")

    if False:
        l = np.sqrt(nparticles)
        rho = np.asarray(phi_ss_packed[phi_ss_packed > phi_c])
        # rho = rho[30:-30]
        # ycut = np.asarray([y[30:-30] for y in yspl_list])
        a = np.asarray([dp.logp_var[:, 0][dp.phi_ss_packed > phi_c] for dp in datasets])
        print a.shape
        da = np.ones(a.shape) * 1e-1
        rho_c0 = 0.84
        nu0 = 1. / 2
        zeta0 = 1. / 3
        ret = fssa.autoscale(l, rho, a, da, rho_c0, nu0, zeta0)
        print "rho: {} +/- {}".format(ret.rho, ret.drho)
        print "nu: {} +/- {}".format(ret.nu, ret.dnu)
        print "zeta: {} +/- {}".format(ret.zeta, ret.dzeta)
        print ret.fun
        auto_scaled_data = fssa.scaledata(l, rho, a, da, ret.rho, ret.nu, ret.zeta)
        # critical exponents and errors, quality of data collapse
        fig8 = plt.figure()
        ax8 = fig8.add_subplot(111)
        ax8.set_prop_cycle(get_cycler(ncolors=len(nparticles)))
        ax8.plot(np.log(auto_scaled_data.x.T), np.log(auto_scaled_data.y.T), '.', rasterized=True)
        fig8.savefig("{}/{}".format(figdir, "lnphi_lnpvar_rescaling.pdf"))

    if True:
        from scipy.stats import moment
        # computation of the fourth order cumulants
        fig9 = plt.figure()
        ax9 = fig9.add_subplot(111)
        ax9.set_prop_cycle(get_cycler(ncolors=len(nparticles)))
        left, bottom, width, eight = [0.5, 0.2, 0.4, 0.4]
        # ax9inset = fig9.add_axes([left, bottom, width, eight])
        # ax9inset.set_prop_cycle(get_cycler(ncolors=len(nparticles)))
        fig11 = plt.figure()
        ax11 = fig11.add_subplot(111)
        fig12 = plt.figure()
        ax12 = fig12.add_subplot(111)
        fig13 = plt.figure()
        ax13 = fig13.add_subplot(111)
        fig14 = plt.figure()
        ax14 = fig14.add_subplot(111)
        fig15 = plt.figure()
        ax15 = fig15.add_subplot(111)
        fig16 = plt.figure()
        ax16 = fig16.add_subplot(111)
        fig17 = plt.figure()
        ax17 = fig17.add_subplot(111)
        fig18 = plt.figure()
        ax18 = fig18.add_subplot(111)
        color_cycle = get_color_cycle(ncolors=len(nparticles))
        phi_max = 0.87
        varlnp_argmax, varprel_argmax = [], []
        varlnp_max, varprel_max = [], []
        for i, dp in enumerate(sorted(datasets, key=lambda data: data.nparticles)):
            phi_ss_packed, logp_var, ptyp_var, logp_mean = dp.phi_ss_packed[dp.phi_ss_packed < phi_max], \
                                                           dp.logp_var[dp.phi_ss_packed < phi_max], \
                                                           dp.p_rel_var[dp.phi_ss_packed < phi_max], \
                                                           dp.logp_mean[dp.phi_ss_packed < phi_max]
            p_var, p_mean = dp.p_var[dp.phi_ss_packed < phi_max], \
                            dp.p_mean[dp.phi_ss_packed < phi_max]
            prel_var = np.asarray([np.var(p) / (np.mean(p) ** 2) for p in dp.pressure])[dp.phi_ss_packed < phi_max]
            lnp_u4_list, p_u4_list = [], []
            weights = []
            for phi_ss, pressure in zip(phi_ss_packed, dp.pressure):
                if phi_ss < phi_max:
                    lnp = np.log(pressure)
                    relp = np.asarray(pressure)/ np.mean(pressure)
                    m2c= moment(lnp, 2)
                    m4c = moment(lnp, 4)
                    lnp_u4 = 1 - m4c/(3*(m2c**2))
                    p_u4 = 1 - moment(relp, 4)/(3*(moment(relp, 2)**2))
                    lnp_u4_list.append(lnp_u4)
                    p_u4_list.append(p_u4)
                    weights.append(np.sqrt(len(pressure)))
                    # u4_err.append(bootstrap.ci(lnp, lambda x : 1 - moment(x,4)/(3*moment(x, 2)**2), n_samples=1000))
                    # chi_err.append(bootstrap.ci(lnp, lambda x: np.var(np.abs(x)), n_samples=1000 ))
            color = color_cycle.next()
            x = np.asarray(phi_ss_packed)
            xx = np.linspace(np.amin(x), np.amax(x), 10000)
            y, yerr = np.asarray(lnp_u4_list), []
            spl = UnivariateSpline(x, y, s=1e3, k=3, w=weights)
            ax9.errorbar(x, y, marker='o', linestyle='', label='N={}'.format(nparticles[i]), color=color, rasterized=True)
            ax9.plot(xx, spl(xx), linewidth=2, color=color, rasterized=True)
            # ax9inset.errorbar(x[x>0.84], y[x>0.84], fmt='o', label='N={}'.format(nparticles[i]), rasterized=True)
            y, yerr = np.asarray(p_u4_list), []
            spl = UnivariateSpline(x, y, s=5e6, k=5, w=weights)
            ax16.errorbar(x, y, marker='o', linestyle='', label='N={}'.format(nparticles[i]), color=color,
                         rasterized=True)
            ax16.plot(xx, spl(xx), linewidth=2, color=color, rasterized=True)
            # mean of lnP
            y = logp_mean[:,0]
            yerr = np.asarray([logp_mean[:, 1], logp_mean[:, 2]])/3
            w = 0.5*(logp_mean[:, 2]-logp_mean[:, 1])/np.sqrt(weights)
            spl = UnivariateSpline(x, y, s=10, k=1, w=w)
            ax11.errorbar(x, y, yerr=yerr, marker='o', linestyle='', label='N={}'.format(nparticles[i]),
                          color=color, rasterized=True)
            ax11.plot(xx, spl(xx), linewidth=2, color=color, rasterized=True)
            # var of lnP
            y = logp_var[:,0]*nparticles[i]
            yerr = np.asarray([logp_var[:, 2]*nparticles[i], logp_var[:, 2]*nparticles[i]])/3
            w = 0.5*(logp_var[:, 2]-logp_var[:, 1])/np.sqrt(weights)
            spl = UnivariateSpline(x, y, s=1e6, k=5, w=w)
            yspl = spl(xx)
            ax12.errorbar(x, y, yerr=yerr, marker='o', linestyle='', label='N={}'.format(nparticles[i]),
                          color=color, rasterized=True)
            ax12.plot(xx, yspl, linewidth=2, color=color, rasterized=True)
            varlnp_argmax.append(xx[np.argmax(yspl)])
            varlnp_max.append(np.amax(yspl)/nparticles[i])
            # mean of P
            y = p_mean[:,0]
            yerr = np.asarray([p_mean[:, 1], p_mean[:, 2]])/3
            w = 0.5 * (p_mean[:, 2] - p_mean[:, 1]) / np.sqrt(weights)
            spl = UnivariateSpline(x, y, s=10, k=1, w=w)
            ax13.errorbar(x, y, yerr=yerr, marker='o', linestyle='', label='N={}'.format(nparticles[i]),
                          color=color, rasterized=True)
            ax13.plot(xx, spl(xx), linewidth=2, color=color, rasterized=True)
            ax13.set_yscale('log')
            # var of rel P
            y = nparticles[i] * prel_var
            spl = UnivariateSpline(x, y, s=3.8e6, k=5, w=weights)
            yspl = spl(xx)
            ax14.errorbar(x, y, marker='o', linestyle='', label='N={}'.format(nparticles[i]),
                          color=color, rasterized=True)
            ax14.plot(xx, yspl, linewidth=2, color=color, rasterized=True)
            varprel_argmax.append(xx[np.argmax(yspl)])
            varprel_max.append(np.amax(yspl)/nparticles[i])
        color_cycle = get_color_cycle(ncolors=3)
        color = color_cycle.next()
        x = np.log(np.sqrt(nparticles))
        y = np.log(0.844 - np.asarray(varlnp_argmax))
        ax15.plot(x, y, marker='o', color=color, label=r'$\sigma^2_{\Lambda}$')
        fit_fn, fit_params, fit_err, rho = poly_fit(x, y, order=1)
        ax15.plot(x, fit_fn(x), marker='', linewidth=2, color=color)
        varlnp_argmax_fn = fit_fn
        print "varlnP_argmax fit params: ", fit_params, " fit_err: ", fit_err
        # f = lambda L, a, b, c: a - b * (L ** (-1 / c))
        # popt, pcov = curve_fit(f, np.sqrt(nparticles), np.asarray(varlnp_argmax),
        #                        p0=[0.828, 2, 0.5], maxfev=3000)
        # print "varlnP_argmax phi_c, c, nu ", popt, " err ", np.sqrt(np.diag(pcov))
        # ax18.plot(np.sqrt(nparticles), np.asarray(varlnp_argmax), marker='o', color='b')
        # ax18.plot(np.exp(xnew), f(np.exp(xnew), popt[0],popt[1],popt[2]), linestyle='--', color='b')
        color = color_cycle.next()
        y = np.log(0.844 - np.asarray(varprel_argmax))
        ax15.plot(x, y, marker='s', color=color, label=r'$\sigma^2_{P}/\langle P \rangle^2$')
        fit_fn, fit_params, fit_err, rho = poly_fit(x[1:], y[1:], order=1)
        ax15.plot(x, fit_fn(x), marker='', linewidth=2, color=color)
        print "varprel_argmax fit params: ", fit_params, " fit_err: ", fit_err
        # popt, pcov = curve_fit(f, np.sqrt(nparticles), np.asarray(varprel_argmax),
        #                        p0=[0.828, 2, 0.5], maxfev=3000)
        # print "varprel_argmax phi_c, c, nu ", popt, " err ", np.sqrt(np.diag(pcov))
        # ax18.plot(np.sqrt(nparticles), np.asarray(varprel_argmax), marker='o', color='r')
        # ax18.plot(np.exp(xnew), f(np.exp(xnew), popt[0], popt[1], popt[2]), linestyle='--', color='r')
        color_cycle = get_color_cycle(ncolors=3)
        color = color_cycle.next()
        x = np.log(np.sqrt(nparticles))
        y = np.log(varlnp_max)
        ax17.plot(x, y, marker='o', color=color, label=r'$\sigma^2_{\Lambda}$')
        fit_fn, fit_params, fit_err, rho = poly_fit(x, y, order=1)
        ax17.plot(x, fit_fn(x), marker='', linewidth=2, color=color)
        print "varlnP_max fit params: ", fit_params, " fit_err: ", fit_err
        color = color_cycle.next()
        y = np.log(varprel_max)
        ax17.plot(x, y, marker='s', color=color, label=r'$\sigma^2_{P}/\langle P \rangle^2$')
        fit_fn, fit_params, fit_err, rho = poly_fit(x[1:], y[1:], order=1)
        ax17.plot(x, fit_fn(x), marker='', linewidth=2, color=color)
        varprel_argmax_fn = fit_fn
        print "varprel_max fit params: ", fit_params, " fit_err: ", fit_err

        # ax9inset.set_xlim((0.84,0.87))
        ax9.set_xlim((0.81, phi_max))
        ax11.set_xlim((0.81, phi_max))
        ax12.set_xlim((0.81, phi_max))
        ax13.set_xlim((0.81, phi_max))
        ax14.set_xlim((0.81, phi_max))
        ax16.set_xlim((0.81, phi_max))
        ax9.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                    columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax11.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                    columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax12.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                    columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax13.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                    columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax14.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                    columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax15.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                    columnspacing=0.25, labelspacing=0.25)
        ax16.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                    columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax17.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1,
                    markerscale=1, columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax9.set_ylabel(r'$V_N(\Lambda)$')
        ax9.set_xlabel(r'$\phi$')
        ax16.set_ylabel(r'$V_N(P/\langle P \rangle)$')
        ax16.set_xlabel(r'$\phi$')
        ax11.set_xlabel(r'$\phi$')
        ax11.set_ylabel(r'$\langle \Lambda \rangle$')
        ax12.set_xlabel(r'$\phi$')
        ax12.set_ylabel(r'$N \sigma^2_{\Lambda}$')
        ax13.set_xlabel(r'$\phi$')
        ax13.set_ylabel(r'$\langle P \rangle$')
        ax14.set_xlabel(r'$\phi$')
        ax14.set_ylabel(r'$N \sigma^{2}_P/\langle P \rangle^2$')
        ax15.set_xlabel(r'$\ln(N^{1/d})$')
        ax15.set_ylabel(r'$\ln(\phi_{\arg \max y})$')
        ax17.set_xlabel(r'$\ln(N^{1/d})$')
        ax17.set_ylabel(r'$\ln(y_{max})$')
        fig9.savefig("{}/{}".format(figdir, "lnp_u4.pdf"))
        fig11.savefig("{}/{}".format(figdir, "lnp_mean.pdf"))
        fig12.savefig("{}/{}".format(figdir, "lnp_var.pdf"))
        fig13.savefig("{}/{}".format(figdir, "p_mean.pdf"))
        fig14.savefig("{}/{}".format(figdir, "p_var.pdf"))
        fig15.savefig("{}/{}".format(figdir, "lnp_var_argmax.pdf"))
        fig16.savefig("{}/{}".format(figdir, "p_u4.pdf"))
        fig17.savefig("{}/{}".format(figdir, "lnp_var_max.pdf"))

    if True:
        import matplotlib.gridspec as gridspec
        phi_max = 0.87
        # fig5 = plt.figure(figsize=(8, 8))
        # gs = gridspec.GridSpec(7, 2)
        # ax5 = fig5.add_subplot(gs[:4, :])
        fig51 = plt.figure()
        fig52 = plt.figure()
        fig53 = plt.figure()
        fig54 = plt.figure()
        ax51 = fig51.add_subplot(111)
        ax52 = fig52.add_subplot(111)
        ax53 = fig53.add_subplot(111)
        ax54 = fig54.add_subplot(111)
        fit_params_list, fit_params_list_err = [], []
        color_cycle = get_color_cycle(ncolors=len(datasets))
        color_cycle2 = get_color_cycle(ncolors=len(datasets))
        fss = Result()
        fss.nparticles = nparticles
        fss.pressure = [dp.pressure for dp in sorted(datasets, key=lambda data: data.nparticles)]
        fss.phi_ss_packed = [dp.phi_ss_packed for dp in sorted(datasets, key=lambda data: data.nparticles)]
        fss.phi_c_lnp = np.exp([varlnp_argmax_fn(0.5 * np.log(n)) for n in nparticles])
        fss.phi_c_prel = np.exp([varprel_argmax_fn(0.5 * np.log(n)) for n in nparticles])
        pickle.dump(fss, open(os.path.join(os.getcwd(), "pressure_fss.pickle"), "wb"))
        for i, dp in enumerate(sorted(datasets, key=lambda data: data.nparticles)):
            phi_ss_packed, logp_var, ptyp_var, logp_mean = dp.phi_ss_packed[dp.phi_ss_packed < phi_max], \
                                                           dp.logp_var[dp.phi_ss_packed < phi_max], \
                                                           dp.p_rel_var[dp.phi_ss_packed < phi_max], \
                                                           dp.logp_mean[dp.phi_ss_packed < phi_max]
            prel_var = np.asarray([np.var(p) / (np.mean(p) ** 2) for p in dp.pressure])[dp.phi_ss_packed < phi_max]
            inu = 2.
            zeta = 1./4.
            # lambda
            phi_c = 0.844 #np.exp(varlnp_argmax_fn(0.5 * np.log(nparticles[i])))  #DEBUG
            x = np.power(np.sqrt(nparticles[i]), inu) * np.abs(
            np.asarray(phi_ss_packed) - np.asarray(phi_c)) / phi_c
            logx = x
            y = np.power(np.sqrt(nparticles[i]), zeta * inu) * logp_var[:, 0] #[phi_ss_packed > phi_c]
            logy = y
            logy_err = np.asarray([logp_var[:, 1], logp_var[:, 2]])/3 #* logp_var[:, 0])
            # np.log(p_rel_var[:, 0]c)
            # ax6.errorbar(phi, logp_mean[:, 0], yerr=[logp_mean[:, 1], logp_mean[:, 2]])
            # yerr = [np.log(p_rel_var[:, 1]), np.log(p_rel_var[:, 2])]
            color = color_cycle.next()
            # ax5.errorbar(logx, logy, fmt='o', color=color, yerr=logy_err, rasterized=True)
            ax51.errorbar(logx, logy, fmt='o', color=color, yerr=logy_err,
                          label='N={}'.format(nparticles[i]), rasterized=True)
            ax53.errorbar(logx, logy, fmt='o', color=color, yerr=logy_err,
                          label='N={}'.format(nparticles[i]), rasterized=True)
            # idx = np.argmax(logy < 0.5 * np.amax(logy))  # int(len(logx)*0.2)
            # fit_fn, fit_params, fit_err, rho = poly_fit(logx[idx:], logy[idx:], yerr=logy_err[idx:], order=1)
            # ax5.plot(logx[idx:], fit_fn(logx[idx:]), color=color, linewidth=2,
            #          label="N:{}; {:.3f} ln(x) + {:.3f}".format(dp.nparticles, fit_params[0], fit_params[1]), rasterized=True)
            # ax51.plot(logx[idx:], fit_fn(logx[idx:]), color=color, linewidth=2,
            #           label="{}".format(dp.nparticles), rasterized=True)
            # fit_params_list.append(fit_params)
            # fit_params_list_err.append(fit_err)
            # print "nparticles: {}; ({}+/-{}) x + ({}+/-{})".format(dp.nparticles, fit_params[0], fit_err[0],
            #                                                        fit_params[1], fit_err[1])
            #prel
            phi_c = 0.839 #np.exp(varprel_argmax_fn(0.5 * np.log(nparticles[i])))  # DEBUG
            inu = 2.
            zeta = 1./3.
            x = np.power(np.sqrt(nparticles[i]), inu) * np.abs(np.asarray(phi_ss_packed) - np.asarray(phi_c)) / phi_c
            y = np.power(np.sqrt(nparticles[i]), zeta * inu) * prel_var
            fit_fn, fit_params, fit_err, rho = poly_fit(np.log(x)[phi_ss_packed > phi_c], np.log(y)[phi_ss_packed > phi_c], order=1)
            print "phi > phi_c relp fit paramters: ", fit_params, " fit errs: ", fit_err
            fit_fn, fit_params, fit_err, rho = poly_fit(np.log(x)[phi_ss_packed < phi_c],
                                                        np.log(y)[phi_ss_packed < phi_c], order=1)
            print "phi < phi_c relp fit paramters: ", fit_params, " fit errs: ", fit_err
            color = color_cycle2.next()
            ax52.errorbar(x, y, fmt='o', color=color, yerr=None,
                          label='N={}'.format(nparticles[i]), rasterized=True)
            ax54.errorbar(x, y, fmt='o', color=color, yerr=None,
                          label='N={}'.format(nparticles[i]), rasterized=True)
            # idx = np.argmax(logy < 0.5 * np.amax(logy))  # int(len(logx)*0.2)
            # fit_fn, fit_params, fit_err, rho = poly_fit(logx[idx:], logy[idx:], yerr=None, order=1)
            # ax52.plot(logx[idx:], fit_fn(logx[idx:]), color=color, linewidth=2,
            #           label="{}".format(dp.nparticles), rasterized=True)
            # print "nparticles: {}; ({}+/-{}) x + ({}+/-{})".format(dp.nparticles, fit_params[0], fit_err[0],
            #                                                        fit_params[1], fit_err[1])

        ax51.set_ylabel(r'$\sigma^2_{\Lambda} (N^{1/d})^{\alpha/\nu}$')
        ax51.set_xlabel(r'$\varepsilon (N^{1/d})^{1/\nu}$')
        ax51.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax53.set_yscale('log')
        ax53.set_xscale('log')
        ax53.set_ylabel(r'$\sigma^2_{\Lambda} (N^{1/d})^{\alpha/\nu}$')
        ax53.set_xlabel(r'$\varepsilon (N^{1/d})^{1/\nu}$')
        ax53.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1,
                    markerscale=1, columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax52.set_ylabel(r'$\sigma^2_{P/\langle P \rangle^2} (N^{1/d})^{\alpha/\nu}$')
        ax52.set_xlabel(r'$\varepsilon (N^{1/d})^{1/\nu}$')
        ax52.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                    columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax54.set_yscale('log')
        ax54.set_xscale('log')
        ax54.set_ylabel(r'$\sigma^2_{P/\langle P \rangle^2} (N^{1/d})^{\alpha/\nu}$')
        ax54.set_xlabel(r'$\varepsilon (N^{1/d})^{1/\nu}$')
        ax54.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1,
                    markerscale=1, columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        fig51.savefig("{}/{}".format(figdir, "phi_lnpvar.pdf"))
        fig52.savefig("{}/{}".format(figdir, "phi_prelvar.pdf"))
        fig53.savefig("{}/{}".format(figdir, "lnphi_lnpvar.pdf"))
        fig54.savefig("{}/{}".format(figdir, "lnphi_prelvar.pdf"))

        # ax5.set_title(r'$\nu = 1;~\alpha \approx 0.015$')
        # ax5.set_ylabel(r'$\ln(\sigma^2_{\Lambda} (N^{1/d})^{\alpha/\nu})$')
        # ax5.set_xlabel(r'$\ln((N^{1/d}/\xi)^{1/\nu})$')
        # ax5.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
        #            columnspacing=0.5, labelspacing=0.25, handletextpad=0)
        # ax6 = fig5.add_subplot(gs[4:, 0])
        # ax7 = fig5.add_subplot(gs[4:, 1])
        # color = color_cycle.next()
        # # slopes subplot
        # ax6.errorbar(nparticles, np.asarray(fit_params_list)[:, 0], yerr=np.asarray(fit_params_list_err)[:, 0],
        #              fmt='o', color=color, rasterized=True)
        # fit_fn, fit_params, fit_err, rho = poly_fit(nparticles, np.asarray(fit_params_list)[:, 0],
        #                                             yerr=np.asarray(fit_params_list_err)[:, 0], order=1)
        # ax6.plot(nparticles, fit_fn(nparticles), color=color, linewidth=2,
        #          label="{:.3f} x + {:.3f}".format(fit_params[0], fit_params[1]), rasterized=True)
        # print "fit slopes; ({}+/-{}) x + ({}+/-{})".format(fit_params[0], fit_err[0], fit_params[1], fit_err[1])
        # # intercepts subplot
        # ax7.errorbar(nparticles, np.asarray(fit_params_list)[:, 1], yerr=np.asarray(fit_params_list_err)[:, 1],
        #              fmt='o', color=color, rasterized=True)
        # fit_fn, fit_params, fit_err, rho = poly_fit(nparticles, np.asarray(fit_params_list)[:, 1],
        #                                             yerr=np.asarray(fit_params_list_err)[:, 1], order=1)
        # ax7.plot(nparticles, fit_fn(nparticles), color=color, linewidth=2,
        #          label="{:.3f} x + {:.3f}".format(fit_params[0], fit_params[1]), rasterized=True)
        # print "fit interceps ; ({}+/-{}) x + ({}+/-{})".format(fit_params[0], fit_err[0], fit_params[1], fit_err[1])
        # # ax6 ax7 legends
        # ax6.set_xlabel('N')
        # ax6.set_ylabel('slope')
        # ax6.legend(frameon=False, loc='best', prop={'size': 13}, numpoints=1, scatterpoints=1, markerscale=1,
        #            columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        # ax7.set_xlabel('N')
        # ax7.set_ylabel('intercept')
        # ax7.legend(frameon=False, loc='best', prop={'size': 13}, numpoints=1, scatterpoints=1, markerscale=1,
        #            columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        # fig5.savefig("{}/{}".format(figdir, "lnphi_lnpvar_sub.pdf"))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="plot findjstats")
    parser.add_argument("-n", "--nparticles", type=int, help="number of particles")
    parser.add_argument("--all", action='store_true', help="run over all packings", default=False)
    parser.add_argument("--show", action='store_true', help="show plots", default=False)
    args = parser.parse_args()

    if args.all:
        plot_all()
    else:
        plot(nparticles=args.nparticles)

    if args.show:
        plt.show()

    # x, y = np.log(sim.energy_list), np.log(sim.pressure_list)
    # plt.scatter(x, y)
    # from scipy.optimize import curve_fit
    # def ff(x, a, b):
    #     return a * x + b
    # popt, pcov = curve_fit(ff, x, y)
    # plt.plot(x, ff(x, popt[0], popt[1]), label=r"$\ln P = {}\ln E + {}$".format(popt[0], popt[1]))
    # print "fit: ", popt
    # plt.legend()
    # plt.xlabel(r"$\ln E$")
    # plt.ylabel(r"$\ln P$")
    # plt.show()
