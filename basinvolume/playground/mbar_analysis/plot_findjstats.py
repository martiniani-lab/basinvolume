from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import rc
from pele.optimize import Result
from basinvolume.utils import *
from joblib import Parallel, delayed
import cPickle as pickle
from itertools import cycle
from cycler import cycler
from basinvolume.spheres.find_jstats import SoftPackingDataset, SoftPackingData
from sklearn.neighbors import KernelDensity
from basinvolume.experiment_2d.cross_validation_bandwidth_selection import get_bandwidth_estimate, get_pdf
from scipy import integrate
import scikits.bootstrap as bootstrap
from scipy.optimize import curve_fit
import argparse
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps

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

class DataPlot(object):
    def __init__(self, psuccess, phi_ss, nrattlers, energy, pressure, contacts, contacts_all, prob_min=0.05, bdim=2,
                 nparticles=64):
        self.psuccess = np.array(psuccess)
        self.phi_ss = np.array(phi_ss)
        self.phi_ss_packed = self.phi_ss[self.psuccess > prob_min]
        self.nrattlers = np.array(nrattlers)[self.psuccess > prob_min]
        self.energy = np.array(energy)[self.psuccess > prob_min]
        self.pressure = np.array(pressure)[self.psuccess > prob_min]
        self.contacts = np.array(contacts)[self.psuccess > prob_min]
        self.contacts_all = np.array(contacts_all)[self.psuccess > prob_min]
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
            lnp = np.log(np.array(p))
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
            p_rel = p / np.exp(log_maxp)
            n = int(min(n_samples, lnp.size*25))
            varCIs = bootstrap.ci(p_rel, np.var, n_samples=n)
            self.p_rel_var.append([np.var(p_rel), varCIs[0], varCIs[1]])
            #build array of logp mean, var and maxp
            meanCIs = bootstrap.ci(lnp, np.mean, n_samples=n)
            varCIs = bootstrap.ci(lnp, np.var, n_samples=n)
            self.logp_mean.append([np.mean(lnp), meanCIs[0], meanCIs[1]])
            self.logp_var.append([np.var(lnp), varCIs[0], varCIs[1]])
            self.logp_mode.append(log_maxp)
            meanCIs = bootstrap.ci(p, np.mean, n_samples=n)
            varCIs = bootstrap.ci(p, np.var, n_samples=n)
            self.p_mean.append([np.mean(p), meanCIs[0], meanCIs[1]])
            self.p_var.append([np.var(p), varCIs[0], varCIs[1]])

        self.logp_mean, self.p_mean = np.array(self.logp_mean), np.array(self.p_mean)
        self.logp_var, self.p_var, self.p_rel_var = np.array(self.logp_var), np.array(self.p_var), np.array(self.p_rel_var)
        self.bw = np.array(self.bw)
        self.log_pdf = np.array(self.log_pdf)
        self.log_pdf_x = np.array(self.log_pdf_x)
        self.initialized = True

def collect_data_plot(datasets, phi_max=0.871, bdim=2):
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
        for i, dataset in enumerate(sorted(datasets, key=lambda data: data.phi_ss)):
            if phi_ == dataset.phi_ss:
                tmp = np.array(dataset.success, dtype='int')
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
                    else:
                        tmp[np.argmax(tmp > 0)] = 0
                success_.extend(tmp)
                # plt.scatter(contacts_, np.log(pressure_))
        # plt.show()
        psuccess.append(np.mean(success_))
        nrattlers.append(np.mean(nrattlers_))
        energy.append(energy_)
        pressure.append(pressure_)
        contacts.append(contacts_)
        contacts_all.append(contacts_all_)
    return psuccess, phi_ss, nrattlers, energy, pressure, contacts, contacts_all

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
        psuccess, phi_ss, nrattlers, energy, pressure, contacts, contacts_all = collect_data_plot(datasets)
        dp = DataPlot(psuccess, phi_ss, nrattlers, energy, pressure, contacts, contacts_all, bdim=bdim, nparticles=nparticles)
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
        ax.plot(phi_ss, np.array(psuccess), marker='o', linestyle='', rasterized=True)
        spl = UnivariateSpline(phi_ss, psuccess, bbox=[phi_ss[0], phi_ss[-1]], s=7.5e-4, k=3)
        pickle.dump(spl, open(os.path.join(os.getcwd(), "phi_psuccess_spline.pickle"), "wb"))
        phi_ss_spl = np.linspace(phi_ss[0], phi_ss[-1], 1000)
        ax.plot(phi_ss_spl, spl(phi_ss_spl), marker='', linestyle='-', rasterized=True)
        ax.set_xlim([phi_ss[0],phi_ss[-1]])
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$p_{pack}$")
        fig.savefig("{}/{}".format(figdir, "phi_ppack.pdf"))

    if True:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(phi_ss_packed, np.array(nrattlers), marker='o', rasterized=True)
        spl = UnivariateSpline(phi_ss_packed, nrattlers, bbox=[phi_ss_packed[0], phi_ss_packed[-1]], s=0.05, k=3)
        pickle.dump(spl, open(os.path.join(os.getcwd(), "phi_nrattlers_spline.pickle"), "wb"))
        phi_ss_spl = np.linspace(phi_ss_packed[0], phi_ss_packed[-1], 1000)
        ax1.plot(phi_ss_spl, spl(phi_ss_spl), marker='', linestyle='-', rasterized=True)
        ax1.set_xlim([phi_ss_packed[0], phi_ss_packed[-1]])
        ax1.set_xlabel(r"$\phi$")
        ax1.set_ylabel(r"$n_{rattlers}$")
        fig1.savefig("{}/{}".format(figdir, "phi_nrattlers.pdf"))

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
        popt, pcov = curve_fit(ff, flat_energy[flat_energy<-5], flat_pressure[flat_energy<-5], absolute_sigma=True)
        fit_err = np.sqrt(np.diag(pcov))
        ax2.plot(xnew, ff(xnew, popt[0], popt[1]), color='k', linewidth=2.5, linestyle='--',
                 label="ln(p) = {:.3f} ln(E) + {:.3f}".format(popt[0], popt[1]), rasterized=True)
        # linear fit high P
        xnew = np.linspace(2.3, 10, 10)
        popt, pcov = curve_fit(ff, flat_energy[flat_energy > 5], flat_pressure[flat_energy > 5], absolute_sigma=True)
        fit_err = np.sqrt(np.diag(pcov))
        ax2.plot(xnew, ff(xnew, popt[0], popt[1]), color='r', linewidth=2.5, linestyle='--',
                 label="ln(p) = {:.3f} ln(E) + {:.3f}".format(popt[0], popt[1]), rasterized=True)
        # quadratic fit
        xnew = np.linspace(-0.5, 10, 10)
        popt, pcov = curve_fit(ff2, flat_energy[flat_energy > -0.5], flat_pressure[flat_energy > -0.5], absolute_sigma=True)
        fit_err = np.sqrt(np.diag(pcov))
        ax2.plot(xnew, ff2(xnew, popt[0], popt[1], popt[2]), color='b', linewidth=2.5, linestyle=':',
                 label="ln(p) = {:.3f} ln(E)**2 + {:.3f} ln(E) + {:.3f}".format(popt[0], popt[1], popt[2]),
                 rasterized=True)
        ax2.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                  columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax2.set_xlabel(r"$\ln(E)$")
        ax2.set_ylabel(r"$\ln(P)$")
        fig2.savefig("{}/{}".format(figdir, "lnE_lnP.pdf"))

    if True:
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
        fig3.savefig("{}/{}".format(figdir, "lnP_lnZ.pdf"))

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
        ikappa = lambda phi : 1.79473551963 * phi - 1.45331466838
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

            bias = np.exp(np.log(pressure[i]) * ikappa(phi_ss_packed[i]))
            y = 1./bias
            mu_B, sig_B = np.mean(y), np.var(y)
            mu_U = np.average(y, weights=bias/np.sum(bias))
            sig_U = np.average((y-mu_U)**2, weights=bias/np.sum(bias))
            rel_dmu = 2*np.abs(mu_U - mu_B)/(np.abs(mu_B)+np.abs(mu_U))
            rel_dsig = 2*np.abs(sig_U-sig_B)/(np.abs(sig_B)+np.abs(sig_U))
            v_mean_bias.append(mu_B)
            v_var_bias.append(sig_B)
            v_mean.append(mu_U)
            v_var.append(sig_U)
            ax52.plot(phi_ss_packed[i], np.mean(np.log(pressure[i]))*ikappa(phi_ss_packed[i]), color=color, marker='o', linestyle='',
                      rasterized=True)
            ax522.plot(phi_ss_packed[i], np.var(np.log(pressure[i])), color=color, marker='^',
                      linestyle='', rasterized=True)

        ax43 = ax42.twinx()
        ax42.plot(phi_ss_packed, v_mean_bias, color='b', marker='o', linestyle='',
                  rasterized=True, label=r"$E_B(v)$")
        ax43.plot(phi_ss_packed, v_var_bias, color='r', marker='o', linestyle='',
                  rasterized=True, label=r"$Var_B(v)$")
        ax42.plot(phi_ss_packed, v_mean, color='b', marker='^', linestyle='',
                  rasterized=True, label=r"$E_U(v)$")
        ax43.plot(phi_ss_packed, v_var, color='r', marker='^', linestyle='',
                  rasterized=True, label=r"$Var_U(v)$")
        # ax42.set_yscale('log')
        ax42.legend(frameon=False, loc=2, prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax43.legend(frameon=False, loc=0, prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
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
        fig4.savefig("{}/{}".format(figdir, "lnP_pdf.pdf"))
        fig42.savefig("{}/{}".format(figdir, "muv_varv.pdf"))
        fig5.savefig("{}/{}".format(figdir, "lnP_lnpdf.pdf"))
        fig52.savefig("{}/{}".format(figdir, "muP_kappa.pdf"))


    def ff(x, a, b):
        return a * x + b

    if True:
        phi_c = phi_ss[np.argmin(np.abs(np.array(psuccess)-0.5))]
        print phi_c
        logx = np.log(np.abs(1 - phi_ss_packed/phi_c))
        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111)
        # ax6.errorbar(phi, logp_var[:,0], yerr=[logp_var[:,1], logp_var[:,2]])
        # ax6.errorbar(phi, logp_mean[:, 0], yerr=[logp_mean[:, 1], logp_mean[:, 2]])
        # yerr = [np.log(p_rel_var[:, 1]), np.log(p_rel_var[:, 2])]
        color_cycle = get_color_cycle(ncolors=3)
        color = color_cycle.next()
        logy = np.log(logp_var[:, 0])
        yerr = (logp_var[:,2]-logp_var[:,1])/(2*logp_var[:, 0])
        ax6.errorbar(logx[phi_ss_packed>phi_c], logy[phi_ss_packed>phi_c], yerr=yerr[phi_ss_packed>phi_c], fmt='o', color=color, rasterized=True)
        ax6.errorbar(logx[phi_ss_packed <= phi_c], logy[phi_ss_packed <= phi_c], yerr=yerr[phi_ss_packed <= phi_c], fmt='o',
                     color=color_cycle.next(), rasterized=True)
        # popt, pcov = curve_fit(ff, logx, logy, sigma=yerr, absolute_sigma=True)
        # fit_err = np.sqrt(np.diag(pcov))
        # ax6.plot(logx, ff(logx, popt[0], popt[1]), color=color, linewidth=2)
        # print "({}+/-{}) x + ({}+/-{})".format(popt[0], fit_err[0], popt[1], fit_err[1])

        # logp_mode = np.array(logp_mode)[phi_ss_packed > phi_c]
        # color = color_cycle.next()
        # ax6.scatter(logx, np.log(logp_mode), color=color)
        # popt, pcov = curve_fit(ff, logx, np.log(logp_mode), absolute_sigma=True)
        # fit_err = np.sqrt(np.diag(pcov))
        # ax6.plot(logx, ff(logx, popt[0], popt[1]), color=color, linewidth=2)
        # print "({}+/-{}) x + ({}+/-{})".format(popt[0], fit_err[0], popt[1], fit_err[1])

        ax6.set_xlabel(r"$|1-\phi/\phi_0|$")
        ax6.set_ylabel(r"$\sigma^2_{\Lambda}$")
        fig6.savefig("{}/{}".format(figdir, "phi_varlnP.pdf"))

def plot_all(figdir="figures", bdim=2):
    figdir = os.path.join(os.getcwd(), figdir)
    trymakedir(figdir)
    datasets = collect_data_every_set_all(data_signature='[0-9]*/findjstats.pickle')
    for dp in datasets:
        assert dp.initialized == True
    nparticles = np.array([dp.nparticles for dp in sorted(datasets, key=lambda data: data.nparticles)])

    def ff(x, a, b):
        return a * x + b

    if True:
        phi_max = 0.85
        fig = plt.figure()
        ax = fig.add_subplot(111)
        yder_max = []
        yspl_mid = []
        yspl_list, yder_list = [], []
        phi_ss_all = np.hstack([dp.phi_ss for dp in datasets])
        xspl = np.linspace(np.amin(phi_ss_all), phi_max, 300, endpoint=True)
        color_cycle = get_color_cycle(ncolors=len(datasets))
        for dp in sorted(datasets, key=lambda data: data.nparticles):
            color = color_cycle.next()
            phi_ss, psuccess = dp.phi_ss[dp.phi_ss<phi_max], dp.psuccess[dp.phi_ss<phi_max]
            ax.plot(phi_ss, np.array(psuccess), marker='o', linestyle='None', color=color, rasterized=True)
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
            yder_max.append(xspl[np.argmax(yder)])
            yspl_mid.append(xspl[np.argmin(np.abs(ynew-0.5))])
        ax.set_xlim([phi_ss[0], phi_ss[-1]])
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$p_{pack}$")
        fig.savefig("{}/{}".format(figdir, "phi_ppack_all.pdf"))

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        color_cycle = get_color_cycle(ncolors=4)
        x = np.log(np.sqrt(nparticles))
        xnew = np.log(np.sqrt(np.linspace(np.amin(nparticles), np.amax(nparticles), 10)))
        color = color_cycle.next()
        ax2.plot(x, np.log(yder_max), marker='o', linestyle='None', color=color, rasterized=True)
        popt, pcov = curve_fit(ff, x, np.log(yder_max), absolute_sigma=True)
        fit_err = np.sqrt(np.diag(pcov))
        ax2.plot(xnew, ff(xnew, popt[0], popt[1]), color=color, linewidth=2,
                 label="dermax: : ln(phi) = {:.3f} ln(N) {:.3f}".format(popt[0], popt[1]), rasterized=True)
        color = color_cycle.next()
        ax2.plot(x, np.log(yspl_mid), marker='o', linestyle='None', color=color, rasterized=True)
        popt, pcov = curve_fit(ff, x, np.log(yspl_mid), absolute_sigma=True)
        fit_err = np.sqrt(np.diag(pcov))
        ax2.plot(xnew, ff(xnew, popt[0], popt[1]), color=color, linewidth=2,
                 label="midpoint: ln(phi) = {:.3f} ln(N) {:.3f}".format(popt[0], popt[1]), rasterized=True)
        ax2.set_xlabel(r"$\ln(N^{1/2})$")
        ax2.legend(frameon=False, loc='best', prop={'size':glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                  columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        fig2.savefig("{}/{}".format(figdir, "lnphi_lnppack_scaling.pdf"))
        # ax2.set_ylim([0, 1./(np.amin(nparticles)-5)])
        # ax2.set_yscale('log')
        # ax2.set_xscale('log')

    if True:
        import fssa
        l = np.sqrt(nparticles)
        rho = xspl
        # rho = rho[30:-30]
        # ycut = np.array([y[30:-30] for y in yspl_list])
        a = np.array(yspl_list)
        print a.shape
        da = np.ones(a.shape) * 1e-2
        rho_c0 = 0.847
        nu0 = 0.5
        zeta0 = 0.04
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
        ax3.set_ylabel(r'$\ln (p_{pack} (N^{1/d})^{\beta/\nu})$')
        ax3.set_xlabel(r'$\ln(\varepsilon (N^{1/d})^{1/\nu})$')
        ax3.set_title(r'$\phi_c \approx 0.847;~\nu \approx 1;~\beta \approx 1/20$')
        ax3.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        fig3.savefig("{}/{}".format(figdir, "phi_ppack_rescaling.pdf"))

        # rho = rho[15:-15]
        # ycut = np.array([y[15:-15] for y in yder_list])
        # a = np.array([y / simps(y, rho) for y in ycut])
        # # a = np.array(yder_list)
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
        # fig4.savefig("{}/{}".format(figdir, "phi_ppack_der_rescaling.pdf"))

    if True:
        import matplotlib.gridspec as gridspec
        phi_max=0.875
        fig5 = plt.figure(figsize=(8, 8))
        fig51 = plt.figure()
        gs = gridspec.GridSpec(7, 2)
        ax5 = fig5.add_subplot(gs[:4, :])
        ax51 = fig51.add_subplot(111)
        fit_params, fit_params_err = [], []
        fit_params2, fit_params_err2 = [], []
        color_cycle = get_color_cycle(ncolors=len(datasets))
        for i,dp in enumerate(sorted(datasets, key=lambda data: data.nparticles)):
            phi_ss_packed, logp_var, p_rel_var, logp_mean = dp.phi_ss_packed[dp.phi_ss_packed<phi_max], \
                                                            dp.logp_var[dp.phi_ss_packed<phi_max], \
                                                            dp.p_rel_var[dp.phi_ss_packed<phi_max], \
                                                            dp.logp_mean[dp.phi_ss_packed < phi_max]
            phi_c = yspl_mid[i] #yder_max[i] #DEBUG
            print dp.nparticles, phi_c
            inu = 2
            zeta = 1./8
            x = np.power(np.sqrt(nparticles[i]), inu) * (np.array(phi_ss_packed[phi_ss_packed > phi_c]) - np.array(phi_c)) / phi_c
            logx = np.log(x)
            y = np.power(np.sqrt(nparticles[i]), zeta*inu) * logp_var[:,0][phi_ss_packed > phi_c]
            logy = np.log(y)
            logy_err = (logp_var[:, 2][phi_ss_packed > phi_c] - logp_var[:, 1][phi_ss_packed > phi_c]) / (2 * logp_var[:,0][phi_ss_packed > phi_c])
            # np.log(p_rel_var[:, 0][phi_ss_packed > phi_c])
            # ax6.errorbar(phi, logp_mean[:, 0], yerr=[logp_mean[:, 1], logp_mean[:, 2]])
            # yerr = [np.log(p_rel_var[:, 1]), np.log(p_rel_var[:, 2])]
            color = color_cycle.next()
            ax5.errorbar(logx, logy, fmt='o', color=color, yerr=logy_err, rasterized=True)
            ax51.errorbar(logx, logy, fmt='o', color=color, yerr=logy_err, rasterized=True)
            idx = np.argmax(logy < 0.5*np.amax(logy)) #int(len(logx)*0.2)
            popt, pcov = curve_fit(ff, logx[idx:], logy[idx:],
                                   absolute_sigma=True,
                                   sigma = logy_err[idx:])

            fit_err = np.sqrt(np.diag(pcov))
            ax5.plot(logx[idx:], ff(logx[idx:], popt[0], popt[1]), color=color, linewidth=2,
                     label="N:{}; {:.3f} ln(x) + {:.3f}".format(dp.nparticles, popt[0], popt[1]), rasterized=True)
            ax51.plot(logx[idx:], ff(logx[idx:], popt[0], popt[1]), color=color, linewidth=2,
                     label="{}".format(dp.nparticles), rasterized=True)
            fit_params.append(popt)
            fit_params_err.append(fit_err)
            print "nparticles: {}; ({}+/-{}) x + ({}+/-{})".format(dp.nparticles, popt[0], fit_err[0], popt[1],
                                                                   fit_err[1])

        ax5.set_title(r'$\nu = 1/2;~\beta=1/8$')
        ax5.set_ylabel(r'$\ln(\sigma^2_{\Lambda} (N^{1/d})^{\alpha/\nu})$')
        ax5.set_xlabel(r'$\ln((N^{1/d}/\xi)^{1/\nu})$')
        ax5.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.5, labelspacing=0.25, handletextpad=0)
        ax51.set_ylabel(r'$\ln(\sigma^2_{\Lambda} (N^{1/d})^{\alpha/\nu})$')
        ax51.set_xlabel(r'$\ln((N^{1/d}/\xi)^{1/\nu})$')
        ax51.legend(frameon=False, loc='best', prop={'size': glob_fontsize})
        ax6 = fig5.add_subplot(gs[4:, 0])
        ax7 = fig5.add_subplot(gs[4:, 1])
        color = color_cycle.next()
        # slopes subplot
        ax6.errorbar(nparticles, np.array(fit_params)[:,0], yerr=np.array(fit_params_err)[:,0],
                     fmt='o', color=color, rasterized=True)
        popt, pcov = curve_fit(ff, nparticles, np.array(fit_params)[:,0],
                               absolute_sigma=True,
                               sigma=np.array(fit_params_err)[:,0])
        fit_err = np.sqrt(np.diag(pcov))
        ax6.plot(nparticles, ff(nparticles, popt[0], popt[1]), color=color, linewidth=2,
                 label="{:.3f} x + {:.3f}".format(popt[0], popt[1]), rasterized=True)
        print "fit slopes; ({}+/-{}) x + ({}+/-{})".format(popt[0], fit_err[0], popt[1], fit_err[1])
        # intercepts subplot
        ax7.errorbar(nparticles, np.array(fit_params)[:, 1], yerr=np.array(fit_params_err)[:, 1],
                     fmt='o', color=color, rasterized=True)
        popt, pcov = curve_fit(ff, nparticles, np.array(fit_params)[:, 1],
                               absolute_sigma=True,
                               sigma=np.array(fit_params_err)[:, 1])
        fit_err = np.sqrt(np.diag(pcov))
        ax7.plot(nparticles, ff(nparticles, popt[0], popt[1]), color=color, linewidth=2,
                 label="{:.3f} x + {:.3f}".format(popt[0], popt[1]), rasterized=True)
        print "fit interceps ; ({}+/-{}) x + ({}+/-{})".format(popt[0], fit_err[0], popt[1], fit_err[1])
        # ax6 ax7 legends
        ax6.set_xlabel('N')
        ax6.set_ylabel('slope')
        ax6.legend(frameon=False, loc='best', prop={'size': 13}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax7.set_xlabel('N')
        ax7.set_ylabel('intercept')
        ax7.legend(frameon=False, loc='best', prop={'size': 13}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        fig5.savefig("{}/{}".format(figdir, "lnphi_lnpvar_sub.pdf"))
        fig51.savefig("{}/{}".format(figdir, "lnphi_lnpvar.pdf"))


    if False:
        l = np.sqrt(nparticles)
        rho = np.array(phi_ss_packed[phi_ss_packed > phi_c])
        # rho = rho[30:-30]
        # ycut = np.array([y[30:-30] for y in yspl_list])
        a = np.array([dp.logp_var[:, 0][dp.phi_ss_packed > phi_c] for dp in datasets])
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
        fig10 = plt.figure()
        ax10 = fig10.add_subplot(111)
        color_cycle = get_color_cycle(ncolors=len(nparticles))
        for i, dp in enumerate(sorted(datasets, key=lambda data: data.nparticles)):
            u4_list, u4_err = [], []
            chi_list, chi_err = [], []
            for phi_ss, pressure in zip(dp.phi_ss_packed, dp.pressure):
                lnp = np.log(pressure)
                m2 = np.mean(np.power(lnp,2))
                m2c= moment(lnp, 2)
                m1abs = np.mean(np.abs(lnp))
                m4c = moment(lnp, 4)
                u4 = 1 - m4c/(3*m2c**2)
                chi = m2 - m1abs**2
                u4_list.append(u4)
                chi_list.append(chi)
                # u4_err.append(bootstrap.ci(lnp, lambda x : 1 - moment(x,4)/(3*moment(x, 2)**2), n_samples=1000))
                # chi_err.append(bootstrap.ci(lnp, lambda x: np.var(np.abs(x)), n_samples=1000 ))
            color = color_cycle.next()
            x = np.array(dp.phi_ss_packed)
            y, yerr = np.array(u4_list), np.array(u4_err)
            spl = UnivariateSpline(x, y, s=1.8, k=3)
            ax9.errorbar(x, y, fmt='o', label='N={}'.format(nparticles[i]), color=color, rasterized=True)
            ax9.plot(x, spl(x), linewidth=2, color=color, rasterized=True)
            # ax9inset.errorbar(x[x>0.84], y[x>0.84], fmt='o', label='N={}'.format(nparticles[i]), rasterized=True)
            y, yerr = np.array(chi_list), np.array(chi_err)
            spl = UnivariateSpline(x, y, s=1e-2, k=3)
            ax10.errorbar(x, chi_list, fmt='o', label='N={}'.format(nparticles[i]), color=color, rasterized=True)
            ax10.plot(x, spl(x), linewidth=2, color=color, rasterized=True)
        # ax9inset.set_xlim((0.84,0.87))
        ax9.set_xlim((0.81, 0.87))
        ax10.set_xlim((0.81, 0.87))
        ax9.legend(frameon=False, loc='best', prop={'size': 15}, numpoints=1, scatterpoints=1, markerscale=1,
                    columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax10.legend(frameon=False, loc='best', prop={'size': 15}, numpoints=1, scatterpoints=1, markerscale=1,
                       columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax9.set_ylabel(r'$V_N$')
        ax9.set_xlabel(r'$\phi$')
        ax10.set_ylabel(r'$\langle \Lambda^2 \rangle - \langle |\Lambda| \rangle^2$')
        ax10.set_xlabel(r'$\phi$')
        fig9.savefig("{}/{}".format(figdir, "lnp_u4.pdf"))
        fig10.savefig("{}/{}".format(figdir, "lnp_chi.pdf"))


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
