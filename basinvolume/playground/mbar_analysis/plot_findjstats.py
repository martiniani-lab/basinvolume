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
def get_color_cycle(ncolors=20):
    cm = plt.get_cmap('Accent')
    color_cycle=cycle([cm(1. * i / ncolors) for i in xrange(ncolors)][::-1])
    return color_cycle
def get_marker_cycle():
    markers = ["o","v","s","h","^","8","p","<","*","D",">",]
    markercycle = cycle(markers)
    return markercycle
def get_line_cycle():
    lines = ["--","-"]
    linecycle = cycle(lines)
    return linecycle
def get_cycler(ncolors=20):
    cm = plt.get_cmap('Accent')
    return cycler('color', [cm(1. * i / ncolors) for i in xrange(ncolors)][::-1])

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
    def __init__(self, psuccess, phi_ss, nrattlers, energy, pressure, contacts, prob_min=0.05, bdim=2,
                 nparticles=64):
        self.psuccess = np.array(psuccess)
        self.phi_ss = np.array(phi_ss)
        self.phi_ss_packed = self.phi_ss[self.psuccess > prob_min]
        self.nrattlers = np.array(nrattlers)[self.psuccess > prob_min]
        self.energy = np.array(energy)[self.psuccess > prob_min]
        self.pressure = np.array(pressure)[self.psuccess > prob_min]
        self.contacts = np.array(contacts)[self.psuccess > prob_min]
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
    phi_ss = np.unique([dataset.phi_ss for dataset in sorted(datasets, key=lambda data: data.phi_ss)])
    phi_ss = phi_ss[phi_ss < phi_max]
    bdim = bdim
    for phi_ in phi_ss:
        success_, nrattlers_, energy_ = [], [], []
        pressure_, contacts_ = [], []
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
    return psuccess, phi_ss, nrattlers, energy, pressure, contacts

def plot(figdir="figures", bdim=2, nparticles=64):
    figdir = os.path.join(os.getcwd(), figdir)
    trymakedir(figdir)
    try:
        dp = pickle.load(open(path, "rb"))
    except Exception, e:
        datasets = collect_data_every_set_all()
        psuccess, phi_ss, nrattlers, energy, pressure, contacts = collect_data_plot(datasets)
        dp = DataPlot(psuccess, phi_ss, nrattlers, energy, pressure, contacts, bdim=bdim, nparticles=nparticles)
        dp.compute_stats()
        pickle.dump(dp, open("findjstats.pickle", "wb"))

    psuccess, phi_ss, phi_ss_packed =  dp.psuccess, dp.phi_ss, dp.phi_ss_packed
    nrattlers, energy, pressure, contacts = dp.nrattlers, dp.energy, dp.pressure, dp.contacts
    assert dp.initialized == True
    log_pdf, log_pdf_x, logp_mean, logp_var = dp.log_pdf, dp.log_pdf_x, dp.logp_mean, dp.logp_var
    p_mean, p_var, p_rel_var, logp_mode = dp.p_mean, dp.p_var, dp.p_rel_var, dp.logp_mode
    assert nparticles == dp.nparticles

    if True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(phi_ss, np.array(psuccess), marker='o')
        ax.set_xlim([phi_ss[0],phi_ss[-1]])
        ax.set_xlabel(r"$\phi_{ss}$")
        ax.set_ylabel(r"$p_{pack}$")
        fig.savefig("{}/{}".format(figdir, "phi_ppack.pdf"))

    if False:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(phi_ss_packed, np.array(nrattlers), marker='o')
        ax1.set_xlim([phi_ss_packed[0], phi_ss_packed[-1]])
        ax1.set_xlabel(r"$\phi_{ss}$")
        ax1.set_ylabel(r"$n_{rattlers}$")
        fig1.savefig("{}/{}".format(figdir, "phi_nrattlers.pdf"))

    if True:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        color = get_color_cycle()
        for e,p in zip(energy, pressure):
            assert len(e) == len(p)
            ax2.scatter(np.log(e), np.log(p), color=color.next())
        ax2.legend(frameon=False, loc='best', prop={'size': 18}, numpoints=1, scatterpoints=1, markerscale=1,
                  columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax2.set_xlabel(r"$\ln(E)$")
        ax2.set_ylabel(r"$\ln(P)$")
        fig2.savefig("{}/{}".format(figdir, "lnE_lnP.pdf"))

    if False:
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        color = get_color_cycle()
        for i, (p, z) in enumerate(zip(pressure, contacts)):
            assert len(p) == len(z)
            ax3.scatter(np.log(p), np.log(z), color=color.next())
        ax3.legend(frameon=False, loc='best', prop={'size': 18}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax3.set_xlabel(r"$\ln(P)$")
        ax3.set_ylabel(r"$\ln(\langle Z \rangle )$")
        fig3.savefig("{}/{}".format(figdir, "lnP_lnZ.pdf"))

    phi_c = 0.832

    if True:
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        color_cycle = get_color_cycle()

        for i in xrange(phi_ss_packed.size):
            # kde histogram
            assert log_pdf[i].size == log_pdf_x[i].size
            log_maxp = log_pdf_x[i][np.argmax(log_pdf[i])]
            color, label = color_cycle.next(), phi_ss_packed[i]
            # ax4.plot(np.log(x_integrate)-np.log(maxp), log_pdf, color=color, label=label)
            ax4.plot(log_pdf_x[i]-log_maxp, np.exp(log_pdf[i]-np.amax(log_pdf[i])), color=color, label=label)
            ax5.plot(log_pdf_x[i]-log_maxp, log_pdf[i] - np.amax(log_pdf[i]), color=color, label=label)
        # (0.686850451878 * np.abs(phi_ss_packed[i] - phi_c) + 3.80454345905)
        # ax4.legend(frameon=False, loc='best', prop={'size': 18}, numpoints=1, scatterpoints=1, markerscale=1,
        #            columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax4.set_xlabel(r"$\ln(P/P_{peak})$")
        ax4.set_ylabel(r"$pdf/\max(pdf)$")
        ax5.set_xlabel(r"$\ln(P/P_{peak})$")
        ax5.set_ylabel(r"$\ln(pdf)-\ln(\max(pdf))$")
        fig4.savefig("{}/{}".format(figdir, "lnP_pdf.pdf"))
        fig5.savefig("{}/{}".format(figdir, "lnP_lnpdf.pdf"))

    def ff(x, a, b):
        return a * x + b

    if True:
        logx = np.log(phi_ss_packed[phi_ss_packed > phi_c]) #-phi_c
        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111)
        # ax6.errorbar(phi, logp_var[:,0], yerr=[logp_var[:,1], logp_var[:,2]])
        # ax6.errorbar(phi, logp_mean[:, 0], yerr=[logp_mean[:, 1], logp_mean[:, 2]])
        # yerr = [np.log(p_rel_var[:, 1]), np.log(p_rel_var[:, 2])]
        color_cycle = get_color_cycle(ncolors=3)
        color = color_cycle.next()
        ax6.errorbar(logx, np.log(p_rel_var[:, 0][phi_ss_packed > phi_c]), yerr=[logp_var[:,1][phi_ss_packed > phi_c],
                                                                                 logp_var[:,2][phi_ss_packed > phi_c]], fmt='o', color=color)
        popt, pcov = curve_fit(ff, logx, np.log(p_rel_var[:, 0][phi_ss_packed > phi_c]), sigma=(logp_var[:,2][phi_ss_packed > phi_c]-logp_var[:,1][phi_ss_packed > phi_c])/2,
                               absolute_sigma=True)
        fit_err = np.sqrt(np.diag(pcov))
        ax6.plot(logx, ff(logx, popt[0], popt[1]), color=color, linewidth=2)
        print "({}+/-{}) x + ({}+/-{})".format(popt[0], fit_err[0], popt[1], fit_err[1])

        # logp_mode = np.array(logp_mode)[phi_ss_packed > phi_c]
        # color = color_cycle.next()
        # ax6.scatter(logx, np.log(logp_mode), color=color)
        # popt, pcov = curve_fit(ff, logx, np.log(logp_mode), absolute_sigma=True)
        # fit_err = np.sqrt(np.diag(pcov))
        # ax6.plot(logx, ff(logx, popt[0], popt[1]), color=color, linewidth=2)
        # print "({}+/-{}) x + ({}+/-{})".format(popt[0], fit_err[0], popt[1], fit_err[1])

        ax6.set_xlabel(r"$\Delta \phi$")
        ax6.set_ylabel(r"$\sigma^2(pdf(\ln P))$")
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
        # from scipy.interpolate import splrep, splev
        from scipy.interpolate import UnivariateSpline
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
            ax.plot(phi_ss, np.array(psuccess), marker='o', linestyle='None', color=color)
            # tck = splrep(phi_ss, psuccess, s=1e-3, k=3)
            # ynew = splev(xspl, tck, der=0)
            spl = UnivariateSpline(phi_ss, psuccess, bbox=[phi_ss[0], phi_ss[-1]], s=7.5e-4, k=3)
            ynew = spl(xspl)
            yspl_list.append(ynew)
            ax.plot(xspl, ynew, marker='None', linestyle='-', color=color)
            # yder = splev(xspl, tck, der=1)
            yder = spl.derivative(1)(xspl)
            yder_list.append(yder)
            ax.plot(xspl, yder/np.amax(yder), marker='None', linestyle='--', color=color)
            yder_max.append(xspl[np.argmax(yder)])
            yspl_mid.append(xspl[np.argmin(np.abs(ynew-0.5))])
        ax.set_xlim([phi_ss[0], phi_ss[-1]])
        ax.set_xlabel(r"$\phi_{ss}$")
        ax.set_ylabel(r"$p_{pack}$")
        fig.savefig("{}/{}".format(figdir, "phi_ppack_all.pdf"))

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        color_cycle = get_color_cycle(ncolors=4)
        x = np.log(np.sqrt(nparticles))
        xnew = np.log(np.sqrt(np.linspace(np.amin(nparticles), np.amax(nparticles), 10)))
        color = color_cycle.next()
        ax2.plot(x, np.log(yder_max), marker='o', linestyle='None', color=color)
        popt, pcov = curve_fit(ff, x, np.log(yder_max), absolute_sigma=True)
        fit_err = np.sqrt(np.diag(pcov))
        ax2.plot(xnew, ff(xnew, popt[0], popt[1]), color=color, linewidth=2,
                 label="dermax: : ln(phi) = {:.3f} ln(N) {:.3f}".format(popt[0], popt[1]))
        color = color_cycle.next()
        ax2.plot(x, np.log(yspl_mid), marker='o', linestyle='None', color=color)
        popt, pcov = curve_fit(ff, x, np.log(yspl_mid), absolute_sigma=True)
        fit_err = np.sqrt(np.diag(pcov))
        ax2.plot(xnew, ff(xnew, popt[0], popt[1]), color=color, linewidth=2,
                 label="midpoint: ln(phi) = {:.3f} ln(N) {:.3f}".format(popt[0], popt[1]))
        ax2.set_xlabel(r"$\ln(N^{1/2})$")
        ax2.legend(frameon=False, loc='best', prop={'size':18}, numpoints=1, scatterpoints=1, markerscale=1,
                  columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        fig2.savefig("{}/{}".format(figdir, "lnphi_lnppack_scaling.pdf"))
        # ax2.set_ylim([0, 1./(np.amin(nparticles)-5)])
        # ax2.set_yscale('log')
        # ax2.set_xscale('log')

    if True:
        import fssa
        from scipy.integrate import simps
        l = np.sqrt(nparticles)
        rho = xspl
        # rho = rho[30:-30]
        # ycut = np.array([y[30:-30] for y in yspl_list])
        a = np.array(yspl_list)
        print a.shape
        da = np.ones(a.shape) * 1e-1
        rho_c0 = 0.84
        nu0 = 1
        zeta0 = 0.02
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
        ax3.plot(auto_scaled_data.x.T, auto_scaled_data.y.T, '.', label=nparticles)
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
        phi_max=0.875
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        color_cycle = get_color_cycle(ncolors=len(datasets))
        for i,dp in enumerate(sorted(datasets, key=lambda data: data.nparticles)):
            phi_ss_packed, logp_var, p_rel_var = dp.phi_ss_packed[dp.phi_ss_packed<phi_max], \
                                                 dp.logp_var[dp.phi_ss_packed<phi_max], \
                                                 dp.p_rel_var[dp.phi_ss_packed<phi_max]
            phi_c = yspl_mid[i]
            logx = np.log(phi_ss_packed[phi_ss_packed > phi_c])
            logy = np.log(logp_var[:,0][phi_ss_packed > phi_c])
            logy_err = (logp_var[:, 2][phi_ss_packed > phi_c] - logp_var[:, 1][phi_ss_packed > phi_c]) / (2 * logp_var[:,0][phi_ss_packed > phi_c])
            # np.log(p_rel_var[:, 0][phi_ss_packed > phi_c])
            # ax6.errorbar(phi, logp_mean[:, 0], yerr=[logp_mean[:, 1], logp_mean[:, 2]])
            # yerr = [np.log(p_rel_var[:, 1]), np.log(p_rel_var[:, 2])]
            color = color_cycle.next()
            ax5.errorbar(logx, logy, fmt='o', color=color, yerr=logy_err)
            idx = int(len(logx)*0.44)
            popt, pcov = curve_fit(ff, logx[:idx], logy[:idx],
                                   absolute_sigma=True,
                                   sigma = logy_err[:idx])

            fit_err = np.sqrt(np.diag(pcov))
            ax5.plot(logx[:idx], ff(logx[:idx], popt[0], popt[1]), color=color, linewidth=2,
                     label="N:{}; {:.3f} ln(x) {:.3f}".format(dp.nparticles, popt[0], popt[1]))
            print "nparticles: {}; ({}+/-{}) x + ({}+/-{})".format(dp.nparticles, popt[0], fit_err[0], popt[1], fit_err[1])
        ax5.set_ylabel(r'$\ln(var(\ln P))$')
        ax5.set_xlabel(r'$\ln(\phi-\phi_{f_J=0.5})$')
        ax5.legend(frameon=False, loc='best', prop={'size': 18}, numpoints=1, scatterpoints=1, markerscale=1,
                   columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        fig5.savefig("{}/{}".format(figdir, "lnphi_lnpvar.pdf"))
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