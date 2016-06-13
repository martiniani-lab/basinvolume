from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import rc
from pele.optimize import Result
from basinvolume.utils import *
from joblib import Parallel, delayed
import cPickle as pickle
from itertools import cycle
from basinvolume.spheres.find_jstats import SoftPackingDataset, SoftPackingData
from sklearn.neighbors import KernelDensity
from basinvolume.experiment_2d.cross_validation_bandwidth_selection import get_bandwidth_estimate, get_pdf
from scipy import integrate
import scikits.bootstrap as bootstrap

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
    cm = plt.get_cmap('Accent')
    color_cycle=cycle([cm(1. * i / 20) for i in xrange(20)][::-1])
    return color_cycle
def get_marker_cycle():
    markers = ["o","v","s","h","^","8","p","<","*","D",">",]
    markercycle = cycle(markers)
    return markercycle
def get_line_cycle():
    lines = ["--","-"]
    linecycle = cycle(lines)
    return linecycle
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
    def __init__(self, psuccess, phi_ss, nrattlers, energy, pressure, contacts, bdim=2, nparticles=64):
        self.psuccess = np.array(psuccess)
        self.phi_ss = np.array(phi_ss)
        self.nrattlers = np.array(nrattlers)
        self.energy = np.array(energy)
        self.pressure = np.array(pressure)
        self.contacts = np.array(contacts)
        self.bdim = bdim
        self.nparticles = nparticles
        self.bw = []
        self.logp_mean = []
        self.logp_var = []
        self.log_pdf = []
        self.log_pdf_x = []
        self.initialized = False

    def compute_stats(self, n_samples=2.5e4, n_integrate=None):
        for i, p in enumerate(self.pressure):
            if n_integrate is None:
                n_integrate = 2 ** 14 + 1
            else:
                n_integrate = n_integrate
            lnp = np.log(np.array(p))
            meanCIs = bootstrap.ci(lnp, np.mean, n_samples=int(n_samples))
            varCIs = bootstrap.ci(lnp, np.var, n_samples=int(n_samples))
            self.logp_mean.append([np.mean(lnp), meanCIs[0], meanCIs[1]])
            self.logp_var.append([np.var(lnp), varCIs[0], varCIs[1]])
            bw = get_bandwidth_estimate(lnp, kernel="gaussian", method="cross_validation")
            self.bw.append(bw)
            kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(lnp[:, np.newaxis])
            x_integrate = np.linspace(np.amin(lnp), np.amax(lnp), n_integrate)
            log_pdf = kde.score_samples(x_integrate[:, np.newaxis])
            self.log_pdf.append(log_pdf)
            self.log_pdf_x.append(x_integrate)
        self.logp_mean = np.array(self.logp_mean)
        self.logp_var = np.array(self.logp_var)
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

def plot(path, figdir="figures", bdim=2, nparticles=64):
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

    psuccess, phi_ss, nrattlers, energy, pressure, contacts =  dp.psuccess, dp.phi_ss, dp.nrattlers, dp.energy, dp.pressure, dp.contacts
    assert dp.initialized == True
    log_pdf, log_pdf_x, logp_mean, logp_var = dp.log_pdf, dp.log_pdf_x, dp.logp_mean, dp.logp_var

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
        ax1.plot(phi_ss, np.array(nrattlers), marker='o')
        ax1.set_xlim([phi_ss[0], phi_ss[-1]])
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

    if True:
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        color_cycle = get_color_cycle()

        for i in xrange(phi_ss.size):
            # kde histogram
            assert log_pdf[i].size == log_pdf_x[i].size
            maxp = log_pdf_x[i][np.argmax(log_pdf[i])]
            color, label = color_cycle.next(), phi_ss[i]
            # ax4.plot(np.log(x_integrate)-np.log(maxp), log_pdf, color=color, label=label)
            ax4.plot(log_pdf_x[i] - maxp, np.exp(log_pdf[i]-np.amax(log_pdf[i])), color=color, label=label)
            ax5.plot(log_pdf_x[i] - maxp, log_pdf[i] - np.amax(log_pdf[i]), color=color, label=label)
        # ax4.legend(frameon=False, loc='best', prop={'size': 18}, numpoints=1, scatterpoints=1, markerscale=1,
        #            columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax4.set_xlabel(r"$\ln(P/P_{peak})$")
        ax4.set_ylabel(r"$pdf/\max(pdf)$")
        ax5.set_xlabel(r"$\ln(P/P_{peak})$")
        ax5.set_ylabel(r"$\ln(pdf)-\ln(\max(pdf))$")
        fig4.savefig("{}/{}".format(figdir, "lnP_pdf.pdf"))
        fig5.savefig("{}/{}".format(figdir, "lnP_lnpdf.pdf"))

    if True:
        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111)
        ax6.errorbar(phi_ss, logp_var[:,0], yerr=[logp_var[:,1], logp_var[:,2]])
        ax6.errorbar(phi_ss, logp_mean[:, 0], yerr=[logp_mean[:, 1], logp_mean[:, 2]])
        ax6.set_xlabel(r"$\phi$")
        ax6.set_ylabel(r"$\sigma^2(pdf(\ln P))$")
        # ax6.set_xscale('log')
        # ax6.set_yscale('log')
        fig6.savefig("{}/{}".format(figdir, "phi_varlnP.pdf"))

if __name__=="__main__":
    path = os.path.join(os.getcwd(),'findjstats.pickle')
    plot(path)
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