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

def plot(datasets, figdir="figures"):
    figdir = os.path.join(os.getcwd(), figdir)
    trymakedir(figdir)

    psuccess = []
    phi_ss = []
    nrattlers = []
    energy = []
    pressure = []
    contacts = []
    phi_ss = np.unique([dataset.phi_ss for dataset in sorted(datasets, key=lambda data: data.phi_ss)])
    bdim = 2

    for phi_ in phi_ss:
        success_, nrattlers_, energy_ = [], [], []
        pressure_, contacts_ = [], []
        for i, dataset in enumerate(sorted(datasets, key=lambda data: data.phi_ss)):
            if phi_ == dataset.phi_ss:
                tmp = np.array(dataset.success,dtype='int')
                for data in dataset.packings_data:
                    N_contacts = int(np.sum(data.Z))
                    no_stable = len(data.Z)
                    N_min = int(2 * (bdim * (no_stable - 1) + 1))
                    if N_contacts >= N_min:
                        nrattlers_.append(data.nrattlers)
                        energy_.append(data.energy)
                        pressure_.append(data.pressure)
                        contacts_.append(np.mean(data.Z ))
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
        n_integrate = 2 ** 14 + 1
        p_var = []
        for i, p in enumerate(pressure):
            # kde histogram
            p = np.log(np.array(p))
            CIs = bootstrap.ci(p, np.var, n_samples=int(2.5e4))
            p_var.append([np.var(p), CIs[0], CIs[1]])
            bw = get_bandwidth_estimate(p, kernel="gaussian", method="cross_validation")
            print "bandwidth estimate: ", bw
            kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(p[:, np.newaxis])
            # plot histograms
            x_integrate = np.linspace(np.amin(p), np.amax(p), n_integrate)
            log_pdf = kde.score_samples(x_integrate[:, np.newaxis])
            # norm = np.log(integrate.romb(np.exp(log_pdf), dx=x_integrate[1] - x_integrate[0]))
            maxp = x_integrate[np.argmax(log_pdf)]
            color, label = color_cycle.next(), phi_ss[i]
            # ax4.plot(np.log(x_integrate)-np.log(maxp), log_pdf, color=color, label=label)
            ax4.plot(x_integrate - maxp, np.exp(log_pdf-np.amax(log_pdf)), color=color, label=label)
            ax5.plot(x_integrate - maxp, log_pdf - np.amax(log_pdf), color=color, label=label)
        # ax4.legend(frameon=False, loc='best', prop={'size': 18}, numpoints=1, scatterpoints=1, markerscale=1,
        #            columnspacing=0.25, labelspacing=0.25, handletextpad=0)
        ax4.set_xlabel(r"$\ln(P/P_{peak})$")
        ax4.set_ylabel(r"$pdf/\max(pdf)$")
        ax5.set_xlabel(r"$\ln(P/P_{peak})$")
        ax5.set_ylabel(r"$\ln(pdf)-\ln(\max(pdf))$")
        fig4.savefig("{}/{}".format(figdir, "lnP_pdf.pdf"))
        fig5.savefig("{}/{}".format(figdir, "lnP_lnpdf.pdf"))
        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111)
        p_var = np.array(p_var)
        ax6.errorbar(phi_ss, p_var[:,0], yerr=[p_var[:,1],p_var[:,2]])
        ax6.set_xlabel(r"$\phi$")
        ax6.set_ylabel(r"$\sigma^2(pdf(\ln P))$")
        fig6.savefig("{}/{}".format(figdir, "phi_varlnP.pdf"))

if __name__=="__main__":
    datasets = collect_data_every_set_all()
    plot(datasets)
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