from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import rc
from pele.optimize import Result
from basinvolume.utils import *
from joblib import Parallel, delayed
import cPickle as pickle
from itertools import cycle
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
    for i, data in enumerate(sorted(datasets, key=lambda data: data.phi_ss)):
        phi_ss.append(data.phi_ss)
        psuccess.append(np.mean(data.success))
        nrattlers.append(np.mean(data.nrattlers))
        energy.append(data.energy)
        pressure.append(data.pressure)
        contacts.append(data.Z)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(phi_ss, np.array(psuccess), marker='o')
    ax.set_xlim([phi_ss[0],phi_ss[-1]])
    ax.set_xlabel(r"$\phi_{ss}$")
    ax.set_ylabel(r"$p_{pack}$")
    fig.savefig("{}/{}".format(figdir, "phi_ppack.pdf"))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(phi_ss, np.array(nrattlers), marker='o')
    ax1.set_xlim([phi_ss[0], phi_ss[-1]])
    ax1.set_xlabel(r"$\phi_{ss}$")
    ax1.set_ylabel(r"$n_{rattlers}$")
    fig1.savefig("{}/{}".format(figdir, "phi_nrattlers.pdf"))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    color = get_color_cycle()
    for e, p in zip(energy, pressure):
        assert len(e) == len(p)
        ax2.scatter(np.log(e), np.log(p), color=color.next())
    ax2.legend(frameon=False, loc='best', prop={'size': 18}, numpoints=1, scatterpoints=1, markerscale=1,
              columnspacing=0.25, labelspacing=0.25, handletextpad=0)
    ax2.set_xlabel(r"$\ln(E)$")
    ax2.set_ylabel(r"$\ln(P)$")
    fig2.savefig("{}/{}".format(figdir, "lnE_lnP.pdf"))

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    color = get_color_cycle()
    for z, p in zip(contacts, pressure):
        assert len(p) == len(z)
        ax3.scatter(np.log(p), np.log([np.mean(x) for x in z]), color=color.next())
    ax3.legend(frameon=False, loc='best', prop={'size': 18}, numpoints=1, scatterpoints=1, markerscale=1,
               columnspacing=0.25, labelspacing=0.25, handletextpad=0)
    ax3.set_xlabel(r"$\ln(P)$")
    ax3.set_ylabel(r"$\ln(\langle Z \rangle )$")
    fig3.savefig("{}/{}".format(figdir, "lnP_lnZ.pdf"))
    plt.show()

if __name__=="__main__":
    datasets = collect_data_every_set_all()
    plot(datasets)
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