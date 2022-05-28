from __future__ import division
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter
from basinvolume.utils import trymakedir, Bunch, sort_pair
from basinvolume.utils import log_volume_nball
import numpy as np
import os
import glob
import gc
import cPickle as pickle
from itertools import cycle
from cycler import cycler
import argparse

from basinvolume.monte_carlo import CheckExponentiallyDecayingProfile

#######################SET LATEX OPTIONS###################
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
#rc('text.latex',preamble=r'\usepackage{times}')
glob_fontsize=24
plt.rcParams.update({'font.size': glob_fontsize})
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams.update({'figure.autolayout': True})
##########################################################

class FixedOrderFormatter(ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of
    magnitude"""
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset,
                                 useMathText=useMathText)
    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag

####SET COLOUR MAP######
def get_color_cycle(ncolors=20, reverse=True):
    cm = plt.get_cmap('Paired')
    if reverse:
        color_cycle=cycle([cm(1. * (i+0.5) / float(ncolors)) for i in xrange(ncolors)][::-1])
    else:
        color_cycle = cycle([cm(1. * (i - 0.5) / float(ncolors)) for i in xrange(ncolors)])
    return color_cycle
def get_marker_cycle():
    markers = ["s","o","v","^","<",">","*","D","h","8","p"]
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

def plot_sphere(figdir="figures"):
    figdir = os.path.join(os.getcwd(), figdir)
    trymakedir(figdir)

    listfiles = glob.glob("*pickle")
    datasets = [pickle.load(open(f, "r")) for f in listfiles]

    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    left, bottom, width, eight = [0.23, 0.255, 0.35, 0.35] #[0.23, 0.53, 0.35, 0.35]
    ax0inset = fig0.add_axes([left, bottom, width, eight])

    datasets = sorted(datasets, key=lambda d: d.pt)[:2]

    color_cycle = get_color_cycle(ncolors=len(datasets)+1, reverse=False)
    marker_cycle = get_marker_cycle()

    d = datasets[0]
    y, y2 = [], []
    x = sorted(d.ndim_list)
    color = color_cycle.next()
    for i, n in enumerate(x):
        y.append(log_volume_nball(d.geom_params_list[i][0], n))
        y2.append(n * d.geom_params_list[i][0] ** 2 / (n + 2))
    ax0.errorbar(x, y, fmt='-', color=color, linewidth=2.5,
                 label='exact', rasterized=True, markeredgecolor=color)
    ax0inset.errorbar(x, y2, fmt='-', color=color, linewidth=2.5,
                      label='exact', rasterized=True, markeredgecolor=color)

    for d in datasets:
        # here loop through curves
        if not d.pt:
            label = "$k={}$".format(d.cloud_points)
        else:
            label = "$k={}, PT$".format(d.cloud_points)
        color = color_cycle.next()
        marker = marker_cycle.next()
        x, y = sort_pair(d.ndim_list, d.f_list, reverse=False)
        x, y_err = sort_pair(d.ndim_list, d.ferr_list, reverse=False)
        ax0.errorbar(x, -y, marker=marker, color=color, yerr=2*y_err, linestyle="",
                      label=label, rasterized=True, markeredgecolor=color)

        x, y = sort_pair(d.ndim_list, d.mean_r2_list, reverse=False)
        x, y_err = sort_pair(d.ndim_list, d.std_err_r2_list, reverse=False)
        ax0inset.errorbar(x, y, marker=marker, color=color, yerr=2*y_err, linestyle="",
                      label=label, rasterized=True, markeredgecolor=color)

    ax0.set_ylabel(r'$\ln V$')
    ax0.set_xlabel(r'$n$')
    ax0.set_xlim(xmin=2, xmax=20)
    ax0.legend(frameon=False, loc=1, prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1,
               markerscale=1, columnspacing=0.25, labelspacing=0.25, handlelength=0.6)
    ax0inset.autoscale(enable=True, tight=False)
    ax0inset.set_ylabel(r'$\langle |{\bf x} - {\bf x}_0|^2 \rangle$')
    ax0inset.set_xlabel(r'$n$')
    ax0inset.yaxis.tick_right()
    ax0inset.xaxis.tick_top()
    ax0inset.yaxis.set_label_position("left")
    ax0inset.locator_params(axis='x', nbins=5)
    ax0inset.locator_params(axis='y', nbins=4)
    ax0inset.set_xlim(xmin=2, xmax=20)
    fig0.savefig("{}/{}".format(figdir, "sphere_plot.pdf"))


def plot_sphere_exp_decay(figdir="figures"):
    figdir = os.path.join(os.getcwd(), figdir)
    trymakedir(figdir)

    listfiles = glob.glob("*pickle")
    datasets = [pickle.load(open(f, "r")) for f in listfiles]

    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)

    datasets = sorted(datasets, key=lambda d: (d.pt, d.cloud_points))

    color_cycle = get_color_cycle(ncolors=len(datasets)+1, reverse=False)
    marker_cycle = get_marker_cycle()

    d = datasets[0]
    y= []
    x = sorted(d.ndim_list)
    color = color_cycle.next()
    for i, n in enumerate(x):
        ts = CheckExponentiallyDecayingProfile(np.zeros(n), d.geom_params_list[i][0],
                                               d.geom_params_list[i][1], cubic=False)
        y.append(np.log(ts.get_exact_volume()))
    ax0.errorbar(x, y, fmt='-', color=color, linewidth=2.5,
                 label='exact', rasterized=True, markeredgecolor=color)

    color_cycle = get_color_cycle(ncolors=len(datasets) // 2, reverse=True)

    for d in datasets:
        # here loop through curves
        # if not d.pt:
        #     label = "$k={}$".format(d.cloud_points)
        if d.pt:
            label = "$k={}, PT$".format(d.cloud_points)
            color = color_cycle.next()
            marker = marker_cycle.next()
            x, y = sort_pair(d.ndim_list, d.f_list, reverse=False)
            x, y_err = sort_pair(d.ndim_list, d.ferr_list, reverse=False)
            ax0.errorbar(x, -y, marker=marker, color=color, yerr=2*y_err,
                          label=label, rasterized=True, markeredgecolor=color)

    ax0.set_ylabel(r'$\ln V$')
    ax0.set_xlabel(r'$n$')
    ax0.set_xlim(xmin=2, xmax=20)
    ax0.legend(frameon=False, loc='best', prop={'size': glob_fontsize}, numpoints=1, scatterpoints=1,
               markerscale=1, columnspacing=0.25, labelspacing=0.25, handlelength=0.6, ncol=2)
    fig0.savefig("{}/{}".format(figdir, "sphere_exp_decay_plot.pdf"))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="plot oracle")
    parser.add_argument("system_name", type=str, help="system name: sphere, sphere_exp_decay, \
        sphere_pow_decay, cube, cube_exp_decay")
    parser.add_argument("--show", action='store_true', help="show plots", default=False)
    args = parser.parse_args()

    if args.system_name == "sphere":
        plot_sphere()
    elif args.system_name == "sphere_exp_decay":
        plot_sphere_exp_decay()
    else:
        raise NotImplementedError

    if args.show:
        plt.show(args.system_name)