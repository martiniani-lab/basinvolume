from __future__ import division
from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import zip
from builtins import next
from builtins import range
from builtins import object

try:
    import numpy as np
    import argparse
    import configparser
    import os
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from itertools import cycle
    from basinvolume.utils import *
    import scipy
    from scipy.stats import t
    from scipy.interpolate import spline
    from itertools import chain
    from scipy.optimize import curve_fit
except ImportError as err:
    print(err)

#######################SET LATEX OPTIONS###################
rc("text", usetex=True)
rc("font", **{"family": "serif"})
# rc('text.latex',preamble=r'\usepackage{times}')
plt.rcParams.update({"font.size": 22})
plt.rcParams["xtick.major.pad"] = 8
plt.rcParams["ytick.major.pad"] = 8
plt.rcParams.update({"figure.autolayout": True})
##########################################################
####SET COLOUR MAP######
def get_color_cycle():
    cm = plt.get_cmap("Set2")
    color_cycle = cycle([cm(1.0 * i / 7) for i in range(7)])
    return color_cycle


########################
"""
for plotting a linear fit with intervals of confidence see http://nbviewer.ipython.org/url/bagrow.com/dsv/LEC10_notes_2014-02-13.ipynb
"""


def _read_nparticles(folder):
    nparticles = ""
    for char in folder[1:]:
        if char == "_":
            break
        nparticles += char
    nparticles = int(nparticles)
    return nparticles


def studentTCI(y, yerr, alpha=0.95):
    """
    return lower and upper bound of confidence interval
    see https://en.wikipedia.org/wiki/Confidence_interval Thoretical example
    """
    y = np.array(y)
    yerr = np.array(yerr)
    ts = t.interval(np.array([alpha] * len(y)), len(y) - 1)
    return y + alpha * ts[0] * yerr, y + alpha * ts[1] * yerr


def _sort_pair(x, y):
    """
    sorts x and moves elements of y accordingly
    """
    xc = np.array(x)
    points = list(zip(xc, y))
    sorted_points = sorted(points, key=lambda x: x[0])
    new_x = np.array([point[0] for point in sorted_points])
    new_y = np.array([point[1] for point in sorted_points])
    return new_x, new_y


class EntropyData(object):
    def __init__(
        self,
        nparticles,
        path,
        apf_file="entropy_APF",
        apf_file_title="ENTROPY_APF",
        kd_file="entropy_kernel_density",
        kd_file_title="LOG_OMEGA_KERNEL_DENSITY",
        lo_file="entropy_LogOmega",
        lo_file_title="ENTROPY_LOG_OMEGA",
        loml_file="entropy_ML_LogOmega",
        loml_file_title="LOG_OMEGA_ML",
        gen_gaussian_param_names=[
            ("mu", "mu_error"),
            ("alpha", "alpha_error"),
            ("zeta", "zeta_error"),
        ],
    ):
        self.nparticles = nparticles
        self.path = path
        self.apf = Bunch(file=apf_file, title=apf_file_title, entropy=[0.0, 0.0])
        self.kd = Bunch(file=kd_file, title=kd_file_title, entropy=[0.0, 0.0])
        parameters = dict(mu=[0.0, 0.0], alpha=[0.0, 0.0], zeta=[0.0, 0.0])
        self.lo = Bunch(
            file=lo_file,
            title=lo_file_title,
            entropy=[0.0, 0.0],
            parameters=parameters,
        )  # parameters are mu,alpha,zeta
        self.loml = Bunch(
            file=loml_file,
            title=loml_file_title,
            entropy=[0.0, 0.0],
            parameters=parameters,
        )  # parameters are mu,alpha,zeta
        self.gen_gaussian_param_names = gen_gaussian_param_names

    def add_all_entropy(self):
        self.add_entropy(self.apf)
        self.add_entropy(self.kd)
        self.add_entropy_with_parameters(self.lo)
        self.add_entropy_with_parameters(self.loml)

    def add_entropy(self, member):
        """
        member should be the appropriate class member
        n : packing number
        example
        -------
        data = EntropyData()
        data.add_entropy(S, ds, data.apf)
        """
        configf = configparser.ConfigParser()
        fpath = os.path.join(self.path, member.file)
        if os.path.isfile(fpath):
            configf.read(fpath)
            try:
                member.entropy[0] = configf.getfloat(member.title, "S_star")
                member.entropy[1] = configf.getfloat(member.title, "error_S_star")
            except Exception as e:
                print(e)

    def add_entropy_with_parameters(self, member):
        """
        member should be the appropriate class member
        parameters: list ot tuples
            (value, error)
        n : packing number
        example
        -------
        data = EntropyData()
        data.add_entropy(S, ds, data.apf)
        """
        configf = configparser.ConfigParser()
        fpath = os.path.join(self.path, member.file)
        if os.path.isfile(fpath):
            configf.read(fpath)
            try:
                member.entropy[0] = configf.getfloat(member.title, "S_star")
                member.entropy[1] = configf.getfloat(member.title, "error_S_star")
            except Exception as e:
                print(e)
            for name in self.gen_gaussian_param_names:
                try:
                    val = configf.getfloat(member.title, name[0])
                    member.parameters[name[0]][0] = val
                    errval = configf.getfloat(member.title, name[1])
                    member.parameters[name[0]][1] = errval
                except Exception:
                    pass


class plot_entropy(object):
    def __init__(
        self,
        workdir=None,
        analysis_folder="entropy_analysis_all",
        Nrange=(0, 128),
    ):
        if not workdir:
            workdir = os.getcwd()
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
        self.workdir = workdir

        self.Nrange = Nrange
        self.analysis_folder = analysis_folder
        self.entropy_data = []
        markers = ["o", "^", "s", "x", "+"]
        self.markercycler = cycle(markers)

        subdirs = get_immediate_subdirectories(workdir)
        for folder in subdirs:
            if folder[1].isdigit():
                path = os.path.join(workdir, folder, self.analysis_folder)
                if os.path.isdir(path):
                    n = _read_nparticles(folder)
                    if self.Nrange[0] <= n <= self.Nrange[1]:
                        data = EntropyData(n, path)
                        data.add_all_entropy()
                        #                        #this is a hack to get the volume cavity
                        #                        configf = ConfigParser.ConfigParser()
                        #                        configf.read(os.path.join(workdir,folder, 'jammed_packings', 'jammed_packing0.config'))
                        #                        boxv = configf.get('JAMMED_PACKING','boxv')
                        #                        boxv = np.array([float(x) for x in boxv.split()])
                        #                        boxv = np.prod(boxv)
                        #                        data.lo.parameters['mu'][0] += n * np.log(boxv)
                        #                        data.loml.parameters['mu'][0] += n * np.log(boxv)
                        #                        ####
                        self.entropy_data.append(data)
        self.apf_entropy = list(
            chain.from_iterable(
                (data.nparticles, data.apf.entropy[0], data.apf.entropy[1])
                for data in self.entropy_data
            )
        )
        self.kd_entropy = list(
            chain.from_iterable(
                (data.nparticles, data.kd.entropy[0], data.kd.entropy[1])
                for data in self.entropy_data
            )
        )
        self.lo_entropy = list(
            chain.from_iterable(
                (data.nparticles, data.lo.entropy[0], data.lo.entropy[1])
                for data in self.entropy_data
            )
        )
        self.loml_entropy = list(
            chain.from_iterable(
                (data.nparticles, data.loml.entropy[0]) for data in self.entropy_data
            )
        )
        self.all_entropies_err = [
            (self.apf_entropy, r"$\sum p \ln p$", "apf"),
            (self.kd_entropy, r"$\ln \Omega_{KDE}$", "kde"),
            (self.lo_entropy, r"$\ln \Omega_G$", "logomega"),
        ]
        self.all_entropies = [(self.loml_entropy, r"$\ln \Omega_{GML}$", "logomegaml")]
        self.lo_parameters = list(
            chain.from_iterable(
                (
                    data.nparticles,
                    data.lo.parameters["mu"][0],
                    data.lo.parameters["mu"][1],
                    data.lo.parameters["alpha"][0],
                    data.lo.parameters["alpha"][1],
                    data.lo.parameters["zeta"][0],
                    data.lo.parameters["zeta"][1],
                )
                for data in self.entropy_data
            )
        )
        self.loml_parameters = list(
            chain.from_iterable(
                (
                    data.nparticles,
                    data.loml.parameters["mu"][0],
                    data.loml.parameters["alpha"][0],
                    data.loml.parameters["zeta"][0],
                )
                for data in self.entropy_data
            )
        )

    def _ploterr(
        self,
        entropy_array,
        xlabel=r"$N$",
        ylabel=r"$S$",
        raw=True,
        title=None,
        show=False,
        ax=None,
        marker="o",
        raw_marker="^",
    ):
        color_cycle = get_color_cycle()
        nparticles = np.array(entropy_array[0::3])
        nmax = np.amax(nparticles)
        trialx = np.linspace(0, nmax + 2, 1000)
        # extensive
        if ax is None:
            print("here")
            fig = plt.figure()
            ax = fig.add_subplot(111)
        y = np.array(entropy_array[1::3]) - log_factorial(np.array(nparticles))
        yerr = np.array(entropy_array[2::3])
        xa, y = _sort_pair(nparticles, y)
        x, yerr = _sort_pair(nparticles, yerr)
        # print ylabel, len(x), len(y)
        # print x,"\n", y
        # fit = np.polyfit(x, y, 1, w=1./np.array(yerr))
        # ynew = trialx * fit[0] + fit[1]
        def ff(x, a):
            return a * x

        popt, pcov = curve_fit(ff, x, y, sigma=yerr, absolute_sigma=True)
        ynew = ff(trialx, popt[0])
        color = next(color_cycle)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            marker=marker,
            linestyle="",
            ms=12,
            label=r"$S^\star -\ln N!$",
        )
        ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        # raw
        if raw:
            y = np.array(entropy_array[1::3])
            x, y = _sort_pair(nparticles, y)
            # fit = np.polyfit(x, y, 1, w=1./np.array(yerr))
            # ynew = trialx * fit[0] + fit[1]
            popt, pcov = curve_fit(ff, x, y, sigma=yerr, absolute_sigma=True)
            ynew = ff(trialx, popt[0])
            raw_color = color  # color_cycle.next()
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=raw_color,
                marker=raw_marker,
                linestyle="",
                ms=12,
            )
            ax.plot(trialx, ynew, "--", color=raw_color, linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.legend(
            frameon=False,
            loc="best",
            prop={"size": 28},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        if title:
            plt.title(title)
        if show:
            plt.show()
        return ax

    def _plot(
        self,
        entropy_array,
        xlabel=r"$N$",
        ylabel=r"$S$",
        raw=True,
        title=None,
        show=False,
    ):
        color_cycle = get_color_cycle()
        nparticles = entropy_array[::2]
        nmax = np.amax(nparticles)
        trialx = np.linspace(0, nmax, 1000)
        fig = plt.figure()
        # extensive
        ax = fig.add_subplot(111)
        y = np.array(entropy_array[1::2]) - log_factorial(np.array(nparticles))
        fit = np.polyfit(nparticles, y, 1)
        ynew = trialx * fit[0] + fit[1]
        color = next(color_cycle)
        ax.errorbar(
            nparticles,
            y,
            marker="o",
            linestyle="",
            color=color,
            ms=12,
            label=r"$S^\star -\ln N!$",
        )
        ax.plot(trialx, ynew, "b--", linewidth=2)
        # raw
        if raw:
            y = np.array(entropy_array[1::2])
            fit = np.polyfit(nparticles, y, 1)
            ynew = trialx * fit[0] + fit[1]
            color = next(color_cycle)
            ax.errorbar(
                nparticles,
                y,
                marker="^",
                linestyle="",
                color=color,
                ms=12,
                label=r"$S^\star$",
            )
            ax.plot(trialx, ynew, "r--", linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.legend(
            frameon=False,
            loc="best",
            prop={"size": 28},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        if title:
            plt.title(title)
        if show:
            plt.show()
        return ax

    def _plot_apf(
        self,
        ax=None,
        xlabel=r"$N$",
        ylabel=r"$S$",
        raw=False,
        marker="o",
        raw_marker="^",
    ):
        entropy_array, label, plot_label = self.all_entropies_err[0]
        ax = self._ploterr(
            entropy_array,
            xlabel=xlabel,
            ylabel=label,
            raw=raw,
            ax=ax,
            marker=marker,
            raw_marker=raw_marker,
        )
        return ax

    def plot_single(self, show=False, savefig=True):
        for item in self.all_entropies_err:
            entropy_array, label, plot_label = item
            self._ploterr(entropy_array, ylabel=label)
            if savefig:
                plt.savefig("plot_{}.pdf".format(plot_label))
        for item in self.all_entropies:
            entropy_array, label, plot_label = item
            self._plot(entropy_array, ylabel=label)
            if savefig:
                plt.savefig("plot_{}.pdf".format(plot_label))
        if show:
            plt.show()

    def plot_all(
        self,
        xlabel=r"$N$",
        ylabel=r"$S$",
        title=None,
        show=False,
        savefig=False,
    ):
        color_cycle = get_color_cycle()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for item in self.all_entropies_err:
            entropy_array, label, plot_label = item
            nparticles = np.array(entropy_array[0::3])
            nmax = np.amax(nparticles)
            trialx = np.linspace(0, nmax, 1000)
            # extensive
            y = np.array(entropy_array[1::3]) - log_factorial(np.array(nparticles))
            yerr = np.array(entropy_array[2::3])
            xa, y = _sort_pair(nparticles, y)
            x, yerr = _sort_pair(nparticles, yerr)
            fit = np.polyfit(x, y, 1, w=1.0 / np.array(yerr))
            ynew = trialx * fit[0] + fit[1]
            m = next(self.markercycler)
            color = next(color_cycle)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=m,
                linestyle="",
                color=color,
                ms=12,
                label=label,
            )
            ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        # loop over entropies without an associated error
        for item in self.all_entropies:
            entropy_array, label, plot_label = item
            nparticles = np.array(entropy_array[0::2])
            nmax = np.amax(nparticles)
            trialx = np.linspace(0, nmax, 1000)
            # extensive
            y = np.array(entropy_array[1::2]) - log_factorial(np.array(nparticles))
            x, y = _sort_pair(nparticles, y)
            fit = np.polyfit(x, y, 1)
            ynew = trialx * fit[0] + fit[1]
            m = next(self.markercycler)
            color = next(color_cycle)
            ax.errorbar(
                x,
                y,
                marker=m,
                linestyle="",
                color=color,
                ms=12,
                mew=2,
                label=label,
            )
            ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.legend(
            frameon=False,
            loc="best",
            prop={"size": 28},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        if title:
            plt.title(title)
        if show:
            plt.show()
        if savefig:
            plt.savefig("compare_all.pdf")
        return ax

    def plot_lo_param(self, xlabel=r"$N$", show=False, savefig=False, mlplot=False):
        nparticles = np.array(self.lo_parameters[::7])
        nparticlesml = np.array(self.loml_parameters[::4])
        nmax = np.amax(np.append(nparticles, nparticlesml))
        nmin = np.amin(np.append(nparticles, nparticlesml))
        trialx = np.linspace(0, nmax, 1000)
        # mu
        color_cycle = get_color_cycle()
        mu = np.array(self.lo_parameters[1::7])
        mu_err = np.array(self.lo_parameters[2::7])
        xa, mu = _sort_pair(nparticles, mu)
        x, mu_err = _sort_pair(nparticles, mu_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        color = next(color_cycle)
        ax.errorbar(
            x,
            mu,
            yerr=mu_err,
            marker="o",
            linestyle="",
            color=color,
            ms=12,
            label=r"$\mu$",
        )
        fit = np.polyfit(x, mu, 1, w=1.0 / np.array(mu_err))
        ynew = trialx * fit[0] + fit[1]
        ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        # muML
        if mlplot:
            mu = np.array(self.loml_parameters[1::4])
            x, mu = _sort_pair(nparticlesml, mu)
            color = next(color_cycle)
            ax.errorbar(
                x,
                mu,
                marker="^",
                linestyle="",
                color=color,
                ms=12,
                label=r"$\mu_{ML}$",
            )
            fit = np.polyfit(x, mu, 1)
            ynew = trialx * fit[0] + fit[1]
            ax.plot(trialx, ynew, "r--", color=color, linewidth=2)
            ax.legend(
                frameon=False,
                loc="best",
                prop={"size": 28},
                numpoints=1,
                scatterpoints=1,
                markerscale=1,
                columnspacing=0.25,
                labelspacing=0.25,
                handletextpad=0,
            )
        plt.xlabel(xlabel)  # plot
        plt.ylabel(r"$\mu$")
        if savefig:
            plt.savefig("lo_mu.pdf")
        # alpha
        color_cycle = get_color_cycle()
        alpha = np.array(self.lo_parameters[3::7])
        alpha_err = np.array(self.lo_parameters[4::7])
        xa, alpha = _sort_pair(nparticles, alpha)
        x, alpha_err = _sort_pair(nparticles, alpha_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        color = next(color_cycle)
        ax.errorbar(
            x,
            alpha,
            yerr=alpha_err,
            marker="o",
            linestyle="",
            color=color,
            ms=12,
            label=r"$\sigma$",
        )
        fit = np.polyfit(x, alpha, 1, w=1.0 / np.array(alpha_err))
        ynew = trialx * fit[0] + fit[1]
        ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        # alphaML
        if mlplot:
            alpha = np.array(self.loml_parameters[2::4])
            x, alpha = _sort_pair(nparticlesml, alpha)
            color = next(color_cycle)
            ax.errorbar(
                x,
                alpha,
                marker="^",
                linestyle="",
                color=color,
                ms=12,
                label=r"$\sigma_{ML}$",
            )
            fit = np.polyfit(x, alpha, 1)
            ynew = trialx * fit[0] + fit[1]
            ax.plot(trialx, ynew, "--", color=color, linewidth=2)
            ax.legend(
                frameon=False,
                loc="best",
                prop={"size": 28},
                numpoints=1,
                scatterpoints=1,
                markerscale=1,
                columnspacing=0.25,
                labelspacing=0.25,
                handletextpad=0,
            )
        plt.xlabel(xlabel)
        plt.ylabel(r"$\sigma$")
        if savefig:
            plt.savefig("lo_alpha.pdf")
        # zeta
        color_cycle = get_color_cycle()
        trialx = np.linspace(0, 1 / nmin, 1000)
        zeta = np.array(self.lo_parameters[5::7])
        zeta_err = np.array(self.lo_parameters[6::7])
        xa, zeta = _sort_pair(nparticles, zeta)
        x, zeta_err = _sort_pair(nparticles, zeta_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        szeta = 2 - zeta
        color = next(color_cycle)
        ax.errorbar(
            1.0 / x,
            szeta,
            yerr=zeta_err,
            marker="o",
            linestyle="",
            color=color,
            ms=12,
            label=r"$2-\zeta$",
        )
        fit = np.polyfit(1.0 / x, szeta, 1, w=1.0 / np.array(zeta_err))
        ynew = trialx * fit[0] + fit[1]
        ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        # zetaML
        if mlplot:
            zeta = np.array(self.loml_parameters[3::4])
            x, zeta = _sort_pair(nparticlesml, zeta)
            szeta = 2 - zeta
            color = next(color_cycle)
            ax.errorbar(
                1.0 / x,
                szeta,
                marker="^",
                linestyle="",
                color=color,
                ms=12,
                label=r"$2-\zeta_{ML}$",
            )
            fit = np.polyfit(1.0 / x, szeta, 1, w=1.0 / np.array(zeta_err))
            ynew = trialx * fit[0] + fit[1]
            ax.plot(trialx, ynew, "--", color=color, linewidth=2)
            ax.legend(
                frameon=False,
                loc="best",
                prop={"size": 28},
                numpoints=1,
                scatterpoints=1,
                markerscale=1,
                columnspacing=0.25,
                labelspacing=0.25,
                handletextpad=0,
            )
        plt.xlabel(r"1/N")
        plt.ylabel(r"$2-\zeta$")
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        if show:
            plt.show()
        if savefig:
            plt.savefig("lo_zeta.pdf")

    def plot_compare_apf2D(self, ax=None, show=False, savefig=True):
        """
        plot a comparison to the data provided by D. Asenjo for 2D packings
        for data with soft to hard ration 1.12
        ##### N, mean, err_mean, alpha^2, err_alpha^2, beta, err_beta
        """
        fpath = os.path.join(self.workdir, "apf_prl_data/N_mean_alpha_beta_dense.dat")
        if not os.path.isfile(fpath):
            raise Exception("{} not a file".format(fpath))
        dat_dense = np.loadtxt(fpath)
        f_ex_dense = 3.39558433477
        kmax_dense = np.array([40000.0, 50000.0, 90000.0, 180000.0, 400000.0])
        f0_dense = dat_dense[-1, 1] / 128.0 - f_ex_dense
        # plot all
        entropy_array, label, plot_label = self.all_entropies_err[0]

        ax = self._ploterr(entropy_array, title=None, raw=False, raw_marker="o", show=False)
        trialx = np.linspace(0, 130, 1000)
        # d+1/d-1 equation
        #        ynew = trialx * 1./2
        #        ax.plot(trialx, ynew,'r--', linewidth=2, label=r'$\frac{d-1}{d+1}f(\phi)N$')
        #        ynew = trialx * 1./3
        #        ax.plot(trialx, ynew,'r--', linewidth=2)
        # plot apf_prl
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        x = dat_dense[:, 0]
        y_apf = (
            dat_dense[:, 1]
            - dat_dense[:, 0] * f_ex_dense
            - dat_dense[:, 0] * np.log(dat_dense[:, 0])
            + dat_dense[:, 0]
            - np.log(dat_dense[:, 0])
            + np.log(2.0 * np.pi / kmax_dense)
        )
        y_lo = (
            dat_dense[:, 7]
            + dat_dense[:, 1]
            - dat_dense[:, 0] * f_ex_dense
            - (dat_dense[:, 0] * np.log(dat_dense[:, 0]))
            + dat_dense[:, 0]
            - np.log(dat_dense[:, 0])
            + np.log(2.0 * np.pi / kmax_dense)
        )
        y_apf_err = dat_dense[:, 2]
        # 2D apf
        color_cycle = get_color_cycle()
        color = next(color_cycle)
        color = next(color_cycle)
        ax.errorbar(
            x,
            y_apf,
            marker="^",
            yerr=y_apf_err,
            color=color,
            linestyle="",
            markersize=12,
            label=r"$S^\star_{2D} -\ln N!$",
        )
        # fit = np.polyfit(x, y_apf, 1)
        # ynew = trialx * fit[0] + fit[1]
        def ff(x, a):
            return a * x

        popt, pcov = curve_fit(ff, x, y_apf, sigma=y_apf_err, absolute_sigma=True)
        ynew = ff(trialx, popt[0])
        ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        # raw
        if False:
            y_apf = (
                dat_dense[:, 1]
                - dat_dense[:, 0] * f_ex_dense
                - np.log(dat_dense[:, 0])
                + np.log(2.0 * np.pi / kmax_dense)
            )
            color_cycle = get_color_cycle()
            ax.errorbar(
                x,
                y_apf,
                marker="^",
                yerr=y_apf_err,
                color=color,
                markersize=12,
            )
            # fit = np.polyfit(x, y_apf, 1)
            # ynew = trialx * fit[0] + fit[1]
            popt, pcov = curve_fit(ff, x, y_apf, sigma=y_apf_err, absolute_sigma=True)
            ynew = ff(trialx, popt[0])
            ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        # 2D LogOmega
        #        ax.plot(x, y_lo, 'g*', markersize=15, label=r"PRL(\ln \Omega_G)_{2D}")
        #        fit = np.polyfit(x, y_lo, 1)
        #        ynew = trialx * fit[0] + fit[1]
        #        ax.plot(trialx,ynew,'g--', linewidth=2)
        L = ax.legend(
            frameon=False,
            loc="best",
            prop={"size": 18},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        L.get_texts()[1].set_text("2D")
        L.get_texts()[0].set_text("3D")
        if show:
            plt.show()
        if savefig:
            plt.savefig("compare_entropy.pdf")
        return ax

    def plot_lo_param_compareAPF(self, xlabel=r"$N$", show=False, savefig=False):
        """
        N, mean, err_mean, alpha^2, err_alpha^2, beta, err_beta
        """
        fpath = os.path.join(self.workdir, "apf_prl_data/N_mean_alpha_beta_dense.dat")
        if not os.path.isfile(fpath):
            raise Exception("{} not a file".format(fpath))
        dat_dense = np.loadtxt(fpath)

        nparticles = np.array(self.lo_parameters[::7])
        nparticlesml = np.array(self.loml_parameters[::4])
        nmax = np.amax(np.append(nparticles, nparticlesml))
        nmin = np.amin(np.append(nparticles, nparticlesml))
        trialx = np.linspace(0, nmax, 1000)
        # mu
        color_cycle = get_color_cycle()
        mu = np.array(self.lo_parameters[1::7])
        mu_err = np.array(self.lo_parameters[2::7])
        xa, mu = _sort_pair(nparticles, mu)
        x, mu_err = _sort_pair(nparticles, mu_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        color = next(color_cycle)
        ax.errorbar(
            x,
            mu,
            yerr=mu_err,
            marker="o",
            linestyle="",
            color=color,
            ms=12,
            label=r"$3D$",
        )
        fit = np.polyfit(x, mu, 1, w=1.0 / np.array(mu_err))
        ynew = trialx * fit[0] + fit[1]
        ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        # muAPF
        x = np.array(dat_dense[:, 0])
        mu = np.array(dat_dense[:, 1])
        mu_err = np.array(dat_dense[:, 4])
        color = next(color_cycle)
        ax.errorbar(
            x,
            mu,
            yerr=mu_err,
            marker="^",
            linestyle="",
            color=color,
            ms=12,
            label=r"$2D$",
        )
        fit = np.polyfit(x, mu, 1, w=1.0 / np.array(mu_err))
        ynew = trialx * fit[0] + fit[1]
        ax.plot(trialx, ynew, "r--", color=color, linewidth=2)
        ax.legend(
            frameon=False,
            loc="best",
            prop={"size": 28},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        plt.xlabel(xlabel)  # plot
        plt.ylabel(r"$\mu$")
        if savefig:
            plt.savefig("lo_mu_compare.pdf")
        # alpha
        color_cycle = get_color_cycle()
        alpha = np.array(self.lo_parameters[3::7])
        alpha_err = np.array(self.lo_parameters[4::7])
        xa, alpha = _sort_pair(nparticles, alpha)
        x, alpha_err = _sort_pair(nparticles, alpha_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        color = next(color_cycle)
        ax.errorbar(
            x,
            alpha,
            yerr=alpha_err,
            marker="o",
            linestyle="",
            color=color,
            ms=12,
            label=r"$3D$",
        )
        fit = np.polyfit(x, alpha, 1, w=1.0 / np.array(alpha_err))
        ynew = trialx * fit[0] + fit[1]
        ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        # alphaAPF
        alpha = np.sqrt(np.array(dat_dense[:, 3]))
        alpha_err = np.sqrt(np.array(dat_dense[:, 4]))
        x = np.array(dat_dense[:, 0])
        color = next(color_cycle)
        ax.errorbar(
            x,
            alpha,
            yerr=alpha_err,
            marker="^",
            linestyle="",
            color=color,
            ms=12,
            label=r"$2D$",
        )
        fit = np.polyfit(x, alpha, 1, w=1.0 / np.array(alpha_err))
        ynew = trialx * fit[0] + fit[1]
        ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        ax.legend(
            frameon=False,
            loc="best",
            prop={"size": 28},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        plt.xlabel(xlabel)
        plt.ylabel(r"$\sigma$")
        if savefig:
            plt.savefig("lo_alpha_compare.pdf")
        # zeta
        color_cycle = get_color_cycle()
        trialx = np.linspace(0, 1 / nmin, 1000)
        zeta = np.array(self.lo_parameters[5::7])
        zeta_err = np.array(self.lo_parameters[6::7])
        xa, zeta = _sort_pair(nparticles, zeta)
        x, zeta_err = _sort_pair(nparticles, zeta_err)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        szeta = 2 - zeta
        color = next(color_cycle)
        ax.errorbar(
            1.0 / x,
            szeta,
            yerr=zeta_err,
            marker="o",
            linestyle="",
            color=color,
            ms=12,
            label=r"$3D$",
        )
        fit = np.polyfit(1.0 / x, szeta, 1, w=1.0 / np.array(zeta_err))
        ynew = trialx * fit[0] + fit[1]
        ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        # zetaAPF
        zeta = np.array(dat_dense[:, 5])
        zeta_err = np.array(dat_dense[:, 6])
        x = np.array(dat_dense[:, 0])
        szeta = 2 - zeta
        color = next(color_cycle)
        ax.errorbar(
            1.0 / x,
            szeta,
            yerr=zeta_err,
            marker="^",
            linestyle="",
            color=color,
            ms=12,
            label=r"$2D$",
        )
        fit = np.polyfit(1.0 / x, szeta, 1, w=1.0 / np.array(zeta_err))
        trialx = np.linspace(0, 1 / np.amin(x), 1000)
        ynew = trialx * fit[0] + fit[1]
        ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        ax.legend(
            frameon=False,
            loc="best",
            prop={"size": 28},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        plt.xlabel(r"1/N")
        plt.ylabel(r"$2-\zeta$")
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        if show:
            plt.show()
        if savefig:
            plt.savefig("lo_zeta_compare.pdf")

    def plot_paper_all(self, show=False, savefig=False, legendzise=20):
        fpath = os.path.join(self.workdir, "apf_prl_data/N_mean_alpha_beta_dense.dat")
        if not os.path.isfile(fpath):
            raise Exception("{} not a file".format(fpath))
        dat_dense = np.loadtxt(fpath)

        nparticles = np.array(self.lo_parameters[::7])
        nparticlesml = np.array(self.loml_parameters[::4])
        nmax = np.amax(np.append(nparticles, nparticlesml))
        nmin = np.amin(np.append(nparticles, nparticlesml))
        trialx = np.linspace(0, nmax, 1000)
        fig = plt.figure(figsize=(10, 8))
        # mu
        color_cycle = get_color_cycle()
        mu = np.array(self.lo_parameters[1::7])
        mu_err = np.array(self.lo_parameters[2::7])
        xa, mu = _sort_pair(nparticles, mu)
        x, mu_err = _sort_pair(nparticles, mu_err)
        ax1 = fig.add_subplot(222)
        color = next(color_cycle)
        ax1.errorbar(
            x,
            mu,
            yerr=mu_err,
            marker="o",
            linestyle="",
            color=color,
            ms=14,
            label=r"$3D$",
        )
        fit = np.polyfit(x, mu, 1, w=1.0 / np.array(mu_err))
        ynew = trialx * fit[0] + fit[1]
        ax1.plot(trialx, ynew, "--", color=color, linewidth=2)
        # muAPF
        x = np.array(dat_dense[:, 0])
        mu = np.array(dat_dense[:, 1]) - x * np.log(x * np.pi * (1.12**2) / 0.88)
        mu_err = np.array(dat_dense[:, 4])
        color = next(color_cycle)
        ax1.errorbar(
            x,
            mu,
            yerr=mu_err,
            marker="^",
            linestyle="",
            color=color,
            ms=14,
            label=r"$2D$",
        )
        fit = np.polyfit(x, mu, 1, w=1.0 / np.array(mu_err))
        ynew = trialx * fit[0] + fit[1]
        ax1.plot(trialx, ynew, "r--", color=color, linewidth=2)
        ax1.legend(
            frameon=False,
            loc=2,
            prop={"size": legendzise},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        ax1.set_xlabel(r"$N$")  # plot
        ax1.set_ylabel(r"$\mu$")
        ax1.locator_params(axis="x", nbins=4)
        ax1.locator_params(axis="y", nbins=4)
        ax1.set_ylim((0, 320))
        ax1.set_xlim((0, 140))
        # alpha
        color_cycle = get_color_cycle()
        alpha = np.array(self.lo_parameters[3::7])
        alpha_err = np.array(self.lo_parameters[4::7])
        xa, alpha = _sort_pair(nparticles, alpha)
        x, alpha_err = _sort_pair(nparticles, alpha_err)
        ax2 = fig.add_subplot(223)
        color = next(color_cycle)
        ax2.errorbar(
            x,
            alpha,
            yerr=alpha_err,
            marker="o",
            linestyle="",
            color=color,
            ms=14,
            label=r"$3D$",
        )
        fit = np.polyfit(x, alpha, 1, w=1.0 / np.array(alpha_err))
        ynew = trialx * fit[0] + fit[1]
        ax2.plot(trialx, ynew, "--", color=color, linewidth=2)
        # alphaAPF
        alpha = np.sqrt(np.array(dat_dense[:, 3]))
        alpha_err = np.sqrt(np.array(dat_dense[:, 4]))
        x = np.array(dat_dense[:, 0])
        color = next(color_cycle)
        ax2.errorbar(
            x,
            alpha,
            yerr=alpha_err,
            marker="^",
            linestyle="",
            color=color,
            ms=14,
            label=r"$2D$",
        )
        fit = np.polyfit(x, alpha, 1, w=1.0 / np.array(alpha_err))
        ynew = trialx * fit[0] + fit[1]
        ax2.plot(trialx, ynew, "--", color=color, linewidth=2)
        ax2.legend(
            frameon=False,
            loc=2,
            prop={"size": legendzise},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        ax2.set_xlabel(r"$N$")
        ax2.set_ylabel(r"$\sigma$")
        ax2.locator_params(axis="x", nbins=4)
        ax2.locator_params(axis="y", nbins=4)
        ax2.set_xlim((0, 140))
        # zeta
        color_cycle = get_color_cycle()
        trialx = np.linspace(0, 1 / nmin, 1000)
        zeta = np.array(self.lo_parameters[5::7])
        zeta_err = np.array(self.lo_parameters[6::7])
        xa, zeta = _sort_pair(nparticles, zeta)
        x, zeta_err = _sort_pair(nparticles, zeta_err)
        ax3 = fig.add_subplot(224)
        szeta = 2 - zeta
        color = next(color_cycle)
        ax3.errorbar(
            1.0 / x,
            szeta,
            yerr=zeta_err,
            marker="o",
            linestyle="",
            color=color,
            ms=14,
            label=r"$3D$",
        )
        fit = np.polyfit(1.0 / x, szeta, 1, w=1.0 / np.array(zeta_err))
        ynew = trialx * fit[0] + fit[1]
        ax3.plot(trialx, ynew, "--", color=color, linewidth=2)
        # zetaAPF
        zeta = np.array(dat_dense[:, 5])
        zeta_err = np.array(dat_dense[:, 6])
        x = np.array(dat_dense[:, 0])
        szeta = 2 - zeta
        color = next(color_cycle)
        ax3.errorbar(
            1.0 / x,
            szeta,
            yerr=zeta_err,
            marker="^",
            linestyle="",
            color=color,
            ms=14,
            label=r"$2D$",
        )
        fit = np.polyfit(1.0 / x, szeta, 1, w=1.0 / np.array(zeta_err))
        trialx = np.linspace(0, 1 / np.amin(x), 1000)
        ynew = trialx * fit[0] + fit[1]
        ax3.plot(trialx, ynew, "--", color=color, linewidth=2)
        ax3.legend(
            frameon=False,
            loc=2,
            prop={"size": legendzise},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        ax3.set_xlabel(r"1/N")
        ax3.set_ylabel(r"$2-\zeta$")
        ax3.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax3.locator_params(axis="x", nbins=4)
        ax3.locator_params(axis="y", nbins=4)
        ax3.set_xlim((0, 0.07))
        ax3.set_ylim((-1, 1.5))
        # plt.ticklabel_format(style='sci',axis='x', scilimits=(0,0))

        ax = fig.add_subplot(221)
        color_cycle = get_color_cycle()
        markers = ["o", "^", "s", "x", "+"]
        self.markercycler = cycle(markers)
        for item in self.all_entropies_err:
            entropy_array, label, plot_label = item
            nparticles = np.array(entropy_array[0::3])
            nmax = np.amax(nparticles)
            trialx = np.linspace(0, nmax, 1000)
            # extensive
            y = np.array(entropy_array[1::3]) - log_factorial(np.array(nparticles))
            yerr = np.array(entropy_array[2::3])
            xa, y = _sort_pair(nparticles, y)
            x, yerr = _sort_pair(nparticles, yerr)
            fit = np.polyfit(x, y, 1, w=1.0 / np.array(yerr))
            ynew = trialx * fit[0] + fit[1]
            m = next(self.markercycler)
            color = next(color_cycle)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=m,
                linestyle="",
                color=color,
                ms=14,
                label=label,
            )
            ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        # loop over entropies without an associated error
        for item in self.all_entropies:
            entropy_array, label, plot_label = item
            nparticles = np.array(entropy_array[0::2])
            nmax = np.amax(nparticles)
            trialx = np.linspace(0, nmax, 1000)
            # extensive
            y = np.array(entropy_array[1::2]) - log_factorial(np.array(nparticles))
            x, y = _sort_pair(nparticles, y)
            fit = np.polyfit(x, y, 1)
            ynew = trialx * fit[0] + fit[1]
            m = next(self.markercycler)
            color = next(color_cycle)
            ax.errorbar(
                x,
                y,
                marker=m,
                linestyle="",
                color=color,
                ms=14,
                mew=2,
                label=label,
            )
            ax.plot(trialx, ynew, "--", color=color, linewidth=2)
        plt.xlabel(r"$N$")
        plt.ylabel(r"$S$")
        ax.legend(
            frameon=False,
            prop={"size": legendzise},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
            bbox_to_anchor=[0.52, 1.05],
        )
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=4)
        ax.set_ylim((0, 120))
        ax.set_xlim((0, 140))
        if show:
            plt.show()
        if savefig:
            plt.savefig("entropy_all_paper.pdf")


if __name__ == "__main__":
    show = False
    savefig = True
    pe = plot_entropy(analysis_folder="entropy_analysis_all")
    pe_msf = plot_entropy(analysis_folder="msf_entropy_analysis_all")
    # pe.plot_single(show=True, savefig=True)
    # pe.plot_compare_apf2D(show=show,savefig=savefig)
    # pe.plot_all(show=show,savefig=savefig)
    # pe.plot_lo_param(show=show,savefig=savefig)
    # pe.plot_lo_param_compareAPF(show=show,savefig=savefig)
    pe.plot_paper_all(show=show, savefig=savefig)
#    #COMPARE MSF TO NUMERICAL
#    raw=False
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    print ax
#    ax = pe._plot_apf(ax=ax, color='b', raw_color='b', raw=raw)
#    print ax
#    ax = pe_msf._plot_apf(ax=ax, color='g', raw_color='g', raw=raw)
#    #plt.show()
#    plt.savefig("plot_apf_predicted.pdf")
