from __future__ import division
from __future__ import print_function

from future import standard_library

standard_library.install_aliases()
from builtins import zip
from builtins import next
from builtins import range
import numpy as np
import configparser
import os
import re
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
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
import pickle as pickle
from basinvolume.post_processing import PackingData, PackingDataSet, BasinAnalysis
from joblib import Parallel, delayed
from basinvolume.experiment_2d.cross_validation_bandwidth_selection import (
    get_bandwidth_estimate,
    get_pdf,
)
from basinvolume.post_processing import (
    GeneralisedLogNormal,
    GeneralisedGauss,
    LogNormal,
    OutlierRemovalUnbiasingEntropyLogOmega,
)
import vegas
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet, EllipticEnvelope
from cycler import cycler
from scipy.stats.stats import pearsonr
from uncertainties import ufloat
from uncertainties import umath
import uncertainties.unumpy as unp

# except ImportError as err:
#     print err
#######################SET LATEX OPTIONS###################
rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
# rc('text.latex',preamble=r'\usepackage{times}')
glob_fontsize = 30
plt.rcParams.update({"font.size": glob_fontsize})
plt.rcParams["xtick.major.pad"] = 8
plt.rcParams["ytick.major.pad"] = 8
plt.rcParams.update({"figure.autolayout": True})
##########################################################
def get_color_cycle(ncolors=8, reverse=True):
    cm = plt.get_cmap("Paired")
    if reverse:
        color_cycle = cycle(
            [cm(1.0 * (i + 0.5) / float(ncolors)) for i in range(ncolors)][::-1]
        )
    else:
        color_cycle = cycle(
            [cm(1.0 * (i - 0.5) / float(ncolors)) for i in range(ncolors)]
        )
    return color_cycle


def get_marker_cycle():
    markers = [
        "o",
        "v",
        "s",
        "h",
        "^",
        "8",
        "p",
        "<",
        "*",
        "D",
        ">",
    ]
    markercycle = cycle(markers)
    return markercycle


def get_line_cycle():
    lines = ["--", "-"]
    linecycle = cycle(lines)
    return linecycle


class FixedOrderFormatter(ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of
    magnitude"""

    def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset, useMathText=useMathText)

    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag


class EdwardsGeneralisedLogNormal(GeneralisedLogNormal):
    def __init__(
        self,
        mu_initial=1,
        alpha_initial=1,
        zeta_initial=1,
        alpha_min=1e-10,
        zeta_min=1e-10,
        verbose=False,
    ):
        super(EdwardsGeneralisedLogNormal, self).__init__(
            mu_initial=mu_initial,
            alpha_initial=alpha_initial,
            zeta_initial=zeta_initial,
            alpha_min=alpha_min,
            zeta_min=zeta_min,
            verbose=verbose,
        )
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
            if t > 0.0:
                logt = np.log(t)
                logintegrand = (
                    (n / kappa) * logt
                    - ang * vcavity * t
                    - 0.5 * np.power(np.abs((logt - mu) / a), z)
                    - norm
                )
                return logintegrand
            else:
                return -1e300

        def log_integrand_den(tarr, norm):
            t = tarr[0]
            if t > 0.0:
                logt = np.log(t)
                logintegrand = (
                    (n / kappa - 1.0) * logt
                    - ang * vcavity * t
                    - 0.5 * np.power(np.abs((logt - mu) / a), z)
                    - norm
                )
                return logintegrand
            else:
                return -1e300

        # import matplotlib.pyplot as plt
        if ang < 1e-6:
            x = np.linspace(0, 1e7, 100000)
        else:
            x = np.linspace(0, self.glob_x, 100000)
        y = np.array([log_integrand_den([xi], 1) for xi in x])
        zargmax = np.argmax(y)
        maxy = np.amax(y)
        y = np.array([log_integrand_den([xi], maxy) for xi in x])
        zend = next(i for i, yy in enumerate(y[zargmax:]) if yy < np.log(1e-5))
        zend += zargmax

        y = np.array([log_integrand_nom([xi], 1) for xi in x])
        argmax = np.argmax(y)
        maxy = np.amax(y)
        print(np.amax(y))
        y = np.array([log_integrand_nom([xi], maxy) for xi in x])
        end = next(i for i, yy in enumerate(y[argmax:]) if yy < np.log(1e-5))
        end += argmax
        # plt.plot(x, y)
        # plt.show()

        if ang > 1e-6:
            self.glob_x = x[max(zend, end) + 1]

        zx = np.linspace(0, x[zend], 100000)
        y = np.array([log_integrand_den([xi], 1) for xi in zx])
        maxy = np.amax(y)
        print("zxmax", zx[-1])
        print("xmax", x[end])

        intnval = 10000
        integ = vegas.Integrator([[0.0, x[end]]])
        result = integ(
            lambda t: np.exp(log_integrand_nom(t, maxy)),
            nitn=20,
            neval=intnval,
            alpha=0.2,
            beta=1,
        )
        Pea, errPea = result.mean, result.sdev
        integ = vegas.Integrator([[0.0, zx[-1]]])
        result = integ(
            lambda t: np.exp(log_integrand_den(t, maxy)),
            nitn=20,
            neval=intnval,
            alpha=0.2,
            beta=1,
        )
        zPea, errzPea = result.mean, result.sdev
        print("{} \pm {} \n {} \pm {}".format(Pea, errPea, zPea, errzPea))
        return np.log(Pea) - np.log(zPea), 0.434 * (errPea / Pea + errzPea / zPea)


# cdf = CDFAccumulator()
# cdf.add_array(x)
# x, cdf_x = cdf.get_vecdata()


def sigmoid(x, x0, k, ymax, ymin, v):
    return ymax - (ymax - ymin) / np.power(1.0 + np.exp(-k * (x - x0)), 1.0 / v)


def sigmoid_d1(x, x0, k, ymax, ymin, v):
    return (
        -(k / v)
        * (ymax - ymin)
        * np.exp(-k * (x - x0))
        / np.power(1.0 + np.exp(-k * (x - x0)), 1.0 + 1.0 / v)
    )


def remove_outliers_cluster(x, y, yerr):
    x, y = np.asarray(x), np.asarray(y)
    y_outliers = OutlierDetection(y, p=0.5, D=4 * np.std(y))
    x_outliers = OutlierDetection(x, p=0.5, D=4 * np.std(x))
    non_outliers_indexes = list(
        set(y_outliers.non_outliers_indexes).intersection(
            x_outliers.non_outliers_indexes
        )
    )
    x = np.array(x)[np.array(non_outliers_indexes, dtype="i")]
    y = np.array(y)[np.array(non_outliers_indexes, dtype="i")]
    yerr = np.array(yerr)[np.array(non_outliers_indexes, dtype="i")]
    return x, y, yerr, non_outliers_indexes


def remove_outliers_mcd(x, y, yerr, support_fraction=0.99, contamination=0.1):
    x, y = np.asarray(x), np.asarray(y)
    classifier = EllipticEnvelope(
        contamination=contamination, support_fraction=support_fraction, random_state=42
    )
    features = np.vstack((x, y)).T
    classifier.fit(features)
    decision = classifier.predict(features)
    mask = decision > 0
    return x[mask], y[mask], yerr[mask], mask


def remove_outliers_pca_cluster(x, y, yerr):
    pca = PCA(n_components=2)
    x, y = np.asarray(x), np.asarray(y)
    features = np.vstack((x, y)).T
    pca.fit(features)
    pca_projection1 = np.array(
        [np.dot(xx, pca.components_[0]) for xx in np.vstack((x, y)).T]
    )
    pca_projection2 = np.array(
        [np.dot(xx, pca.components_[1]) for xx in np.vstack((x, y)).T]
    )
    outliers_pca1 = OutlierDetection(
        pca_projection1, p=0.5, D=3 * np.std(pca_projection1)
    )
    outliers_pca2 = OutlierDetection(
        pca_projection2, p=0.5, D=3 * np.std(pca_projection2)
    )
    non_outliers_indexes_pca = list(
        set(outliers_pca1.non_outliers_indexes).intersection(
            outliers_pca2.non_outliers_indexes
        )
    )
    x = np.array(x)[np.array(non_outliers_indexes_pca, dtype="i")]
    y = np.array(y)[np.array(non_outliers_indexes_pca, dtype="i")]
    yerr = np.array(yerr)[np.array(non_outliers_indexes_pca, dtype="i")]
    # plt.scatter(pca_projection1[np.array(non_outliers_indexes_pca, dtype="i")],
    #             pca_projection2[np.array(non_outliers_indexes_pca, dtype="i")])
    # plt.show()
    return x, y, yerr


def poly_fit(x, y, yerr=None, order=1):
    w = 1.0 / np.array(yerr) if yerr is not None else None
    if len(x) - order - 2 < 2:
        # this hack was taken from
        # https://stackoverflow.com/questions/27230285/numpy-polyfit-gives-useful-fit-but-infinite-covariance-matrix
        assert order == 2, "hack is only implemented for order=2"
        w = [1 for _ in range(len(x))].append(0)
        x = np.append(x, x[-1])
        y = np.append(y, y[-1])
    fit_params, cov = np.polyfit(x, y, order, w=w, cov=True)
    fit_err = np.sqrt(np.diag(cov))
    fit_fn = np.poly1d(fit_params)
    rho = pearsonr(x, y)[0]
    return fit_fn, fit_params, fit_err, rho


def lmms_fit(x, y, support_fraction=0.99):
    # linear minimum mean square error estimator
    x, y = np.asarray(x), np.asarray(y)
    robust_cov = MinCovDet(support_fraction=support_fraction, random_state=42).fit(
        np.vstack((x, y)).T
    )
    cov = robust_cov.covariance_[0, 1]
    mean_x, var_x = robust_cov.location_[0], robust_cov.covariance_[0, 0]
    mean_y, var_y = robust_cov.location_[1], robust_cov.covariance_[1, 1]
    robust_rho = cov / np.sqrt(var_x * var_y)
    print(" robust rho", robust_rho)
    m = cov / var_x
    interc = mean_y - m * mean_x
    fit_fn = lambda xnew: m * (np.asarray(xnew) - mean_x) + mean_y
    # http://athenasc.com/Bivariate-Normal.pdf
    mserr = np.sqrt(var_y) * np.sqrt(1 - robust_rho**2)
    # https://stats.stackexchange.com/questions/44838/how-are-the-standard-errors-of-coefficients-calculated-in-a-regression
    err = (
        mserr**2
        * np.array([[np.sum(x**2), -np.sum(x)], [-np.sum(x), x.size]])
        / (x.size * np.sum(x**2) - np.sum(x) ** 2)
    )
    fit_err = np.sqrt(np.diag(err))
    return fit_fn, (m, interc), fit_err[::-1], robust_rho


def robust_mean_var(x, y, support_fraction=0.99):
    x, y = np.asarray(x), np.asarray(y)
    robust_cov = MinCovDet(support_fraction=support_fraction, random_state=42).fit(
        np.vstack((x, y)).T
    )
    cov = robust_cov.covariance_[0, 1]
    mean_x, var_x = robust_cov.location_[0], robust_cov.covariance_[0, 0]
    mean_y, var_y = robust_cov.location_[1], robust_cov.covariance_[1, 1]
    logl = robust_cov.score(np.vstack((x, y)).T)
    return (mean_x, var_x), (mean_y, var_y), cov, logl


def find_roots(params, err):
    # see http://kitchingroup.cheme.cmu.edu/blog/2013/07/05/Uncertainty-in-polynomial-roots/
    assert len(params) == len(err)
    if len(params) == 2:
        # equation of a line
        a, b = ufloat(params[0], err[0]), ufloat(params[1], err[1])
        r1 = -b / a
        roots = [(r1.nominal_value, r1.std_dev)]
    elif len(params == 3):
        a, b, c = (
            ufloat(params[0], err[0]),
            ufloat(params[1], err[1]),
            ufloat(params[2], err[2]),
        )
        roots = []
        try:
            r1 = (-b + umath.pow(b**2 - 4 * a * c, 0.5)) / (2 * a)
            roots.append((r1.nominal_value, r1.std_dev))
        except Exception:
            pass
        try:
            r2 = (-b - umath.pow(b**2 - 4 * a * c, 0.5)) / (2 * a)
            roots.append[(r2.nominal_value, r2.std_dev)]
        except Exception:
            pass
    else:
        raise NotImplementedError
    return roots


def plot(packing_datasets, figdir="figures", phi_min=0.825, phi_max=0.865):
    from scipy.optimize import curve_fit

    if not os.path.isabs(figdir):
        figdir = os.path.join(os.getcwd(), figdir)
    trymakedir(figdir)

    if True:
        # plot free energy vs P for all packings
        import matplotlib.gridspec as gridspec

        color_cycle = get_color_cycle()
        # fig = plt.figure(figsize=(8, 8))
        # gs = gridspec.GridSpec(7, 2)
        # ax = fig.add_subplot(gs[:4, :])
        fig0 = plt.figure()
        ax = fig0.add_subplot(111)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        fig11 = plt.figure()
        ax11 = fig11.add_subplot(111)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        fig22 = plt.figure()
        ax22 = fig22.add_subplot(111)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        fig32 = plt.figure()
        ax32 = fig32.add_subplot(111)
        fig33 = plt.figure()
        ax33 = fig33.add_subplot(111)
        fig34 = plt.figure()
        ax34 = fig34.add_subplot(111)
        fig35 = plt.figure()
        ax35 = fig35.add_subplot(111)
        fig36 = plt.figure()
        ax36 = fig36.add_subplot(111)
        left, bottom, width, eight = [0.2, 0.6, 0.3, 0.3]
        # ax36inset = fig36.add_axes([left, bottom, width, eight])

        Sg, Sb_gauss, Sb_kde = [], [], []
        phi, avgz = [], []
        pmin, pmax = 1e100, -1e100
        p_minmax_list = []
        meanvar_f_list, meanvar_pi_list = [], []
        histograms_nsamples = []
        cov_f_pi_list, logl_list = [], []
        bw_list = []
        pjam_spline = pickle.load(open("phi_psuccess_spline.pickle", "rb"))
        for i, dataset in enumerate(
            sorted(packing_datasets, key=lambda data: data.ss_phi)
        ):
            if len(dataset.free_energies) > 0 and phi_min < dataset.ss_phi < phi_max:
                print("set name ", dataset.set_name)
                nparticles = dataset.nparticles
                j = 0
                Facc = None
                vcavity = None
                while Facc is None or vcavity is None:
                    Facc = dataset.packing_data[j].Facc - np.log(
                        pjam_spline(dataset.ss_phi)
                    )
                    vcavity = dataset.packing_data[j].vcavity
                    j += 1
                print(Facc)
                print(nparticles)
                if (
                    0.86 < dataset.ss_phi < phi_max and "fire" not in dataset.set_name
                ) or (phi_min < dataset.ss_phi < 0.865 and "fire" in dataset.set_name):
                    # should remoe both outliers in pressure and in volume
                    (
                        p_raw,
                        f_raw,
                        ferr_raw,
                        non_outliers_indexes,
                    ) = remove_outliers_cluster(
                        np.log(dataset.pressures),
                        np.array(dataset.free_energies),
                        np.array(dataset.free_energies_err),
                    )
                    print(
                        dataset.ss_phi,
                        " cluster 4s n_outliers: ",
                        np.array(dataset.free_energies).size - f_raw.size,
                    )
                    # fix units
                    u, s = 1, 0.1  # E(x^2) = V(x) + (E(x))^2
                    p_raw, f_raw = p_raw + np.log(
                        np.pi * (u**2 + s**2)
                    ), f_raw + nparticles * np.log(
                        np.pi * (u**2 + s**2)
                    )  # DEBUG: here I supposedly adjust the units, check this
                    Facc += nparticles * np.log(
                        np.pi * (u**2 + s**2)
                    )  # DEBUG: check here too
                    # end fix units
                    p, f, ferr = p_raw, f_raw, ferr_raw
                    p = np.exp(p)
                    pmin, pmax = min(pmin, np.amin(p)), max(pmax, np.amax(p))
                    p_minmax_list.append([np.amin(p), np.amax(p)])
                    p = np.log(p)
                    y = Facc - f
                    y_raw = Facc - f_raw
                    if "fire" in dataset.set_name:
                        marker = "o"
                        label = "{:.3f}".format(dataset.ss_phi)
                    else:
                        marker = "o"
                        label = "cgd {:.3f}".format(dataset.ss_phi)
                    color = next(color_cycle)
                    ax.scatter(p_raw, -y_raw, label=label, marker=marker, color=color)
                    fit_fn, fit_params, fit_err, rho = lmms_fit(p, y)
                    ax.plot(p_raw, -fit_fn(p_raw), color="k", linestyle="-")

                    (mean_p, var_p), (mean_f, var_f), cov, logl = robust_mean_var(
                        p, f_raw
                    )
                    meanvar_f_list.append([mean_f, var_f])
                    meanvar_pi_list.append([mean_p, var_p])
                    cov_f_pi_list.append(cov)
                    logl_list.append(logl)

                    u = ufloat(
                        mean_f, np.sqrt(var_f / (0.9 * f.size))
                    )  # debug: hardcoded 0.9 parameter
                    s = u - Facc - log_factorial(nparticles)
                    Sg.append([s.nominal_value, s.std_dev])
                    # S.append(- Facc - log_factorial(dataset.nparticles))
                    phi.append(dataset.ss_phi)
                    # now fit the actual power laws, not the probabilities
                    fit_fn, fit_params, fit_err, rho = lmms_fit(p, f)
                    # fit_fn, fit_params, fit_err, rho = lmms_fit(x, f)
                    dataset.add_extras((fit_params, fit_err, rho))
                    print(dataset.extras)

                    contacts = np.array(dataset.contacts)[
                        np.array(non_outliers_indexes, dtype="i")
                    ]
                    assert len(contacts) == len(f_raw)
                    avgz.append(np.mean(contacts))

                    # FIT DISTRIBUTIONS
                    # remove outliers as for robust_mean_var
                    # to avoid fit issues
                    p, f, ferr, mask = remove_outliers_mcd(p_raw, f_raw, ferr_raw)
                    print("n outliers mcd: ", mask.tolist().count(False))
                    histograms_nsamples.append(len(f))
                    print("n samples", len(f))
                    print("min f, max f", np.amin(f), np.amax(f))
                    # f: fit to kde
                    bw = get_bandwidth_estimate(
                        f, kernel="gaussian", method="cross_validation"
                    )
                    bw_list.append(bw[0])
                    edges = np.linspace(120, 160, 10000)
                    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(
                        f[:, np.newaxis]
                    )
                    kdehist = np.exp(kde.score_samples(edges[:, np.newaxis]))
                    ax3.plot(
                        edges, kdehist, label=dataset.ss_phi, color=color, linewidth=3
                    )
                    kdefunc = lambda x: np.exp(np.add(kde.score_samples([[x]]), x))
                    kde_integral, kde_integral_error = integrate.quad(
                        kdefunc,
                        Facc,
                        np.amax(f) * 100,
                        points=[np.amin(f), np.amax(f), np.mean(f)],
                    )
                    u = ufloat(kde_integral, np.sqrt(np.var(np.exp(f)) / f.size))
                    s = umath.log(u) - Facc - log_factorial(nparticles)
                    Sb_kde.append([s.nominal_value, s.std_dev])
                    # fit to generalised gaussian

                    # np.random.seed(42)
                    # n_resample = 20
                    # n = 0
                    # fintegral_list = []
                    # while n < n_resample:
                    #     generalised_gauss = GeneralisedGauss(alpha_min=0.00001, zeta_min=0.1)
                    #     cdf = CDFAccumulator()
                    #     idx = np.random.choice(f.size, f.size)
                    #     _, f_resampled, _ = remove_outliers_mcd(p_raw[idx], f_raw[idx], ferr_raw[idx])
                    #     cdf.add_array(f_resampled)
                    #     x, cdf_x = cdf.get_vecdata()
                    #     generalised_gauss.fit_cdf(x, cdf_x)
                    #     # compute boltzmann entropy from generalised gaussian
                    #     fintegral, fintegral_error = integrate.quad(generalised_gauss.get_fitted_times_expx,
                    #                                                 Facc, np.amax(f_resampled) * 100,
                    #                                                 points=[np.amin(f_resampled),
                    #                                                         np.amax(f_resampled), np.mean(f_resampled)])
                    #     print fintegral, fintegral_error
                    #     if np.isfinite(fintegral) and fintegral_error < 1e55:
                    #         fintegral_list.append(fintegral)
                    #         n += 1
                    generalised_gauss = GeneralisedGauss(
                        alpha_min=0.00001, zeta_min=0.1
                    )
                    cdf = CDFAccumulator()
                    cdf.add_array(f)
                    x, cdf_x = cdf.get_vecdata()
                    generalised_gauss.fit_cdf(x, cdf_x)
                    # compute boltzmann entropy from generalised gaussian
                    fintegral, fintegral_error = integrate.quad(
                        generalised_gauss.get_fitted_times_expx,
                        Facc,
                        np.amax(f) * 100,
                        points=[np.amin(f), np.amax(f), np.mean(f)],
                    )
                    u = ufloat(fintegral, np.sqrt(np.var(np.exp(f)) / f.size))
                    s = umath.log(u) - Facc - log_factorial(nparticles)
                    Sb_gauss.append([s.nominal_value, s.std_dev])
                    hist = np.array(
                        [generalised_gauss.get_fitted(edge) for edge in edges]
                    )
                    ax3.plot(edges, hist, linestyle="--", color=color, linewidth=3)

                    # mean_func = lambda x: generalised_gauss.get_fitted_times_expx(x) * x / fintegral
                    # mean, mean_error = integrate.quad(mean_func, Facc, np.amax(f) * 100,
                    #                                   points=[np.amin(f), np.amax(f), np.mean(f)])
                    # var_func = lambda x: generalised_gauss.get_fitted_times_expx(x) * ((x - mean) ** 2) / fintegral
                    # fvar, fvar_error = integrate.quad(var_func, Facc, np.amax(f) * 100,
                    #                                   points=[np.amin(f), np.amax(f), np.mean(f)])
                    # print "fvar", fvar
                    # meanvar_f_u_list.append([mean, fvar])

                    # f: fit weighted kde
                    hist = kdehist * np.exp(edges) / kde_integral
                    hist /= simps(hist, edges)
                    ax33.plot(
                        edges, hist, color=color, label=dataset.ss_phi, linewidth=3
                    )
                    # wpdf = weighted_gaussian_kde(f, weights=np.exp(f), bw_method=bw[0])
                    # ax33.plot(edges, wpdf(edges), label=dataset.ss_phi, color=color, linewidth=3)
                    hist = (
                        np.array(
                            [
                                generalised_gauss.get_fitted_times_expx(edge)
                                for edge in edges
                            ]
                        )
                        / fintegral
                    )
                    hist /= simps(hist, edges)
                    ax33.plot(edges, hist, linestyle="--", color=color, linewidth=3)
                    # u_set = wpdf.resample(size=1e4)
                    # meanvar_f_u_list.append([np.mean(u_set), np.var(u_set)])

                    # p: fit to kde
                    bw = get_bandwidth_estimate(
                        p, kernel="gaussian", method="cross_validation"
                    )
                    edges = np.linspace(-2, 8, 10000)
                    kdehist = get_pdf(p, edges, bandwidth=bw, kernel="gaussian")
                    ax32.plot(
                        edges, kdehist, label=dataset.ss_phi, color=color, linewidth=3
                    )
                    # fit to generalised gaussian
                    generalised_gauss = GeneralisedGauss(
                        alpha_min=0.00001, zeta_min=0.1
                    )
                    cdf = CDFAccumulator()
                    cdf.add_array(p)
                    x, cdf_x = cdf.get_vecdata()
                    generalised_gauss.fit_cdf(x, cdf_x)
                    print("mu: ", generalised_gauss.mu_fit)
                    print("alpha: ", generalised_gauss.alpha_fit)
                    print("zeta: ", generalised_gauss.zeta_fit)
                    hist = np.array(
                        [generalised_gauss.get_fitted(edge) for edge in edges]
                    )
                    ax32.plot(edges, hist, linestyle="--", color=color, linewidth=3)

                    pintegral, pintegral_error = integrate.quad(
                        generalised_gauss.get_fitted_times_expx,
                        -1e2,
                        np.amax(p) * 100,
                        points=[np.amin(p), np.amax(p), np.mean(p)],
                    )
                    # mean_func = lambda x: generalised_gauss.get_fitted_times_expx(x) * x / integral
                    # mean, mean_error = integrate.quad(mean_func, -1e4, np.amax(p) * 100,
                    #                                   points=[np.amin(p), np.amax(p), np.mean(p)])
                    # var_func = lambda x: generalised_gauss.get_fitted_times_expx(x) * (x - mean) ** 2 / integral
                    # var, var_error = integrate.quad(var_func, -1e4, np.amax(p) * 100,
                    #                                 points=[np.amin(p), np.amax(p), np.mean(p)])
                    # meanvar_pi_u_list.append([mean, var])

                    # p: fit weighted kde
                    wpdf = weighted_gaussian_kde(p, weights=np.exp(f), bw_method=bw[0])
                    ax34.plot(
                        edges,
                        wpdf(edges),
                        label=dataset.ss_phi,
                        color=color,
                        linewidth=3,
                    )
                    # the following is incorrect because to unbiase need to multiply the distribution by pi/kappa
                    # hist = np.array([generalised_gauss.get_fitted_times_expx(edge) for edge in edges])/pintegral
                    # ax34.plot(edges, hist, linestyle='--', color=color, linewidth=3)
                    # u_set = wpdf.resample(size=1e4)
                    # meanvar_pi_u_list.append([np.mean(u_set), np.var(u_set)])

        print("bw_list ", bw_list)
        ax3.set_xlabel(r"$F$")
        ax3.set_ylabel(r"$\mathcal{B}(F)$")
        ax3.legend(
            frameon=False,
            loc="best",
            prop={"size": glob_fontsize},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handlelength=0.5,
        )
        fig3.savefig("{0}/plot_{1}.pdf".format(figdir, "f_obs_dist"))
        ax32.set_xlabel(r"$\Lambda$")
        ax32.set_ylabel(r"$\mathcal{B}(\Lambda)$")
        ax32.legend(
            frameon=False,
            loc=2,
            prop={"size": glob_fontsize},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handlelength=0.5,
        )
        fig32.savefig("{0}/plot_{1}.pdf".format(figdir, "pi_obs_dist"))
        ax33.set_xlabel(r"$F$")
        ax33.set_ylabel(r"$\mathcal{DOS}(F)$")
        ax33.legend(
            frameon=False,
            loc=2,
            prop={"size": glob_fontsize},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handlelength=0.5,
        )
        fig33.savefig("{0}/plot_{1}.pdf".format(figdir, "f_dos"))
        ax34.set_xlabel(r"$\Lambda$")
        ax34.set_ylabel(r"$\mathcal{DOS}(\Lambda)$")
        ax34.legend(
            frameon=False,
            loc=2,
            prop={"size": glob_fontsize},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handlelength=0.5,
        )
        fig34.savefig("{0}/plot_{1}.pdf".format(figdir, "pi_dos"))
        print(phi)

        axbox = ax.get_position()
        ax.legend(
            frameon=False,
            loc=(axbox.x0 - 0.18, axbox.x1 - 0.475),
            prop={"size": glob_fontsize},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0,
            labelspacing=0.012,
            handletextpad=-0.45,
        )
        ax.set_ylabel(r"$-\ln p_i$")
        # ax.set_ylabel(r"$F$")
        ax.set_xlabel(r"$\Lambda$")
        ax.locator_params(axis="y", nbins=6)
        ax.set_ylim((205, 241))
        ax.set_xlim((-5, 7.5))
        print("extras", dataset.extras)
        fig0.savefig("{0}/plot_{1}.pdf".format(figdir, "f_logp"))

        ax35.scatter(phi, np.array(logl_list), s=100)
        ax35.set_xlabel(r"$\phi$")
        ax35.set_ylabel(r"$\mathcal{L}(X|\hat{X})$")
        fig35.savefig("{0}/plot_{1}.pdf".format(figdir, "log_likelihood_set"))

        meanz_spline = pickle.load(open("phi_meanZ_spline.pickle", "rb"))
        meanz_data = pickle.load(open("phi_meanZ_data.pickle", "rb"))
        meanz_spline_err = lambda x, xstd: np.sqrt(
            meanz_spline.derivative()(x) ** 2 * xstd**2
        )
        # assume that error in entropy is proportional to standard error of the mean for all of them
        phi_star = 0.823
        Sg = np.array(Sg)
        uSg = unp.uarray(Sg[:, 0], Sg[:, 1])
        yerr = Sg[:, 1]
        iyerr = unp.std_devs(1.0 / uSg)
        print("iyerr: ", iyerr)
        fit_fn, fit_params, fit_err, rho = poly_fit(
            phi, 1.0 / Sg[:, 0], yerr=iyerr, order=2
        )
        print("fit_params", fit_params)
        print("fit_err", fit_err)
        # phi_star, phi_star_err = find_roots(fit_params, fit_err)[0]
        # print "Sg phi* = {} \pm {}".format(phi_star, phi_star_err)
        x = np.linspace(phi_star, np.amax(phi), 1000)
        z = np.linspace(4, np.amax(avgz), 1000)
        color_cycle = get_color_cycle(ncolors=3, reverse=False)
        color = next(color_cycle)
        ax1.errorbar(
            phi,
            Sg[:, 0],
            yerr=yerr,
            color=color,
            markeredgecolor=color,
            label=r"$S_G$",
            fmt="o",
            markersize=15,
        )
        ax1.plot(
            x, 1.0 / fit_fn(x), marker="", linewidth=3, linestyle="--", color=color
        )
        a1 = ufloat(fit_params[0], fit_err[0])
        b1 = ufloat(fit_params[1], fit_err[1])
        c1 = ufloat(fit_params[2], fit_err[2])
        fit_fn, fit_params, fit_err, rho = poly_fit(
            avgz, 1.0 / Sg[:, 0], yerr=iyerr, order=2
        )
        ax11.errorbar(
            avgz,
            Sg[:, 0],
            yerr=yerr,
            color=color,
            markeredgecolor=color,
            label=r"$S_G$",
            fmt="o",
            markersize=15,
        )
        ax11.plot(
            z, 1.0 / fit_fn(z), marker="", linewidth=3, linestyle="--", color=color
        )
        za1 = ufloat(fit_params[0], fit_err[0])
        zb1 = ufloat(fit_params[1], fit_err[1])
        zc1 = ufloat(fit_params[2], fit_err[2])

        Sb_gauss = np.array(Sb_gauss)
        uSb = unp.uarray(Sb_gauss[:, 0], Sb_gauss[:, 1])
        yerr = Sb_gauss[:, 1]
        iyerr = unp.std_devs(1.0 / uSb)
        print("iyerr: ", iyerr)
        fit_fn, fit_params, fit_err, rho = poly_fit(
            phi, 1.0 / Sb_gauss[:, 0], yerr=iyerr, order=2
        )
        # phi_star, phi_star_err = find_roots(fit_params, fit_err)[0]
        # print "Sb-gauss phi* = {} \pm {}".format(phi_star, phi_star_err)
        x = np.linspace(phi_star, np.amax(phi), 1000)
        z = np.linspace(4, np.amax(avgz), 1000)
        color = next(color_cycle)
        ax1.errorbar(
            phi,
            Sb_gauss[:, 0],
            yerr=yerr,
            color=color,
            markeredgecolor=color,
            label=r"$S_B^{(Gauss)}$",
            fmt="o",
            markersize=15,
        )
        ax1.plot(
            x, 1.0 / fit_fn(x), marker="", linewidth=3, linestyle="--", color=color
        )
        a2 = ufloat(fit_params[0], fit_err[0])
        b2 = ufloat(fit_params[1], fit_err[1])
        c2 = ufloat(fit_params[2], fit_err[2])
        # find intersection
        a, b, c = a1 - a2, b1 - b2, c1 - c2
        r = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)
        print("phi* intersection gauss: {} \pm {}".format(r.nominal_value, r.std_dev))
        print(
            "z(\phi*) intersection gauss: {} \pm {}".format(
                meanz_spline(r.nominal_value),
                meanz_spline_err(r.nominal_value, r.std_dev),
            )
        )

        fit_fn, fit_params, fit_err, rho = poly_fit(
            avgz, 1.0 / Sb_gauss[:, 0], yerr=iyerr, order=2
        )
        ax11.errorbar(
            avgz,
            Sb_gauss[:, 0],
            yerr=yerr,
            color=color,
            markeredgecolor=color,
            label=r"$S_B^{(Gauss)}$",
            fmt="o",
            markersize=15,
        )
        ax11.plot(
            z, 1.0 / fit_fn(z), marker="", linewidth=3, linestyle="--", color=color
        )
        a2 = ufloat(fit_params[0], fit_err[0])
        b2 = ufloat(fit_params[1], fit_err[1])
        c2 = ufloat(fit_params[2], fit_err[2])
        # find intersection
        a, b, c = za1 - a2, zb1 - b2, zc1 - c2
        r = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)
        print("z* intersection gauss: {} \pm {}".format(r.nominal_value, r.std_dev))

        Sb_kde = np.array(Sb_kde)
        uSb = unp.uarray(Sb_kde[:, 0], Sb_kde[:, 1])
        yerr = Sb_kde[:, 1]
        iyerr = unp.std_devs(1.0 / uSb)
        fit_fn, fit_params, fit_err, rho = poly_fit(
            phi, 1.0 / Sb_kde[:, 0], yerr=iyerr, order=2
        )
        # phi_star, phi_star_err = find_roots(fit_params, fit_err)[0]
        # print "Sb-kde phi* = {} \pm {}".format(phi_star, phi_star_err)
        x = np.linspace(phi_star, np.amax(phi), 1000)
        z = np.linspace(4, np.amax(avgz), 1000)
        color = next(color_cycle)
        ax1.errorbar(
            phi,
            Sb_kde[:, 0],
            yerr=yerr,
            color=color,
            markeredgecolor=color,
            label=r"$S_B^{(KDE)}$",
            fmt="o",
            markersize=15,
        )
        ax1.plot(
            x, 1.0 / fit_fn(x), marker="", linewidth=3, linestyle="--", color=color
        )
        a2 = ufloat(fit_params[0], fit_err[0])
        b2 = ufloat(fit_params[1], fit_err[1])
        c2 = ufloat(fit_params[2], fit_err[2])
        # find intersection
        a, b, c = a1 - a2, b1 - b2, c1 - c2
        r = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)
        print("phi* intersection kde: {} \pm {}".format(r.nominal_value, r.std_dev))
        print(
            "z(\phi*) intersection kde: {} \pm {}".format(
                meanz_spline(r.nominal_value),
                meanz_spline_err(r.nominal_value, r.std_dev),
            )
        )

        fit_fn, fit_params, fit_err, rho = poly_fit(
            avgz, 1.0 / Sb_kde[:, 0], yerr=iyerr, order=2
        )
        ax11.errorbar(
            avgz,
            Sb_kde[:, 0],
            yerr=yerr,
            color=color,
            markeredgecolor=color,
            label=r"$S_B^{(KDE)}$",
            fmt="o",
            markersize=15,
        )
        ax11.plot(
            z, 1.0 / fit_fn(z), marker="", linewidth=3, linestyle="--", color=color
        )
        a2 = ufloat(fit_params[0], fit_err[0])
        b2 = ufloat(fit_params[1], fit_err[1])
        c2 = ufloat(fit_params[2], fit_err[2])
        # find intersection
        a, b, c = za1 - a2, zb1 - b2, zc1 - c2
        r = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)
        print("z* intersection kde: {} \pm {}".format(r.nominal_value, r.std_dev))

        # fit = np.polyfit(phi[2:], S[2:], 1)
        # fit_fn = np.poly1d(fit)
        # ax1.plot(np.linspace(phi[0],1,20), fit_fn(np.linspace(phi[0],1,20)), color='k')
        # ax1.plot([0.825,1],[0,0],lw=1,color='black')
        ax1.set_xlim((phi_star, 0.865))
        ax1.set_ylim((12.5, 28.5))
        ax1.locator_params(axis="x", nbins=8)
        ax1.set_ylabel(r"$S$")
        ax1.set_xlabel(r"$\phi$")
        ax1.legend(
            frameon=False,
            loc="best",
            prop={"size": glob_fontsize},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        fig1.savefig("{0}/plot_{1}.pdf".format(figdir, "s_phi"))

        # entropy vs coordination number

        ax11.set_ylim((12.5, 28.5))
        ax11.set_xlim((4, 4.4))
        ax11.locator_params(axis="x", nbins=8)
        ax11.set_ylabel(r"$S$")
        ax11.set_xlabel(r"$\overline{z}$")
        ax11.legend(
            frameon=False,
            loc="best",
            prop={"size": glob_fontsize},
            numpoints=1,
            scatterpoints=1,
            markerscale=1,
            columnspacing=0.25,
            labelspacing=0.25,
            handletextpad=0,
        )
        fig11.savefig("{0}/plot_{1}.pdf".format(figdir, "s_z"))

        print("avgz", avgz)
        print("phi", phi)
        print("Sg", Sg[:, 0])
        print("Sb_gauss", Sb_gauss[:, 0])
        print("Sb_kde", Sb_kde[:, 0])

        # subplots
        # subplots
        if True:
            # plot power law exponent
            color_cycle = get_color_cycle(ncolors=3, reverse=False)
            color_marker = next(color_cycle)
            next(color_cycle)
            color_fit = next(color_cycle)
            # ax3 = fig.add_subplot(gs[4:, 0])
            fig01 = plt.figure()
            ax3 = fig01.add_subplot(111)
            x, y, yerr, y2, y2err, rho = [], [], [], [], [], []
            for i, dataset in enumerate(
                sorted(packing_datasets, key=lambda data: data.ss_phi)
            ):
                if (
                    len(dataset.free_energies) > 0
                    and phi_min < dataset.ss_phi < phi_max
                ):
                    if (
                        0.86 < dataset.ss_phi < phi_max
                        and "fire" not in dataset.set_name
                    ) or (
                        phi_min < dataset.ss_phi < 0.865 and "fire" in dataset.set_name
                    ):
                        y.append(dataset.extras[0][0])
                        yerr.append(dataset.extras[1][0])
                        y2.append(dataset.extras[0][1])
                        y2err.append(dataset.extras[1][1])
                        rho.append(dataset.extras[2])
                        x.append(dataset.ss_phi)

            x, y, yerr, y2, y2err = (
                np.array(x),
                np.array(y),
                np.array(yerr),
                np.array(y2),
                np.array(y2err),
            )
            rho = np.array(rho)

            # ax2.plot(x, rho, marker='o', linestyle='', ms=12, color=color_marker)
            fig03 = plt.figure()
            ax31 = fig03.add_subplot(111)

            y /= nparticles
            yerr /= nparticles
            y2 /= nparticles
            y2err /= nparticles
            # ax3.scatter(meanz_data[0], y[-1]* (meanz_data[1]/meanz_spline(x[-1])), s=25)
            ax3.errorbar(
                x,
                y,
                yerr,
                marker="o",
                linestyle="",
                ms=15,
                color=color_marker,
                markeredgecolor=color_marker,
            )
            fit_fn, fit_params, fit_err, rho = poly_fit(x, y, yerr=yerr)
            ax3.plot(x, fit_fn(x), color=color_fit)
            glob_phi_j, glob_phi_j_std = find_roots(fit_params, fit_err)[0]
            print("1/k(phi) = {} phi + {}".format(fit_params[0], fit_params[1]))
            print(
                "1/k: phi*: {} \pm {}, beta: {}".format(
                    glob_phi_j, glob_phi_j_std, fit_params[1]
                )
            )
            print(
                "1/k: z(\phi*): {} \pm {}".format(
                    meanz_spline(glob_phi_j),
                    meanz_spline_err(glob_phi_j, glob_phi_j_std),
                )
            )
            ax3.set_xlabel(r"$\phi$", size=glob_fontsize)
            ax3.set_ylabel(r"$\lambda$", size=glob_fontsize)
            ax3.locator_params(axis="x", nbins=4)
            ax3.locator_params(axis="y", nbins=3)
            ax3.tick_params(axis="both", which="major", labelsize=glob_fontsize)
            ax3.set_xlim((phi_min, phi_max))
            ax3.set_ylim((0, 0.11))
            ax3.ticklabel_format(axis="y", style="sci")

            ax31.errorbar(
                avgz,
                y,
                yerr,
                marker="o",
                linestyle="",
                ms=15,
                color=color_marker,
                markeredgecolor=color_marker,
            )
            fit_fn, fit_params, fit_err, rho = poly_fit(avgz, y, yerr=yerr)
            ax31.plot(avgz, fit_fn(avgz), color=color_fit)
            glob_z_j, glob_z_j_std = find_roots(fit_params, fit_err)[0]
            print("1/k(z) = {} z + {}".format(fit_params[0], fit_params[1]))
            print(
                "1/k: z*: {} \pm {}, beta: {}".format(
                    glob_z_j, glob_z_j_std, fit_params[1]
                )
            )
            ax31.set_xlabel(r"$\overline{z}$", size=glob_fontsize)
            ax31.set_ylabel(r"$\lambda$", size=glob_fontsize)
            ax31.locator_params(axis="x", nbins=4)
            ax31.locator_params(axis="y", nbins=3)
            ax31.tick_params(axis="both", which="major", labelsize=glob_fontsize)
            ax31.set_xlim((4, 4.4))
            ax31.set_ylim((0, 0.11))
            ax31.ticklabel_format(axis="y", style="sci")

            # ax4 = fig.add_subplot(gs[4:, 1])
            fig02 = plt.figure()
            ax4 = fig02.add_subplot(111)
            fig04 = plt.figure()
            ax41 = fig04.add_subplot(111)
            ax4.errorbar(
                x,
                y2,
                y2err,
                marker="o",
                linestyle="",
                ms=15,
                color=color_marker,
                markeredgecolor=color_marker,
            )
            fit_fn, fit_params, fit_err, rho = poly_fit(x, y2, yerr=yerr)
            ax4.plot(x, fit_fn(x), color=color_fit)
            print("c(phi) = {} phi + {}".format(fit_params[0], fit_params[1]))
            a, b = ufloat(fit_params[0], fit_err[0]), ufloat(fit_params[1], fit_err[1])
            r1 = (2.0 - b) / a
            phi_c1, phi_c1err = r1.nominal_value, r1.std_dev
            print(
                "c: phi_c1: {} \pm {}, beta: {}".format(
                    phi_c1, phi_c1err, fit_params[1]
                )
            )
            ax4.set_xlabel(r"$\phi$", size=glob_fontsize)
            ax4.set_ylabel(r"$c$", size=glob_fontsize)
            ax4.locator_params(axis="x", nbins=4)
            ax4.locator_params(axis="y", nbins=3)
            ax4.tick_params(axis="both", which="major", labelsize=glob_fontsize)
            ax4.set_xlim((phi_min, phi_max))
            ax4.set_ylim((1.7, 2.08))

            ax41.errorbar(
                avgz,
                y2,
                y2err,
                marker="o",
                linestyle="",
                ms=15,
                color=color_marker,
                markeredgecolor=color_marker,
            )
            fit_fn, fit_params, fit_err, rho = poly_fit(avgz, y2, yerr=yerr)
            ax41.plot(avgz, fit_fn(avgz), color=color_fit)
            print("c(z) = {} z + {}".format(fit_params[0], fit_params[1]))
            a, b = ufloat(fit_params[0], fit_err[0]), ufloat(fit_params[1], fit_err[1])
            r1 = (2.0 - b) / a
            z_c1, z_c1err = r1.nominal_value, r1.std_dev
            print("c: z_c1: {} \pm {}, beta: {}".format(z_c1, z_c1err, fit_params[1]))
            ax41.set_xlabel(r"$\overline{z}$", size=glob_fontsize)
            ax41.set_ylabel(r"$c$", size=glob_fontsize)
            ax41.locator_params(axis="x", nbins=4)
            ax41.locator_params(axis="y", nbins=3)
            ax41.tick_params(axis="both", which="major", labelsize=glob_fontsize)
            ax41.set_xlim((4, 4.4))
            ax41.set_ylim((1.7, 2.08))

            fig0.savefig("{0}/plot_{1}.pdf".format(figdir, "f_logp"))
            fig01.savefig("{0}/plot_{1}.pdf".format(figdir, "f_logp_kappa"))
            fig02.savefig("{0}/plot_{1}.pdf".format(figdir, "f_logp_c"))
            fig03.savefig("{0}/plot_{1}.pdf".format(figdir, "f_logp_kappa_z"))
            fig04.savefig("{0}/plot_{1}.pdf".format(figdir, "f_logp_c_z"))

            ax36.scatter(meanz_data[0], meanz_data[1], s=25, color=color_fit)
            xx = np.linspace(0.81, 0.865, 1000)
            y = meanz_data[1]
            popt, pcov = curve_fit(
                sigmoid,
                meanz_data[0],
                y,
                p0=[0.845, 1, np.amax(y), np.amin(y), 1],
                maxfev=3000,
            )
            yspl = np.vectorize(sigmoid)(xx, *popt)
            yder = np.vectorize(sigmoid_d1)(xx, *popt)
            ax36.plot(xx, yspl, color=color_fit, marker="", linestyle="-")
            # ax36inset.plot(xx, yder, color=color_fit, marker='', linestyle='-')
            ax36.scatter(phi, avgz, s=100, marker="^", color=color_marker, zorder=10)
            ax36.set_xlabel(r"$\phi$")
            ax36.set_ylabel(r"$\langle z \rangle_{\mathcal{B}}$")
            ax36.set_xlim((0.815, 0.865))
            ax36.set_ylim((3.98, 4.44))
            # ax36inset.autoscale(enable=True, axis='x', tight=True)
            # ax36inset.set_xlim((0.815, 0.865))
            # ax36inset.locator_params(axis='x', nbins=5)
            # ax36inset.yaxis.set_ticks([])
            # ax36.yaxis.set_major_formatter(FixedOrderFormatter(3))
            # ax36inset.set_xlabel(r'$\phi$')
            # ax36inset.set_ylabel(r'$\partial_\phi \langle z \rangle_{\mathcal{B}}$')
            # ax36inset.yaxis.set_label_position("right")
            # ax36inset.xaxis.set_label_position("top")

            fig36.savefig("{0}/plot_{1}.pdf".format(figdir, "z_phi"))

            color_cycle = get_color_cycle(ncolors=3, reverse=False)
            next(color_cycle)
            next(color_cycle)
            colorr = next(color_cycle)
            colorl = colorr

            x = np.linspace(np.amin(phi), np.amax(phi), 1000)
            y3 = np.array(meanvar_f_list)[:, 0] / dataset.nparticles
            fit_fn, fit_params, fit_err, rho = poly_fit(
                phi, y3, yerr=1.0 / np.sqrt(histograms_nsamples), order=2
            )
            ax2.plot(
                phi,
                y3,
                marker="o",
                markersize=10,
                linestyle="",
                color=colorl,
                markeredgecolor=colorl,
            )
            ax2.plot(x, fit_fn(x), marker="", linewidth=3, linestyle="--", color=colorl)
            fit_params = np.array(fit_params)
            print(
                "mu_f = {} phi^2 + {} phi + {} ".format(
                    fit_params[0], fit_params[1], fit_params[2]
                )
            )
            y3 = np.array(meanvar_pi_list)[:, 0]
            fit_fn, fit_params, fit_err, rho = poly_fit(
                phi, y3, yerr=1.0 / np.sqrt(histograms_nsamples), order=1
            )
            ax22.plot(
                phi,
                y3,
                marker="o",
                markersize=10,
                linestyle="",
                color=colorr,
                markeredgecolor=colorr,
            )
            ax22.plot(
                x, fit_fn(x), marker="", linewidth=3, linestyle="--", color=colorr
            )
            print("mu_pi = {} phi + {} ".format(fit_params[0], fit_params[1]))

            ax2.set_ylabel(r"$\mu_f$")
            # for tl in ax2.get_yticklabels():
            #     tl.set_color(colorl)
            ax22.set_ylabel(r"$\mu_\Lambda$")
            # for tl in ax22.get_yticklabels():
            #     tl.set_color(colorr)
            ax2.set_xlabel(r"$\phi$")
            ax22.set_xlabel(r"$\phi$")
            ax2.set_xlim(xmax=phi_max)
            ax22.set_xlim(xmax=phi_max)
            ax2.locator_params(axis="x", nbins=7)
            ax22.locator_params(axis="x", nbins=7)
            fig2.savefig("{0}/plot_{1}.pdf".format(figdir, "muf"))
            fig22.savefig("{0}/plot_{1}.pdf".format(figdir, "mupi"))

            fig23 = plt.figure()
            ax23 = fig23.add_subplot(111)
            fig24 = plt.figure()
            ax24 = fig24.add_subplot(111)
            # ax24 = ax23.twinx()

            y3 = np.array(meanvar_f_list)[:, 1] / dataset.nparticles**2
            fit_fn, fit_params, fit_err, rho = poly_fit(
                phi, y3, yerr=1.0 / np.sqrt(histograms_nsamples), order=2
            )
            phi_star, phi_star_err = find_roots(fit_params, fit_err)[0]
            print("sig_f phi* = {} \pm {}".format(phi_star, phi_star_err))
            x = np.linspace(phi_star, np.amax(phi), 1000)
            ax24.plot(
                phi,
                y3,
                marker="o",
                markersize=10,
                linestyle="",
                color=colorr,
                markeredgecolor=colorr,
            )
            ax24.plot(
                x, fit_fn(x), marker="", linewidth=3, linestyle="--", color=colorr
            )
            print(
                "sigma_f = {} phi^2 + {} phi + {} ".format(
                    fit_params[0], fit_params[1], fit_params[2]
                )
            )

            y3 = np.array(meanvar_pi_list)[:, 1]
            fit_fn, fit_params, fit_err, rho = poly_fit(
                phi, y3, yerr=1.0 / np.sqrt(histograms_nsamples), order=2
            )
            ax23.plot(
                phi,
                y3,
                marker="o",
                markersize=10,
                linestyle="",
                color=colorl,
                markeredgecolor=colorl,
            )
            ax23.plot(
                x, fit_fn(x), marker="", linewidth=3, linestyle="--", color=colorl
            )
            fit_params = np.array(fit_params)
            print(
                "var_pi = {} phi^2 + {} phi + {} ".format(
                    fit_params[0], fit_params[1], fit_params[2]
                )
            )

            ax23.set_ylabel(r"$\sigma^2_\Lambda$")
            # for tl in ax23.get_yticklabels():
            #     tl.set_color(colorl)
            ax24.set_ylabel(r"$\sigma^2_f$")
            # for tl in ax24.get_yticklabels():
            #     tl.set_color(colorr)
            ax23.set_xlabel(r"$\phi$")
            ax24.set_xlabel(r"$\phi$")
            ax23.set_xlim(xmax=phi_max)
            ax24.set_xlim(xmax=phi_max)
            ax24.set_ylim(ymin=0)
            fig23.savefig("{0}/plot_{1}.pdf".format(figdir, "varpi"))
            fig24.savefig("{0}/plot_{1}.pdf".format(figdir, "varf"))

        # fig25 = plt.figure()
        # ax25 = fig25.add_subplot(111)
        # ax26 = ax25.twinx()
        fig26 = plt.figure()
        ax26 = fig26.add_subplot(111)
        fig27 = plt.figure()
        ax27 = fig27.add_subplot(111)

        y3 = np.array(cov_f_pi_list) / dataset.nparticles
        fit_fn, fit_params, fit_err, rho = poly_fit(
            phi, y3, yerr=1.0 / np.sqrt(histograms_nsamples), order=2
        )
        phi_star, phi_star_err = find_roots(fit_params, fit_err)[0]
        print("cov_fpi phi* = {} \pm {}".format(phi_star, phi_star_err))
        x = np.linspace(phi_star, np.amax(phi), 1000)
        ax26.plot(
            phi,
            y3,
            marker="o",
            markersize=10,
            linestyle="",
            color=colorr,
            markeredgecolor=colorr,
        )
        ax26.plot(x, fit_fn(x), marker="", linewidth=3, linestyle="--", color=colorr)
        print(
            "cov_fpi = {} phi^2 + {} phi + {} ".format(
                fit_params[0], fit_params[1], fit_params[2]
            )
        )

        y3 = np.array(cov_f_pi_list) / np.sqrt(
            np.asarray(meanvar_f_list)[:, 1] * np.asarray(meanvar_pi_list)[:, 1]
        )
        fit_fn, fit_params, fit_err, rho = poly_fit(
            phi, y3, yerr=1.0 / np.sqrt(histograms_nsamples), order=2
        )
        phi_star, phi_star_err = find_roots(fit_params, fit_err)[0]
        print("rho phi* = {} \pm {}".format(phi_star, phi_star_err))
        x = np.linspace(phi_star, np.amax(phi), 1000)
        ax27.plot(
            phi,
            y3,
            marker="o",
            markersize=10,
            linestyle="",
            color=colorr,
            markeredgecolor=colorr,
        )
        ax27.plot(x, fit_fn(x), marker="", linewidth=3, linestyle="--", color=colorr)
        print(
            "rho = {} phi^2 + {} phi + {} ".format(
                fit_params[0], fit_params[1], fit_params[2]
            )
        )

        # y3 = np.array(meanvar_pi_list)[:, 1]
        # fit_fn, fit_params, fit_err, rho = poly_fit(phi, y3, yerr=1./np.sqrt(histograms_nsamples), order=2)
        # ax25.plot(phi, y3, marker='o', markersize=10, linestyle='', color=colorl, markeredgecolor=colorl)
        # ax25.plot(x, fit_fn(x), marker='', linewidth=3, linestyle='--', color=colorl)
        # fit_params = np.array(fit_params)
        # print "var_pi = {} phi^2 + {} phi + {} ".format(fit_params[0], fit_params[1], fit_params[2])
        # unbiased variance
        # y3 = np.array(meanvar_pi_u_list)[:, 1] / dataset.nparticles
        # fit_fn, fit_params, fit_err, rho = poly_fit(phi, y3, yerr=1./np.sqrt(histograms_nsamples), order=2)
        # ax25.plot(phi, y3, marker='^', markersize=10, linestyle='', color='b')
        # ax25.plot(x, fit_fn(x), marker='', linewidth=3, linestyle='--', color='b')
        # fit_params = np.array(fit_params)
        # print "var_pi_u = {} phi^2 + {} phi + {} ".format(fit_params[0], fit_params[1], fit_params[2])

        # ax25.set_ylabel(r"$\sigma^2_\Lambda$", color=colorl)
        # for tl in ax25.get_yticklabels():
        #     tl.set_color(colorl)
        # for tl in ax26.get_yticklabels():
        #     tl.set_color(colorr)
        # ax25.set_xlabel(r"$\phi$")
        # ax25.set_xlim(xmax=phi_max)
        ax26.set_ylabel(r"$\sigma^2_{f \Lambda}$")
        ax26.set_xlabel(r"$\phi$")
        ax26.set_xlim(xmax=phi_max)
        ax26.set_ylim(ymin=0)
        fig26.savefig("{0}/plot_{1}.pdf".format(figdir, "covfpi"))
        ax27.set_ylabel(r"$\rho_{f\Lambda}$")
        ax27.set_xlabel(r"$\phi$")
        ax27.set_xlim(xmax=phi_max)
        ax27.set_ylim(ymin=0)
        fig27.savefig("{0}/plot_{1}.pdf".format(figdir, "rho"))

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
            fig7 = plt.figure()
            ax7 = fig7.add_subplot(111)
            fit_params = []
            fit_params_std = []
            ss_phi_list = []
            unpad_log_omega_p_hist = []
            kde_list = []
            iNK_list = []
            log_omega_list = []
            maxp, max_logdos = [], []  # pressure for which kde is max
            x_integrate_list = []
            if len(dataset.free_energies) > 0 and phi_min < dataset.ss_phi < phi_max:
                if (
                    0.86 < dataset.ss_phi < phi_max and "fire" not in dataset.set_name
                ) or (phi_min < dataset.ss_phi < 0.865 and "fire" in dataset.set_name):
                    j = 0
                    Facc = None
                    vcavity = None
                    while Facc is None or vcavity is None:
                        Facc = dataset.packing_data[j].Facc
                        vcavity = dataset.packing_data[j].vcavity
                        j += 1
                    nparticles = dataset.nparticles
                    vcavity = dataset.packing_data[0].vcavity
                    if (
                        0.855 < dataset.ss_phi < phi_max
                        and "fire" not in dataset.set_name
                    ) or (
                        phi_min < dataset.ss_phi < 0.86 and "fire" in dataset.set_name
                    ):
                        print("n:", nparticles)
                        # fit slopes to find inverse kappa and the intercept
                        x_raw = dataset.pressures
                        f_outliers = OutlierDetection(
                            dataset.free_energies,
                            p=0.5,
                            D=2 * np.std(dataset.free_energies),
                        )
                        x_outliers = OutlierDetection(
                            np.log(x_raw), p=0.5, D=2 * np.std(np.log(x_raw))
                        )
                        non_outliers_indexes = list(
                            set(f_outliers.non_outliers_indexes).intersection(
                                x_outliers.non_outliers_indexes
                            )
                        )
                        x = np.array(x_raw)[np.array(non_outliers_indexes, dtype="i")]
                        f = np.array(dataset.free_energies)[
                            np.array(non_outliers_indexes, dtype="i")
                        ]
                        weights = (
                            1.0
                            / np.array(dataset.free_energies_err)[
                                np.array(non_outliers_indexes, dtype="i")
                            ]
                        )
                        x = np.log(x)
                        fit, cov = np.polyfit(x, f, 1, w=weights, cov=True)
                        fit_err = np.sqrt(np.diag(cov))

                        iNK, NC = fit  # N/K, C(N)
                        iNKstd, NCstd = fit_err
                        iNK_list.append(iNK)
                        K = nparticles / iNK
                        C = NC / nparticles
                        fit_params.append([K, C])
                        fit_params_std.append(fit_err)
                        ss_phi_list.append(dataset.ss_phi)

                        # free energy non parametric log omega
                        bw = get_bandwidth_estimate(
                            np.array(f), kernel="gaussian", method="cross_validation"
                        )
                        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(
                            f[:, np.newaxis]
                        )
                        # plot histograms
                        n_integrate = 2**14 + 1
                        x_integrate = np.linspace(Facc, np.amax(f) * 100, n_integrate)
                        log_pdf = kde.score_samples(x_integrate[:, np.newaxis])
                        integrand = np.exp(np.add(log_pdf, x_integrate))
                        integral = integrate.romb(
                            integrand, dx=x_integrate[1] - x_integrate[0]
                        )
                        log_omega = -Facc + np.log(integral)
                        log_omega_list.append(log_omega)

                        # kde histogram
                        bw = get_bandwidth_estimate(
                            np.array(x), kernel="gaussian", method="cross_validation"
                        )
                        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(
                            x[:, np.newaxis]
                        )
                        kde_list.append(kde)
                        # plot histograms
                        n_integrate = 2**14 + 1
                        x_integrate = np.linspace(
                            np.amin(x) * 0.5,
                            min(np.amax(x) * 15, pmax * 5),
                            n_integrate,
                        )
                        x_integrate_list.append(x_integrate)
                        # x_integrate = np.linspace(0, 1000, n_integrate)
                        log_pdf = kde.score_samples(x_integrate[:, np.newaxis])
                        log_hist = log_pdf + iNK * np.log(x_integrate)
                        norm = np.log(
                            integrate.romb(
                                np.exp(log_hist), dx=x_integrate[1] - x_integrate[0]
                            )
                        )
                        unpad_log_omega_p_hist.append(log_hist - norm + log_omega)
                        ax2.plot(
                            x_integrate,
                            log_hist - norm + log_omega,
                            label=dataset.ss_phi,
                            color=next(color_cycle),
                            linewidth=3,
                        )
                        maxp.append(x_integrate[np.argmax(log_hist - norm + log_omega)])
                        max_logdos.append(np.amax(log_hist - norm + log_omega))
                        # color_cycle = get_color_cycle()

            xmin, xmax = np.amin(x_integrate_list), np.amax(x_integrate_list)
            print(xmin, xmax)
            pad_log_omega_p_hist = []
            for x_, x_integrate_ in zip(unpad_log_omega_p_hist, x_integrate_list):
                dx = x_integrate_[1] - x_integrate_[0]
                for j, xx_ in enumerate(x_):
                    x_[j] = 0 if xx_ < 0 else xx_
                pad_log_omega_p_hist.append(
                    np.pad(
                        x_,
                        (
                            int(np.floor((np.amin(x_integrate_) - xmin) / dx)),
                            int(np.floor((xmax - np.amax(x_integrate_)) / dx)),
                        ),
                        "constant",
                        constant_values=(0, 0),
                    )
                )

            maxsize = len(max(pad_log_omega_p_hist, key=len)) // 10
            from skimage.transform import resize

            x_integrate = np.linspace(xmin, xmax, maxsize)
            for i, x_ in enumerate(pad_log_omega_p_hist):
                a = np.array(x_).reshape((1, len(x_)))
                pad_log_omega_p_hist[i] = resize(a, (1, maxsize)).flatten()
                ax2.plot(
                    x_integrate,
                    pad_log_omega_p_hist[i],
                    label=dataset.ss_phi,
                    color=next(color_cycle),
                    linewidth=1,
                )
            ax2.set_xscale("log")

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

            color = next(color_cycle)
            x = np.array(phi)
            y = np.array(fit_params)[:, 0]
            y = np.exp(np.array(1.0 / y))
            ax5.scatter(x, y, color=color, label=r"$\exp(\kappa)$", s=100, alpha=0.5)
            weights = 1.0 / np.array(fit_params_std)[:, 0]
            fit, cov = np.polyfit(
                x, y, 1, w=weights, cov=True
            )  # if I use dphi instead of phi it's not a power law
            fit_fn = np.poly1d(fit)
            ax5.plot(x, fit_fn(x), color=color, linewidth=3)
            # ax5.axhline(0, color='k')
            print("phi max(p) fit: ", fit)

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

            ax5.legend(
                frameon=False,
                loc=2,
                prop={"size": glob_fontsize},
                numpoints=1,
                scatterpoints=1,
                markerscale=1,
                columnspacing=0.25,
                labelspacing=0.25,
                handletextpad=0,
            )

            x, y = dphi, max_logdos
            color = next(color_cycle)
            ax6.scatter(x, y, color=color, s=100)
            fit, cov = np.polyfit(x, y, 1, cov=True)
            fit_fn = np.poly1d(fit)
            ax6.plot(x, fit_fn(x), color=color, linewidth=3)
            ax6.set_xlabel(r"$\log DOS_{max}$")
            ax6.set_xlabel(r"$\Delta\phi$")
            print("phi max dos fit: ", fit)

            alpha, beta = 200, 100
            color_cycle = get_color_cycle()
            for i, (kde, iNK) in enumerate(zip(kde_list, iNK_list)):
                x_integrate = np.linspace(
                    p_minmax_list[i][0] * 0.5,
                    min(p_minmax_list[i][1] * 15, pmax * 5),
                    2**14 + 1,
                )
                # x_integrate = np.linspace(p_minmax_list[i][0], p_minmax_list[i][1], 2**14+1)
                log_pdf = kde.score_samples(x_integrate[:, np.newaxis])
                log_hist = log_pdf + iNK * np.log(x_integrate)
                log_norm = np.log(
                    integrate.romb(np.exp(log_hist), dx=x_integrate[1] - x_integrate[0])
                )
                logg = (log_hist - log_norm + log_omega_list[i]) + alpha * dphi[i]
                # print np.amax(logg), alpha * dphi[i]
                ax7.plot(
                    np.log(x_integrate) - beta * dphi[i],
                    logg,
                    label=phi[i],
                    color=next(color_cycle),
                    linewidth=3,
                )
            ax7.legend(
                frameon=False,
                loc=2,
                prop={"size": glob_fontsize},
                numpoints=1,
                scatterpoints=1,
                markerscale=1,
                columnspacing=0.25,
                labelspacing=0.25,
                handletextpad=0,
            )
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
                X, Y, = np.array(
                    x_integrate
                ), np.array(ss_phi_list)
                Z = np.array(pad_log_omega_p_hist)
                ax3.imshow(
                    Z,
                    vmin=np.abs(Z).min(),
                    vmax=np.abs(Z).max(),
                    origin="lower",
                    extent=[X.min(), X.max(), Y.min(), Y.max()],
                    interpolation="gaussian",
                    aspect="auto",
                )

                from matplotlib import cm

                norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
                cmap = cm.jet
                levels = np.arange(
                    np.amin(pad_log_omega_p_hist),
                    np.amax(pad_log_omega_p_hist) * 1.01,
                    10,
                )
                ax4.contourf(
                    X,
                    Y,
                    pad_log_omega_p_hist,
                    norm=norm,
                    cmap=cmap,  # cm.get_cmap(cmap, len(levels) - 1),
                    levels=levels,
                )
                ax4.set_xscale("log")

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
            # fit the (N/k)\ln(P)+N model to each and compute the average of the kappas

            fig = plt.figure()
            ax = fig.add_subplot(111)
            avg = 0
            for i, dataset in enumerate(
                sorted(packing_datasets, key=lambda data: data.nparticles)
            ):
                if len(dataset.free_energies) > 0:
                    nparticles = dataset.nparticles
                    print(nparticles)
                    outliers = OutlierDetection(
                        dataset.free_energies,
                        p=0.5,
                        D=3 * np.std(dataset.free_energies),
                    )
                    x = np.array(dataset.pressures)[
                        np.array(outliers.non_outliers_indexes, dtype="i")
                    ]
                    y = np.array(dataset.free_energies)[
                        np.array(outliers.non_outliers_indexes, dtype="i")
                    ]
                    print(len(x), len(y))
                    x = np.log(x)
                    ax.scatter(x, y, color=next(color_cycle))

                    def ff_msf(x, a):
                        return nparticles * (a * x + 1)

                    popt, pcov = curve_fit(ff_msf, x, y)
                    ax.plot(
                        x,
                        ff_msf(x, popt[0]),
                        label="F = N/{:.3f}ln(P) + N".format(1.0 / popt[0]),
                    )
                    avg += popt[0]
            avg /= i + 1
            print("avg kappa = ", 1 / avg)
            ax.legend(
                frameon=False,
                loc="best",
                numpoints=1,
                markerscale=0.5,
                columnspacing=0.25,
                labelspacing=0.25,
            )
            plt.ylabel(r"$F$")
            plt.xlabel(r"$\ln \mathcal{P}$")


if __name__ == "__main__":
    import dill

    pts = BasinAnalysis()
    pts.collect_data_every_set_all(data_name="basin_analysis.pickle")

    plot(pts.packing_datasets)
    plt.show()
    plt.close()

    # print pts.packing_datasets[0].packing_data[0].coords
    # print pts.packing_datasets[0].packing_data[0].packing_frac
    #
    # with open("packing_datasets.pickle", 'wb') as f:
    #     pickle.dump(pts.packing_datasets, f, protocol=-1)
    #
    # with open("packing_datasets.pickle", 'rb') as f:
    #     dsets = pickle.load(f)
    #
    # print dsets[0].packing_data[0].coords
    # print dsets[0].packing_data[0].packing_frac
