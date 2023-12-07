from __future__ import division
from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import input
from builtins import str
from builtins import next
from builtins import zip
from builtins import map
from builtins import range
from past.builtins import basestring
from builtins import object
import numpy as np
import os
from scipy.special import gamma, gammaln
from scipy.spatial import Delaunay, ConvexHull
import subprocess
import platform
import basinvolume
import pele
import mcpele
import sys, traceback
import configparser
import pandas as pd
import glob
import logging
from itertools import chain
from basinvolume.utils._utils_cpp import read_txt
from pele.distance import get_distance, put_in_box, Distance

try:
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError as err:
    print(err)


# given the number of particles
# return the corresponding tolerance for >95 percent accuracy
# for the CVODE integrator for mapping basins of attraction
# for the inversepower potential
INVERSE_POWER_CVODE_95_ACC = {
    2: 1e-7,
    8: 1e-7,
    16: 1e-7,
    32: 1e-7,
    64: 1e-7,
    128: 1e-7,
    256: 1e-8,
    512: 1e-9,
    1024: 1e-10,
}


def get_mxd_t(ndim):
    """Get number for checking convergence for mixed descent."""
    if ndim < 600:
        return 50
    else:
        return 200


class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self


class Result(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if list(self.keys()):
            m = max(list(map(len, list(self.keys())))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(v) for k, v in list(self.items())]
            )
        else:
            return self.__class__.__name__ + "()"


def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


def _sort_pair(x, y):
    """
    sorts x and moves elements of y accordingly
    """
    xc = np.array(x)
    points = list(zip(xc, y))
    sorted_points = sorted(points)
    new_x = np.array([point[0] for point in sorted_points])
    new_y = np.array([point[1] for point in sorted_points])
    return new_x, new_y


def write_csv_xy(x, y, xerr=None, yerr=None, fit=None, fname="data.csv"):
    """
    fit is the y values of the fit to the xy plot
    """
    if xerr is None:
        xerr = np.zeros(len(x))
    if yerr is None:
        yerr = np.zeros(len(x))
    if fit is None:
        fit = np.zeros(len(x))
    data = (
        np.array(x),
        np.array(xerr),
        np.array(y),
        np.array(yerr),
        np.array(fit),
    )
    np.savetxt(fname, np.column_stack(data), delimiter=",")


def read_csv_xy(fname):
    x, xerr, y, yerr, fit = np.loadtxt(fname, delimiter=",", unpack=True)
    return x, xerr, y, yerr, fit


def volume_nball(radius, n):
    volume = np.power(np.pi, n / 2) * np.power(radius, n) / gamma(n / 2 + 1)
    return volume


def surface_nball(radius, n):
    return 2 * np.pi * volume_nball(radius, n - 1)


def log_surface_nball(radius, n):
    log_surface = np.log(2 * np.pi) + log_volume_nball(radius, n - 1)
    return log_surface


def log_volume_nball(radius, n):
    log_volume = n / 2.0 * np.log(np.pi) + n * np.log(radius) - gammaln(n / 2 + 1)
    return log_volume


def log_factorial(x):
    return gammaln(x + 1)


def cround(r):
    if r > 0.0:
        r = np.floor(r + 0.5)
    else:
        r = np.ceil(r - 0.5)
    return r


def trymakedir(path):
    """this function deals with common race conditions"""
    while True:
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                break
            except OSError as e:
                if e.errno != 17:
                    raise
                # time.sleep might help here
                pass
        else:
            break


def view_traceback():
    ex_type, ex, tb = sys.exc_info()
    print("exception type:", ex_type)
    print("exception:", ex)
    print("Traceback:")
    traceback.print_tb(tb)
    del tb


def read_xyd(fname):
    coords = []
    radii = []
    if not os.path.isfile(fname):
        raise IOError("The xyd file '{}' does not exist.".format(fname))
    f = open(fname, "r")
    while True:
        xyd = f.readline()
        if not xyd:
            break
        x, y, d = xyd.split()
        coords.extend([float(x), float(y)])
        radii.extend([float(d)])
    return np.array(coords, dtype="d"), np.array(radii, dtype="d")


def read_xydf(fname):
    coords = []
    radii = []
    frozen = []
    if not os.path.isfile(fname):
        raise IOError("The xydf file '{}' does not exist.".format(fname))
    f = open(fname, "r")
    i = 0
    while True:
        xydf = f.readline()
        if not xydf:
            break
        x, y, d, fr = xydf.split()
        coords.extend([float(x), float(y)])
        radii.extend([float(d)])
        if bool(int(fr)):
            frozen.extend([i])
        i += 1
    return (
        np.array(coords, dtype="d"),
        np.array(radii, dtype="d"),
        np.array(frozen, dtype="int"),
    )


def read_xyzd(fname):
    coords = []
    radii = []
    if not os.path.isfile(fname):
        raise IOError("The xyzd file '{}' does not exist.".format(fname))
    f = open(fname, "r")
    while True:
        xyzd = f.readline()
        if not xyzd:
            break
        x, y, z, d = xyzd.split()
        coords.extend([float(x), float(y), float(z)])
        radii.extend([float(d)])
    return np.array(coords, dtype="d"), np.array(radii, dtype="d")


def read_xyzdf(fname):
    coords = []
    radii = []
    frozen = []
    if not os.path.isfile(fname):
        raise IOError("The xyzdf file '{}' does not exist.".format(fname))
    f = open(fname, "r")
    i = 0
    while True:
        xyzdf = f.readline()
        if not xyzdf:
            break
        x, y, z, d, fr = xyzdf.split()
        coords.extend([float(x), float(y), float(z)])
        radii.extend([float(d)])
        if bool(int(fr)):
            frozen.extend([i])
        i += 1
    return (
        np.array(coords, dtype="d"),
        np.array(radii, dtype="d"),
        np.array(frozen, dtype="int"),
    )


def read_xydr(fname, etol=1.0, bdim=2):
    coords = []
    radii = []
    stable_atoms = []
    if not os.path.isfile(fname):
        raise IOError("The xydr file '{}' does not exist.".format(fname))
    f = open(fname, "r")
    while True:
        xydr = f.readline()
        if not xydr:
            break
        # print 'xydr ',xydr
        x, y, d, r = xydr.split()
        coords.extend([float(x), float(y)])
        radii.extend([float(d)])
        stable = float(float(r) >= etol)
        for _ in range(bdim):
            stable_atoms.extend([stable])
    return (
        np.array(coords, dtype="d"),
        np.array(radii, dtype="d"),
        np.array(stable_atoms, dtype="d"),
    )


def read_xydfr(fname, etol=1.0, bdim=2):
    coords = []
    radii = []
    frozen = []
    stable_atoms = []
    if not os.path.isfile(fname):
        raise IOError("The xydfr file '{}' does not exist.".format(fname))
    f = open(fname, "r")
    i = 0
    while True:
        xydfr = f.readline()
        if not xydfr:
            break
        # print 'xydr ',xydr
        x, y, d, fr, r = xydfr.split()
        coords.extend([float(x), float(y)])
        radii.extend([float(d)])
        if bool(int(fr)):
            frozen.extend([i])
        stable = float(float(r) >= etol)
        for _ in range(bdim):
            stable_atoms.extend([stable])
        i += 1
    return (
        np.array(coords, dtype="d"),
        np.array(radii, dtype="d"),
        np.array(frozen, dtype="int"),
        np.array(stable_atoms, dtype="d"),
    )


def read_xyzdr(fname, etol=1.0, bdim=3):
    coords = []
    radii = []
    stable_atoms = []
    if not os.path.isfile(fname):
        raise IOError("The xyzdr file '{}' does not exist.".format(fname))
    f = open(fname, "r")
    while True:
        xyzdr = f.readline()
        if not xyzdr:
            break
        x, y, z, d, r = xyzdr.split()
        coords.extend([float(x), float(y), float(z)])
        radii.extend([float(d)])
        stable = float(float(r) >= etol)
        for _ in range(bdim):
            stable_atoms.extend([stable])
    return (
        np.array(coords, dtype="d"),
        np.array(radii, dtype="d"),
        np.array(stable_atoms, dtype="d"),
    )


def read_xyzdfr(fname, etol=1.0, bdim=3):
    coords = []
    radii = []
    frozen = []
    stable_atoms = []
    if not os.path.isfile(fname):
        raise IOError("The xyzdfr file '{}' does not exist.".format(fname))
    f = open(fname, "r")
    i = 0
    while True:
        xyzdfr = f.readline()
        if not xyzdfr:
            break
        x, y, z, d, fr, r = xyzdfr.split()
        coords.extend([float(x), float(y), float(z)])
        radii.extend([float(d)])
        if bool(int(fr)):
            frozen.extend([i])
        stable = float(float(r) >= etol)
        for _ in range(bdim):
            stable_atoms.extend([stable])
        i += 1
    return (
        np.array(coords, dtype="d"),
        np.array(radii, dtype="d"),
        np.array(frozen, dtype="int"),
        np.array(stable_atoms, dtype="d"),
    )


def import_packing(fname, jammed, bdim, sca=0.0):
    results = {}
    if jammed:
        if bdim == 2:
            (
                results["coords"],
                hs_diameters,
                results["stable_atoms_float_bdim"],
            ) = read_xydr(fname)
        elif bdim == 3:
            (
                results["coords"],
                hs_diameters,
                results["stable_atoms_float_bdim"],
            ) = read_xyzdr(fname)
        else:
            raise NotImplementedError("bdim={} not implemented".format(bdim))
        results["ss_radii"] = hs_diameters * 0.5 * (1 + sca)
        results["stable_atoms"] = []
        for i in range(0, len(results["stable_atoms_float_bdim"]), bdim):
            results["stable_atoms"].append(results["stable_atoms_float_bdim"][i] == 1.0)
    else:
        if bdim == 2:
            results["coords"], hs_diameters = read_xyd(fname)
        elif bdim == 3:
            results["coords"], hs_diameters = read_xyzd(fname)
        else:
            raise NotImplementedError("bdim={} not implemented".format(bdim))
    results["hs_radii"] = hs_diameters * 0.5
    return results


def read_single_column_coords(fname):
    coords = []
    f = open(fname, "r")
    while True:
        line = f.readline()
        if not line:
            break
        coords.append(float(line))
    f.close()
    return np.array(coords, dtype="d")


def read_multi_column(fname):
    coords = []
    f = open(fname, "r")
    while True:
        line = f.readline()
        if not line:
            break
        coords.append([float(x) for x in line.split()])
    f.close()
    return np.array(coords, dtype="d")


def reduce_coordinates(mylist, indexes, bdim):
    """
    remove coordinates of frozen atoms
    """
    newlist = mylist.copy().tolist()
    for index in sorted(indexes, reverse=True):
        del newlist[index * bdim : index * bdim + bdim]
    return np.array(newlist)


def full_coordinates(reduced_list, old_full_list, indexes, bdim):
    """
    add coordinates of frozen atoms to reduced coordinates
    """
    newlist = reduced_list.copy()
    for index in sorted(indexes, reverse=False):
        newlist = np.insert(
            newlist,
            index * bdim,
            old_full_list[index * bdim : index * bdim + bdim],
        )
    return np.array(newlist)


def plot_disks(coords, radii, boxv, colors=None, sca=0):
    import matplotlib
    from matplotlib.patches import Circle
    import pylab

    def myscatter(ax, colormap, x, y, radii, colors):
        for x1, y1, r, c in zip(x, y, radii, colormap(colors)):
            ax.add_patch(Circle((x1, y1), r, fc=c))

    coords = put_in_box(coords, 2, Distance.PERIODIC, boxv)
    coords = np.reshape(coords, (len(radii), 2))
    fig = pylab.figure()
    ax = fig.add_subplot(111, aspect="equal")
    myscatter(
        ax,
        matplotlib.cm.jet,
        coords[:, 0],
        coords[:, 1],
        radii * sca,
        np.ones(len(radii)),
    )
    myscatter(
        ax,
        matplotlib.cm.jet,
        coords[:, 0],
        coords[:, 1],
        radii,
        np.ones(len(radii)) * -1,
    )
    ax.axis("equal")
    if boxv is not None:
        ax.axes.set_xlim([-boxv[0] / 2, boxv[0] / 2])
        ax.axes.set_ylim([-boxv[1] / 2, boxv[1] / 2])
        ax.plot(
            [-boxv[0] / 2, -boxv[0] / 2],
            [-boxv[1] / 2, boxv[1] / 2],
            color="k",
            linestyle="-",
            linewidth=2,
        )
        ax.plot(
            [boxv[0] / 2, boxv[0] / 2],
            [-boxv[1] / 2, boxv[1] / 2],
            color="k",
            linestyle="-",
            linewidth=2,
        )
        ax.plot(
            [-boxv[0] / 2, boxv[0] / 2],
            [-boxv[1] / 2, -boxv[1] / 2],
            color="k",
            linestyle="-",
            linewidth=2,
        )
        ax.plot(
            [-boxv[0] / 2, boxv[0] / 2],
            [boxv[1] / 2, boxv[1] / 2],
            color="k",
            linestyle="-",
            linewidth=2,
        )
    pylab.show()


#
# Make the git revision visible.  Most of this is copied from scipy
# In turn, most of this is copied from pele.
#
# Return the git revision as a string
def get_git_version_direct(repository="basinvolume"):
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        repo_path = None
        try:
            if repository == "basinvolume":
                repo_path = os.path.dirname(basinvolume.__file__)[:-12]
            elif repository == "pele":
                repo_path = os.path.dirname(pele.__file__)[:-5]
            elif repository == "mcpele":
                repo_path = os.path.dirname(mcpele.__file__)[:-7]
        except:
            sys.stderr.write("WARNING: could't find path to" + repository + "\n")
            sys.exit()
        repo_path = os.path.abspath(repo_path)
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env, cwd=repo_path
        ).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_git_version_from_build(repository="basinvolume"):
    repo_path = None
    try:
        if repository == "basinvolume":
            repo_path = os.path.dirname(basinvolume.__file__)[:-12]
        elif repository == "pele":
            repo_path = os.path.dirname(pele.__file__)[:-5]
        elif repository == "mcpele":
            repo_path = os.path.dirname(mcpele.__file__)[:-7]
        repo_path = os.path.abspath(repo_path)
    except:
        sys.stderr.write("WARNING: could't find path to" + repository + "\n")
        sys.exit()
    result = "Unknown"
    version_path = os.path.abspath(repo_path + "/" + repository + "/version.py")
    try:
        f = open(version_path, "r")
        result = (f.readlines()[2].strip().split("=")[1]).split("'")[1]
        f.close()
    except (OSError, IOError) as e:
        sys.stderr.write(
            "WARNING: no version.py file found\n path: " + version_path + "\n"
        )
        print("error", e)
    return result


def get_git_version(repository="basinvolume", from_build=True):
    if from_build:
        return get_git_version_from_build(repository=repository)
    else:
        return get_git_version_direct(repository=repository)


def get_python_version():
    return platform.python_version()


def get_cython_version():
    try:
        from Cython.Compiler.Version import version
    except Exception as e:
        print(e)
        version = "not known"
    return version


def to_string(inp, digits_after_point=16):
    if isinstance(inp, basestring):
        return inp
    format_string = "{0:."
    format_string += str(digits_after_point)
    format_string += "f}"
    return format_string.format(inp)


def save_pdf(fig, file_name):
    pdf = PdfPages(file_name)
    fig.savefig(pdf, format="pdf")
    pdf.close()


class ResultsFile(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.f = open(self.file_name, "w")
        self.f.write("#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n")

    def set_heading(self, title):
        self.f.write("[" + title + "]\n")

    def to_file(self, name, value):
        self.f.write((name + ": {}\n").format(to_string(value)))

    def to_file_plain(self, name, value):
        self.f.write((name + ": {}\n").format(value))

    def close(self):
        self.f.close()


class OutlierDetection(object):
    """
    Classification of an array of numbers in outliers and non-outliers
    according to the definition in Knorr98,
    http://www.vldb.org/conf/1998/p392.pdf
    An object O in a dataset T is a DB(p,D) outlier if at least fraction p of
    the obects in T lies greater than distance D from O.
    Parameters are p and D.
    """

    def __init__(self, data, p=0.1, D=1, verbose=False):
        if p < 0 or p > 1:
            raise Exception("OutlierDetection: illegal input: p")
        if D < 0:
            raise Exception("OutlierDetection: illegal input: D")
        self.data = data
        self.p = p
        self.D = D
        self.verbose = verbose
        self.distant_point_maximum = len(self.data) * self.p
        self.find_outliers()

    def find_outliers(self):
        self.non_outliers = []
        self.outliers = []
        self.outliers_indexes = []
        self.non_outliers_indexes = []
        for i, datum in enumerate(self.data):
            if self.is_outlier(datum):
                self.outliers.append(datum)
                self.outliers_indexes.append(i)
            else:
                self.non_outliers.append(datum)
                self.non_outliers_indexes.append(i)
        if self.verbose:
            self.print_parameters_statistics()

    def is_outlier(self, central_datum):
        distant_points = 0
        for other_datum in self.data:
            if self.is_distant_point(central_datum, other_datum):
                distant_points += 1
            if distant_points >= self.distant_point_maximum:
                return True
        return False

    def is_distant_point(self, central_datum, other_datum):
        return np.abs(central_datum - other_datum) > self.D

    def print_parameters_statistics(self):
        print("OutlierDetection:")
        print("parameter p:", self.p)
        print("parameter D:", self.D)
        print("number of outliers:", len(self.outliers))
        if len(self.outliers) + len(self.non_outliers) > 0:
            print(
                "fraction of outliers:",
                len(self.outliers) / (len(self.outliers) + len(self.non_outliers)),
            )
        print("mean of non_outliers:", np.mean(self.non_outliers))
        print("mean of outliers:", np.mean(self.outliers))
        print("outliers:", self.outliers)


class MomentsAcc(object):
    def __init__(self):
        self.mean = 0
        self.mean2 = 0
        self.count = 0

    def update(self, inp):
        self.mean = (self.mean * self.count + inp) / (self.count + 1)
        self.mean2 = (self.mean2 * self.count + (inp * inp)) / (self.count + 1)
        self.count += 1

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.mean2 - self.mean * self.mean

    def get_std(self):
        return np.sqrt(self.get_variance())

    def get_error(self):
        return np.sqrt(self.get_variance() / self.count)


class MedianAcc(object):
    def __init__(self):
        self.data = []

    def update(self, inp):
        self.data.append(inp)

    def get_median(self):
        return np.median(np.asarray(self.data))


class CDFAccumulator(object):
    """
    Preliminary version; maybe there will be a better implementation.
    """

    def __init__(self):
        self.total_number = 0
        self.data_x = dict()

    def add_array(self, inp):
        for x in inp:
            self.add(x)

    def add(self, inp):
        already_in = inp in self.data_x
        if already_in:
            self.data_x[inp] += 1
        else:
            self.data_x[inp] = 1
        self.total_number += 1

    def get_vecdata(self):
        x = list(self.data_x.keys())
        x = sorted(x)
        cdf_x = []
        remaining_x = self.total_number
        for xi in x:
            cdf_x.append(remaining_x / self.total_number)
            remaining_x -= self.data_x[xi]
            del self.data_x[xi]
        return x, cdf_x


def gen_gauss(x, pars):
    mu = pars[0]
    alpha = pars[1]
    zeta = pars[2]
    return (
        zeta
        / (2 * alpha * gamma(1 / zeta))
        * np.exp(-np.power((np.abs(x - mu) / alpha), zeta))
    )


def get_gauss_times_expx(x, pars):
    mu = pars[0]
    alpha = pars[1]
    zeta = pars[2]
    return (
        zeta
        / (2 * alpha * gamma(1 / zeta))
        * np.exp(-np.power((np.abs(x - mu) / alpha), zeta) + x)
    )


def log_gen_gauss(x, pars):
    mu = pars[0]
    alpha = pars[1]
    zeta = pars[2]
    return (
        -np.power((np.abs(x - mu) / alpha), zeta)
        + np.log(zeta)
        - np.log(2 * alpha)
        - gammaln(1 / zeta)
    )


# replaced by a c++ function
# class CrossValidationCost(BasePotential):
#    """
#    Use leave-one-out cross validation to estimate bandwidth for kernel density.
#
#    Parameters
#    ----------
#    data : array of floats
#        The observed data.
#    kernel : string, optional
#        The used kernel type.
#
#    Examples
#    --------
#    To get a bandwidth estimate, do e.g.:
#
#        pot = CrossValidationCost(data)
#        optimizer = LBFGS_CPP(h_initial, pot)
#        result = optimizer.run()
#        opt_h = result.coords
#
#    References
#    ----------
#    http://en.wikipedia.org/wiki/Kernel_density_estimation
#    http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/ebooks/html/spm/spmhtmlnode15.html
#    http://www.control.aau.dk/~tk/undervisning/PhDAdvSI/Litterature/MadsenAndHolst2006.pdf
#    http://www.jstor.org/stable/2336252
#    """
#    def __init__(self, data, kernel="gaussian"):
#        self.data = data
#        self.N = len(self.data)
#        self.kernel = kernel
#        if self.kernel != "gaussian":
#            raise Exception("CrossValidationCost: convolution only implemented for gaussian kernel")
#
#    def getEnergy(self, h):
#        """
#        See: http://www.jstor.org/stable/2336252
#        """
#        self.h = h
#        self.compute_sums()
#        return 1 / (self.N - 1) * self.term_A + (self.N - 2) / (self.N * (self.N - 1) ** 2) * self.term_B - 2 / (self.N * (self.N - 1)) * self.term_C
#
#    def compute_sums(self):
#        def nd(x, h2):
#            return np.exp(-0.5 * x**2 / h2) / np.sqrt(2 * np.pi * h2)
#        self.term_A = nd(0, 2 * self.h**2)
#        self.term_B = 0
#        self.term_C = 0
#        for ii in xrange(self.N):
#            for jj in xrange(ii+1, self.N):
#                self.term_B += 2*nd(self.data[ii] - self.data[jj], 2 * self.h**2)
#                self.term_C += 2*nd(self.data[ii] - self.data[jj], self.h**2)


def simple_overlap_check(coords, radii, boxlength):
    """
    Perform overlap check.

    This is not very efficient, it is just a direct implementation of a
    double loop.
    Returns True if there is at least one overlap.
    Returns False if there is no overlap.
    Assums periodic boundary conditions in box of length boxlength

    Parameters
    ----------
    coords : array
        The coordinates of the centers of the considerd spheres.
    radii : array
        The considered sphere radii.
    boxlength: real
        The side length of the periodic cubic box.
    """
    nr_dof = len(coords)
    nr_particles = len(radii)
    boxdim = int(nr_dof / nr_particles)
    if nr_dof != boxdim * nr_particles:
        raise Exception("simple_overlap_check: illegal input: coords vs radii")

    def get_dist2(a, b):
        def box(input):
            boxed = np.fmod(input, boxlength)
            if boxed < 0:
                boxed += boxlength
            if boxed > 0.5 * boxlength:
                return boxed - boxlength
            return boxed

        return np.sum(
            np.array(
                [
                    np.square(box(coords[a * boxdim + ii] - coords[b * boxdim + ii]))
                    for ii in range(boxdim)
                ]
            )
        )

    def pair_is_overlapping(a, b):
        radii_sum = radii[a] + radii[b]
        return get_dist2(a, b) < radii_sum**2

    for ii in range(nr_particles - 1):
        for jj in range(ii + 1, nr_particles):
            if pair_is_overlapping(ii, jj):
                return True  # At least one overlap.
    return False  # No overlap.


def check_kmax_reasonable(kmax_configpath, max_kmax=1e7):
    """
    checks whether the value for kmax is reasonable. If it can't
    read kmax then it assumes that it is reasonable. It is essential
    that if reading kmax_configpath fail this functions returns True
    """
    configf = configparser.ConfigParser()
    try:
        configf.read(str(kmax_configpath))
        kmax = configf.getfloat("FINDK", "kmax")
    except:
        return True
    if not (0.0 < kmax <= max_kmax):
        return False
    return True


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def asphericity_factor(evals):
    """
    A measure of the gross anisotropy in a random walk.
    Asphericity has 0 as its lower bound, achieved for a walk that is
    spherical, and has an upper bound of 1, achieved when the walk is extended
    in one dimension only.

    Parameters
    ----------
    evals : array
        list of eigenvalues obtained from PCA of random walk
    """
    evals = np.sort(np.array(evals))[::-1]
    ndof = evals.size
    A = 0.0
    for i in range(ndof):
        for j in range(i, ndof):
            A += (evals[i] - evals[j]) ** 2
    A /= (ndof - 1) * np.sum(evals) ** 2
    return A


def trajectory_pca(traj):
    """
    Perform Principal Component Analysis

    returns eigenvalues and eigenvectors from
    principal componenent analysis of a trajectory

    Parameters
    ----------
    traj : 2d array
        array of containing trajectory with shape (npoints, ndof)
    """
    ndof = np.shape(traj)[1]
    cov_mat = np.cov([traj[:, i] for i in range(ndof)])
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    idx = eig_val_cov.argsort()[::-1]
    eig_val_cov = eig_val_cov[idx]
    eig_vec_cov = eig_vec_cov[:, idx]
    return eig_val_cov, eig_vec_cov


def write_2d_array_to_hdf5(array, key, path):
    assert array.ndim == 2
    nind, ncol = array.shape
    ind = [i for i in range(nind)]
    col = [i for i in range(ncol)]
    df = pd.DataFrame(array, index=ind, columns=col)
    df.to_hdf(path, key)


def read_hdf5_to_2d_array(path, key):
    df = pd.read_hdf(path, key)
    array = np.array(df.values)
    return array


def import_pt_time_series(
    explore_dir,
    adjustf_niter,
    max_series_size=0,
    ncores=4,
    del_raw=False,
    crop_adjustf_niter=False,
):
    """
    to import without loss of data set max_series_size=0 and crop_adjustf_niter=False
    if max_series_size=0 and raw timeseries are imported then the timeseries will not be cropped
    therefore max_series_size=0 indicates that there is no loss from raw to hdf5.
    If want to remove the equilibration region when importing the full dataset in hdf5 format set
    crop_adjustf_niter=True

    explore_dir string
        path to the directory containing raw data
    max_series_size int
        maximum size of array to import, set to 0 to import the whole thing
    adjustf_niter int
        number of steps to remove from timeseries because used to adjust
        stepsize
    delraw bool
        delete raw timeseries
    """
    assert (max_series_size > 0 and crop_adjustf_niter is True) or (
        max_series_size == 0 and crop_adjustf_niter is False
    )
    tsframe = os.path.join(explore_dir, "timeseries.h5")
    if os.path.isfile(tsframe):
        try:
            timeseries = read_hdf5_to_2d_array(tsframe, "ts")
            if crop_adjustf_niter:
                logging.info("cropping adjustf_niter")
                timeseries = timeseries[:, adjustf_niter:]
            logging.info("Old timeseries shape: {}".format(np.shape(timeseries)))
            if max_series_size > 0 and np.shape(timeseries)[1] > max_series_size:
                # need subsample and probably crop
                tsl = np.shape(timeseries)[1]
                logging.info(
                    "subsampling timeseries because np.shape(timeseries)[1] > max_series_size"
                )
                logging.info(
                    "subsampling every {} steps".format(int(tsl / max_series_size))
                )
                timeseries = timeseries[:, :: max(int(tsl / max_series_size), 1)]
        except Exception:
            traceback.print_exc(file=sys.stdout)
            try:
                timeseries = import_pt_time_series_raw(
                    explore_dir,
                    adjustf_niter,
                    max_series_size=max_series_size,
                    ncores=ncores,
                )
                write_2d_array_to_hdf5(timeseries, "ts", tsframe)
            except Exception:
                traceback.print_exc(file=sys.stdout)
                sys.exit(0)
    else:
        try:
            timeseries = import_pt_time_series_raw(
                explore_dir,
                adjustf_niter,
                max_series_size=max_series_size,
                ncores=ncores,
            )
            write_2d_array_to_hdf5(timeseries, "ts", tsframe)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)
    if del_raw:
        try:
            del_pt_time_series_raw(explore_dir)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)
    return timeseries


def import_pt_time_series_raw(explore_dir, adjustf_niter, max_series_size=0, ncores=7):
    """
    max_series_size int
        when set to 0 the whole time series is imported and there is a lossless conversion from
        raw to hdf5, otherwise the equilibration region needs to be necessarily removed
    """
    timeseries = []
    series_order = []
    for subdir, dirs, files in os.walk(explore_dir):
        print(explore_dir)
        for dir in dirs:
            if dir.isdigit():
                logging.debug("importing replica %s" % dir)
                series_order.append(int(dir))
                path = os.path.join(explore_dir, dir)
                # HACK for when a new directory is created
                if not os.path.isdir(path):
                    path = os.path.join(explore_dir, os.path.basename(explore_dir), dir)
                file_list = glob.glob(path + "/TimeSeries*")
                file_list = sorted(file_list, key=lambda x: int(x.split(".")[-1]))
                tot_size = int(file_list[-1].split(".")[-1]) - adjustf_niter
                print(tot_size)
                init_size = int(file_list[0].split(".")[-1]) - adjustf_niter
                if max_series_size > 0:
                    init_max_size = int(max_series_size * init_size / tot_size)
                    other_max_size = int(
                        (max_series_size - init_max_size) / len(file_list[1:])
                    )
                else:
                    # import all and don't crop (this is a bit hacky)
                    other_max_size = 0
                    adjustf_niter = 0
                series = []
                series.extend(
                    read_txt(file_list[0], adjustf_niter, max_series_size).tolist()
                )
                results = Parallel(n_jobs=ncores)(
                    delayed(read_txt)(series_path, 0, other_max_size)
                    for series_path in file_list[1:]
                )
                series.extend(list(chain.from_iterable(results)))
                timeseries.append(series)
    X = np.array(timeseries)
    Y = series_order
    timeseries = np.array([x for (y, x) in sorted(zip(Y, X))])
    return timeseries


def del_pt_time_series_raw(explore_dir):
    del_dirs = ""
    for subdir, dirs, files in os.walk(explore_dir):
        for dir in dirs:
            if dir.isdigit():
                del_dirs += dir + ", "
                path = os.path.join(explore_dir, dir)
                filelist = glob.glob(os.path.join(path, "TimeSeries.*"))
                for f in filelist:
                    os.remove(f)
    logging.info("Delete raw timeseries: {}".format(del_dirs[:-2]))


def get_uniform_in_sphere(radius, dim):
    x = np.random.normal(0, 1, dim)
    return x / np.linalg.norm(x) * radius * np.power(np.random.uniform(0, 1), 1 / dim)


class BasicPlot(object):
    """
    Set up reasonable font sizes and pdf saving etc.
    """

    def __init__(self):
        self.setup()

    def setup(self):
        plt.rcParams.update({"font.size": 17})
        plt.rcParams.update({"figure.autolayout": True})

    def save_and_close(self, loc=2):
        plt.legend(loc=loc, prop={"size": 14})
        pdf = PdfPages(self.out_name)
        plt.savefig(pdf, format="pdf")
        pdf.close()
        plt.close()


def in_hull(p, hull):
    """
    http://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    Test if points in `p` are in (the convex hull defined by) `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def is_left(coords, index1, index2, point):
    """
    Checks if a point is left of the line defined by the 2d vertices at
    coords[index1] and coords[index2]
    Returns:
    > 0 for point left of the line
    = 0 for point on the line
    < 0 for point right of the line
    """
    return (coords[index2][0] - coords[index1][0]) * (point[1] - coords[index1][1]) - (
        point[0] - coords[index1][0]
    ) * (coords[index2][1] - coords[index1][1])


def origin_is_left(coords, index1, index2):
    """
    Checks if the origin is left of the line defined by the 2d vertices at
    coords[index1] and coords[index2]
    Returns:
    > 0 for origin left of the line
    = 0 for origin on the line
    < 0 for origin right of the line
    """
    return coords[index1][0] * coords[index2][1] - coords[index2][0] * coords[index1][1]


def in_hull_2d(point, vertices):
    """
    Checks if a point is inside the convex hull defined by the 2d vertices.
    """
    hull = ConvexHull(vertices)
    for pnt in range(len(vertices)):  # traverse hull vertices counter-clockwise
        next_pnt = pnt + 1 if pnt + 1 < len(vertices) else 0
        if is_left(vertices, hull.vertices[pnt], hull.vertices[next_pnt], point) < 0:
            return False  # Point right of line
    return True


def sort_circle(vertices):
    """
    Sorts vertices by polar angle
    """
    tans = (y / (x) for [x, y] in vertices)
    rights = {}
    lefts = {}
    for i, coord in enumerate(vertices):
        if coord[0] >= 0:
            rights[next(tans)] = i
        else:
            lefts[next(tans)] = i
    return [vertices[rights[tan]] for tan in sorted(rights.keys())] + [
        vertices[lefts[tan]] for tan in sorted(lefts.keys())
    ]


def origin_in_hull_2d(vertices):
    """
    Checks if the origin is inside the convex hull defined by the 2d vertices.
    """
    sorted_vertices = sort_circle(vertices)
    for pnt in range(len(sorted_vertices)):  # traverse hull vertices counter-clockwise
        next_pnt = pnt + 1 if pnt + 1 < len(sorted_vertices) else 0
        if origin_is_left(sorted_vertices, pnt, next_pnt) < 0:
            return False  # Origin right of line
    return True


def calc_distance(coord1, coord2, bdim, distance_method, box, pot_kwargs={}):
    if distance_method is Distance.LEES_EDWARDS:
        return get_distance(
            coord1,
            coord2,
            bdim,
            distance_method,
            box=box,
            shear=pot_kwargs["shear"],
        )
    else:
        return get_distance(coord1, coord2, bdim, distance_method, box=box)


def find_neighbors_slow(
    coords,
    radii,
    bdim,
    box,
    distance_method=Distance.PERIODIC,
    pot_kwargs={"shear": 0.0},
    include=None,
    cutoff_factor=1.0,
):
    nparticles = radii.size
    neighbor_distancess = [[] for _ in range(nparticles)]
    neighbor_indicess = [[] for _ in range(nparticles)]

    # Only include given particles (e.g. for excluding rattlers)
    if include is None:
        atom_labels = list(range(nparticles))
    else:
        atom_labels = [i for i in range(nparticles) if include[i]]

    # Loop over all unique pairs of different particles
    for i, atomi in enumerate(atom_labels):
        for atomj in atom_labels[i + 1 :]:
            # Calculate distance
            dij = calc_distance(
                coords[atomi * bdim : (atomi + 1) * bdim],
                coords[atomj * bdim : (atomj + 1) * bdim],
                bdim,
                distance_method,
                box,
                pot_kwargs,
            )
            dijnorm = np.linalg.norm(dij)

            # Check if this particle lies within neighbor range
            dmax = cutoff_factor * (radii[atomi] + radii[atomj])
            if dijnorm <= dmax:
                neighbor_distancess[atomi].append(dij)
                neighbor_distancess[atomj].append(-dij)
                neighbor_indicess[atomi].append(atomj)
                neighbor_indicess[atomj].append(atomi)

    return neighbor_indicess, neighbor_distancess


def conf_get_default(configf, region, option, default):
    if configf.has_option(region, option):
        return configf.get(region, option)
    else:
        return default


def conf_getint_default(configf, region, option, default):
    if configf.has_option(region, option):
        return configf.getint(region, option)
    else:
        return default


def conf_getfloat_default(configf, region, option, default):
    if configf.has_option(region, option):
        return configf.getfloat(region, option)
    else:
        return default


def conf_getboolean_default(configf, region, option, default):
    if configf.has_option(region, option):
        return configf.getboolean(region, option)
    else:
        return default
