from __future__ import division
import numpy as np
import os
from scipy.special import gamma, gammaln
import subprocess
import platform
import basinvolume
import pele
import mcpele
from pele.potentials import BasePotential
import copy
import sys, traceback
from bisect import bisect_left
import ConfigParser
import csv
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError as err:
    print err
    
def write_csv_xy(x, y, xerr=None, yerr=None, fit=None, fname='data.csv'):
    """
    fit is the y values of the fit to the xy plot
    """
    if xerr is None:
        xerr = np.zeros(len(x))
    if yerr is None:
        yerr = np.zeros(len(x))
    if fit is None:
        fit = np.zeros(len(x))
    data = (np.array(x),np.array(xerr),np.array(y),np.array(yerr), np.array(fit))
    np.savetxt(fname, np.column_stack(data), delimiter=',')

def read_csv_xy(fname):
    x, xerr, y, yerr, fit = np.loadtxt(fname, delimiter=',', unpack=True)
    return x, xerr, y, yerr, fit

def volume_nball(radius, n):
    volume = np.power(np.pi, n / 2) * np.power(radius, n) / gamma(n / 2 + 1)
    return volume

def surface_nball(radius, n):
    return 2*np.pi*volume_nball(radius, n-1)

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
            except OSError, e:
                if e.errno != 17:
                    raise
                # time.sleep might help here
                pass
        else:
            break

def view_traceback():
    ex_type, ex, tb = sys.exc_info()
    traceback.print_tb(tb)
    del tb

def put_in_box(x, boxvec):
    x = x.reshape(-1, len(boxvec))
    x -= boxvec * np.round(x / boxvec)
    
def read_xyd(fname):
    coords = []
    radii = []
    f = open(fname, "r")
    while True:
        xyd = f.readline()
        if not xyd:
            break
        x, y, d = xyd.split()
        coords.extend([float(x), float(y)])
        radii.extend([float(d)])
    return np.array(coords, dtype='d'), np.array(radii, dtype='d')

def read_xydf(fname):
    coords = []
    radii = []
    frozen = []
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
        i+=1
    return np.array(coords, dtype='d'), np.array(radii, dtype='d'), np.array(frozen, dtype='int')

def read_xyzd(fname):
    coords = []
    radii = []
    f = open(fname, "r")
    while True:
        xyzd = f.readline()
        if not xyzd:
            break
        x, y, z, d = xyzd.split()
        coords.extend([float(x), float(y), float(z)])
        radii.extend([float(d)])
    return np.array(coords, dtype='d'), np.array(radii, dtype='d')

def read_xyzdf(fname):
    coords = []
    radii = []
    frozen = []
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
        i+=1
    return np.array(coords, dtype='d'), np.array(radii, dtype='d'), np.array(frozen, dtype='int')

def read_xydr(fname, etol=1.0, bdim=2):
    coords = []
    radii = []
    rattlers = []
    f = open(fname, "r")
    while True:
        xydr = f.readline()
        if not xydr:
            break
        #print 'xydr ',xydr
        x, y, d, r = xydr.split()
        coords.extend([float(x), float(y)])
        radii.extend([float(d)])
        rattler = float(float(r) >= etol)
        for _ in xrange(bdim): 
            rattlers.extend([rattler])
    return np.array(coords, dtype='d'), np.array(radii, dtype='d'), np.array(rattlers, dtype='d')

def read_xydfr(fname, etol=1.0, bdim=2):
    coords = []
    radii = []
    frozen = []
    rattlers = []
    f = open(fname, "r")
    i=0
    while True:
        xydfr = f.readline()
        if not xydfr:
            break
        #print 'xydr ',xydr
        x, y, d, fr, r = xydfr.split()
        coords.extend([float(x), float(y)])
        radii.extend([float(d)])
        if bool(int(fr)):
            frozen.extend([i])
        rattler = float(float(r) >= etol)
        for _ in xrange(bdim): 
            rattlers.extend([rattler])
        i+=1
    return np.array(coords, dtype='d'), np.array(radii, dtype='d'), np.array(frozen, dtype='int'), np.array(rattlers, dtype='d')

def read_xyzdr(fname, etol=1.0, bdim=3):
    coords = []
    radii = []
    rattlers = []
    f = open(fname, "r")
    while True:
        xyzdr = f.readline()
        if not xyzdr:
            break
        x, y, z, d, r = xyzdr.split()
        coords.extend([float(x), float(y), float(z)])
        radii.extend([float(d)])
        rattler = float(float(r) >= etol)
        for _ in xrange(bdim): 
            rattlers.extend([rattler])
    return np.array(coords, dtype='d'), np.array(radii, dtype='d'), np.array(rattlers, dtype='d')

def read_xyzdfr(fname, etol=1.0, bdim=3):
    coords = []
    radii = []
    frozen = []
    rattlers = []
    f = open(fname, "r")
    i=0
    while True:
        xyzdfr = f.readline()
        if not xyzdfr:
            break
        x, y, z, d, fr, r = xyzdfr.split()
        coords.extend([float(x), float(y), float(z)])
        radii.extend([float(d)])
        if bool(int(fr)):
            frozen.extend([i])
        rattler = float(float(r) >= etol)
        for _ in xrange(bdim): 
            rattlers.extend([rattler])
        i+=1
    return np.array(coords, dtype='d'), np.array(radii, dtype='d'), np.array(frozen, dtype='int'), np.array(rattlers, dtype='d')

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
        newlist = np.insert(newlist, index * bdim, old_full_list[index * bdim : index * bdim + bdim])
    return np.array(newlist)

def plot_disks(coords, radii, boxv, colors=None, sca=0):
    import matplotlib 
    from matplotlib.patches import Circle 
    import pylab 
    def myscatter(ax, colormap, x, y, radii, colors): 
        for x1,y1,r,c in zip(x, y, radii, colormap(colors)): 
            ax.add_patch(Circle((x1,y1), r, fc=c)) 
    put_in_box(coords, boxv)
    coords = np.reshape(coords, (len(radii), 2))
    fig=pylab.figure() 
    ax=fig.add_subplot(111, aspect='equal') 
    myscatter(ax, matplotlib.cm.jet, coords[:,0],coords[:,1], radii * sca, np.ones(len(radii)))
    myscatter(ax, matplotlib.cm.jet, coords[:,0],coords[:,1], radii, np.ones(len(radii)) * -1) 
    ax.axis('equal')
    if boxv is not None:
        ax.axes.set_xlim([-boxv[0] / 2, boxv[0] / 2])
        ax.axes.set_ylim([-boxv[1] / 2, boxv[1] / 2])
        ax.plot([-boxv[0] / 2, -boxv[0] / 2], [-boxv[1] / 2, boxv[1] / 2], color='k', linestyle='-', linewidth=2)
        ax.plot([boxv[0] / 2, boxv[0] / 2], [-boxv[1] / 2,boxv[1] / 2], color='k', linestyle='-', linewidth=2)
        ax.plot([-boxv[0] / 2, boxv[0] / 2], [-boxv[1] / 2, -boxv[1] / 2], color='k', linestyle='-', linewidth=2)
        ax.plot([-boxv[0] / 2, boxv[0] / 2], [boxv[1] / 2,boxv[1] / 2], color='k', linestyle='-', linewidth=2)
    pylab.show()

#
# Make the git revision visible.  Most of this is copied from scipy
# In turn, most of this is copied from pele.
# 
# Return the git revision as a string
def get_git_version_direct(repository='basinvolume'):
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        repo_path = None
        try:
            if repository is "basinvolume":
                repo_path = os.path.dirname(basinvolume.__file__)[:-12]
            elif repository is "pele":
                repo_path = os.path.dirname(pele.__file__)[:-5]
            elif repository is "mcpele":
                repo_path = os.path.dirname(mcpele.__file__)[:-7]
        except:
            sys.stderr.write("WARNING: could't find path to" + repository + "\n")
            sys.exit()
        repo_path = os.path.abspath(repo_path)
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env, cwd=repo_path).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def get_git_version_from_build(repository="basinvolume"):
    repo_path = None
    try:
        if repository is "basinvolume":
            repo_path = os.path.dirname(basinvolume.__file__)[:-12]
        elif repository is "pele":
            repo_path = os.path.dirname(pele.__file__)[:-5]
        elif repository is "mcpele":
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
        sys.stderr.write("WARNING: no version.py file found\n path: " + version_path + "\n")
        print "error", e
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
    except Exception, e:
        print e
        version="not known"
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
        self.f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
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
        for datum in self.data:
            if self.is_outlier(datum):
                self.outliers.append(datum)
            else:
                self.non_outliers.append(datum)
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
        print "OutlierDetection:"
        print "parameter p:", self.p
        print "parameter D:", self.D
        print "number of outliers:", len(self.outliers)
        if len(self.outliers) + len(self.non_outliers) > 0:
            print "fraction of outliers:", len(self.outliers) / (len(self.outliers) + len(self.non_outliers))
        print "mean of non_outliers:", np.mean(self.non_outliers)
        print "mean of outliers:", np.mean(self.outliers)
        print "outliers:", self.outliers
        
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
        already_in = (inp in self.data_x)
        if already_in:
            self.data_x[inp] += 1
        else:
            self.data_x[inp] = 1
        self.total_number += 1
    def get_vecdata(self):
        x = self.data_x.keys()
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
    return zeta / (2 * alpha * gamma(1 / zeta)) * np.exp(-np.power((np.abs(x - mu) / alpha), zeta))

def get_gauss_times_expx(x, pars):
    mu = pars[0]
    alpha = pars[1]
    zeta = pars[2]
    return zeta / (2 * alpha * gamma(1 / zeta)) * np.exp(-np.power((np.abs(x - mu) / alpha), zeta) + x)

def log_gen_gauss(x, pars):
    mu = pars[0]
    alpha = pars[1]
    zeta = pars[2]
    return -np.power((np.abs(x - mu) / alpha), zeta) + np.log(zeta) - np.log(2 * alpha) - gammaln(1 / zeta)
    
class CrossValidationCost(BasePotential):
    """
    Use leave-one-out cross validation to estimate bandwidth for kernel density.
    
    Parameters
    ----------
    data : array of floats
        The observed data.
    kernel : string, optional
        The used kernel type.
    
    Examples
    --------
    To get a bandwidth estimate, do e.g.:
    
        pot = CrossValidationCost(data)
        optimizer = LBFGS_CPP(h_initial, pot)
        result = optimizer.run()
        opt_h = result.coords
        
    References
    ----------
    http://en.wikipedia.org/wiki/Kernel_density_estimation
    http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/ebooks/html/spm/spmhtmlnode15.html
    http://www.control.aau.dk/~tk/undervisning/PhDAdvSI/Litterature/MadsenAndHolst2006.pdf
    http://www.jstor.org/stable/2336252
    """
    def __init__(self, data, kernel="gaussian"):
        self.data = data
        self.N = len(self.data)
        self.kernel = kernel
        if self.kernel != "gaussian":
            raise Exception("CrossValidationCost: convolution only implemented for gaussian kernel")
        from sklearn.neighbors import KernelDensity
    def getEnergy(self, h):
        """
        See: http://www.jstor.org/stable/2336252
        """
        self.h = h
        self.compute_sums()
        return 1 / (self.N - 1) * self.term_A + (self.N - 2) / (self.N * (self.N - 1) ** 2) * self.term_B - 2 / (self.N * (self.N - 1)) * self.term_C
    def compute_sums(self):
        def nd(x, h2):
            return np.exp(-0.5 * x**2 / h2) / np.sqrt(2 * np.pi * h2)
        self.term_A = nd(0, 2 * self.h**2)
        self.term_B = 0
        self.term_C = 0
        for ii in xrange(self.N):
            for jj in xrange(self.N):
                if ii != jj:
                    self.term_B += nd(self.data[ii] - self.data[jj], 2 * self.h**2)
                    self.term_C += nd(self.data[ii] - self.data[jj], self.h**2)

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
        return np.sum(np.array([np.square(box(coords[a * boxdim + ii] - coords[b * boxdim + ii])) for ii in xrange(boxdim)]))
    def pair_is_overlapping(a, b):
        radii_sum = radii[a] + radii[b]
        return get_dist2(a, b) < radii_sum**2
    for ii in xrange(nr_particles - 1):
        for jj in xrange(ii + 1, nr_particles):
            if pair_is_overlapping(ii, jj):
                return True # At least one overlap.
    return False # No overlap.

def check_kmax_reasonable(kmax_configpath, max_kmax=1e5):
    """
    checks whether the value for kmax is reasonable. If it can't
    read kmax then it assumes that it is reasonable. It is essential
    that if reading kmax_configpath fail this functions returns True
    """
    configf = ConfigParser.ConfigParser()
    try:
        configf.read(str(kmax_configpath))
        kmax = configf.getfloat('FINDK','kmax')
    except:
        return True
    if not (0. < kmax <= max_kmax):
        return False
    return True
