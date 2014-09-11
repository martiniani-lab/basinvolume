from __future__ import division
import numpy as np
import os
from scipy.special import gamma, gammaln
import subprocess
import platform
import basinvolume
import pele
import mcpele

def volume_nball(radius, n):
    volume = np.power(np.pi, n / 2) * np.power(radius, n) / gamma(n / 2 + 1)
    return volume

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

def put_in_box(x, boxvec):
    x = x.reshape(-1, len(boxvec))
    x -= boxvec * np.round(x / boxvec)
    
def read_xyd(fname):
    coords = []
    radii = []
    f = open(fname, "r")
    while True:
        xyd = f.readline()
        if not xyd: break
        x, y, d = xyd.split()
        coords.extend([float(x),float(y)])
        radii.extend([float(d)])
    return np.array(coords, dtype='d'), np.array(radii, dtype='d')

def read_xyzd(fname):
    coords = []
    radii = []
    f = open(fname, "r")
    while True:
        xyzd = f.readline()
        if not xyzd: break
        x, y, z, d = xyzd.split()
        coords.extend([float(x),float(y),float(z)])
        radii.extend([float(d)])
    return np.array(coords, dtype='d'), np.array(radii, dtype='d')

def read_xydr(fname, etol=1.0, bdim=2):
    coords = []
    radii = []
    rattlers = []
    f = open(fname, "r")
    while True:
        xydr = f.readline()
        if not xydr: break
        #print 'xydr ',xydr
        x, y, d, r = xydr.split()
        coords.extend([float(x),float(y)])
        radii.extend([float(d)])
        rattler = float(float(r)>=etol)
        for _ in xrange(bdim): 
            rattlers.extend([rattler])
    return np.array(coords, dtype='d'), np.array(radii, dtype='d'), np.array(rattlers, dtype='d')

def read_xyzdr(fname, etol=1.0, bdim=3):
    coords = []
    radii = []
    rattlers = []
    f = open(fname, "r")
    while True:
        xyzdr = f.readline()
        if not xyzdr: break
        x, y, z, d, r = xyzdr.split()
        coords.extend([float(x),float(y),float(z)])
        radii.extend([float(d)])
        rattler = float(float(r)>=etol)
        for _ in xrange(bdim): 
            rattlers.extend([rattler])
    return np.array(coords, dtype='d'), np.array(radii, dtype='d'), np.array(rattlers, dtype='d')

#
# Make the git revision visible.  Most of this is copied from scipy
# In turn, most of this is copied from pele.
# 
# Return the git revision as a string
def get_git_version(repository = 'basinvolume'):
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
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env, cwd=repo_path).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def get_python_version():
    return platform.python_version()

def get_cython_version():
    from Cython.Compiler.Version import version
    return version

def to_string(inp, digits_after_point = 16):
    format_string = "{0:."
    format_string += str(digits_after_point)
    format_string += "f}"
    return format_string.format(inp)

class ResultsFile(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.f = open(self.file_name, "w")
        self.f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
    def set_heading(self, title):
        self.f.write("[" + title + "]\n")
    def to_file(self, name, value):
        self.f.write((name + ": {}\n").format(to_string(value)))
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
    def __init__(self, data, p = 0.1, D = 1, verbose = False):
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
    def get_variance(self):
        return self.mean2 - self.mean * self.mean
    def get_error(self):
        return np.sqrt(self.get_variance() / self.count)
        
