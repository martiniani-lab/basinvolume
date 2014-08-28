from __future__ import division
import numpy as np
import os
from scipy.special import gamma, gammaln
import subprocess
import platform
import basinvolume
import pele
import mcpele

def volume_nball(radius,n):
    volume = np.power(np.pi,n/2)*np.power(radius,n)/gamma(n/2+1)
    return volume

def log_volume_nball(radius,n):
    log_volume = n/2.0 * np.log(np.pi) + n * np.log(radius) - gammaln(n/2+1)
    return log_volume

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
