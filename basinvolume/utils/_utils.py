from __future__ import division
import numpy as np
import os
from scipy.special import gamma, gammaln

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

def read_xyzdr(fname, etol=1., bdim=3):
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
