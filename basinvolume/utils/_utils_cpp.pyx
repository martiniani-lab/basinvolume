# distutils: language = c++

cimport cython
import sys
import numpy as np
cimport numpy as np
from pele.potentials import _pele

@cython.boundscheck(False)
@cython.wraparound(False)
def get_dist_com(coords, origin, bdim):
    cdef np.ndarray[double, ndim=1] coordsc = np.array(coords, dtype=float)
    cdef np.ndarray[double, ndim=1] originc = np.array(origin, dtype=float)
    cdef size_t cbdim = bdim
    
    dist = get_distance_com(_pele.Array[double](<double*> coordsc.data, coordsc.size),
                         _pele.Array[double](<double*> originc.data, originc.size), cbdim)
    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
def read_txt(fname):
    cdef _pele.Array[double] cseries = cread_txt(fname)
    cdef double *seriesdata = cseries.data()
    cdef size_t ndof = cseries.size()
    cdef np.ndarray[double, ndim=1, mode="c"] series = np.zeros(ndof)
    cdef size_t i
    for i in xrange(ndof):
        series[i] = seriesdata[i]
    return series

@cython.boundscheck(False)
@cython.wraparound(False)
def statisticalInefficiency(A, B=None, cbool fast=True, size_t mintime=10):
    if B is None:
        B = A
    cdef np.ndarray[double, ndim=1] Ac = np.array(A, dtype=float)
    cdef np.ndarray[double, ndim=1] Bc = np.array(B, dtype=float)
    g = statistical_inefficiency(_pele.Array[double](<double*> Ac.data, Ac.size),
                                 _pele.Array[double](<double*> Bc.data, Bc.size),
                                 fast, mintime)
    return g

@cython.boundscheck(False)
@cython.wraparound(False)
def integratedAutocorrelationTime(A_n, B_n=None, fast=True, mintime=10):
    """Estimate the integrated autocorrelation time."""
    g = statisticalInefficiency(A_n, B_n, fast=fast, mintime=mintime)
    tau = (g - 1.0) / 2.0
    return tau

@cython.boundscheck(False)
@cython.wraparound(False)
def detectEquilibration(A, cbool fast=True, size_t nskip=1):
    """
    """
    cdef np.ndarray[double, ndim=1] Ac = np.array(A, dtype=float)
    cdef _pele.Array[double] cseries = detect_equilibration(_pele.Array[double](<double*> Ac.data, Ac.size), fast, nskip)
    cdef double *seriesdata = cseries.data()
    cdef size_t ndof = cseries.size()
    cdef np.ndarray[double, ndim=1, mode="c"] series = np.zeros(ndof)
    cdef size_t i
    for i in xrange(ndof):
        series[i] = seriesdata[i]
    return series
