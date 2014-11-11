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
    return series;