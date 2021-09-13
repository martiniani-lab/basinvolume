"""
# distutils: language = C++
"""

from ctypes import c_size_t as size_t
cimport numpy as np
import numpy as np

#===============================================================================
# shared pointer
#===============================================================================
cdef extern from "<memory>" namespace "std":
    cdef cppclass shared_ptr[T]:
        shared_ptr() except+
        shared_ptr(T*) except+
        T* get() except+
        # T& operator*() # doesn't do anything
        # Note: operator->, operator= are not supported

#===============================================================================
# pele::Array
#===============================================================================
cdef extern from "pele/array.hpp" namespace "pele":
    cdef cppclass Array[dtype] :
        Array() except +
        Array(size_t) except +
        Array(dtype*, size_t n) except +
        size_t size() except +
        Array[dtype] copy() except +
        dtype *data() except +
        dtype & operator[](size_t) except +

#===============================================================================
# conversion routines between numpy arrays and pele::Array
#===============================================================================
cdef inline Array[double] array_wrap_np(np.ndarray[double] v) except *:
    """return a pele Array which wraps the data in a numpy array

    Notes
    -----
    We must be careful we only wrap the existing data.
    """
    if not v.flags["FORC"]:
        raise ValueError("the numpy array is not c-contiguous.  copy it into a contiguous format before wrapping with pele::Array")
    return Array[double](<double *> v.data, v.size)

#===============================================================================
# conversion routines between numpy arrays and pele::Array
#===============================================================================
cdef inline Array[int] array_wrap_np_int(np.ndarray[int] v) except *:
    """return a pele Array which wraps the data in a numpy array

    Notes
    -----
    We must be careful we only wrap the existing data.
    """
    if not v.flags["FORC"]:
        raise ValueError("the numpy array is not c-contiguous.  copy it into a contiguous format before wrapping with pele::Array")
    return Array[int](<int *> v.data, v.size)

cdef inline np.ndarray[double, ndim=1] pele_array_to_np(Array[double] v):
    """copy the data in a pele::Array into a new numpy array
    """
    cdef int i
    cdef int N = v.size()
    cdef np.ndarray[double, ndim=1] vnew = np.zeros(N)
    for i in xrange(N):
        vnew[i] = v[i]
    return vnew

#===============================================================================
# Independence sampling method in C++
#===============================================================================
cdef extern from "basinvolume/independence_sampling.h" namespace "bv":
    cdef cppclass  cppIndependenceSampling "bv::IndependenceSampling":
        cppIndependenceSampling(size_t) except+
        int exchange (Array[int], Array[double], Array[double], int)

#===============================================================================
# cython IndependenceSampling
#===============================================================================
cdef class IndependenceSampling:
    cdef shared_ptr[cppIndependenceSampling] thisptr
