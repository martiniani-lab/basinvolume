"""
# distutils: language = C++
# cython: language_level=3str
"""

cimport numpy as np
import numpy as np

cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr, array_wrap_np

cdef extern from "basinvolume/independence_sampling.h" namespace "basinvolume":
    cdef cppclass cppIndependenceSampling "basinvolume::IndependenceSampling":
        cppIndependenceSampling(size_t) except +
        int exchange(_pele.Array[int], _pele.Array[double], _pele.Array[double], int) except +

# Use the locally defined array_wrap_np_int from the .pxd file
cdef inline _pele.Array[int] array_wrap_np_int(np.ndarray[int] v) except *:
    """return a pele Array which wraps the data in a numpy array"""
    if not v.flags["FORC"]:
        raise ValueError("the numpy array is not c-contiguous.  copy it into a contiguous format before wrapping with pele::Array")
    return _pele.Array[int](<int *> v.data, v.size)

cdef class IndependenceSampling(object):
    """this class defines the python interface for c++ independence sampling

    Notes
    -----
    for direct access to the underlying c++ class use self.thisptr
    """

    def __cinit__(self, size_t seed):
        self.thisptr = shared_ptr[cppIndependenceSampling](<cppIndependenceSampling*> new
            cppIndependenceSampling(seed))

    def exchange(self, np.ndarray[int, ndim=1] exchange_pattern not None,
                 np.ndarray[double, ndim=1] dxs not None,
                 np.ndarray[double, ndim=1] betas not None, int nexchanges):
        return self.thisptr.get().exchange(array_wrap_np_int(exchange_pattern),
                                           array_wrap_np(dxs), array_wrap_np(betas),
                                           nexchanges)
