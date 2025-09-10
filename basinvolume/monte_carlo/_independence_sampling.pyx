# cython: language_level=3str
# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""
# distutils: language = C++
"""

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
