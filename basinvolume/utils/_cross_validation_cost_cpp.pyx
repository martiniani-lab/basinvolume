# distutils: language = c++
# cython: language_level=3str

import numpy as np
cimport numpy as np
import sys
from ctypes import c_size_t as size_t

cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr, array_wrap_np, array_wrap_np_long, array_wrap_np_size_t, BasePotential
from libcpp.string cimport string

cimport cython

from pymbar.timeseries import statistical_inefficiency_fft

cdef extern from "basinvolume/cross_validation_cost.h" namespace "bv":
    cdef cppclass cCrossValidationCost "bv::CrossValidationCost":
        double get_h() except +
    cdef cppclass cGaussianCrossValidationCost "bv::GaussianCrossValidationCost":
        cGaussianCrossValidationCost(_pele.Array[double] data) except +

cdef class CrossValidationCost(BasePotential):
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
    http://sfb649.wiwi.hu-berlin.de/xplore/ebooks/html/spm/spmhtmlnode15.html
    http://www.control.aau.dk/~tk/undervisning/PhDAdvSI/Litterature/MadsenAndHolst2006.pdf
    http://www.jstor.org/stable/2336252
    """
    cdef cCrossValidationCost* newptr
    cdef data
    cdef kernel
    
    def __cinit__(self, data, kernel="gaussian"):
        
        cdef np.ndarray[double, ndim=1] cdata = data
        cdef _pele.Array[double] cd_ = array_wrap_np(data)
        self.kernel = kernel
        
        if self.kernel == "gaussian":
            self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*>new cGaussianCrossValidationCost(cd_))
        else:
            raise Exception("CrossValidationCost: illegal kernel")
            
        self.data = cdata
        self.newptr = <cCrossValidationCost*> self.thisptr.get()
        
    def get_h(self):
        h = self.newptr.get_h()
        return h
    
    def __reduce__(self):
        return (CrossValidationCost,(self.data))
    
