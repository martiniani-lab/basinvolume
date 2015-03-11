import numpy as np
cimport numpy as np
from ctypes import c_size_t as size_t
from ctypes import c_double as double

cimport pele.potentials._pele as _pele

cdef extern from "basinvolume/sumgaussianpot.h" namespace "bv":
    cdef cppclass cppSumGaussianPot "bv::SumGaussianPot":
        cppSumGaussianPot(size_t, _pele.Array[double], _pele.Array[double]) except+
