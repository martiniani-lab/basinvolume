# distutils: language = c++
cimport cython
import sys
from libcpp cimport bool as cbool
import numpy as np
cimport numpy as np
from pele.potentials import _pele
cimport pele.potentials._pele as _pele
cimport pele.optimize._pele_opt as _pele_opt
from mcpele.monte_carlo._pele_mc cimport cppConfTest,_Cdef_ConfTest

cdef extern from "basinvolume/conf_test.h" namespace "bv":
    cdef cppclass cppCheckHyperSphericalContainer "bv::CheckHyperSphericalContainer":
        cppCheckHyperSphericalContainer(_pele.Array[double], double) except+
    cdef cppclass cppCheckOverlap "bv::CheckOverlap":
        cppCheckOverlap(_pele.Array[double], _pele.Array[double]) except+
    cdef cppclass cppCheckSameMinimum "bv::CheckSameMinimum":
        cppCheckSameMinimum(_pele_opt.cGradientOptimizer *, _pele.Array[double], _pele.Array[double],
                            _pele.Array[double], _pele.Array[double] , double) except+

#===============================================================================
# Check hyper spherical container
#===============================================================================

cdef class _Cdef_CheckHyperSphericalContainer(_Cdef_ConfTest):
    """This class is the python interface for the c++ pele::CheckHyperSphericalContainer configuration test class implementation
    """
    cdef cppCheckHyperSphericalContainer* newptr
    def __cinit__(self, origin, radius):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        self.thisptr = <cppConfTest*>new cppCheckHyperSphericalContainer(_pele.Array[double](<double*> orginc.data, orginc.size), radius)
        self.newptr = <cppCheckHyperSphericalContainer*> self.thisptr
        
class CheckHyperSphericalContainer(_Cdef_CheckHyperSphericalContainer):
    """This class is the python interface for the c++ CheckHyperSphericalContainer implementation."""

#===============================================================================
# Check Overlap
#===============================================================================

cdef class _Cdef_CheckOverlap(_Cdef_ConfTest):
    """This class is the python interface for the c++ pele::CheckOverlap configuration test class implementation
    """
    cdef cppCheckOverlap* newptr
    def __cinit__(self, hs_radii, boxvec):
        cdef np.ndarray[double, ndim=1] hs_radiic = np.array(hs_radii, dtype=float)
        cdef np.ndarray[double, ndim=1] boxvecc = np.array(boxvec, dtype=float)
        self.thisptr = <cppConfTest*>new cppCheckOverlap(_pele.Array[double](<double*> hs_radiic.data, hs_radiic.size),
                                                             _pele.Array[double](<double*> boxvecc.data, boxvecc.size))
        self.newptr = <cppCheckOverlap*> self.thisptr
        
class CheckOverlap(_Cdef_CheckOverlap):
    """This class is the python interface for the c++ CheckOverlap implementation."""

#===============================================================================
# Check same minimum
#===============================================================================

cdef class _Cdef_CheckSameMinimum(_Cdef_ConfTest):
    """This class is the python interface for the c++ bv::CheckSameMinimum configuration test class implementation
    """
    
    cdef _pele_opt.GradientOptimizer opt # this is stored so that the memory is not freed
    cdef cppCheckSameMinimum* newptr
    def __cinit__(self, optimizer, origin, hs_radii, boxvec, rattlers, dtol):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef np.ndarray[double, ndim=1] hs_radiic = np.array(hs_radii, dtype=float)
        cdef np.ndarray[double, ndim=1] boxvecc = np.array(boxvec, dtype=float)
        cdef np.ndarray[double, ndim=1] rattlersc = np.array(rattlers, dtype=float)
        cdef _pele_opt.GradientOptimizer opt = optimizer
        #print rattlers
        self.thisptr = <cppConfTest*>new cppCheckSameMinimum(opt.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
                                                             _pele.Array[double](<double*> hs_radiic.data, hs_radiic.size),
                                                             _pele.Array[double](<double*> boxvecc.data, boxvecc.size), 
                                                             _pele.Array[double](<double*> rattlersc.data, rattlersc.size), dtol)
        self.newptr = <cppCheckSameMinimum*> self.thisptr 
        
class CheckSameMinimum(_Cdef_CheckSameMinimum):
    """This class is the python interface for the c++ CheckSameMinimum implementation.
    """