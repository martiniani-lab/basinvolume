# distutils: language = c++
# distutils: sources = ['check_same_minimum.cpp', 'check_hyper_spherical_container'] 

cimport cython
import sys
import numpy as np
cimport numpy as np
from pele.potentials import _pele
    
#===============================================================================
# Check hyper spherical container
#===============================================================================

cdef class _Cdef_CheckHyperSphericalContainer(_Cdef_ConfTest):
    """This class is the python interface for the c++ pele::CheckHyperSphericalContainer configuration test class implementation
    """
    cdef cppCheckHyperSphericalContainer* newptr
    def __cinit__(self, origin, radius, ndim):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckHyperSphericalContainer(_pele.Array[double](<double*> orginc.data, orginc.size), 
                                                                                                 radius, ndim)
                                               )
        self.newptr = <cppCheckHyperSphericalContainer*> self.thisptr.get()
        
class CheckHyperSphericalContainer(_Cdef_CheckHyperSphericalContainer):
    """This class is the python interface for the c++ CheckHyperSphericalContainer implementation."""

#===============================================================================
# Check Overlap
#===============================================================================

cdef class _Cdef_CheckOverlap(_Cdef_ConfTest):
    """This class is the python interface for the c++ pele::CheckOverlap configuration test class implementation
    """
    #cdef cppCheckOverlap* newptr
    def __cinit__(self, hs_radii, boxvec):
        cdef np.ndarray[double, ndim=1] hs_radiic = np.array(hs_radii, dtype=float)
        cdef np.ndarray[double, ndim=1] bv = np.array(boxvec, dtype=float)
        if (len(boxvec) == 2):
            self.thisptr = <cppConfTest*>new cppCheckOverlap2D(_pele.Array[double](<double*> hs_radiic.data, hs_radiic.size),
                                                             <double*> bv.data)
        else:
            assert(len(boxvec) == 3)
            self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckOverlap3D(_pele.Array[double](<double*> hs_radiic.data, hs_radiic.size),
                                                             <double*> bv.data)
                                                   )
        #self.newptr = <cppCheckOverlap*> self.thisptr
        
class CheckOverlap(_Cdef_CheckOverlap):
    """This class is the python interface for the c++ CheckOverlap implementation."""

#===============================================================================
# Check same minimum
#===============================================================================

cdef class _Cdef_CheckSameMinimum(_Cdef_ConfTest):
    """This class is the python interface for the c++ bv::CheckSameMinimum configuration test class implementation
    """
    
    cdef _pele_opt.GradientOptimizer opt # this is stored so that the memory is not freed
    cdef _pele.BasePotential potential
    
    #cdef cppCheckSameMinimum* newptr
    def __cinit__(self, optimizer, pot, origin, hs_radii, rattlers, dtol, boxvec=None, bdim=3):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef np.ndarray[double, ndim=1] hs_radiic = np.array(hs_radii, dtype=float)
        cdef np.ndarray[double, ndim=1] rattlersc = np.array(rattlers, dtype=float)
        cdef _pele_opt.GradientOptimizer opt = optimizer
        cdef _pele.BasePotential potential = pot
        cdef np.ndarray[double, ndim=1] bv
        self.opt = optimizer
        self.potential = pot
        #print rattlers
        
        if boxvec is None:
            if (bdim == 2):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimum2D(opt.thisptr, potential.thisptr,_pele.Array[double](<double*> orginc.data, orginc.size),
                                                                     _pele.Array[double](<double*> hs_radiic.data, hs_radiic.size),
                                                                     _pele.Array[double](<double*> rattlersc.data, rattlersc.size), dtol)
                                                       )
            else:
                assert(bdim == 3)
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimum3D(opt.thisptr, potential.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
                                                                     _pele.Array[double](<double*> hs_radiic.data, hs_radiic.size),
                                                                     _pele.Array[double](<double*> rattlersc.data, rattlersc.size), dtol)
                                                       )
        else:    
            bv = np.array(boxvec, dtype=float)
            if (len(boxvec) == 2):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimumPeriodic2D(opt.thisptr, potential.thisptr,_pele.Array[double](<double*> orginc.data, orginc.size),
                                                                     _pele.Array[double](<double*> hs_radiic.data, hs_radiic.size),
                                                                     <double*> bv.data, 
                                                                     _pele.Array[double](<double*> rattlersc.data, rattlersc.size), dtol)
                                                       )
            else:
                assert(len(boxvec) == 3)
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimumPeriodic3D(opt.thisptr, potential.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
                                                                     _pele.Array[double](<double*> hs_radiic.data, hs_radiic.size),
                                                                     <double*> bv.data, 
                                                                     _pele.Array[double](<double*> rattlersc.data, rattlersc.size), dtol)
                                                       )
        #self.newptr = <cppCheckSameMinimum*> self.thisptr
        
class CheckSameMinimum(_Cdef_CheckSameMinimum):
    """This class is the python interface for the c++ CheckSameMinimum implementation.
    """