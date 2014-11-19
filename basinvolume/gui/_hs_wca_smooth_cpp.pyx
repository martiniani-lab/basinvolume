# distutils: language = c++

cimport cython
import numpy as np
from ctypes import c_size_t as size_t

cdef class HS_WCA_Smooth(BasePotential):
    """define the python interface to the c++ HS_WCA_Smooth implementation
    """
    cpdef bool periodic 
    #thickness of the wca shell is sca * R where R is the hard core radius of the sphere
    def __cinit__(self, eps, sca, radii, ndim=3, boxvec=None, boxl=None):
        assert not (boxvec is not None and boxl is not None)
        if boxl is not None:
            boxvec = [boxl] * ndim
        cdef np.ndarray[double, ndim=1] bv
        cdef np.ndarray[double, ndim=1] radiic = np.array(radii, dtype=float)    
        
        if boxvec is None:
            self.periodic = False
            if ndim == 2:
                self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cHS_WCA_Smooth2D(eps, sca, _pele.Array[double](<double*> radiic.data, radiic.size)) )
            else:
                self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cHS_WCA_Smooth(eps, sca, _pele.Array[double](<double*> radiic.data, radiic.size)) )
        else:
            self.periodic = True
            ndim = len(boxvec)
            bv = np.array(boxvec, dtype=float)
            if ndim == 2:
                self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new 
                         cHS_WCA_SmoothPeriodic2D(eps, sca, _pele.Array[double](<double*> radiic.data, radiic.size), _pele.Array[double](<double*> bv.data, bv.size)) )
            else:
                self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new 
                         cHS_WCA_SmoothPeriodic(eps, sca, _pele.Array[double](<double*> radiic.data, radiic.size), _pele.Array[double](<double*> bv.data, bv.size)) )

cdef class HS_WCA_SmoothFrozen(BasePotential):
    """define the python interface to the c++ HS_WCA_SmoothFrozen implementation
    """
    cpdef bool periodic
    def __cinit__(self, np.ndarray[double, ndim=1] reference_coords, frozen_atoms, eps, sca, np.ndarray[double, ndim=1] radii, ndim=3, boxvec=None, boxl=None):
        assert not (boxvec is not None and boxl is not None)
        cdef np.ndarray[long, ndim=1] frozen_dof
        frozen_dof = np.array([range(ndim*i,ndim*i+ndim) for i in frozen_atoms], dtype=int).reshape(-1) 
                    
        if boxl is not None:
            boxvec = np.array([boxl] * ndim)
        cdef np.ndarray[double, ndim=1] bv
          
        if boxvec is None:
            self.periodic = False
            if ndim==2:
                self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*> new 
                     cHS_WCA_Smooth2DFrozen(eps, sca, _pele.Array[double](<double*> radii.data, radii.size), 
                                     _pele.Array[double](<double *> reference_coords.data, reference_coords.size),
                                     _pele.Array[size_t](<size_t *> frozen_dof.data, frozen_dof.size) ) )
            elif ndim==3:
                self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*> new 
                     cHS_WCA_SmoothFrozen(eps, sca, _pele.Array[double](<double*> radii.data, radii.size), 
                                   _pele.Array[double](<double *> reference_coords.data, reference_coords.size),
                                   _pele.Array[size_t](<size_t *> frozen_dof.data, frozen_dof.size) ) )
            else:
                raise Exception("HS_WCA_SmoothFrozen: illegal ndim")
        else:
            self.periodic = True
            bv = np.array(boxvec, dtype=float)
            assert bv.size == ndim
            if ndim==2:
                self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*> new 
                     cHS_WCA_SmoothPeriodic2DFrozen(eps, sca, _pele.Array[double](<double*> radii.data, radii.size), 
                                             _pele.Array[double](<double*> bv.data, bv.size), 
                                             _pele.Array[double](<double *> reference_coords.data, reference_coords.size),
                                             _pele.Array[size_t](<size_t *> frozen_dof.data, frozen_dof.size) ) )
            elif ndim==3:
                self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*> new 
                     cHS_WCA_SmoothPeriodicFrozen(eps, sca, _pele.Array[double](<double*> radii.data, radii.size),
                                           _pele.Array[double](<double*> bv.data, bv.size), 
                                           _pele.Array[double](<double *> reference_coords.data, reference_coords.size),
                                           _pele.Array[size_t](<size_t *> frozen_dof.data, frozen_dof.size) ) )
            else:
                raise Exception("HS_WCA_SmoothFrozen: illegal ndim")
