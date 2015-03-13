# distutils: language = c++
# distutils: sources = ['check_same_minimum.cpp', 'check_hyper_spherical_container.cpp'] 

cimport cython
import sys
import numpy as np
cimport numpy as np
from pele.potentials import _pele
from pele.storage import Database
from pele.storage.database import Minimum
from pele.potentials._pele cimport array_wrap_np
from pele.potentials._pele cimport array_wrap_np_long, array_wrap_np_size_t
from ctypes import c_size_t as size_t
    
#===============================================================================
# Check hyper spherical container
#===============================================================================

cdef class _Cdef_CheckHyperSphericalContainer(_Cdef_ConfTest):
    """This class is the python interface for the c++ pele::CheckHyperSphericalContainer configuration test class implementation
    """
    cdef cppCheckHyperSphericalContainer* newptr
    def __cinit__(self, origin, radius, ndim):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckHyperSphericalContainer(_pele.Array[double](<double*> orginc.data, orginc.size), radius, ndim))
        self.newptr = <cppCheckHyperSphericalContainer*> self.thisptr.get()
        
class CheckHyperSphericalContainer(_Cdef_CheckHyperSphericalContainer):
    """This class is the python interface for the c++ CheckHyperSphericalContainer implementation."""


#===============================================================================
# Check Overlap Periodic
#===============================================================================

cdef class _Cdef_CheckOverlapPeriodic(_Cdef_ConfTest):
    """This class is the python interface for the c++ pele::CheckOverlap configuration test class implementation
    """
    #cdef cppCheckOverlap* newptr
    def __cinit__(self, hs_radii, boxvec, use_frozen=False, reference_coords=None, frozen_atoms=None):
        cdef np.ndarray[size_t, ndim=1] frozen_dof
        cdef size_t ndim = len(boxvec)
        cdef _pele.Array[double] rd_ = array_wrap_np(hs_radii)
        cdef _pele.Array[double] bv_ = array_wrap_np(boxvec)
        cdef _pele.Array[double] rc_
        cdef _pele.Array[size_t] fd_ 
        if not use_frozen:
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckOverlapPeriodic[INT2](rd_, bv_))
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckOverlapPeriodic[INT3](rd_, bv_))
            else:
                    raise Exception("CheckOverlap: illegal boxdimension")
        else:
            assert reference_coords is not None and frozen_atoms is not None, " warning: initialising frozen particle conf test \
                                                                                without frozen particles or reference coordinates"
            frozen_dof = np.array([range(ndim * i, ndim * i + ndim) for i in frozen_atoms], dtype=size_t).reshape(-1)
            fd_ = array_wrap_np_size_t(frozen_dof)
            rc_ = array_wrap_np(reference_coords)
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckOverlapPeriodicFrozen[INT2]
                                                       (rd_, bv_, rc_, fd_))
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckOverlapPeriodicFrozen[INT3]
                                                       (rd_, bv_, rc_, fd_))
            else:
                    raise Exception("CheckOverlap: illegal boxdimension")
        #self.newptr = <cppCheckOverlap*> self.thisptr
        
class CheckOverlapPeriodic(_Cdef_CheckOverlapPeriodic):
    """This class is the python interface for the c++ CheckOverlap implementation."""

# Check overlap cartesian

cdef class _Cdef_CheckOverlapCartesian(_Cdef_ConfTest):
    """
    Python interface for c++ CheckOverlapCartesian
    """
    def __cinit__(self, hs_radii, boxdim, use_frozen=False, reference_coords=None, frozen_atoms=None):
        cdef np.ndarray[size_t, ndim=1] frozen_dof
        cdef size_t ndim = boxdim
        cdef _pele.Array[double] rd_ = array_wrap_np(hs_radii)
        cdef _pele.Array[double] rc_
        cdef _pele.Array[size_t] fd_
        if not use_frozen:
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new
                               cppCheckOverlapCartesian[INT2](rd_)) 
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new
                               cppCheckOverlapCartesian[INT3](rd_))
            else:
                raise Exception("CheckOverlapCartesian: illegal box_dimension")
        else:
            assert reference_coords is not None and frozen_atoms is not None, " warning: initialising frozen particle conf test \
                                                                                without frozen particles or reference coordinates"
            frozen_dof = np.array([range(ndim * i, ndim * i + ndim) for i in frozen_atoms], dtype=size_t).reshape(-1)
            fd_ = array_wrap_np_size_t(frozen_dof)
            rc_ = array_wrap_np(reference_coords)
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckOverlapCartesianFrozen[INT2]
                                                       (rd_, rc_, fd_))
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckOverlapCartesianFrozen[INT3]
                                                       (rd_, rc_, fd_))
            else:
                    raise Exception("CheckOverlap: illegal boxdimension")

class CheckOverlapCartesian(_Cdef_CheckOverlapCartesian):
    """
    Python interface for c++ CheckOverlapCartesian
    """

# Check overlap cell lists

cdef class _Cdef_CheckOverlapPeriodicCellLists(_Cdef_ConfTest):
    """define the python interface to the c++ CheckOverlapCellLists implementation
    """
    def __cinit__(self, hs_radii, boxvec, ncellx_scale=1.0, use_frozen=False, frozen_atoms=None, reference_coords=None):
        cdef np.ndarray[size_t, ndim=1] frozen_dof
        cdef size_t ndim = len(boxvec)
        cdef _pele.Array[double] rd_ = array_wrap_np(hs_radii)
        cdef _pele.Array[double] bv_ = array_wrap_np(boxvec)
        cdef _pele.Array[double] rc_
        cdef _pele.Array[size_t] fd_
        
        if not use_frozen:
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapPeriodicCellLists[INT2]
                                                        (rd_, bv_, ncellx_scale)) 
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapPeriodicCellLists[INT3]
                                                        (rd_, bv_, ncellx_scale)) 
            else:
                raise Exception("CheckOverlapCellLists: illegal boxdimension")
        else:
            assert frozen_atoms is not None, " warning: initialising frozen particle conf test without frozen particles"
            frozen_dof = np.array([range(ndim * i, ndim * i + ndim) for i in frozen_atoms], dtype=size_t).reshape(-1)
            fd_ = array_wrap_np_size_t(frozen_dof)
            if reference_coords is None:
                raise Exception("CheckOverlapPeriodicCellLists: no reference_coords specified")
            rc_ = array_wrap_np(reference_coords)
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapPeriodicCellListsFrozen[INT2]
                                                        (rc_, fd_, rd_, bv_, ncellx_scale)) 
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapPeriodicCellListsFrozen[INT3]
                                                        (rc_, fd_, rd_, bv_, ncellx_scale)) 
            else:
                raise Exception("CheckOverlapCellLists: illegal boxdimension")

class CheckOverlapPeriodicCellLists(_Cdef_CheckOverlapPeriodicCellLists):
    """This class is the python interface for the c++ CheckOverlapCellLists implementation."""

#
# CheckOverlapCartesianCellLists
#

cdef class _Cdef_CheckOverlapCartesianCellLists(_Cdef_ConfTest):
    """
    CheckOverlapCartesianCellLists
    """
    def __cinit__(self, hs_radii, boxvec, ncellx_scale=1.0, use_frozen=False, frozen_atoms=None, reference_coords=None):
        cdef np.ndarray[size_t, ndim=1] frozen_dof
        cdef size_t ndim = len(boxvec)
        cdef _pele.Array[double] rd_ = array_wrap_np(hs_radii)
        cdef _pele.Array[double] bv_ = array_wrap_np(boxvec)
        cdef _pele.Array[double] rc_
        cdef _pele.Array[size_t] fd_
        if not use_frozen:
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapCartesianCellLists[INT2]
                                                        (rd_, bv_, ncellx_scale)) 
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapCartesianCellLists[INT3]
                                                        (rd_, bv_, ncellx_scale)) 
            else:
                raise Exception("CheckOverlapCellLists: illegal boxdimension")
        else:
            assert frozen_atoms is not None, " warning: initialising frozen particle conf test without frozen particles"
            frozen_dof = np.array([range(ndim * i, ndim * i + ndim) for i in frozen_atoms], dtype=size_t).reshape(-1)
            fd_ = array_wrap_np_size_t(frozen_dof)
            if reference_coords is None:
                raise Exception("CheckOverlapCartesianCellLists: no reference_coords specified")
            rc_ = array_wrap_np(reference_coords)
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapCartesianCellListsFrozen[INT2]
                                                        (rc_, fd_, rd_, bv_, ncellx_scale)) 
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapCartesianCellListsFrozen[INT3]
                                                        (rc_, fd_, rd_, bv_, ncellx_scale))
            else:
                raise Exception("CheckOverlapCellLists: illegal boxdimension")

class CheckOverlapCartesianCellLists(_Cdef_CheckOverlapCartesianCellLists):
    """This class is the python interface for the c++ CheckOverlapCartesianCellLists implementation."""
        
#===============================================================================
# Check same minimum config
#===============================================================================

cdef class _Cdef_CheckSameMinimumConfig(_Cdef_ConfTest):
    cdef _pele_opt.GradientOptimizer optimizer # this is stored so that the memory is not freed
    cdef _pele.BasePotential potential
    cdef cppCheckSameMinimumInterface* newptr
    def __cinit__(self, pot, origin, dtol, opt=None, opt_tol=1e-4, opt_maxiter=1e5):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef np.ndarray[double, ndim=1] rattlersc = np.ones(origin.size)
        self.optimizer = opt
        self.potential = pot
        self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimumCartesian[INT1](self.optimizer.thisptr, self.potential.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
                                               _pele.Array[double](<double*> rattlersc.data, rattlersc.size), dtol, 0, False, False))
                                               
                                               
class CheckSameMinimumConfig(_Cdef_CheckSameMinimumConfig):
    """interface
    """

#===============================================================================
# Check same minimum
#===============================================================================

cdef class _Cdef_CheckSameMinimum(_Cdef_ConfTest):
    """This class is the python interface for the c++ bv::CheckSameMinimum configuration test class implementation
    """
    
    cdef _pele_opt.GradientOptimizer optimizer # this is stored so that the memory is not freed
    cdef _pele.BasePotential potential
    
    cdef cppCheckSameMinimumInterface* newptr
    def __cinit__(self, pot, origin, rattlers, dtol, opt=None, bdim=3, eqsteps=0, opt_tol=1e-4, 
                  opt_maxiter=1e5, use_cgd=False, cbool perform_convergence_test=False, 
                  cbool collect_minima_list=False):
        if opt is None:
            assert use_cgd is True
        if len(origin) != len(rattlers):
            raise Exception("_Cdef_CheckSameMinimum: illegal input: origin vs rattlers")
        if len(origin) % bdim != 0:
            raise Exception("_Cdef_CheckSameMinimum: illegal input: origin vs bdim")
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef np.ndarray[double, ndim=1] rattlersc = np.array(rattlers, dtype=float)
        self.optimizer = opt
        self.potential = pot
        #print rattlers
        
        if use_cgd:
            if (bdim == 2):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimumCGDCartesian[INT2](self.potential.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
                                                                     _pele.Array[double](<double*> rattlersc.data, rattlersc.size), opt_tol, dtol, opt_maxiter, 0, eqsteps, 
                                                                     perform_convergence_test, collect_minima_list)
                                                       )
            else:
                assert(bdim == 3)
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimumCGDCartesian[INT3](self.potential.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
                                                                     _pele.Array[double](<double*> rattlersc.data, rattlersc.size), opt_tol, dtol, opt_maxiter, 0, eqsteps, 
                                                                     perform_convergence_test, collect_minima_list)
                                                   )
        else:
            if (bdim == 2):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimumCartesian[INT2](self.optimizer.thisptr, self.potential.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
                                                                     _pele.Array[double](<double*> rattlersc.data, rattlersc.size), dtol, eqsteps, 
                                                                     perform_convergence_test, collect_minima_list)
                                                       )
            else:
                assert(bdim == 3)
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimumCartesian[INT3](self.optimizer.thisptr, self.potential.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
                                                                     _pele.Array[double](<double*> rattlersc.data, rattlersc.size), dtol, eqsteps,
                                                                     perform_convergence_test, collect_minima_list)
                                                       )
        self.newptr = <cppCheckSameMinimumInterface*> self.thisptr.get()
    
    @cython.boundscheck(False)
    @cython.wraparound(False) 
    def dump_minima(self, minima_dicts):
        cdef size_t nr_neighboring_minima = self.newptr.ml_nr_distinct_minima()
        cdef cppMinimum* minimumi
        cdef _pele.Array[double] coori
        cdef double* coordata
        cdef np.ndarray[double, ndim=1, mode="c"] coor
        cdef size_t ii
        cdef _pele.Array[cppMinimum *] minima = self.newptr.get_array_of_minima()
        for i in xrange(nr_neighboring_minima):
            minimumi = minima[i]
            coori = minimumi.get_coor()
            coordata = coori.data()
            coor = np.zeros(coori.size())
            for ii in xrange(coori.size()):
                coor[ii] = coordata[ii]
            mindicti = dict(energy=minimumi.energy(), coords=coor, user_data=dict(count=minimumi.count(), distance=minimumi.delta_x()))
            minima_dicts.append(mindicti)
        assert len(minima_dicts) == nr_neighboring_minima + 1 #in the minima_dicts list, there is also the original minimum
    
    def ml_nr_distinct_minima(self):
        cdef nr_distinct_minima = self.newptr.ml_nr_distinct_minima()
        return nr_distinct_minima
    
    def get_failed_quench_frac(self):
        frac = self.newptr.get_failed_quench_frac()
        return frac
        
class CheckSameMinimum(_Cdef_CheckSameMinimum):
    """This class is the python interface for the c++ CheckSameMinimum implementation.
    """

#        else:    
#            bv = np.array(boxvec, dtype=float)
#            if use_cgd:
#                if (len(boxvec) == 2):
#                    self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimumCGDPeriodic[INT2](self.potential.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
#                                                                         _pele.Array[double](<double*> bv.data, bv.size), _pele.Array[double](<double*> rattlersc.data, rattlersc.size), 
#                                                                         opt_tol, dtol, opt_maxiter, 0, eqsteps, perform_convergence_test, collect_minima_list)
#                                                           )
#                else:
#                    assert(len(boxvec) == 3)
#                    self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimumCGDPeriodic[INT3](self.potential.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
#                                                                         _pele.Array[double](<double*> bv.data, bv.size), _pele.Array[double](<double*> rattlersc.data, rattlersc.size), 
#                                                                         opt_tol, dtol, opt_maxiter, 0, eqsteps, perform_convergence_test, collect_minima_list)
#            else:
#                if (len(boxvec) == 2):
#                    self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimumPeriodic[INT2](self.optimizer.thisptr, self.potential.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
#                                                                         _pele.Array[double](<double*> bv.data, bv.size), 
#                                                                         _pele.Array[double](<double*> rattlersc.data, rattlersc.size), dtol, eqsteps,
#                                                                         perform_convergence_test, collect_minima_list)
#                                                           )
#                else:
#                    assert(len(boxvec) == 3)
#                    self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckSameMinimumPeriodic[INT3](self.optimizer.thisptr, self.potential.thisptr, _pele.Array[double](<double*> orginc.data, orginc.size),
#                                                                         _pele.Array[double](<double*> bv.data, bv.size), 
#                                                                         _pele.Array[double](<double*> rattlersc.data, rattlersc.size), dtol, eqsteps,
#                                                                         perform_convergence_test, collect_minima_list)
#                                                       )
