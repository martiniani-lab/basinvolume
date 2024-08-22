# distutils: language = c++
# distutils: sources = ['check_same_minimum.cpp', 'check_hyper_spherical_container.cpp']

from __future__ import division

cimport cython
import sys
import numpy as np
cimport numpy as np
from scipy.special import gamma
from scipy.special import rgamma
from scipy.special import gammaincc
from pele.potentials import _pele
from pele.storage import Database
from pele.storage.database import Minimum
from pele.potentials._pele cimport array_wrap_np
from pele.potentials._pele cimport array_wrap_np_long
from pele.potentials._pele cimport array_wrap_np_size_t
from ctypes import c_size_t as size_t, c_double as double

#===============================================================================
# Check hyper spherical container
#===============================================================================

cdef class _Cdef_CheckHyperSphericalContainer(_Cdef_ConfTest):
    """This class is the python interface for the c++ pele::CheckHyperSphericalContainer configuration test class implementation
    """
    cdef cppCheckHyperSphericalContainer* newptr
    def __cinit__(self, origin, radius, ndim):
        cdef _pele.Array[double] ori_ = array_wrap_np(origin)
        self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckHyperSphericalContainer(ori_, radius, ndim))
        self.newptr = <cppCheckHyperSphericalContainer*> self.thisptr.get()

class CheckHyperSphericalContainer(_Cdef_CheckHyperSphericalContainer):
    """This class is the python interface for the c++ CheckHyperSphericalContainer implementation."""

#===============================================================================
# Check exponentially decaying profile
#===============================================================================

cdef class _Cdef_CheckExponentiallyDecayingProfile(_Cdef_ConfTest):
    """
    Python interface to the C++ bv::CheckExponentiallyDecayingProfile configuration test implementation.
    """
    cdef cppCheckExponentiallyDecayingProfile* newptr
    def __cinit__(self, origin, unity_radius, decay_length, cbool cubic=False):
        cdef _pele.Array[double] origin_ = array_wrap_np(origin)
        self.n = len(origin)
        self.u = unity_radius
        self.d = decay_length
        self.cubic = cubic
        self.thisptr = shared_ptr[cppConfTest](<cppConfTest*> new cppCheckExponentiallyDecayingProfile(origin_, unity_radius, decay_length, cubic))
        self.newptr = <cppCheckExponentiallyDecayingProfile*> self.thisptr.get()
    def get_exact_volume(self):
        if self.cubic:
            return (2 * (self.u + self.d)) ** self.n
        else:
            return 2 * np.pi ** (self.n / 2) * self.inverse_gamma(self.n / 2) * (self.u ** self.n / self.n + self.d ** self.n * np.exp(self.u / self.d) * self.incomplete_gamma(self.n, self.u / self.d))
    def inverse_gamma(self, z):
        return rgamma(z)
    def incomplete_gamma(self, a, z):
        return gamma(a) * gammaincc(a, z)

class CheckExponentiallyDecayingProfile(_Cdef_CheckExponentiallyDecayingProfile):
    """
    Python interface for above.
    """

#===============================================================================
# Check hyper cubic container
#===============================================================================

cdef class _Cdef_CheckHyperCubicContainer(_Cdef_ConfTest):
    """This class is the python interface for the c++ pele::CheckHyperCubicContainer configuration test class implementation
    """
    cdef cppCheckHyperCubicContainer* newptr
    def __cinit__(self, origin, sidelengths, ndim):
        cdef _pele.Array[double] ori_ = array_wrap_np(origin)
        if isinstance(sidelengths, double):
            sidelengths_array = np.full(len(origin), sidelengths)
        else:
            sidelengths_array = sidelengths
        cdef _pele.Array[double] sls_ = array_wrap_np(sidelengths_array)
        self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckHyperCubicContainer(
            ori_, sls_, ndim, False))
        self.newptr = <cppCheckHyperCubicContainer*> self.thisptr.get()

class CheckHyperCubicContainer(_Cdef_CheckHyperCubicContainer):
    """This class is the python interface for the c++ CheckHyperCubicContainer implementation."""

#===============================================================================
# Check Overlap Periodic
#===============================================================================

cdef class _Cdef_CheckHyperCubicContainerGMC(_Cdef_GMCConfTest):
    """This class is the python interface for the c++ pele::CheckHyperCubicContainer configuration test class implementation
    """
    cdef cppCheckHyperCubicContainer* newptr
    def __cinit__(self, origin, sidelengths, ndim, use_powered_cosine_sum=False):
        cdef _pele.Array[double] ori_ = array_wrap_np(origin)
        if isinstance(sidelengths, double):
            sidelengths_array = np.full(len(origin), sidelengths)
        else:
            sidelengths_array = sidelengths
        cdef _pele.Array[double] sls_ = array_wrap_np(sidelengths_array)
        self.thisptr = shared_ptr[cppGMCConfTest](<cppGMCConfTest*>new cppCheckHyperCubicContainer(
            ori_, sls_, ndim, use_powered_cosine_sum))
        self.newptr = <cppCheckHyperCubicContainer*> self.thisptr.get()

class CheckHyperCubicContainerGMC(_Cdef_CheckHyperCubicContainerGMC):
    pass

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
    def __cinit__(self, hs_radii, boxvec, use_frozen=False, reference_coords=None, frozen_atoms=None):
        cdef np.ndarray[size_t, ndim=1] frozen_dof
        cdef size_t ndim = len(boxvec)
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


# Check Overlap Lees-Edwards

cdef class _Cdef_CheckOverlapLeesEdwards(_Cdef_ConfTest):
    """This class is the python interface for the c++ pele::CheckOverlap configuration test class implementation
    """
    #cdef cppCheckOverlap* newptr
    def __cinit__(self, hs_radii, boxvec, shear=0.0, use_frozen=False, reference_coords=None,
                  frozen_atoms=None):
        cdef np.ndarray[size_t, ndim=1] frozen_dof
        cdef size_t ndim = len(boxvec)
        cdef _pele.Array[double] rd_ = array_wrap_np(hs_radii)
        cdef _pele.Array[double] bv_ = array_wrap_np(boxvec)
        cdef _pele.Array[double] rc_
        cdef _pele.Array[size_t] fd_
        if not use_frozen:
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckOverlapLeesEdwards[INT2](rd_, bv_, shear))
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckOverlapLeesEdwards[INT3](rd_, bv_, shear))
            else:
                    raise Exception("CheckOverlap: illegal boxdimension")
        else:
            assert reference_coords is not None and frozen_atoms is not None, " warning: initialising frozen particle conf test \
                                                                                without frozen particles or reference coordinates"
            frozen_dof = np.array([range(ndim * i, ndim * i + ndim) for i in frozen_atoms], dtype=size_t).reshape(-1)
            fd_ = array_wrap_np_size_t(frozen_dof)
            rc_ = array_wrap_np(reference_coords)
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckOverlapLeesEdwardsFrozen[INT2]
                                                       (rd_, bv_, rc_, fd_, shear))
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new cppCheckOverlapLeesEdwardsFrozen[INT3]
                                                       (rd_, bv_, rc_, fd_, shear))
            else:
                    raise Exception("CheckOverlap: illegal boxdimension")
        #self.newptr = <cppCheckOverlap*> self.thisptr

class CheckOverlapLeesEdwards(_Cdef_CheckOverlapLeesEdwards):
    """This class is the python interface for the c++ CheckOverlap implementation."""

# Check overlap cell lists

cdef class _Cdef_CheckOverlapPeriodicCellLists(_Cdef_ConfTest):
    """define the python interface to the c++ CheckOverlapCellLists implementation
    """
    def __cinit__(self, hs_radii, boxvec, cbool specific=True, ncellx_scale=1.0, use_frozen=False, frozen_atoms=None, reference_coords=None):
        cdef np.ndarray[size_t, ndim=1] frozen_dof
        cdef size_t ndim = len(boxvec)
        cdef _pele.Array[double] rd_ = array_wrap_np(hs_radii)
        cdef _pele.Array[double] bv_ = array_wrap_np(boxvec)
        cdef _pele.Array[double] rc_
        cdef _pele.Array[size_t] fd_

        if not use_frozen:
            if (ndim == 2):
                print("here 2")
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapPeriodicCellLists[INT2]
                                                        (rd_, bv_, specific, ncellx_scale))
            elif (ndim == 3):
                print("here 3")
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapPeriodicCellLists[INT3]
                                                        (rd_, bv_, specific, ncellx_scale))
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
                                                        (rc_, fd_, rd_, bv_, specific, ncellx_scale))
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapPeriodicCellListsFrozen[INT3]
                                                        (rc_, fd_, rd_, bv_, specific, ncellx_scale))
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
    def __cinit__(self, hs_radii, boxvec, cbool specific=True, ncellx_scale=1.0, use_frozen=False, frozen_atoms=None, reference_coords=None):
        cdef np.ndarray[size_t, ndim=1] frozen_dof
        cdef size_t ndim = len(boxvec)
        cdef _pele.Array[double] rd_ = array_wrap_np(hs_radii)
        cdef _pele.Array[double] bv_ = array_wrap_np(boxvec)
        cdef _pele.Array[double] rc_
        cdef _pele.Array[size_t] fd_
        if not use_frozen:
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapCartesianCellLists[INT2]
                                                        (rd_, bv_, specific, ncellx_scale))
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapCartesianCellLists[INT3]
                                                        (rd_, bv_, specific, ncellx_scale))
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
                                                        (rc_, fd_, rd_, bv_, specific, ncellx_scale))
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapCartesianCellListsFrozen[INT3]
                                                        (rc_, fd_, rd_, bv_, specific, ncellx_scale))
            else:
                raise Exception("CheckOverlapCellLists: illegal boxdimension")

class CheckOverlapCartesianCellLists(_Cdef_CheckOverlapCartesianCellLists):
    """This class is the python interface for the c++ CheckOverlapCartesianCellLists implementation."""


# Check overlap cell lists with Lees-Edwards

cdef class _Cdef_CheckOverlapLeesEdwardsCellLists(_Cdef_ConfTest):
    """define the python interface to the c++ CheckOverlapCellLists implementation
    """
    def __cinit__(self, hs_radii, boxvec, shear=0.0, cbool specific=True, ncellx_scale=1.0, use_frozen=False, frozen_atoms=None, reference_coords=None):
        cdef np.ndarray[size_t, ndim=1] frozen_dof
        cdef size_t ndim = len(boxvec)
        cdef _pele.Array[double] rd_ = array_wrap_np(hs_radii)
        cdef _pele.Array[double] bv_ = array_wrap_np(boxvec)
        cdef _pele.Array[double] rc_
        cdef _pele.Array[size_t] fd_

        if not use_frozen:
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapLeesEdwardsCellLists[INT2]
                                                        (rd_, bv_, shear, specific, ncellx_scale))
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapLeesEdwardsCellLists[INT3]
                                                        (rd_, bv_, shear, specific, ncellx_scale))
            else:
                raise Exception("CheckOverlapCellLists: illegal boxdimension")
        else:
            assert frozen_atoms is not None, " warning: initialising frozen particle conf test without frozen particles"
            frozen_dof = np.array([range(ndim * i, ndim * i + ndim) for i in frozen_atoms], dtype=size_t).reshape(-1)
            fd_ = array_wrap_np_size_t(frozen_dof)
            if reference_coords is None:
                raise Exception("CheckOverlapLeesEdwardsCellLists: no reference_coords specified")
            rc_ = array_wrap_np(reference_coords)
            if (ndim == 2):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapLeesEdwardsCellListsFrozen[INT2]
                                                        (rc_, fd_, rd_, bv_, specific, ncellx_scale, shear))
            elif (ndim == 3):
                self.thisptr = shared_ptr[cppConfTest]( <cppConfTest*>new cppCheckOverlapLeesEdwardsCellListsFrozen[INT3]
                                                        (rc_, fd_, rd_, bv_, specific, ncellx_scale, shear))
            else:
                raise Exception("CheckOverlapCellLists: illegal boxdimension")

class CheckOverlapLeesEdwardsCellLists(_Cdef_CheckOverlapLeesEdwardsCellLists):
    """This class is the python interface for the c++ CheckOverlapCellLists implementation."""

#===============================================================================
# Check same minimum config
#===============================================================================

cdef class _Cdef_CheckSameMinimumConfig(_Cdef_ConfTest):
    cdef _pele_opt.GradientOptimizer optimizer # this is stored so that the memory is not freed
    cdef _pele.BasePotential potential
    cdef cppCheckSameMinimumConfig* newptr
    def __cinit__(self, pot, origin, dtol, opt=None, opt_tol=1e-4, opt_maxiter=1e5):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        self.optimizer = opt
        self.potential = pot
        self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new
            cppCheckSameMinimumConfig(self.optimizer.thisptr,
            self.potential.thisptr,
            _pele.Array[double](<double*> orginc.data, orginc.size), dtol))
        self.newptr = <cppCheckSameMinimumConfig*>self.thisptr.get()
    def get_nfev(self):
        nfev = self.newptr.get_nfev()
        return nfev
    def get_failed_quench_fraction(self):
        failed_quench_fraction = self.newptr.get_failed_quench_fraction()
        return failed_quench_fraction
    def get_origin(self):
        cdef _pele.Array[double] origin = self.newptr.get_origin()
        cdef double* origin_data = origin.data()
        cdef np.ndarray[double, ndim=1, mode="c"] origin_result = np.zeros(origin.size())
        cdef size_t i
        for i in xrange(origin.size()):
            origin_result[i] = origin_data[i]
        return origin_result

class CheckSameMinimumConfig(_Cdef_CheckSameMinimumConfig):
    """interface
    """

cdef class _Cdef_CheckSameMinimumConfigGMC(_Cdef_GMCConfTest):
    cdef _pele_opt.GradientOptimizer optimizer # this is stored so that the memory is not freed
    cdef _pele.BasePotential potential
    cdef cppCheckSameMinimumConfig* newptr
    def __cinit__(self, pot, origin, dtol, opt=None, opt_tol=1e-4, opt_maxiter=1e5):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        self.optimizer = opt
        self.potential = pot
        self.thisptr = shared_ptr[cppGMCConfTest](<cppGMCConfTest*>new
            cppCheckSameMinimumConfig(self.optimizer.thisptr,
            self.potential.thisptr,
            _pele.Array[double](<double*> orginc.data, orginc.size), dtol))
        self.newptr = <cppCheckSameMinimumConfig*>self.thisptr.get()
    def get_nfev(self):
        nfev = self.newptr.get_nfev()
        return nfev
    def get_failed_quench_fraction(self):
        failed_quench_fraction = self.newptr.get_failed_quench_fraction()
        return failed_quench_fraction
    def get_origin(self):
        cdef _pele.Array[double] origin = self.newptr.get_origin()
        cdef double* origin_data = origin.data()
        cdef np.ndarray[double, ndim=1, mode="c"] origin_result = np.zeros(origin.size())
        cdef size_t i
        for i in xrange(origin.size()):
            origin_result[i] = origin_data[i]
        return origin_result

class CheckSameMinimumConfigGMC(_Cdef_CheckSameMinimumConfigGMC):
    pass

#===============================================================================
# Check HCP compatible
#===============================================================================
cdef class _Cdef_CheckMinimumIsHCP(_Cdef_ConfTest):
    cdef _pele_opt.GradientOptimizer optimizer
    cdef cppCheckMinimumIsHCP* newptr
    cdef _pele.Array[double] bv_
    def __cinit__(self, optimizer=None, Q4tol=1e-10, boxvec=None, rcut=2, verbose=False, cbool fixed_distance_cutoff=False, cbool record_q4_histogram=False, nr_bins=14):
        if len(boxvec) != 3:
            raise Exception("CheckMinimumIsHCP: illegal input: boxdim must be 3, boxvec must be provided.")
        bv_ = array_wrap_np(boxvec)
        self.optimizer = optimizer
        self.thisptr = shared_ptr[cppConfTest](<cppConfTest*>new
            cppCheckMinimumIsHCP(self.optimizer.thisptr, Q4tol, bv_,
            rcut, fixed_distance_cutoff, record_q4_histogram, nr_bins))
        self.newptr = <cppCheckMinimumIsHCP*> self.thisptr.get()
        if verbose:
            self.newptr.set_verbose()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_hist_x(self):
        cdef _pele.Array[double] histi = self.newptr.get_hist_x()
        cdef double *histdata = histi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] hist = np.zeros(histi.size())
        cdef size_t i
        for i in xrange(histi.size()):
            hist[i] = histdata[i]
        return hist

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_hist_y(self):
        cdef _pele.Array[double] histi = self.newptr.get_hist_y()
        cdef double *histdata = histi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] hist = np.zeros(histi.size())
        cdef size_t i
        for i in xrange(histi.size()):
            hist[i] = histdata[i]
        return hist

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_hist_ey(self):
        cdef _pele.Array[double] histi = self.newptr.get_hist_ey()
        cdef double *histdata = histi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] hist = np.zeros(histi.size())
        cdef size_t i
        for i in xrange(histi.size()):
            hist[i] = histdata[i]
        return hist

class CheckMinimumIsHCP(_Cdef_CheckMinimumIsHCP):
    """interface"""

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
                self.thisptr = shared_ptr[cppConfTest](
                    <cppConfTest*>new cppCheckSameMinimumCGDCartesian[INT2](
                        self.potential.thisptr,
                        _pele.Array[double](<double*> orginc.data, orginc.size),
                        _pele.Array[double](<double*> rattlersc.data, rattlersc.size),
                        opt_tol, dtol, opt_maxiter, 0, eqsteps, perform_convergence_test,
                        collect_minima_list)
                    )
            else:
                assert(bdim == 3)
                self.thisptr = shared_ptr[cppConfTest](
                    <cppConfTest*>new cppCheckSameMinimumCGDCartesian[INT3](
                        self.potential.thisptr,
                        _pele.Array[double](<double*> orginc.data, orginc.size),
                        _pele.Array[double](<double*> rattlersc.data, rattlersc.size),
                        opt_tol, dtol, opt_maxiter, 0, eqsteps, perform_convergence_test,
                        collect_minima_list)
                    )
        else:
            if (bdim == 2):
                self.thisptr = shared_ptr[cppConfTest](
                    <cppConfTest*>new cppCheckSameMinimumCartesian[INT2](
                        self.optimizer.thisptr, self.potential.thisptr,
                        _pele.Array[double](<double*> orginc.data, orginc.size),
                        _pele.Array[double](<double*> rattlersc.data, rattlersc.size),
                        dtol, eqsteps, perform_convergence_test, collect_minima_list)
                    )
            else:
                assert(bdim == 3)
                self.thisptr = shared_ptr[cppConfTest](
                    <cppConfTest*>new cppCheckSameMinimumCartesian[INT3](
                        self.optimizer.thisptr, self.potential.thisptr,
                        _pele.Array[double](<double*> orginc.data, orginc.size),
                        _pele.Array[double](<double*> rattlersc.data, rattlersc.size),
                        dtol, eqsteps, perform_convergence_test, collect_minima_list)
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
