from libcpp cimport bool as cbool
cimport pele.potentials._pele as _pele
cimport pele.optimize._pele_opt as _pele_opt
from mcpele.galilean_monte_carlo._gmc_cpp cimport cppGMCConfTest, _Cdef_GMCConfTest
from mcpele.monte_carlo._pele_mc cimport cppConfTest,_Cdef_ConfTest,shared_ptr
from ctypes import c_size_t as size_t

cdef extern from "basinvolume/check_hyper_spherical_container.h" namespace "bv":
    cdef cppclass cppCheckHyperSphericalContainer "bv::CheckHyperSphericalContainer":
        cppCheckHyperSphericalContainer(_pele.Array[double], double, size_t) except +

cdef extern from "basinvolume/check_exponentially_decaying_profile.h" namespace "bv":
    cdef cppclass cppCheckExponentiallyDecayingProfile "bv::CheckExponentiallyDecayingProfile":
        cppCheckExponentiallyDecayingProfile(_pele.Array[double], double, double, cbool) except+

cdef extern from "basinvolume/check_hyper_cubic_container.h" namespace "bv":
    cdef cppclass cppCheckHyperCubicContainer "bv::CheckHyperCubicContainer":
        cppCheckHyperCubicContainer(_pele.Array[double], _pele.Array[double], size_t, cbool,
                                    _pele.Array[double]) except +

# cython has no support for integer template argument.  This is a hack to get around it
# https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
# Basically you fool cython into thinking INT2 is the type integer,
# but in the generated c++ code you use 2 instead.
# The cython code MyClass[INT2] will create c++ code MyClass<2>.
cdef extern from *:
    ctypedef int INT1 "1"    # a fake type
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type

cdef extern from "basinvolume/check_overlap.h" namespace "bv":
    cdef cppclass cppCheckOverlapPeriodic "bv::CheckOverlapPeriodic"[ndim]:
        cppCheckOverlapPeriodic(_pele.Array[double], _pele.Array[double]) except+
    cdef cppclass cppCheckOverlapCartesian "bv::CheckOverlapCartesian"[ndim]:
        cppCheckOverlapCartesian(_pele.Array[double]) except +
    cdef cppclass cppCheckOverlapLeesEdwards "bv::CheckOverlapLeesEdwards"[ndim]:
        cppCheckOverlapLeesEdwards(_pele.Array[double], _pele.Array[double], const double) except+
    cdef cppclass cppCheckOverlapPeriodicFrozen "bv::CheckOverlapPeriodicFrozen"[ndim]:
        cppCheckOverlapPeriodicFrozen(_pele.Array[double], _pele.Array[double],
        _pele.Array[double], _pele.Array[size_t]) except+
    cdef cppclass cppCheckOverlapCartesianFrozen "bv::CheckOverlapCartesianFrozen"[ndim]:
        cppCheckOverlapCartesianFrozen(_pele.Array[double], _pele.Array[double],
        _pele.Array[size_t]) except +
    cdef cppclass cppCheckOverlapLeesEdwardsFrozen "bv::CheckOverlapLeesEdwardsFrozen"[ndim]:
        cppCheckOverlapLeesEdwardsFrozen(_pele.Array[double], _pele.Array[double],
        _pele.Array[double], _pele.Array[size_t], const double) except+

cdef extern from "basinvolume/check_overlap_cell_lists.h" namespace "bv":
    cdef cppclass cppCheckOverlapPeriodicCellLists "bv::CheckOverlapPeriodicCellLists"[ndim]:
        cppCheckOverlapPeriodicCellLists(_pele.Array[double] radii,
        _pele.Array[double] boxvec, cbool specific, double ncellx_scale) except +
    cdef cppclass cppCheckOverlapCartesianCellLists "bv::CheckOverlapCartesianCellLists"[ndim]:
        cppCheckOverlapCartesianCellLists(_pele.Array[double] radii,
        _pele.Array[double] boxvec, cbool specific, double ncellx_scale) except +
    cdef cppclass cppCheckOverlapLeesEdwardsCellLists "bv::CheckOverlapLeesEdwardsCellLists"[ndim]:
        cppCheckOverlapLeesEdwardsCellLists(_pele.Array[double] radii,
        _pele.Array[double] boxvec, const double shear, cbool specific, double ncellx_scale) except +
    cdef cppclass cppCheckOverlapPeriodicCellListsFrozen "bv::CheckOverlapPeriodicCellListsFrozen"[ndim]:
        cppCheckOverlapPeriodicCellListsFrozen(_pele.Array[double] reference_coords,
        _pele.Array[size_t] frozen_ndof, _pele.Array[double] radii, _pele.Array[double] boxvec,
        cbool specific, double ncellx_scale) except +
    cdef cppclass cppCheckOverlapCartesianCellListsFrozen "bv::CheckOverlapCartesianCellListsFrozen"[ndim]:
        cppCheckOverlapCartesianCellListsFrozen(_pele.Array[double] reference_coords,
        _pele.Array[size_t] frozen_ndof, _pele.Array[double] radii, _pele.Array[double] boxvec,
        cbool specific, double ncellx_scale) except +
    cdef cppclass cppCheckOverlapLeesEdwardsCellListsFrozen "bv::CheckOverlapLeesEdwardsCellListsFrozen"[ndim]:
        cppCheckOverlapLeesEdwardsCellListsFrozen(_pele.Array[double] reference_coords,
        _pele.Array[size_t] frozen_ndof, _pele.Array[double] radii, _pele.Array[double] boxvec,
        const double shear, cbool specific, double ncellx_scale) except +

cdef extern from "basinvolume/minimum.h" namespace "bv":
    cdef cppclass cppMinimum "bv::Minimum":
        double delta_x() except +
        double energy() except +
        size_t count() except +
        _pele.Array[double] get_coor() except +

#CheckSameMinimum2D(bool perform_convergence_test=false, bool collect_minima_list=false)

cdef extern from "basinvolume/check_same_minimum_config.h" namespace "bv":
    cdef cppclass cppCheckSameMinimumConfig "bv::CheckSameMinimumConfig":
        size_t get_nfev() except +
        double get_failed_quench_fraction() except +
        _pele.Array[double] get_origin() except +
        cppCheckSameMinimumConfig(shared_ptr[_pele_opt.cGradientOptimizer],
                                  shared_ptr[_pele.cBasePotential], _pele.Array[double],
                                  double) except +

cdef extern from "basinvolume/check_minimum_is_hcp.h" namespace "bv":
    cdef cppclass cppCheckMinimumIsHCP "bv::CheckMinimumIsHCP":
        cppCheckMinimumIsHCP(shared_ptr[_pele_opt.cGradientOptimizer],
        double, _pele.Array[double], double, cbool, cbool, size_t) except +
        void set_verbose() except +
        _pele.Array[double] get_hist_x() except+
        _pele.Array[double] get_hist_y() except+
        _pele.Array[double] get_hist_ey() except+

cdef extern from "basinvolume/check_same_minimum.h" namespace "bv":
    cdef cppclass cppCheckSameMinimumInterface "bv::CheckSameMinimumInterface":
        size_t ml_nr_distinct_minima() except +
        _pele.Array[cppMinimum *] get_array_of_minima() except +
        double get_failed_quench_frac() except+

    cdef cppclass cppCheckSameMinimumCartesian "bv::CheckSameMinimumCartesian"[ndim]:
        cppCheckSameMinimumCartesian(shared_ptr[_pele_opt.cGradientOptimizer],
                                     shared_ptr[_pele.cBasePotential], _pele.Array[double],
                                     _pele.Array[double], double, size_t, cbool, cbool) except+
    cdef cppclass cppCheckSameMinimumCGDCartesian "bv::CheckSameMinimumCGDCartesian"[ndim]:
        cppCheckSameMinimumCGDCartesian(shared_ptr[_pele.cBasePotential], _pele.Array[double],
                                     _pele.Array[double], double, double, size_t, size_t,
                                     size_t, cbool, cbool) except+

#    cdef cppclass cppCheckSameMinimumPeriodic "bv::CheckSameMinimumPeriodic"[ndim]:
#        cppCheckSameMinimumPeriodic(shared_ptr[_pele_opt.cGradientOptimizer],
#                                    shared_ptr[_pele.cBasePotential], _pele.Array[double],
#                                    _pele.Array[double], _pele.Array[double],
#                                    double, size_t, cbool, cbool) except+
#    cdef cppclass cppCheckSameMinimumCGDPeriodic "bv::CheckSameMinimumCGDPeriodic"[ndim]:
#        cppCheckSameMinimumCGDCartesian(shared_ptr[_pele.cBasePotential], _pele.Array[double],
#                                        _pele.Array[double], _pele.Array[double], double,
#                                        double, size_t, size_t, size_t, cbool, cbool) except+
#
