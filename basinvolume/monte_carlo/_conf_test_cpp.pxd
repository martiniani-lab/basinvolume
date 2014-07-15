from libcpp cimport bool as cbool
cimport pele.potentials._pele as _pele
cimport pele.optimize._pele_opt as _pele_opt
from mcpele.monte_carlo._pele_mc cimport cppConfTest,_Cdef_ConfTest,shared_ptr

cdef extern from "basinvolume/check_hyper_spherical_container.h" namespace "bv":
    cdef cppclass cppCheckHyperSphericalContainer "bv::CheckHyperSphericalContainer":
        cppCheckHyperSphericalContainer(_pele.Array[double], double, size_t) except +

# cython has no support for integer template argument.  This is a hack to get around it
# https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
# Basically you fool cython into thinking INT2 is the type integer,
# but in the generated c++ code you use 2 instead.
# The cython code MyClass[INT2] will create c++ code MyClass<2>.
cdef extern from *:
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type

cdef extern from "basinvolume/check_overlap.h" namespace "bv":
    cdef cppclass cppCheckOverlapPeriodic "bv::CheckOverlapPeriodic"[ndim]:
        cppCheckOverlapPeriodic(_pele.Array[double], _pele.Array[double]) except+




cdef extern from "basinvolume/minimum.h" namespace "bv":
    cdef cppclass cppMinimum "bv::Minimum":
        double delta_x() except +
        double energy() except +
        size_t count() except +
        _pele.Array[double] get_coor() except + 

#CheckSameMinimum2D(bool perform_convergence_test=false, bool collect_minima_list=false)

cdef extern from "basinvolume/check_same_minimum.h" namespace "bv":
    cdef cppclass cppCheckSameMinimum "bv::CheckSameMinimum":
        cppCheckSameMinimum(shared_ptr[_pele.cBasePotential], _pele.Array[double], 
                            _pele.Array[double], _pele.Array[double],
                            _pele.Array[double] , double, size_t, cbool, cbool) except+
        size_t ml_nr_distinct_minima() except +
        _pele.Array[cppMinimum *] get_array_of_minima() except +
        double get_failed_quench_frac() except+
        
    cdef cppclass cppCheckSameMinimumCartesian "bv::CheckSameMinimumCartesian"[ndim]:
        cppCheckSameMinimumCartesian(shared_ptr[_pele_opt.cGradientOptimizer], 
                                     shared_ptr[_pele.cBasePotential], _pele.Array[double], 
                                     _pele.Array[double], _pele.Array[double] , double, size_t, cbool, cbool) except+
    cdef cppclass cppCheckSameMinimumPeriodic "bv::CheckSameMinimumPeriodic"[ndim]:
        cppCheckSameMinimumPeriodic(shared_ptr[_pele_opt.cGradientOptimizer], 
                                    shared_ptr[_pele.cBasePotential], _pele.Array[double], 
                                    _pele.Array[double], _pele.Array[double], _pele.Array[double], 
                                    double, size_t, cbool, cbool) except+
