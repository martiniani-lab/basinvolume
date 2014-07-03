from libcpp cimport bool as cbool
cimport pele.potentials._pele as _pele
cimport pele.optimize._pele_opt as _pele_opt
from mcpele.monte_carlo._pele_mc cimport cppConfTest,_Cdef_ConfTest, shared_ptr

cdef extern from "basinvolume/check_hyper_spherical_container.h" namespace "bv":
    cdef cppclass cppCheckHyperSphericalContainer "bv::CheckHyperSphericalContainer":
        cppCheckHyperSphericalContainer(_pele.Array[double], double, size_t) except +

cdef extern from "basinvolume/check_overlap.h" namespace "bv":
    cdef cppclass cppCheckOverlap2D "bv::CheckOverlap2D":
        cppCheckOverlap2D(_pele.Array[double], double*) except+
    cdef cppclass cppCheckOverlap3D "bv::CheckOverlap3D":
        cppCheckOverlap3D(_pele.Array[double], double*) except+

#CheckSameMinimum2D(bool perform_convergence_test=false, bool collect_minima_list=false)

cdef extern from "basinvolume/check_same_minimum.h" namespace "bv":
    cdef cppclass cppCheckSameMinimum "bv::CheckSameMinimum":
        cppCheckSameMinimum(_pele_opt.cGradientOptimizer *, _pele.cBasePotential *, _pele.Array[double], _pele.Array[double],
                            _pele.Array[double] , double, bool, bool) except+
        size_t ml_nr_distinct_minima() except +
        void ml_reset_minima_iterator() except +
        cppMinimum* ml_next_minimum() except + 
    cdef cppclass cppCheckSameMinimum2D "bv::CheckSameMinimum2D":
        cppCheckSameMinimum2D(_pele_opt.cGradientOptimizer *, _pele.cBasePotential *, _pele.Array[double], _pele.Array[double],
                            _pele.Array[double] , double, bool, bool) except+
    cdef cppclass cppCheckSameMinimum3D "bv::CheckSameMinimum3D":
        cppCheckSameMinimum3D(_pele_opt.cGradientOptimizer *, _pele.cBasePotential *, _pele.Array[double], _pele.Array[double],
                            _pele.Array[double] , double, bool, bool) except+
    cdef cppclass cppCheckSameMinimumPeriodic2D "bv::CheckSameMinimumPeriodic2D":
        cppCheckSameMinimumPeriodic2D(_pele_opt.cGradientOptimizer *, _pele.cBasePotential *, _pele.Array[double], _pele.Array[double],
                            double*, _pele.Array[double] , double, bool, bool) except+
    cdef cppclass cppCheckSameMinimumPeriodic3D "bv::CheckSameMinimumPeriodic3D":
        cppCheckSameMinimumPeriodic3D(_pele_opt.cGradientOptimizer *, _pele.cBasePotential *, _pele.Array[double], _pele.Array[double],
                            double*, _pele.Array[double] , double, bool, bool) except+

cdef extern from "basinvolume/minimum.h" namespace "bv":
    cdef cppclass cppMinimum "bv::Minimum":
        double delta_x() except +
        double energy() except +
        size_t count() except +
        _pele.Array[double] get_coor() except + 
