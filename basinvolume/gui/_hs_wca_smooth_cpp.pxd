cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport BasePotential
from pele.potentials._pele cimport shared_ptr
cimport numpy as np
from cpython cimport bool

# use external c++ class
cdef extern from "basinvolume/hs_wca_smooth.h" namespace "bv":
    cdef cppclass  cHS_WCA_SMOOTH "bv::HS_WCA_SMOOTH":
        cHS_WCA_SMOOTH(double eps, double sca, _pele.Array[double] radii) except +
    cdef cppclass  cHS_WCA_SMOOTHPeriodic "bv::HS_WCA_SMOOTHPeriodic":
        cHS_WCA_SMOOTHPeriodic(double eps, double sca, _pele.Array[double] radii, _pele.Array[double] boxvec) except +
    cdef cppclass  cHS_WCA_SMOOTH2D "bv::HS_WCA_SMOOTH2D":
        cHS_WCA_SMOOTH2D(double eps, double sca, _pele.Array[double] radii) except +
    cdef cppclass  cHS_WCA_SMOOTHPeriodic2D "bv::HS_WCA_SMOOTHPeriodic2D":
        cHS_WCA_SMOOTHPeriodic2D(double eps, double sca, _pele.Array[double] radii, _pele.Array[double] boxvec) except +
    cdef cppclass  cHS_WCA_SMOOTHNeighborList "bv::HS_WCA_SMOOTHNeighborList":
        cHS_WCA_SMOOTHNeighborList(_pele.Array[long] & ilist, double eps, double sca, _pele.Array[double] radii) except +    
    cdef cppclass  cHS_WCA_SMOOTHFrozen "bv::HS_WCA_SMOOTHFrozen":
        cHS_WCA_SMOOTHFrozen(double eps, double sca, _pele.Array[double] radii, _pele.Array[double]& reference_coords, _pele.Array[size_t]& frozen_dof) except +
    cdef cppclass  cHS_WCA_SMOOTH2DFrozen "bv::HS_WCA_SMOOTH2DFrozen":
        cHS_WCA_SMOOTH2DFrozen(double eps, double sca, _pele.Array[double] radii, _pele.Array[double]& reference_coords, _pele.Array[size_t]& frozen_dof) except +
    cdef cppclass  cHS_WCA_SMOOTHPeriodicFrozen "bv::HS_WCA_SMOOTHPeriodicFrozen":
        cHS_WCA_SMOOTHPeriodicFrozen(double eps, double sca, _pele.Array[double] radii, _pele.Array[double] boxvec, _pele.Array[double]& reference_coords, _pele.Array[size_t]& frozen_dof) except +
    cdef cppclass  cHS_WCA_SMOOTHPeriodic2DFrozen "bv::HS_WCA_SMOOTHPeriodic2DFrozen":
        cHS_WCA_SMOOTHPeriodic2DFrozen(double eps, double sca, _pele.Array[double] radii, _pele.Array[double] boxvec, _pele.Array[double]& reference_coords, _pele.Array[size_t]& frozen_dof) except +