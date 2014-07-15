cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport BasePotential
from pele.potentials._pele cimport shared_ptr
cimport numpy as np
from cpython cimport bool

# use external c++ class
cdef extern from "basinvolume/hs_wca_smooth.h" namespace "bv":
    cdef cppclass  cHS_WCA_Smooth "bv::HS_WCA_Smooth":
        cHS_WCA_Smooth(double eps, double sca, _pele.Array[double] radii) except +
    cdef cppclass  cHS_WCA_SmoothPeriodic "bv::HS_WCA_SmoothPeriodic":
        cHS_WCA_SmoothPeriodic(double eps, double sca, _pele.Array[double] radii, _pele.Array[double] boxvec) except +
    cdef cppclass  cHS_WCA_Smooth2D "bv::HS_WCA_Smooth2D":
        cHS_WCA_Smooth2D(double eps, double sca, _pele.Array[double] radii) except +
    cdef cppclass  cHS_WCA_SmoothPeriodic2D "bv::HS_WCA_SmoothPeriodic2D":
        cHS_WCA_SmoothPeriodic2D(double eps, double sca, _pele.Array[double] radii, _pele.Array[double] boxvec) except +
    cdef cppclass  cHS_WCA_SmoothNeighborList "bv::HS_WCA_SmoothNeighborList":
        cHS_WCA_SmoothNeighborList(_pele.Array[long] & ilist, double eps, double sca, _pele.Array[double] radii) except +    
    cdef cppclass  cHS_WCA_SmoothFrozen "bv::HS_WCA_SmoothFrozen":
        cHS_WCA_SmoothFrozen(double eps, double sca, _pele.Array[double] radii, _pele.Array[double]& reference_coords, _pele.Array[size_t]& frozen_dof) except +
    cdef cppclass  cHS_WCA_Smooth2DFrozen "bv::HS_WCA_Smooth2DFrozen":
        cHS_WCA_Smooth2DFrozen(double eps, double sca, _pele.Array[double] radii, _pele.Array[double]& reference_coords, _pele.Array[size_t]& frozen_dof) except +
    cdef cppclass  cHS_WCA_SmoothPeriodicFrozen "bv::HS_WCA_SmoothPeriodicFrozen":
        cHS_WCA_SmoothPeriodicFrozen(double eps, double sca, _pele.Array[double] radii, _pele.Array[double] boxvec, _pele.Array[double]& reference_coords, _pele.Array[size_t]& frozen_dof) except +
    cdef cppclass  cHS_WCA_SmoothPeriodic2DFrozen "bv::HS_WCA_SmoothPeriodic2DFrozen":
        cHS_WCA_SmoothPeriodic2DFrozen(double eps, double sca, _pele.Array[double] radii, _pele.Array[double] boxvec, _pele.Array[double]& reference_coords, _pele.Array[size_t]& frozen_dof) except +