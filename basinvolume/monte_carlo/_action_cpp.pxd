cimport pele.potentials._pele as _pele
from libcpp cimport bool as cbool
cimport mcpele.monte_carlo._pele_mc as _pele_mc
from mcpele.monte_carlo._pele_mc cimport cppAction,_Cdef_Action, shared_ptr
from mcpele.monte_carlo._action_cpp cimport cppRecordEnergyHistogram

#derives from record energy histogram
cdef extern from "basinvolume/record_disp2_histogram.h" namespace "bv":
    cdef cppclass cppRecordDisp2Histogram "bv::RecordDisp2Histogram":
        cppRecordDisp2Histogram(_pele.Array[double],_pele.Array[double], size_t, double, double, double, size_t) except +

cdef extern from "basinvolume/findk.h" namespace "bv":    
    cdef cppclass cppFindk "bv::Findk":
        cppFindk(_pele.Array[double], _pele.Array[double], size_t, size_t, double, double, size_t, double, double, double, double) except+
        double get_prob() except+
        double get_mean() except+
        double get_variance() except+
        _pele.Array[double] get_histogram() except +
        int get_entries() except+

cdef extern from "basinvolume/record_displacement_timeseries.h" namespace "bv":    
    cdef cppclass cppRecordDisplacementTimeseries "bv::RecordDisplacementTimeseries":
        cppRecordDisplacementTimeseries(_pele.Array[double], const size_t, const size_t, const size_t) except +
        _pele.Array[double] get_time_series() except +
        void clear() except +