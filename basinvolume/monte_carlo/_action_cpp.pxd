cimport pele.potentials._pele as _pele
from libcpp cimport bool as cbool
cimport mcpele.monte_carlo._pele_mc as _pele_mc
from mcpele.monte_carlo._pele_mc cimport cppAction,_Cdef_Action,shared_ptr
from mcpele.monte_carlo._action_cpp cimport cppRecordEnergyHistogram, cppRecordScalarTimeseries

# cython has no support for integer template argument.  This is a hack to get around it
# https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
# Basically you fool cython into thinking INT2 is the type integer,
# but in the generated c++ code you use 2 instead.
# The cython code MyClass[INT2] will create c++ code MyClass<2>.
cdef extern from *:
    ctypedef int INT1 "1"    # a fake type
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type

#derives from record energy histogram
cdef extern from "basinvolume/record_disp2_histogram.h" namespace "bv":
    cdef cppclass cppRecordDisp2Histogram "bv::RecordDisp2Histogram":
        cppRecordDisp2Histogram(_pele.Array[double],_pele.Array[double], size_t, double, double, double, size_t, cbool) except +
        int get_count() except +

cdef extern from "basinvolume/record_acceptance_histogram.h" namespace "bv":
    cdef cppclass cppRecordAcceptanceHistogram "bv::RecordAcceptanceHistogram":
        cppRecordAcceptanceHistogram(_pele.Array[double], double, double, size_t, size_t) except+
        _pele.Array[double] get_acceptance_distance_values() except+
        _pele.Array[double] get_acceptance_fraction_values() except+

cdef extern from "basinvolume/findk.h" namespace "bv":
    cdef cppclass cppFindk "bv::Findk":
        cppFindk(_pele.Array[double], _pele.Array[double], size_t, double, size_t, double, double, double, double, cbool) except+
        double get_prob() except+
        double get_mean() except+
        double get_variance() except+
        _pele.Array[double] get_histogram() except +
        int get_entries() except+

cdef extern from "basinvolume/record_displacement_timeseries.h" namespace "bv":
    cdef cppclass cppRecordDisplacementTimeseries "bv::RecordDisplacementTimeseries":
        cppRecordDisplacementTimeseries(_pele.Array[double], size_t, size_t, size_t, cbool) except +

cdef extern from "basinvolume/record_steps_timeseries.h" namespace "bv":
    cdef cppclass cppRecordStepsTimeseries "bv::RecordStepsTimeseries"[ndim]:
        cppRecordStepsTimeseries(_pele.Array[double], _pele.Array[double], size_t, size_t) except +

cdef extern from "basinvolume/find_nr_decorrelation_steps.h" namespace "bv":
    cdef cppclass cppFindNrDecorrelationSteps "bv::FindNrDecorrelationSteps":
        cppFindNrDecorrelationSteps(double, size_t, size_t,
            _pele.Array[double], size_t) except +
        size_t get_nr_decorrelation_steps() except +
        cbool done() except +
