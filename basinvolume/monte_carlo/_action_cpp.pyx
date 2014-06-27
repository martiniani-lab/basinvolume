# distutils: language = c++
import numpy as np
cimport numpy as np
from pele.potentials import _pele
cimport pele.potentials._pele as _pele
cimport cython
import sys
from libcpp cimport bool as cbool
cimport mcpele.monte_carlo._pele_mc as _pele_mc
from mcpele.monte_carlo._pele_mc cimport cppAction,_Cdef_Action

cdef extern from "mcpele/actions.h" namespace "mcpele":
    cdef cppclass cppRecordEnergyHistogram "mcpele::RecordEnergyHistogram":
        cppRecordEnergyHistogram(double, double, double, size_t) except +
        _pele.Array[double] get_histogram() except +
        void print_terminal(size_t) except+
        double get_max() except+
        double get_min() except+
        double get_mean() except+
        double get_variance() except+
        
#===============================================================================
# RecordDisp2Histogram
#===============================================================================
#derives from record energy histogram
cdef extern from "basinvolume/record_disp2_histogram.h" namespace "bv":
    cdef cppclass cppRecordDisp2Histogram "bv::RecordDisp2Histogram":
        cppRecordDisp2Histogram(_pele.Array[double],_pele.Array[double], size_t, double, double, double, size_t) except +
        
cdef class _Cdef_RecordDisp2Histogram(_Cdef_Action):
    """This class is the python interface for the c++ bv::RecordDisp2Histogram acceptance test class implementation
    """
    cdef cppRecordEnergyHistogram* newptr
    def __cinit__(self, origin, rattlers, ndim, min, max, bin, eqsteps):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef np.ndarray[double, ndim=1] rattlersc = np.array(rattlers, dtype=float)
        
        self.thisptr = <cppAction*>new cppRecordDisp2Histogram(_pele.Array[double](<double*> orginc.data, orginc.size),
                                                               _pele.Array[double](<double*> rattlersc.data, rattlersc.size),
                                                               ndim, min, max, bin, eqsteps)
        self.newptr = <cppRecordEnergyHistogram*> self.thisptr
        
    @cython.boundscheck(False)
    def get_histogram(self):
        """return a histogram array"""
        cdef _pele.Array[double] histi = self.newptr.get_histogram()
        cdef double *histdata = histi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] hist = np.zeros(histi.size())
        cdef size_t i
        for i in xrange(histi.size()):
            hist[i] = histdata[i]
              
        return hist
        
    def print_terminal(self, ntot):
        self.newptr.print_terminal(ntot)
    
    def get_bounds_val(self):
        dmin = self.newptr.get_min()
        dmax = self.newptr.get_max()
        return dmin, dmax
    
    def get_mean_variance(self):
        mean = self.newptr.get_mean()
        variance = self.newptr.get_variance()
        return mean, variance
    
class RecordDisp2Histogram(_Cdef_RecordDisp2Histogram):
    """This class is the python interface for the c++ RecordDisp2Histogram implementation.
    """
    
#===============================================================================
# Findk
#===============================================================================

cdef extern from "basinvolume/findk.h" namespace "bv":    
    cdef cppclass cppFindk "bv::Findk":
        cppFindk(_pele.Array[double], _pele.Array[double], size_t, size_t, double, double, size_t, double, double, double, double) except+
        double get_prob() except+
        double get_mean() except+
        double get_variance() except+
        _pele.Array[double] get_histogram() except +
        int get_entries() except+
        
cdef class _Cdef_Findk(_Cdef_Action):
    """This class is the python interface for the c++ bv::cppFindk action class implementation
    """
    cdef cppFindk* newptr
    def __cinit__(self, origin, rattlers, bdim, avgcount, target, factor, navg, tol, min, max, bin):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef np.ndarray[double, ndim=1] rattlersc = np.array(rattlers, dtype=float)
        self.thisptr = <cppAction*>new cppFindk(_pele.Array[double](<double*> orginc.data, orginc.size),
                                                _pele.Array[double](<double*> rattlersc.data, rattlersc.size), 
                                                bdim, avgcount, target, factor, navg, tol, min, max, bin)
        self.newptr = <cppFindk*> self.thisptr
    
    def get_prob(self):
        """
        returns the probability of being in the basin at optimized kmax
        """
        prob = self.newptr.get_prob()
        return prob
    
    def get_entries(self):
        entries = self.newptr.get_entries()
        return entries
    
    def get_mean_variance(self):
        mean = self.newptr.get_mean()
        variance = self.newptr.get_variance()
        return mean, variance
    
    @cython.boundscheck(False)
    def get_histogram(self):
        """return a histogram array"""
        cdef _pele.Array[double] histi = self.newptr.get_histogram()
        cdef double *histdata = histi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] hist = np.zeros(histi.size())
        cdef size_t i
        for i in xrange(histi.size()):
            hist[i] = histdata[i]
              
        return hist
    
class Findk(_Cdef_Findk):
    """This class is the python interface for the c++ Findk implementation.
    """
    
#===============================================================================
# RecordEnergyTimeseries
#===============================================================================

cdef extern from "basinvolume/record_displacement_timeseries.h" namespace "bv":    
    cdef cppclass cppRecordDisplacementTimeseries "bv::RecordDisplacementTimeseries":
        cppRecordDisplacementTimeseries(_pele.Array[double], const size_t, const size_t, const size_t) except +
        _pele.Array[double] get_time_series() except +
        void clear() except +
        
cdef class _Cdef_RecordDisplacementTimeseries(_Cdef_Action):
    """This class is the python interface for the c++ bv::RecordDisplacementTimeseries action class implementation
    """
    cdef cppRecordDisplacementTimeseries* newptr
    def __cinit__(self, origin, bdim, niter, record_every):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef size_t cbdim = bdim
        cdef size_t cniter = niter
        cdef size_t crecord_every = record_every
        
        self.thisptr = <cppAction*>new cppRecordDisplacementTimeseries(_pele.Array[double](<double*> orginc.data, orginc.size),
                                                                       cbdim, cniter, crecord_every)
        self.newptr = <cppRecordDisplacementTimeseries*> self.thisptr
        
    @cython.boundscheck(False)
    def get_time_series(self):
        """return a energy time series array"""
        cdef _pele.Array[double] seriesi = self.newptr.get_time_series()
        cdef double *seriesdata = seriesi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] series = np.zeros(seriesi.size())
        cdef size_t i
        for i in xrange(seriesi.size()):
            series[i] = seriesdata[i]
              
        return series
    
    def clear(self):
        """clears time series"""
        self.newptr.clear()
    
class RecordDisplacementTimeseries(_Cdef_RecordDisplacementTimeseries):
    """This class is the python interface for the c++ RecordDisplacementTimeseries implementation.
    """