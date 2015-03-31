# distutils: language = c++
# distutils: sources = ['record_disp2_histogram.cpp', 'record_displacement_timeseries.cpp', 'findk.cpp'] 

import numpy as np
cimport numpy as np
from pele.potentials import _pele
cimport cython
from pele.potentials._pele cimport array_wrap_np
from pele.potentials._pele cimport array_wrap_np_long, array_wrap_np_size_t
import sys

#===============================================================================
# RecordDisp2Histogram
#===============================================================================
       
cdef class _Cdef_RecordDisp2Histogram(_Cdef_Action):
    """This class is the python interface for the c++ bv::RecordDisp2Histogram acceptance test class implementation
    """
    cdef cbool fix_com
    cdef cppRecordEnergyHistogram* newptr
    def __cinit__(self, origin, rattlers, ndim, min, max, bin, eqsteps, fix_com=True):
        if len(origin) != len(rattlers):
            raise Exception("_Cdef_RecordDisp2Histogram: illegal input: origin, ndim, rattlers")
        if len(origin) % ndim != 0:
            raise Exception("_Cdef_RecordDisp2Histogram: illegal input: origin, ndim")
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef np.ndarray[double, ndim=1] rattlersc = np.array(rattlers, dtype=float)
        
        self.thisptr = shared_ptr[cppAction](<cppAction*>new cppRecordDisp2Histogram(_pele.Array[double](<double*> orginc.data, orginc.size),
                                                               _pele.Array[double](<double*> rattlersc.data, rattlersc.size),
                                                               ndim, min, max, bin, eqsteps, fix_com)
                                             )
        self.newptr = <cppRecordEnergyHistogram*> self.thisptr.get()
    
    @cython.boundscheck(False)
    @cython.wraparound(False) 
    def get_histogram(self):
        """return a histogram array"""
        cdef _pele.Array[double] histi = self.newptr.get_histogram()
        cdef double *histdata = histi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] hist = np.zeros(histi.size())
        cdef size_t i
        for i in xrange(histi.size()):
            hist[i] = histdata[i]
              
        return hist
        
    def print_terminal(self):
        self.newptr.print_terminal()
    
    def get_bounds_val(self):
        dmin = self.newptr.get_min()
        dmax = self.newptr.get_max()
        return dmin, dmax
    
    def get_mean_variance(self):
        mean = self.newptr.get_mean()
        variance = self.newptr.get_variance()
        return mean, variance
        
    def get_count(self):
        return self.newptr.get_count()
    
class RecordDisp2Histogram(_Cdef_RecordDisp2Histogram):
    """This class is the python interface for the c++ RecordDisp2Histogram implementation.
    """

#===============================================================================
# Findk
#===============================================================================
          
cdef class _Cdef_Findk(_Cdef_Action):
    """This class is the python interface for the c++ bv::cppFindk action class implementation
    """
    cdef cbool fix_com
    cdef cppFindk* newptr
    def __cinit__(self, origin, rattlers, bdim, avgcount, target, navg, tol, min, max, bin, fix_com=True):
        if len(origin) != len(rattlers):
            raise Exception("_Cdef_Findk: illegal input: origin, rattlers, bdim")
        if len(origin) % bdim != 0:
            raise Exception("_Cdef_Findk: illegal input: origin, bdim")
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef np.ndarray[double, ndim=1] rattlersc = np.array(rattlers, dtype=float)
        self.thisptr = shared_ptr[cppAction](<cppAction*>new cppFindk(_pele.Array[double](<double*> orginc.data, orginc.size),
                                                _pele.Array[double](<double*> rattlersc.data, rattlersc.size), 
                                                bdim, avgcount, target, navg, tol, min, max, bin, fix_com)
                                             )
        self.newptr = <cppFindk*> self.thisptr.get()
    
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
    @cython.wraparound(False)
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

cdef class _Cdef_RecordDisplacementTimeseries(_Cdef_Action):
    """This class is the python interface for the c++ bv::RecordDisplacementTimeseries action class implementation
    """
cdef cbool fix_com
cdef cppRecordScalarTimeseries* newptr
    def __cinit__(self, origin, bdim, niter, record_every, fix_com=True):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef size_t cbdim = bdim
        cdef size_t cniter = niter
        cdef size_t crecord_every = record_every
        
        self.thisptr = shared_ptr[cppAction](<cppAction*>new cppRecordDisplacementTimeseries(_pele.Array[double](<double*> orginc.data, orginc.size),
                                                                       cbdim, cniter, crecord_every, fix_com)
                                             )
        self.newptr = <cppRecordScalarTimeseries*> self.thisptr.get()
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
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

#===============================================================================
# RecordStepsTimeseries
#===============================================================================

cdef class _Cdef_RecordStepsTimeseries(_Cdef_Action):
    """This class is the python interface for the c++ bv::RecordStepsTimeseries action class implementation
    """
    cdef cppRecordScalarTimeseries* newptr
    def __cinit__(self, origin, rattlers, bdim, niter, record_every):
        cdef _pele.Array[double] origin_ = array_wrap_np(origin)
        cdef _pele.Array[double] rattlers_ = array_wrap_np(rattlers)
        cdef size_t cbdim = bdim
        cdef size_t cniter = niter
        cdef size_t crecord_every = record_every
        
        if bdim == 1:
            self.thisptr = shared_ptr[cppAction](<cppAction*>new cppRecordStepsTimeseries[INT1](origin_, rattlers_, cniter, crecord_every))
        elif bdim == 2:
            self.thisptr = shared_ptr[cppAction](<cppAction*>new cppRecordStepsTimeseries[INT2](origin_, rattlers_, cniter, crecord_every))
        elif bdim == 3:
            self.thisptr = shared_ptr[cppAction](<cppAction*>new cppRecordStepsTimeseries[INT3](origin_, rattlers_, cniter, crecord_every))
        else:
            raise Exception("RecordStepsTimeseries: illegal boxdimension")
        
        self.newptr = <cppRecordScalarTimeseries*> self.thisptr.get()
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
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
    
class RecordStepsTimeseries(_Cdef_RecordStepsTimeseries):
    """This class is the python interface for the c++ RecordStepsTimeseries implementation.
    """

#===============================================================================
# FindNrDecorrelationStep
#===============================================================================

cdef class _Cdef_FindNrDecorrelationSteps(_Cdef_Action):
    """python interface for bv::FindNrDecorrelationStep"""
    
    cdef cppFindNrDecorrelationSteps* newptr
    
    def __cinit__(self, desired_mean_rsm_displ, nr_iterations_start, nr_samples_avergage,
                  initial_coords, boxdim):
        if len(initial_coords) % boxdim != 0:
            raise Exception("_Cdef_FindNrDecorrelationSteps: illegal input: initial_coords, boxdim")
        cdef np.ndarray[double, ndim=1] initial_coordsc = np.array(initial_coords, dtype=float)
        self.thisptr = shared_ptr[cppAction](<cppAction*>new
                         cppFindNrDecorrelationSteps(desired_mean_rsm_displ, nr_iterations_start,
                                                    nr_samples_avergage, _pele.Array[double](<double*>
                                                         initial_coordsc.data, initial_coordsc.size), boxdim))
        self.newptr = <cppFindNrDecorrelationSteps*> self.thisptr.get()
    
    def get_nr_decorrelation_steps(self):
        cdef steps = self.newptr.get_nr_decorrelation_steps()
        return steps

class FindNrDecorrelationSteps(_Cdef_FindNrDecorrelationSteps):
    """This class is the python interface for the c++ FindNrDecorrelationSteps implementation.
    """

