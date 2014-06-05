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
        void print_terminal(size_t) except +
        double get_max() except +
        double get_min() except +
        double get_mean() except+
        double get_variance() except+
        
#===============================================================================
# RecordDisp2Histogram
#===============================================================================
#derives from record energy histogram
cdef extern from "basinvolume/actions.h" namespace "bv":
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

cdef extern from "basinvolume/actions.h" namespace "bv":    
    cdef cppclass cppFindk "bv::Findk":
        cppFindk(_pele.Array[double], _pele.Array[double], size_t, size_t, double, double, size_t, double) except+
        double get_prob() except+
        double get_mean() except+
        double get_variance() except+
        
cdef class _Cdef_Findk(_Cdef_Action):
    """This class is the python interface for the c++ bv::cppFindk action class implementation
    """
    cdef cppFindk* newptr
    def __cinit__(self, origin, rattlers, bdim, avgcount, target, factor, navg, tol):
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef np.ndarray[double, ndim=1] rattlersc = np.array(rattlers, dtype=float)
        self.thisptr = <cppAction*>new cppFindk(_pele.Array[double](<double*> orginc.data, orginc.size),
                                                _pele.Array[double](<double*> rattlersc.data, rattlersc.size), 
                                                bdim, avgcount, target, factor, navg, tol)
        self.newptr = <cppFindk*> self.thisptr
    
    def get_prob(self):
        """
        returns the probability of being in the basin at optimized kmax
        """
        prob = self.newptr.get_prob()
        return prob
    
    def get_mean_variance(self):
        mean = self.newptr.get_mean()
        variance = self.newptr.get_variance()
        return mean, variance
    
class Findk(_Cdef_Findk):
    """This class is the python interface for the c++ Findk implementation.
    """