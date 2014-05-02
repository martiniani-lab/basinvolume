# distutils: language = c++
import numpy as np
cimport numpy as np
from pele.potentials import _pele
cimport pele.potentials._pele as _pele
cimport cython
import sys
from libcpp cimport bool as cbool
from mcpele.monte_carlo._pele_mc cimport cppAction,_Cdef_Action

cdef extern from "mcpele/actions.h" namespace "mcpele":
    cdef cppclass cppRecordEnergyHistogram "mcpele::RecordEnergyHistogram":
        cppRecordEnergyHistogram(double, double, double, size_t) except +
        _pele.Array[double] get_histogram() except +
        void print_terminal(size_t) except +
        double get_max() except +
        double get_min() except +
        double get_mean() except+
        
#===============================================================================
# RecordDisp2Histogram
#===============================================================================
#derives from record energy histogram
cdef extern from "basinvolume/actions.h" namespace "bv":
    cdef cppclass cppRecordDisp2Histogram "bv::RecordDisp2Histogram":
        cppRecordDisp2Histogram(_pele.Array[double],_pele.Array[double],double, double, double, size_t) except +
    cdef cppclass cppRecordDisp2HistogramPeriodic "bv::RecordDisp2HistogramPeriodic":
        cppRecordDisp2HistogramPeriodic(_pele.Array[double],_pele.Array[double],double, double, double, size_t, double * boxvec) except +
        
cdef class _Cdef_RecordDisp2Histogram(_Cdef_Action):
    """This class is the python interface for the c++ bv::RecordDisp2Histogram acceptance test class implementation
    """
    cpdef cbool periodic
    cdef cppRecordEnergyHistogram* newptr
    def __cinit__(self, origin, rattlers, min, max, bin, eqsteps, boxvec=None, boxl=None):
        assert not (boxvec is not None and boxl is not None)
        if boxl is not None:
            boxvec = [boxl] * 3
        cdef np.ndarray[double, ndim=1] bvec
        
        cdef np.ndarray[double, ndim=1] orginc = np.array(origin, dtype=float)
        cdef np.ndarray[double, ndim=1] rattlersc = np.array(rattlers, dtype=float)
        
        if boxvec is None:
            self.periodic = False
            self.thisptr = <cppAction*>new cppRecordDisp2Histogram(_pele.Array[double](<double*> orginc.data, orginc.size),
                                                               _pele.Array[double](<double*> rattlersc.data, rattlersc.size),
                                                               min, max, bin, eqsteps)
        else:
            self.periodic = True
            bvec = np.array(boxvec, dtype=float)
            self.thisptr = <cppAction*>new cppRecordDisp2HistogramPeriodic(_pele.Array[double](<double*> orginc.data, orginc.size),
                                                               _pele.Array[double](<double*> rattlersc.data, rattlersc.size),
                                                               min, max, bin, eqsteps, <double*> bvec.data)
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
    
    def get_mean(self):
        mean = self.newptr.get_mean()
        return mean
    
class RecordDisp2Histogram(_Cdef_RecordDisp2Histogram):
    """This class is the python interface for the c++ RecordDisp2Histogram implementation.
    """
    