# distutils: language = c++
import numpy as np
cimport numpy as np
from pele.potentials import _pele
cimport cython
import sys
from pymbar.timeseries import statisticalInefficiency_fft
from ctypes import c_size_t as size_t

@cython.boundscheck(False)
@cython.wraparound(False)
def get_dist_com(coords, origin, bdim):
    cdef np.ndarray[double, ndim=1] coordsc = np.array(coords, dtype=float)
    cdef np.ndarray[double, ndim=1] originc = np.array(origin, dtype=float)
    cdef size_t cbdim = bdim
    
    dist = get_distance_com(_pele.Array[double](<double*> coordsc.data, coordsc.size),
                         _pele.Array[double](<double*> originc.data, originc.size), cbdim)
    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
def read_txt(fname):
    cdef _pele.Array[double] cseries = cread_txt(fname)
    cdef double *seriesdata = cseries.data()
    cdef size_t ndof = cseries.size()
    cdef np.ndarray[double, ndim=1, mode="c"] series = np.zeros(ndof)
    cdef size_t i
    for i in xrange(ndof):
        series[i] = seriesdata[i]
    return series

@cython.boundscheck(False)
@cython.wraparound(False)
def statisticalInefficiency(A, B=None, cbool fast=True, size_t mintime=10):
    """Compute the (cross) statistical inefficiency of (two) timeseries.
    c++ implementation adapted from <`pymbar`, https://github.com/choderalab/pymbar>_
    
    Parameters
    ----------
    A_n : np.ndarray, float
        A_n[n] is nth value of timeseries A.  Length is deduced from vector.
    B_n : np.ndarray, float, optional
        B_n[n] is nth value of timeseries B.  Length is deduced from vector.
        If supplied, the cross-correlation of timeseries A and B will be estimated instead of the
        autocorrelation of timeseries A.  
    fast : bool, optional, default=False
        f True, will use faster (but less accurate) method to estimate correlation
        time, described in Ref. [1] (default: False)
    mintime : int, optional, default=3
        minimum amount of correlation function to compute (default: 3)
        The algorithm terminates after computing the correlation time out to mintime when the
        correlation function furst goes negative.  Note that this time may need to be increased
        if there is a strong initial negative peak in the correlation function.
    Returns
    -------
    g : float,
        g is the estimated statistical inefficiency (equal to 1 + 2 tau, where tau is the correlation time).
        We enforce g >= 1.0.
    Notes
    -----
    The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
    The fast method described in Ref [1] is used to compute g.
    References
    ----------
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
        histogram analysis method for the analysis of simulated and parallel tempering simulations.
        JCTC 3(1):26-41, 2007.
    Examples
    --------
    Compute statistical inefficiency of timeseries data with known correlation time.  
    >>> from pymbar.testsystems import correlated_timeseries_example
    >>> A_n = correlated_timeseries_example(N=100000, tau=5.0)
    >>> g = statisticalInefficiency(A_n, fast=True)
    """
    cdef np.ndarray[double, ndim=1] Ac = np.array(A, dtype=float)
    if B is None:
        g = auto_statistical_inefficiency(_pele.Array[double](<double*> Ac.data, Ac.size),
                                          fast, mintime)
        return g
    
    cdef np.ndarray[double, ndim=1] Bc = np.array(B, dtype=float)
    g = statistical_inefficiency(_pele.Array[double](<double*> Ac.data, Ac.size),
                                 _pele.Array[double](<double*> Bc.data, Bc.size),
                                 fast, mintime)
    return g

@cython.boundscheck(False)
@cython.wraparound(False)
def integratedAutocorrelationTime(A_n, B_n=None, fast=True, mintime=10):
    """Estimate the integrated autocorrelation time."""
    g = statisticalInefficiency(A_n, B_n, fast=fast, mintime=mintime)
    tau = (g - 1.0) / 2.0
    return tau

@cython.boundscheck(False)
@cython.wraparound(False)
def integratedAutocorrelationTime_fft(A_n, mintime=10):
    """Estimate the integrated autocorrelation time."""
    g = statisticalInefficiency_fft(A_n, mintime=mintime)
    tau = (g - 1.0) / 2.0
    return tau

@cython.boundscheck(False)
@cython.wraparound(False)
def detectEquilibration(A, cbool fast=True, size_t nskip=1, cbool cprint=False, string fname="detect_equilibration.txt"):
    """Automatically detect equilibrated region of a dataset using a heuristic that maximizes number of effectively uncorrelated samples.
    c++ implementation adapted from <`pymbar`, https://github.com/choderalab/pymbar>_
    
    Parameters
    ----------
    A_t : np.ndarray 
        timeseries
    nskip : int, optional, default=1
        number of samples to sparsify data by in order to speed equilibration detection
    
    Returns
    -------
    t : int
        start of equilibrated data
    g : float
        statistical inefficiency of equilibrated data
    Neff_max : float
        number of uncorrelated samples
    
    Examples
    --------
    Determine start of equilibrated data for a correlated timeseries.
    >>> from pymbar import testsystems
    >>> A_t = testsystems.correlated_timeseries_example(N=1000, tau=5.0) # generate a test correlated timeseries
    >>> [t, g, Neff_max] = detectEquilibration(A_t) # compute indices of uncorrelated timeseries
    Determine start of equilibrated data for a correlated timeseries with a shift.
    >>> from pymbar import testsystems
    >>> A_t = testsystems.correlated_timeseries_example(N=1000, tau=5.0) + 2.0 # generate a test correlated timeseries
    >>> B_t = testsystems.correlated_timeseries_example(N=10000, tau=5.0) # generate a test correlated timeseries
    >>> C_t = numpy.concatenate([A_t, B_t])
    >>> [t, g, Neff_max] = detectEquilibration(C_t, nskip=50) # compute indices of uncorrelated timeseries
    """
    cdef np.ndarray[double, ndim=1] Ac = np.array(A, dtype=float)
    cdef _pele.Array[double] cseries = detect_equilibration(_pele.Array[double](<double*> Ac.data, Ac.size), 
                                                            fast, nskip, cprint, fname)
    cdef double *seriesdata = cseries.data()
    cdef size_t ndof = cseries.size()
    cdef np.ndarray[double, ndim=1, mode="c"] series = np.zeros(ndof)
    cdef size_t i
    for i in xrange(ndof):
        series[i] = seriesdata[i]
    return series