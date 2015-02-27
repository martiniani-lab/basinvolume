# distutils: language = c++
# distutils: sources = takestep.cpp

import sys
from pele.potentials import _pele
from pele.potentials._pele cimport array_wrap_np

cdef class _Cdef_SampleUniformSphereGaussian(_Cdef_TakeStep):
    cdef cppGaussianTakeStep* newptr
    def __cinit__(self, rseed, stepsize, origin):
        cdef _pele.Array[double] origin_ = array_wrap_np(origin)
        self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*> new cppSampleUniformSphereGaussian(rseed, stepsize, origin_))
        self.newptr = <cppGaussianTakeStep*> self.thisptr.get()
    
    def get_seed(self):
        """return random number generator seed
        
        Returns
        -------
        int 
            random number generator seed
        """
        cdef res = self.newptr.get_seed()
        return res
    
    def set_generator_seed(self, input):
        """sets the random number generator seed
        
        Parameters
        ----------
        input : pos int
            random number generator seed
        """
        cdef inp = input
        self.newptr.set_generator_seed(inp)
        
    def get_count(self):
        """get the total count of the number of steps taken
        
        Returns
        -------
        int
            total count of steps taken
        """
        return self.newptr.get_count()
    
    def get_stepsize(self):
        """get the step size
        
        Returns
        -------
        double
            stepsize
        """
        return self.newptr.get_stepsize()

class SampleUniformSphereGaussian(_Cdef_SampleUniformSphereGaussian):
    """Sample uniformly a direction from unit sphere centered at ``origin`` and select distance from N(0,stepsize)
    
    this class is the Python interface for the c++ SampleUniformSphereGaussian implementation.
    Sample uniformly a direction from unit sphere centered at ``origin`` and select distance 
    from N(0,``stepsize``) 
    
    Parameters
    ----------
    rseed : pos int
        seed for the random number generator (std:library 64 bits Merseene Twister)
    stepsize : double
        standard deviation for direct sampling
    origin : numpy.array
        coordinates where the gaussian should be centered
    """