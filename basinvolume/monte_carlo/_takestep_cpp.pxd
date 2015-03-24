cimport pele.potentials._pele as _pele
from mcpele.monte_carlo._pele_mc cimport cppTakeStep,_Cdef_TakeStep, shared_ptr
from mcpele.monte_carlo._takestep_cpp cimport cppGaussianTakeStep

cdef extern from "basinvolume/sample_uniform_sphere_gaussian.h" namespace "bv":
    cdef cppclass cppSampleUniformSphereGaussian "bv::SampleUniformSphereGaussian":
        cppSampleUniformSphereGaussian(size_t, double, _pele.Array[double]) except +
