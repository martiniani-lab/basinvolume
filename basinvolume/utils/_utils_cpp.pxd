cimport pele.potentials._pele as _pele

cdef extern from "basinvolume/utils.h" namespace "bv":
    double get_distance_com(_pele.Array[double], _pele.Array[double], size_t) except+