cimport pele.potentials._pele as _pele
from libcpp.string cimport string

cdef extern from "basinvolume/utils.h" namespace "bv":
    double get_distance_com(_pele.Array[double], _pele.Array[double], size_t) except+
    _pele.Array[double] cread_txt(string) except+