cimport pele.potentials._pele as _pele
from libcpp cimport bool as cbool
from libcpp.string cimport string

cdef extern from "basinvolume/utils.h" namespace "bv":
    double get_distance_com(_pele.Array[double], _pele.Array[double], size_t) except+
    _pele.Array[double] get_distance_vec_com(_pele.Array[double], _pele.Array[double], size_t) except+
    _pele.Array[double] cread_txt(string, size_t, size_t) except+
    double statistical_inefficiency_cpp(_pele.Array[double] tsA, _pele.Array[double] tsB, cbool fast, size_t mintime) except+
    double auto_statistical_inefficiency(_pele.Array[double] tsA, cbool fast, size_t mintime) except+
    _pele.Array[double] detect_equilibration(_pele.Array[double] tsA, cbool fast, size_t nskip, cbool cprint, string fname) except+