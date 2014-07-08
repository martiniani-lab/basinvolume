#ifndef _BV_MINIMUM_H__
#define _BV_MINIMUM_H__

#include "pele/array.h"

namespace bv{

class Minimum{
public:
    typedef double energy_t;
    typedef double coor_t;
    typedef size_t index_t;
private:
    coor_t delta_x_;
    energy_t energy_;
    pele::Array<coor_t> coor_;
    index_t count_;
public:
    Minimum(const coor_t delta_x__, const energy_t energy__, pele::Array<coor_t> coor__, const index_t count__=1)
        : delta_x_(delta_x__), energy_(energy__), coor_(coor__.copy()), count_(count__)
    {}
    coor_t delta_x()const{return delta_x_;}
    energy_t energy()const{return energy_;}
    pele::Array<coor_t> coor()const{return coor_;}
    pele::Array<coor_t> get_coor()const{return coor_.copy();}
    index_t count()const{return count_;}
    void increment_count(){++count_;}
};

}//namespace bv

#endif//#ifndef _BV_MINIMUM_H__
