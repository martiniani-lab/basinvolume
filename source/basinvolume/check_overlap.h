#ifndef _BV_CHECK_OVERLAP_H
#define _BV_CHECK_OVERLAP_H

#include <cmath>
#include <memory>
#include <stdexcept>

#include "pele/array.h"
#include "pele/distance.h"

#include "mcpele/mc.h"
#include "mcpele/conf_test.h"

namespace bv{


/**
 * Test for overlap of the hard sphere cores
 */
template<typename DIST_POL>
class CheckOverlap:public mcpele::ConfTest{
protected:
    const static size_t _ndim = DIST_POL::_ndim;
    Array<double> _hs_radii;
    size_t _nparticles;
    std::shared_ptr<DIST_POL> _periodic_dist;

public:
    CheckOverlap(pele::Array<double> hs_radii, std::shared_ptr<DIST_POL> dist=NULL)
        : _hs_radii(hs_radii.copy()), 
        _nparticles(_hs_radii.size()),
        _periodic_dist(dist)
    {
        if (_periodic_dist == NULL)
            throw std::runtime_error("CheckOverlap::periodic distance uninitialised");
    }

    virtual ~CheckOverlap() {};

    bool test(Array<double> &trial_coords, mcpele::MC * mc)
    {
        size_t i,j, i1, j1;
        double dr[_ndim];
        double dij;

        for (i=0;i<_nparticles;++i){
            i1 = _ndim*i;
            for (j=0;j<_nparticles;++j){
                if (i != j){
                    j1 = _ndim*j;
                    _periodic_dist->get_rij(dr, &trial_coords[i1], &trial_coords[j1]);
                    double dij2 = 0;
                    for (size_t k =0;k<_ndim;++k){
                        dij2 += dr[k]*dr[k];
                    }
                    dij = sqrt(dij2);
                    dij -= (_hs_radii[i] + _hs_radii[j]);
                    if (dij <= 0){
                        //std::cout<<"rejected"<<std::endl;
                        return false;
                    }
                }
            }
        }
        return true;
    }

};

class CheckOverlap2D:public CheckOverlap<pele::periodic_distance<2>>{
public:
    CheckOverlap2D(Array<double> hs_radii, double const *boxvec)
        : CheckOverlap< pele::periodic_distance<2> >(hs_radii,
                std::make_shared<pele::periodic_distance<2>>(boxvec))
    {}
};

class CheckOverlap3D:public CheckOverlap<pele::periodic_distance<3>>{
public:
    CheckOverlap3D(Array<double> hs_radii, double const *boxvec)
        : CheckOverlap< pele::periodic_distance<3>>(hs_radii,
                std::make_shared<pele::periodic_distance<3>>(boxvec))
    {}
};


}//namespace bv

#endif//#ifndef _BV_CHECK_OVERLAP_H
