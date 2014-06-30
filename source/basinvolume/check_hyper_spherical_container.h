#ifndef _BV_CHECK_HYPER_SPHERICAL_CONTAINER_H
#define _BV_CHECK_HYPER_SPHERICAL_CONTAINER_H

#include <memory>

#include "pele/array.h"
#include "pele/optimizer.h"
#include "pele/distance.h"

#include "mcpele/mc.h"
#include "mcpele/conf_test.h"

namespace bv{

class CheckHyperSphericalContainer:public mcpele::ConfTest{
protected:
    void _get_vec_distance(const pele::Array<double>& coords);
    pele::Array<double> _origin, _distance;
    double _radius2;
    size_t _ndim,_N;
public:
    CheckHyperSphericalContainer(pele::Array<double> origin, double radius, size_t ndim);
    virtual bool test(pele::Array<double> &trial_coords, mcpele::MC * mc);
    virtual ~CheckHyperSphericalContainer(){};
};

}//namespace bv

#endif//#ifndef _BV_CHECK_HYPER_SPHERICAL_CONTAINER_H
