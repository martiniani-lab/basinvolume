#ifndef _BV_CHECK_HYPER_CUBIC_CONTAINER_H
#define _BV_CHECK_HYPER_CUBIC_CONTAINER_H

#include <memory>

#include "pele/array.h"
#include "pele/optimizer.h"
#include "pele/distance.h"

#include "mcpele/mc.h"

namespace bv{

class CheckHyperCubicContainer:public mcpele::ConfTest{
protected:
    void _get_vec_distance(const pele::Array<double>& coords);
    pele::Array<double> m_origin, m_distance;
    double m_halfside;
    size_t m_ndim,m_N;
public:
    CheckHyperCubicContainer(pele::Array<double> origin, double sidelength, size_t ndim);
    virtual bool conf_test(pele::Array<double> &trial_coords, mcpele::MC * mc);
    virtual ~CheckHyperCubicContainer(){};
};

}//namespace bv

#endif//#ifndef _BV_CHECK_HYPER_CUBIC_CONTAINER_H
