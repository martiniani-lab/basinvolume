#ifndef _BV_CHECK_HYPER_SPHERICAL_CONTAINER_H
#define _BV_CHECK_HYPER_SPHERICAL_CONTAINER_H

#include <memory>

#include "pele/array.hpp"
#include "pele/distance.hpp"
#include "pele/optimizer.hpp"

#include "mcpele/mc.h"

namespace bv {

class CheckHyperSphericalContainer : public mcpele::ConfTest {
protected:
  void _get_vec_distance(const pele::Array<double> &coords);
  pele::Array<double> m_origin, m_distance;
  double m_radius2;
  size_t m_ndim, m_N;

public:
  CheckHyperSphericalContainer(pele::Array<double> origin, double radius,
                               size_t ndim);
  virtual bool conf_test(pele::Array<double> &trial_coords, mcpele::MC *mc);
  virtual ~CheckHyperSphericalContainer(){};
};

} // namespace bv

#endif //#ifndef _BV_CHECK_HYPER_SPHERICAL_CONTAINER_H
