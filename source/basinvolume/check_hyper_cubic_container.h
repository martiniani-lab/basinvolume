#ifndef _BV_CHECK_HYPER_CUBIC_CONTAINER_H
#define _BV_CHECK_HYPER_CUBIC_CONTAINER_H

#include <memory>

#include "pele/array.hpp"
#include "pele/rosenbrock.hpp"
#include "pele/distance.hpp"
#include "pele/optimizer.hpp"

#include "mcpele/mc.h"
#include "mcpele/gmc.h"

namespace bv {

class CheckHyperCubicContainer : public mcpele::GMCConfTest {
protected:
  void _get_vec_distance(const pele::Array<double> &coords);
  pele::Array<double> m_origin, m_distance;
  const size_t m_ndim, m_N;
  pele::Array<double> m_halfsides;
  const bool m_use_powered_cosine_sum;
  pele::PoweredCosineSum m_powered_cosine_sum;

public:
  CheckHyperCubicContainer(pele::Array<double> origin, pele::Array<double> sidelengths,
                           size_t ndim, bool use_powered_cosine_sum,
                           pele::Array<double> powered_cosine_sum_prefactors);
  virtual bool conf_test(pele::Array<double> &trial_coords, mcpele::MCBase *mc);
  virtual ~CheckHyperCubicContainer(){};
  pele::Array<double> gmc_gradient(pele::Array<double> &coords, mcpele::MCBase *mc) override;
  pele::Array<double> gmc_hessian(pele::Array<double> &coords, mcpele::MCBase *mc) override;
};

} // namespace bv

#endif //#ifndef _BV_CHECK_HYPER_CUBIC_CONTAINER_H
