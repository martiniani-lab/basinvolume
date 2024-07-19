#include "basinvolume/check_hyper_cubic_container.h"
#include "pele/rosenbrock.hpp"

using pele::Array;

namespace bv {

CheckHyperCubicContainer::CheckHyperCubicContainer(pele::Array<double> origin,
                                                   double sidelength,
                                                   size_t ndim,
                                                   const bool use_powered_cosine_sum)
    : m_origin(origin.copy()), m_distance(origin.size(), 0),
      m_halfside(sidelength / 2.0), m_ndim(ndim), m_N((origin.size() / ndim)),
      m_side_length(sidelength), m_use_powered_cosine_sum(use_powered_cosine_sum) {
  std::cout << "m_halfside" << m_halfside << std::endl;
  std::cout << "m_origin.size() " << m_origin.size() << std::endl;
  std::cout << "m_distance.size() " << m_distance.size() << std::endl;
}

bool CheckHyperCubicContainer::conf_test(Array<double> &trial_coords,
                                         mcpele::MCBase *mc) {
  m_distance.assign(m_origin);
  m_distance -= trial_coords;

  for (size_t i = 0; i < m_distance.size(); ++i) {
    double l = m_distance[i];
    bool inside = std::fabs(l) <= m_halfside;
    if (not inside)
      return false;
  }

  return true;
}

pele::Array<double> CheckHyperCubicContainer::gmc_gradient(
    pele::Array<double>& coords, mcpele::MCBase* mc) {
  auto c = coords - m_origin;
  if (not m_use_powered_cosine_sum) {
    pele::Array gradient(c.size(), 0.0);
    size_t min_index = c.size();
    double min_distance = std::numeric_limits<double>::infinity();
    bool positive = true;
    for (size_t i = 0; i < c.size(); ++i) {
      const double positive_distance = m_halfside - c[i];
      const double negative_distance = c[i] + m_halfside;
      assert(positive_distance >= 0.0);
      assert(negative_distance >= 0.0);
      double smaller_distance;
      bool smaller_positive;
      if (positive_distance < negative_distance) {
        smaller_distance = positive_distance;
        smaller_positive = true;
      } else {
        smaller_distance = negative_distance;
        smaller_positive = false;
      }
      if (smaller_distance < min_distance) {
        min_distance = smaller_distance;
        min_index = i;
        positive = smaller_positive;
      }
    }
    assert(min_index < c.size() && min_distance > 0.0 && !std::isinf(min_distance));
    if (positive) {
      gradient[min_index] = -1.0;
    } else {
      gradient[min_index] = 1.0;
    }
    return gradient;
  }
  pele::PoweredCosineSum powered_cosine_sum(c.size(), m_side_length, 0.5, c.size());
  pele::Array<double> gradient(c.size());
  powered_cosine_sum.get_energy_gradient(c, gradient);
  return -gradient;
}

} // namespace bv
