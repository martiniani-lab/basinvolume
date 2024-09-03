#include "basinvolume/check_hyper_cubic_container.h"

using pele::Array;

namespace bv {

CheckHyperCubicContainer::CheckHyperCubicContainer(const Array<double> origin,
                                                   const Array<double> sidelengths,
                                                   const size_t ndim,
                                                   const bool use_powered_cosine_sum,
                                                   const Array<double> powered_cosine_sum_prefactors)
    : m_origin(origin.copy()), m_distance(origin.size(), 0),
      m_ndim(ndim), m_N((origin.size() / ndim)),
      m_use_powered_cosine_sum(use_powered_cosine_sum),
      m_powered_cosine_sum(origin.size(), sidelengths.copy(), powered_cosine_sum_prefactors.copy(), 0.5, 1.0) {
  if (sidelengths.size() != origin.size()) {
    throw std::runtime_error("sidelengths.size() != origin.size()");
  }
  if (powered_cosine_sum_prefactors.size() != origin.size()) {
    throw std::runtime_error("powered_cosine_sum_prefactors.size() != origin.size()");
  }
  m_halfsides = Array<double>(origin.size());
  for (size_t i = 0; i < m_halfsides.size(); ++i) {
    m_halfsides[i] = sidelengths[i] / 2.0;
  }
  std::cout << "m_halfsides " << m_halfsides;
  std::cout << "m_origin.size() " << m_origin.size() << std::endl;
  std::cout << "m_distance.size() " << m_distance.size() << std::endl;
}

bool CheckHyperCubicContainer::conf_test(Array<double> &trial_coords,
                                         mcpele::MCBase *mc) {
  m_distance.assign(m_origin);
  m_distance -= trial_coords;

  for (size_t i = 0; i < m_distance.size(); ++i) {
    const double l = m_distance[i];
    bool inside = std::fabs(l) <= m_halfsides[i];
    if (not inside)
      return false;
  }
  return true;
}

Array<double> CheckHyperCubicContainer::gmc_gradient(Array<double>& coords, mcpele::MCBase* mc) {
  auto c = coords - m_origin;
  if (not m_use_powered_cosine_sum) {
    Array gradient(c.size(), 0.0);
    size_t min_index = c.size();
    double min_distance = std::numeric_limits<double>::infinity();
    bool positive = true;
    for (size_t i = 0; i < c.size(); ++i) {
      const double positive_distance = m_halfsides[i] - c[i];
      const double negative_distance = c[i] + m_halfsides[i];
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
  Array<double> gradient(c.size());
  m_powered_cosine_sum.get_energy_gradient(c, gradient);
  return -gradient;
}

Array<double> CheckHyperCubicContainer::gmc_hessian(Array<double>& coords, mcpele::MCBase* mc) {
  auto c = coords - m_origin;
  if (not m_use_powered_cosine_sum) {
    throw std::runtime_error("CheckHyperCubicContainer::gmc_hessian not implemented for m_use_powered_cosine_sum == false");
  }
  Array<double> hessian(c.size() * c.size());
  m_powered_cosine_sum.get_hessian(c, hessian);
  return hessian;
}

} // namespace bv
