#ifndef _BV_CGD_H__
#define _BV_CGD_H__

#include "PyCG_DESCENT/cg_descent_wrapper.hpp"
#include "pele/distance.hpp"

/* this is an implementation of the cg_descent algorithm
 * with an additional termination condition on the distance
 * from the origin
 */

namespace bv {

template <typename distance_policy>
class BvCGDescent : public pycgd::CGDescent {
protected:
  static const size_t m_ndim = distance_policy::_ndim;
  pele::Array<double> m_origin;
  pele::Array<double> m_rattlers;
  double m_dtol2, m_d2_max;
  size_t m_nparticles, m_maxiter, m_inoratt, m_Nnoratt;
  const std::shared_ptr<distance_policy> m_dist_policy;
  double m_get_d2(pele::Array<double> const &coords);
  double m_get_d2_max(pele::Array<double> const &coords);

public:
  BvCGDescent(std::shared_ptr<pele::BasePotential> potential,
              const pele::Array<double> &x0, pele::Array<double> &origin,
              pele::Array<double> &rattlers,
              std::shared_ptr<distance_policy> const &dist, double tol = 1e-4,
              double dtol = 1e-4, size_t maxiter = 1e6, size_t PrintLevel = 0);

  virtual ~BvCGDescent() {}
  virtual bool test_convergence(double energy, pele::Array<double> const &x);
  inline double get_d2_max() { return m_d2_max; }
  inline double get_d2() { return m_get_d2(m_x); }
};

template <typename distance_policy>
BvCGDescent<distance_policy>::BvCGDescent(
    std::shared_ptr<pele::BasePotential> potential,
    const pele::Array<double> &x0, pele::Array<double> &origin,
    pele::Array<double> &rattlers, std::shared_ptr<distance_policy> const &dist,
    double tol, double dtol, size_t maxiter, size_t PrintLevel)
    : pycgd::CGDescent(potential, x0, tol, PrintLevel), m_origin(origin.copy()),
      m_rattlers(rattlers.size() / m_ndim), m_dtol2(dtol * dtol), m_d2_max(0),
      m_nparticles(origin.size() / m_ndim), m_maxiter(maxiter), m_Nnoratt(0),
      m_dist_policy(dist) {
  if (m_dist_policy == NULL) {
    throw std::runtime_error(
        "BvCGDescent::BvCGDescent: distance policy uninitialised");
  }
  if (m_origin.size() != rattlers.size()) {
    throw std::runtime_error(
        "BvCGDescent::BvCGDescent: illegal input: origin vs rattlers");
  }
  if (m_origin.size() % m_ndim) {
    throw std::runtime_error(
        "BvCGDescent::BvCGDescent: illegal input: origin vs boxdimension");
  }
  bool no_stable_yet = true;
  for (size_t i = 0; i < m_rattlers.size(); ++i) {
    m_rattlers[i] = rattlers[i * m_ndim];
    if (no_stable_yet && m_rattlers[i] != 0) {
      m_inoratt = i;
      no_stable_yet = false;
    }
    m_Nnoratt += m_rattlers[i];
  }
  this->set_maxit(m_maxiter);
  this->set_memory(
      0); // guarantees that memory is set to 0 irrespective of default settings
}

template <typename distance_policy>
double
BvCGDescent<distance_policy>::m_get_d2(pele::Array<double> const &coords) {
  double distance2 = 0;
  pele::VecN<m_ndim, double> dr_align;

  // measure distance between a non-rattler and its origin
  m_dist_policy->get_rij(dr_align.data(), &coords[m_inoratt * m_ndim],
                         &m_origin[m_inoratt * m_ndim]);

  // compute distance between aligned structures
  for (size_t i = 0; i < m_nparticles; ++i) {
    const size_t i1 = i * m_ndim;
    pele::VecN<m_ndim, double> dr, x_aligned;
#pragma unroll
    for (size_t j = 0; j < m_ndim; ++j) {
      x_aligned[j] = coords[i1 + j] - dr_align[j];
    }
    m_dist_policy->get_rij(dr.data(), x_aligned.data(), &m_origin[i1]);
#pragma unroll
    for (size_t j = 0; j < m_ndim; ++j) {
      distance2 += dr[j] * dr[j] * m_rattlers[i];
    }
  }

  // avoid taking square roots by returning squared quantities
  return distance2;
}

template <typename distance_policy>
double
BvCGDescent<distance_policy>::m_get_d2_max(pele::Array<double> const &coords) {
  double distance2 = 0;
  pele::VecN<m_ndim, double> dr_align;

  // measure distance between a non-rattler and its origin
  m_dist_policy->get_rij(dr_align.data(), &coords[m_inoratt * m_ndim],
                         &m_origin[m_inoratt * m_ndim]);

  // compute distance between aligned structures
  for (size_t i = 0; i < m_nparticles; ++i) {
    const size_t i1 = i * m_ndim;
    pele::VecN<m_ndim, double> dr, x_aligned;
#pragma unroll
    for (size_t j = 0; j < m_ndim; ++j) {
      x_aligned[j] = coords[i1 + j] - dr_align[j];
    }
    m_dist_policy->get_rij(dr.data(), x_aligned.data(), &m_origin[i1]);
    double current_distance2 = 0;
#pragma unroll
    for (size_t j = 0; j < m_ndim; ++j) {
      current_distance2 += dr[j] * dr[j];
    }
    if (m_rattlers[i] && current_distance2 > distance2) {
      distance2 = current_distance2;
    }
  }

  // avoid taking square roots by returning squared quantities
  return distance2;
}

template <typename distance_policy>
bool BvCGDescent<distance_policy>::test_convergence(
    double energy, pele::Array<double> const &x) {
  m_d2_max = this->m_get_d2_max(x);
  return (m_d2_max < m_dtol2);
}

} // namespace bv

#endif // #ifndef _BV_CGD_H__
