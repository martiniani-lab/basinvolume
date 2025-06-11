#ifndef _BV_CGD_REFACTORED_H__
#define _BV_CGD_REFACTORED_H__

#include "PyCG_DESCENT/cg_descent_wrapper.hpp"
#include "pele/distance.hpp"
#include "distance_calculator.h"

/* Refactored implementation of the cg_descent algorithm
 * using the shared DistanceCalculator for distance computation
 * with an additional termination condition on the distance
 * from the origin
 */

namespace bv {

template <typename distance_policy>
class BvCGDescentRefactored : public pycgd::CGDescent {
protected:
    DistanceCalculator<distance_policy> _distance_calc;
    double m_d2_max;
    size_t m_maxiter;

public:
    BvCGDescentRefactored(std::shared_ptr<pele::BasePotential> potential,
                         const pele::Array<double> &x0, 
                         pele::Array<double> const &origin,
                         pele::Array<double> const &rattlers,
                         std::shared_ptr<distance_policy> const &dist, 
                         double tol = 1e-4,
                         double dtol = 1e-4, 
                         size_t maxiter = 1e6, 
                         size_t PrintLevel = 0);

    virtual ~BvCGDescentRefactored() = default;
    virtual bool test_convergence(double energy, pele::Array<double> const &x) override;
    
    inline double get_d2_max() { return m_d2_max; }
    inline double get_d2() { 
        return _distance_calc.compute_distance_squared(m_x); 
    }
    
    // Access to the underlying distance calculator for advanced usage
    DistanceCalculator<distance_policy>& get_distance_calculator() { 
        return _distance_calc; 
    }
};

template <typename distance_policy>
BvCGDescentRefactored<distance_policy>::BvCGDescentRefactored(
    std::shared_ptr<pele::BasePotential> potential,
    const pele::Array<double> &x0, 
    pele::Array<double> const &origin,
    pele::Array<double> const &rattlers, 
    std::shared_ptr<distance_policy> const &dist,
    double tol, double dtol, size_t maxiter, size_t PrintLevel)
    : pycgd::CGDescent(potential, x0, tol, PrintLevel),
      _distance_calc(potential, origin, rattlers, dtol, dist),
      m_d2_max(0), m_maxiter(maxiter) {
    
    this->set_maxit(m_maxiter);
    this->set_memory(0); // guarantees that memory is set to 0 irrespective of default settings
}

template <typename distance_policy>
bool BvCGDescentRefactored<distance_policy>::test_convergence(
    double energy, pele::Array<double> const &x) {
    
    m_d2_max = _distance_calc.compute_max_distance_squared(x);
    double dtol2 = _distance_calc.get_dtol() * _distance_calc.get_dtol();
    return (m_d2_max < dtol2);
}

// Factory functions for common distance policies
template <size_t ndim>
std::unique_ptr<BvCGDescentRefactored<pele::cartesian_distance<ndim>>>
make_cartesian_bv_cg_descent(
    std::shared_ptr<pele::BasePotential> potential,
    const pele::Array<double> &x0,
    pele::Array<double> const &origin,
    pele::Array<double> const &rattlers,
    double tol = 1e-4,
    double dtol = 1e-4,
    size_t maxiter = 1e6,
    size_t PrintLevel = 0) {
    
    auto distance_policy = std::make_shared<pele::cartesian_distance<ndim>>();
    return std::make_unique<BvCGDescentRefactored<pele::cartesian_distance<ndim>>>(
        potential, x0, origin, rattlers, distance_policy, 
        tol, dtol, maxiter, PrintLevel);
}

template <size_t ndim>
std::unique_ptr<BvCGDescentRefactored<pele::periodic_distance<ndim>>>
make_periodic_bv_cg_descent(
    std::shared_ptr<pele::BasePotential> potential,
    const pele::Array<double> &x0,
    pele::Array<double> const &origin,
    pele::Array<double> const &boxvec,
    pele::Array<double> const &rattlers,
    double tol = 1e-4,
    double dtol = 1e-4,
    size_t maxiter = 1e6,
    size_t PrintLevel = 0) {
    
    auto distance_policy = std::make_shared<pele::periodic_distance<ndim>>(boxvec);
    return std::make_unique<BvCGDescentRefactored<pele::periodic_distance<ndim>>>(
        potential, x0, origin, rattlers, distance_policy,
        tol, dtol, maxiter, PrintLevel);
}

} // namespace bv

#endif // #ifndef _BV_CGD_REFACTORED_H__ 