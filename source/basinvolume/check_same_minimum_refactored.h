#ifndef _BV_CHECK_SAME_MINIMUM_REFACTORED_H
#define _BV_CHECK_SAME_MINIMUM_REFACTORED_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "pele/distance.hpp"
#include "pele/optimizer.hpp"
#include "pele/vecn.hpp"

#include "mcpele/histogram.h"
#include "mcpele/mc.h"

#include "check_same_minimum.h"
#include "distance_calculator.h"
#include "convergence_test.h"
#include "minima_list.h"

namespace bv {



/*
 * 
*/
template <typename distance_policy, class OPT_T = pele::GradientOptimizer>
class CheckSameMinimumRefactored : public CheckSameMinimumInterface {
protected:
    // Calculates distance based on distance policy
    DistanceCalculator<distance_policy> _distance_calc;
    
    std::shared_ptr<OPT_T> _optimizer;
    pele::Array<double> _new_minimum;
    double _d;
    double _dmax;

    mcpele::Moments m_failed_quench_frac;
    
    // Convergence test classes
    bool _perform_convergence_test;
    convergence_test<OPT_T> _conv_test;
    
    // Minima list
    size_t m_eqsteps;
    bool _collect_minima_list;
    MinimaList _minima_list;

    // Core methods using the distance calculator
    inline void _check_convergence(pele::Array<double> const &quenched_coords);
    bool _quench(pele::Array<double> &trial_coords);

public:
    CheckSameMinimumRefactored(std::shared_ptr<OPT_T> optimizer,
                              std::shared_ptr<pele::BasePotential> potential,
                              pele::Array<double> const &origin, 
                              pele::Array<double> const &rattlers,
                              double dtol, 
                              const size_t eqsteps,
                              std::shared_ptr<distance_policy> const &dist,
                              const bool perform_convergence_test,
                              const bool collect_minima_list);

    virtual bool conf_test(pele::Array<double> &trial_coords, mcpele::MC *mc);
    virtual ~CheckSameMinimumRefactored() = default;
    
    // Accessors
    double get_distance() const { return _d; }
    bool perform_convergence_test() const { return _perform_convergence_test; }
    bool collect_minima_list() const { return _collect_minima_list; }
    
    // Forwarding minima database information to the outside
    virtual size_t ml_nr_distinct_minima() const {
        return _minima_list.nr_distinct_minima();
    }
    virtual double get_failed_quench_frac() const {
        return m_failed_quench_frac.mean();
    }
    virtual pele::Array<Minimum *> get_array_of_minima();
};

template <typename distance_policy, class OPT_T>
CheckSameMinimumRefactored<distance_policy, OPT_T>::CheckSameMinimumRefactored(
    std::shared_ptr<OPT_T> optimizer,
    std::shared_ptr<pele::BasePotential> potential,
    pele::Array<double> const &origin,
    pele::Array<double> const &rattlers,
    double dtol,
    const size_t eqsteps,
    std::shared_ptr<distance_policy> const &dist,
    const bool perform_convergence_test,
    const bool collect_minima_list)
    : _distance_calc(DistanceCalculator<distance_policy>(
          potential, origin, rattlers, dtol, dist)),
      _optimizer(optimizer),
      _new_minimum(origin.size()), _d(0), _dmax(0),
      _perform_convergence_test(perform_convergence_test),
      _conv_test(30, 1e-10, _optimizer->get_tol(), 0.1, 
                 const_cast<pele::Array<double>&>(origin), potential,
                 distance_policy::_ndim),
      m_eqsteps(eqsteps), _collect_minima_list(collect_minima_list),
      _minima_list(dtol * sqrt(origin.size()), _optimizer->get_tol(), dtol) {
}

template <typename distance_policy, class OPT_T>
pele::Array<Minimum *>
CheckSameMinimumRefactored<distance_policy, OPT_T>::get_array_of_minima() {
    pele::Array<Minimum *> minima(_minima_list.nr_distinct_minima());
    size_t i = 0;
    for (auto &m : _minima_list) {
        minima[i++] = &m;
    }
    return minima;
}

template <typename distance_policy, class OPT_T>
void CheckSameMinimumRefactored<distance_policy, OPT_T>::_check_convergence(
    pele::Array<double> const &quenched_coords) {
    _conv_test.check_convergence(quenched_coords, _optimizer);
}

template <typename distance_policy, class OPT_T>
bool CheckSameMinimumRefactored<distance_policy, OPT_T>::_quench(
    pele::Array<double> &trial_coords) {
    
    _optimizer->reset(trial_coords);
    bool success = true;
    double d2max = _distance_calc.compute_max_distance_squared(_optimizer->get_x());
    double dtol2 = _distance_calc.get_dtol() * _distance_calc.get_dtol();
    const size_t opt_maxiter = _optimizer->get_maxiter();

    // Optimization loop using distance calculator
    while (d2max > dtol2 &&
           static_cast<size_t>(_optimizer->get_niter()) < opt_maxiter) {
        if (_optimizer->stop_criterion_satisfied()) {
            success = false;
            break;
        }
        _optimizer->one_iteration();
        d2max = _distance_calc.compute_max_distance_squared(_optimizer->get_x());
    }
    
    // Update distance metrics using the calculator
    _dmax = sqrt(d2max);
    _d = sqrt(_distance_calc.compute_distance_squared(_optimizer->get_x()));

    return success;
}

template <typename distance_policy, class OPT_T>
bool CheckSameMinimumRefactored<distance_policy, OPT_T>::conf_test(
    pele::Array<double> &trial_coords, mcpele::MC *mc) {
    
    bool quench_success;
    bool optimizer_converged = this->_quench(trial_coords);

    quench_success = _dmax <= _distance_calc.get_dtol() || !optimizer_converged;

    // Add number of energy evaluations to mc eval count
    const size_t nfev = _optimizer->get_nfev();
    mc->m_neval += nfev;

    if (_perform_convergence_test) {
        this->_check_convergence(_optimizer->get_x());
    }

    m_failed_quench_frac.update(!quench_success);
    if (!quench_success) {
        return false;
    }

    if (!optimizer_converged) {
        // Save new minimum if collecting minima list
        if (_collect_minima_list && mc->get_iterations_count() > m_eqsteps) {
            _new_minimum.assign(_optimizer->get_x());
            _distance_calc.align_coordinates(_new_minimum);
            _minima_list.insert_minimum(_d, _optimizer->get_f(), _new_minimum,
                                       _distance_calc.get_rattlers());
        }
        return false;
    } else {
        return true;
    }
}

// Factory functions for common cases
template <size_t ndim>
std::unique_ptr<CheckSameMinimumRefactored<pele::cartesian_distance<ndim>>>
make_cartesian_check_same_minimum(
    std::shared_ptr<pele::GradientOptimizer> optimizer,
    std::shared_ptr<pele::BasePotential> potential,
    pele::Array<double> const &origin,
    pele::Array<double> const &rattlers,
    double dtol,
    size_t eqsteps = 0,
    bool perform_convergence_test = false,
    bool collect_minima_list = false) {
    
    auto distance_policy = std::make_shared<pele::cartesian_distance<ndim>>();
    return std::make_unique<CheckSameMinimumRefactored<pele::cartesian_distance<ndim>>>(
        optimizer, potential, origin, rattlers, dtol, eqsteps,
        distance_policy, perform_convergence_test, collect_minima_list);
}

template <size_t ndim>
std::unique_ptr<CheckSameMinimumRefactored<pele::periodic_distance<ndim>>>
make_periodic_check_same_minimum(
    std::shared_ptr<pele::GradientOptimizer> optimizer,
    std::shared_ptr<pele::BasePotential> potential,
    pele::Array<double> const &origin,
    pele::Array<double> const &boxvec,
    pele::Array<double> const &rattlers,
    double dtol,
    size_t eqsteps = 0,
    bool perform_convergence_test = false,
    bool collect_minima_list = false) {
    
    auto distance_policy = std::make_shared<pele::periodic_distance<ndim>>(boxvec);
    return std::make_unique<CheckSameMinimumRefactored<pele::periodic_distance<ndim>>>(
        optimizer, potential, origin, rattlers, dtol, eqsteps,
        distance_policy, perform_convergence_test, collect_minima_list);
}

} // namespace bv

#endif // _BV_CHECK_SAME_MINIMUM_REFACTORED_H 