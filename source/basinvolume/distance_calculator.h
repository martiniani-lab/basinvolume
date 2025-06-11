#ifndef _BV_DISTANCE_CALCULATOR_H
#define _BV_DISTANCE_CALCULATOR_H

#include <memory>
#include <stdexcept>

#include "pele/distance.hpp"
#include "pele/optimizer.hpp"
#include "pele/vecn.hpp"

namespace bv {

/**
 * Abstract base class for distance calculations with potential association
 * This class encapsulates the common distance calculation logic used across
 * CheckSameMinimum and related classes.
 */
template <typename distance_policy>
class DistanceCalculator {
protected:
    static const size_t _ndim = distance_policy::_ndim;
    
    // Core data members
    std::shared_ptr<pele::BasePotential> _potential;
    pele::Array<double> _origin;
    pele::Array<double> _rattlers;
    double _dtol;
    size_t _nparticles;
    size_t _inoratt;  // index of first non-rattler
    size_t _Nnoratt;  // number of non-rattlers
    
    const std::shared_ptr<distance_policy> _dist_policy;

    // Common distance calculation methods
    inline double _get_d2(pele::Array<double> const &coords);
    inline double _get_d2_max(pele::Array<double> const &coords);
    inline void _align_coords(pele::Array<double> &coords);
    
    // Initialization helper
    void _initialize_rattlers(pele::Array<double> const &rattlers);
    void _validate_inputs(pele::Array<double> const &origin, 
                         pele::Array<double> const &rattlers);

public:
    DistanceCalculator(std::shared_ptr<pele::BasePotential> potential,
                      pele::Array<double> const &origin,
                      pele::Array<double> const &rattlers,
                      double dtol,
                      std::shared_ptr<distance_policy> const &dist);
    
    virtual ~DistanceCalculator() = default;
    
    // Accessors
    double get_dtol() const { return _dtol; }
    size_t get_nparticles() const { return _nparticles; }
    size_t get_ndim() const { return _ndim; }
    std::shared_ptr<pele::BasePotential> get_potential() const { return _potential; }
    pele::Array<double> const& get_origin() const { return _origin; }
    pele::Array<double> const& get_rattlers() const { return _rattlers; }
    
    // Distance calculation interface
    double compute_distance_squared(pele::Array<double> const &coords) { 
        return _get_d2(coords); 
    }
    double compute_max_distance_squared(pele::Array<double> const &coords) { 
        return _get_d2_max(coords); 
    }
    void align_coordinates(pele::Array<double> &coords) { 
        _align_coords(coords); 
    }
};

template <typename distance_policy>
DistanceCalculator<distance_policy>::DistanceCalculator(
    std::shared_ptr<pele::BasePotential> potential,
    pele::Array<double> const &origin,
    pele::Array<double> const &rattlers,
    double dtol,
    std::shared_ptr<distance_policy> const &dist)
    : _potential(potential), _origin(origin.copy()), 
      _rattlers(rattlers.size() / _ndim), _dtol(dtol),
      _nparticles(origin.size() / _ndim), _Nnoratt(0),
      _dist_policy(dist) {
    
    _validate_inputs(origin, rattlers);
    _initialize_rattlers(rattlers);
}

template <typename distance_policy>
void DistanceCalculator<distance_policy>::_validate_inputs(
    pele::Array<double> const &origin, 
    pele::Array<double> const &rattlers) {
    
    if (_dist_policy == nullptr) {
        throw std::runtime_error(
            "DistanceCalculator: distance policy uninitialised");
    }
    if (origin.size() != rattlers.size()) {
        throw std::runtime_error(
            "DistanceCalculator: illegal input: origin vs rattlers size mismatch");
    }
    if (origin.size() % _ndim != 0) {
        throw std::runtime_error(
            "DistanceCalculator: illegal input: origin size not divisible by ndim");
    }
}

template <typename distance_policy>
void DistanceCalculator<distance_policy>::_initialize_rattlers(
    pele::Array<double> const &rattlers) {
    
    bool no_stable_yet = true;
    for (size_t i = 0; i < _rattlers.size(); ++i) {
        _rattlers[i] = rattlers[i * _ndim];
        if (no_stable_yet && _rattlers[i] != 0) {
            _inoratt = i;
            no_stable_yet = false;
        }
        _Nnoratt += _rattlers[i];
    }
}

template <typename distance_policy>
void DistanceCalculator<distance_policy>::_align_coords(
    pele::Array<double> &coords) {
    
    pele::VecN<_ndim, double> dr;
    
    // measure distance between a non-rattler and its origin
    _dist_policy->get_rij(dr.data(), &coords[_inoratt * _ndim],
                          &_origin[_inoratt * _ndim]);

    // align structures
    #pragma simd
    for (size_t i = 0; i < _nparticles; ++i) {
        const size_t i1 = i * _ndim;
        #pragma unroll
        for (size_t j = 0; j < _ndim; ++j) {
            coords[i1 + j] -= dr[j];
        }
    }
}

template <typename distance_policy>
double DistanceCalculator<distance_policy>::_get_d2(
    pele::Array<double> const &coords) {
    
    double distance2 = 0;
    pele::VecN<_ndim, double> dr_align;

    // measure distance between a non-rattler and its origin
    _dist_policy->get_rij(dr_align.data(), &coords[_inoratt * _ndim],
                          &_origin[_inoratt * _ndim]);

    // compute distance between aligned structures
    for (size_t i = 0; i < _nparticles; ++i) {
        const size_t i1 = i * _ndim;
        pele::VecN<_ndim, double> dr, x_aligned;
        #pragma unroll
        for (size_t j = 0; j < _ndim; ++j) {
            x_aligned[j] = coords[i1 + j] - dr_align[j];
        }
        _dist_policy->get_rij(dr.data(), x_aligned.data(), &_origin[i1]);
        #pragma unroll
        for (size_t j = 0; j < _ndim; ++j) {
            distance2 += dr[j] * dr[j] * _rattlers[i];
        }
    }

    return distance2;
}

template <typename distance_policy>
double DistanceCalculator<distance_policy>::_get_d2_max(
    pele::Array<double> const &coords) {
    
    double distance2 = 0;
    pele::VecN<_ndim, double> dr_align;

    // measure distance between a non-rattler and its origin
    _dist_policy->get_rij(dr_align.data(), &coords[_inoratt * _ndim],
                          &_origin[_inoratt * _ndim]);

    // compute distance between aligned structures
    for (size_t i = 0; i < _nparticles; ++i) {
        const size_t i1 = i * _ndim;
        pele::VecN<_ndim, double> dr, x_aligned;
        #pragma unroll
        for (size_t j = 0; j < _ndim; ++j) {
            x_aligned[j] = coords[i1 + j] - dr_align[j];
        }
        _dist_policy->get_rij(dr.data(), x_aligned.data(), &_origin[i1]);
        double current_distance2 = 0;
        #pragma unroll
        for (size_t j = 0; j < _ndim; ++j) {
            current_distance2 += dr[j] * dr[j];
        }
        if (_rattlers[i] && current_distance2 > distance2) {
            distance2 = current_distance2;
        }
    }

    return distance2;
}

// Convenience typedefs for common distance policies
template <size_t ndim>
using CartesianDistanceCalculator = DistanceCalculator<pele::cartesian_distance<ndim>>;

template <size_t ndim>
using PeriodicDistanceCalculator = DistanceCalculator<pele::periodic_distance<ndim>>;

} // namespace bv

#endif // _BV_DISTANCE_CALCULATOR_H 