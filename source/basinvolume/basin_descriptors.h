/*
* Abstract class that defines a BasinDescriptor for basin volume calculation
* A BasinDescriptor should be defined by 
* 1. A potential
* 2. An identification function that identifies whether two minimized configurations
*    are the same attractor (i.e account for symmetries etc)
* 3. a descent algorithm, could be an optimization algorithm or an ODE solver
*    though the interface is defined in pele and can be phrased as a descent problem
*/

#include <memory>

#include "pele/array.hpp"
#include "pele/optimizer.hpp"

#include "minima_list.h"
#include "pele/vecn.hpp"


// Rename Minimalist to AttractorList



namespace bv {

// Forward declarations for types that may be defined elsewhere



class AbstractGradientBasin {
    protected:
        const std::shared_ptr<pele::BasePotential> _potential;
        const std::shared_ptr<pele::GradientOptimizer> _optimizer; // Defines the flow on the potential
        pele::Array<double> _attractor;

    public:
        AbstractGradientBasin(std::shared_ptr<pele::BasePotential> potential,
        std::shared_ptr<pele::GradientOptimizer> optimizer,
         pele::Array<double> const &attractor);
        virtual ~AbstractGradientBasin() = default;
        virtual bool is_same_attractor(const pele::Array<double> &candidate_attractor) const = 0;
        inline std::shared_ptr<pele::GradientOptimizer> get_optimizer() const { return _optimizer; }
        inline pele::Array<double> get_attractor() const { return _attractor; }
        inline std::shared_ptr<pele::BasePotential> get_potential() const { return _potential; }
      
};

class AbstractBasinCollector : public AbstractGradientBasin {
public:
    AbstractBasinCollector(std::shared_ptr<pele::BasePotential> potential,
                     std::shared_ptr<pele::GradientOptimizer> optimizer,
                     pele::Array<double> const &attractor)
        : AbstractGradientBasin(potential, optimizer, attractor) {}
    virtual ~AbstractBasinCollector() = default;
    virtual bool collect_attractor(const pele::Array<double> &candidate_attractor, std::shared_ptr<pele::GradientOptimizer> optimizer) = 0;
};


template<typename distance_policy>
class PairPotentialBasin : public AbstractBasinCollector {

protected:
    static const size_t _ndim = distance_policy::_ndim;
    // Functions to compute distances
    void _align_coords(pele::Array<double> &coords);
    double _get_d2(pele::Array<double> const &coords);
    double _get_d2_max(pele::Array<double> const &coords);
    const distance_policy _dist_policy;
    size_t _Nnoratt;
    size_t _inoratt;
    size_t _nparticles;
    double _dtol2;
    pele::Array<double> _rattlers;

    // Statistics
    size_t _m_eq_steps;
    bool _collect_minima_list;
    pele::Array<double> _new_minimum;
    MinimaList _minima_list;



public:
    PairPotentialBasin(std::shared_ptr<pele::BasePotential> potential,
    std::shared_ptr<pele::GradientOptimizer> optimizer,
    pele::Array<double> const &attractor);

    virtual ~PairPotentialBasin() = default;
    virtual bool collect_attractor(const pele::Array<double> &candidate_attractor, std::shared_ptr<pele::GradientOptimizer> optimizer) override;
    bool is_same_attractor(const pele::Array<double> &candidate_attractor);
};


/**
 * aligns structures
 */
template <typename distance_policy>
void PairPotentialBasin<distance_policy>::_align_coords(
    pele::Array<double> &coords) {
  /*assert(coords.size() == _origin.size());
  assert(coords.size() == _ndim * _nparticles);*/
  pele::VecN<_ndim, double> dr;

  // measure distance between a non-rattler and its origin
  _dist_policy.get_rij(dr.data(), &coords[_inoratt * _ndim],
                        &_attractor[_inoratt * _ndim]);

// align structures
#pragma omp simd
  for (size_t i = 0; i < _nparticles; ++i) {
    const size_t i1 = i * _ndim;
#pragma unroll
    for (size_t j = 0; j < _ndim; ++j) {
      coords[i1 + j] -= dr[j];
    }
  }
}

/*compute the squared distance from origin after aligning one particle with its
origin this ignores the rattlers completely*/
template <typename distance_policy>
double PairPotentialBasin<distance_policy>::_get_d2(
    pele::Array<double> const &coords) {
  double distance2 = 0;
  pele::VecN<_ndim, double> dr_align;

  // measure distance between a non-rattler and its origin
  _dist_policy.get_rij(dr_align.data(), &coords[_inoratt * _ndim],
                        &_attractor[_inoratt * _ndim]);

  // compute distance between aligned structures
  for (size_t i = 0; i < _nparticles; ++i) {
    const size_t i1 = i * _ndim;
    pele::VecN<_ndim, double> dr, x_aligned;
#pragma unroll
    for (size_t j = 0; j < _ndim; ++j) {
      x_aligned[j] = coords[i1 + j] - dr_align[j];
    }
    _dist_policy.get_rij(dr.data(), x_aligned.data(), &_attractor[i1]);
#pragma unroll
    for (size_t j = 0; j < _ndim; ++j) {
      distance2 += dr[j] * dr[j] * _rattlers[i];
    }
  }

  // avoid taking square roots by returning squared quantities
  return distance2;
}

/* compute maximum of squared distances between a particle and its origin after
aligning one particle with its origin this ignores the rattlers completely */
template <typename distance_policy>
double PairPotentialBasin<distance_policy>::_get_d2_max(
    pele::Array<double> const &coords) {
  double distance2 = 0;
  pele::VecN<_ndim, double> dr_align;

  // measure distance between a non-rattler and its origin
  _dist_policy.get_rij(dr_align.data(), &coords[_inoratt * _ndim],
                        &_attractor[_inoratt * _ndim]);

  // compute distance between aligned structures
  for (size_t i = 0; i < _nparticles; ++i) {
    const size_t i1 = i * _ndim;
    pele::VecN<_ndim, double> dr, x_aligned;
#pragma unroll
    for (size_t j = 0; j < _ndim; ++j) {
      x_aligned[j] = coords[i1 + j] - dr_align[j];
    }
    _dist_policy.get_rij(dr.data(), x_aligned.data(), &_attractor[i1]);
    double current_distance2 = 0;
#pragma unroll
    for (size_t j = 0; j < _ndim; ++j) {
      current_distance2 += dr[j] * dr[j];
    }
    if (_rattlers[i] && current_distance2 > distance2) {
      distance2 = current_distance2;
    }
  }

  // avoid taking square roots by returning squared quantities
  return distance2;
}

template <typename distance_policy>
bool PairPotentialBasin<distance_policy>::is_same_attractor(const pele::Array<double> &candidate_attractor) {
  // Compute maximum squared distance after alignment
  double d2_max = this->_get_d2_max(candidate_attractor);
  return d2_max <= _dtol2;
}

template <typename distance_policy>
bool PairPotentialBasin<distance_policy>::collect_attractor(
    const pele::Array<double> &candidate_attractor, 
    std::shared_ptr<pele::GradientOptimizer> optimizer) {
  
  // Check if it's the same attractor
  bool same_attractor = this->is_same_attractor(candidate_attractor);
  
  // If quench succeeded and it's NOT the same attractor (new minimum found) and we should collect
  if (optimizer->success()) {
    // Compute distance for minima list entry
    double d = sqrt(this->_get_d2(candidate_attractor));
    
    // Align coordinates for storage
    _new_minimum.assign(candidate_attractor);
    this->_align_coords(_new_minimum);
    
    // Insert into minima list with energy from optimizer
    _minima_list.insert_minimum(d, optimizer->get_f(), _new_minimum, _rattlers);
  }
  
  return same_attractor;
}

} // namespace bv















