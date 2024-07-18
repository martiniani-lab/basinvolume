#ifndef _BV_CHECK_OVERLAP_H
#define _BV_CHECK_OVERLAP_H

#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "pele/array.hpp"
#include "pele/distance.hpp"

#include "mcpele/mc.h"

#include "frozen_wrappers.h"

namespace bv {

/**
 * Test for overlap of the hard sphere cores
 */
template <typename DIST_POL> class CheckOverlap : public mcpele::ConfTest {
protected:
  const static size_t m_ndim = DIST_POL::_ndim;
  pele::Array<double> m_hs_radii;
  const size_t m_nparticles;
  std::shared_ptr<DIST_POL> m_dist;

public:
  virtual ~CheckOverlap(){};
  CheckOverlap(pele::Array<double> hs_radii, std::shared_ptr<DIST_POL> dist)
      : m_hs_radii(hs_radii.copy()), m_nparticles(m_hs_radii.size()),
        m_dist(dist) {
    if (m_dist == NULL) {
      throw std::runtime_error("CheckOverlap: distance uninitialised");
    }
    if (hs_radii.size() == 0) {
      throw std::runtime_error("CheckOverlap: illegal input: hs_radii");
    }
    static_assert(DIST_POL::_ndim > 0,
                  "CheckOverlap: illegal input: distance policy");
  }
  bool conf_test(pele::Array<double> &trial_coords, mcpele::MCBase *mc) {
    if (trial_coords.size() % m_ndim) {
      throw std::runtime_error("CheckOverlap::conf_test: illegal input");
    }
    if (trial_coords.size() / m_ndim != m_nparticles) {
      throw std::runtime_error("CheckOverlap::conf_test: illegal input");
    }
    double dr[m_ndim];
    // std::cout << trial_coords  << "\n";
    const std::vector<size_t> changed_atoms = mc->get_changed_atoms();
    if (changed_atoms.size() == 0) {
      for (size_t i = 0; i < m_nparticles; ++i) {
        const size_t i1 = m_ndim * i;
        for (size_t j = i + 1; j < m_nparticles; ++j) {
          const size_t j1 = m_ndim * j;
          m_dist->get_rij(dr, &trial_coords[i1], &trial_coords[j1]);
          const double dij2 =
              std::inner_product(dr, dr + m_ndim, dr, double(0));
          const double tmp = (m_hs_radii[i] + m_hs_radii[j]);
          if (dij2 < tmp * tmp) {
            return false;
          }
        }
      }
    } else {
      for (const size_t i : changed_atoms) {
        const size_t i1 = m_ndim * i;
        for (size_t j = 0; j < m_nparticles; ++j) {
          if (i != j) {
            const size_t j1 = m_ndim * j;
            m_dist->get_rij(dr, &trial_coords[i1], &trial_coords[j1]);
            const double dij2 =
                std::inner_product(dr, dr + m_ndim, dr, double(0));
            const double tmp = (m_hs_radii[i] + m_hs_radii[j]);
            if (dij2 < tmp * tmp) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }
};

template <size_t ndim>
class CheckOverlapPeriodic
    : public CheckOverlap<pele::periodic_distance<ndim>> {
public:
  CheckOverlapPeriodic(pele::Array<double> hs_radii, pele::Array<double> boxvec)
      : CheckOverlap<pele::periodic_distance<ndim>>(
            hs_radii, std::make_shared<pele::periodic_distance<ndim>>(boxvec)) {
  }
};

template <size_t ndim>
class CheckOverlapPeriodicFrozen
    : public ConfTestFrozenWrapper<CheckOverlapPeriodic<ndim>> {
public:
  CheckOverlapPeriodicFrozen(pele::Array<double> hs_radii,
                             pele::Array<double> boxvec,
                             pele::Array<double> &reference_coords,
                             pele::Array<size_t> &frozen_dof)
      : ConfTestFrozenWrapper<CheckOverlapPeriodic<ndim>>(
            std::make_shared<CheckOverlapPeriodic<ndim>>(hs_radii, boxvec),
            reference_coords.copy(), frozen_dof.copy()) {}
};

template <size_t ndim>
class CheckOverlapCartesian
    : public CheckOverlap<pele::cartesian_distance<ndim>> {
public:
  CheckOverlapCartesian(pele::Array<double> hs_radii)
      : CheckOverlap<pele::cartesian_distance<ndim>>(
            hs_radii, std::make_shared<pele::cartesian_distance<ndim>>()) {}
};

template <size_t ndim>
class CheckOverlapCartesianFrozen
    : public ConfTestFrozenWrapper<CheckOverlapCartesian<ndim>> {
public:
  CheckOverlapCartesianFrozen(pele::Array<double> hs_radii,
                              pele::Array<double> &reference_coords,
                              pele::Array<size_t> &frozen_dof)
      : ConfTestFrozenWrapper<CheckOverlapCartesian<ndim>>(
            std::make_shared<CheckOverlapCartesian<ndim>>(hs_radii),
            reference_coords.copy(), frozen_dof.copy()) {}
};

template <size_t ndim>
class CheckOverlapLeesEdwards
    : public CheckOverlap<pele::leesedwards_distance<ndim>> {
public:
  CheckOverlapLeesEdwards(pele::Array<double> hs_radii,
                          pele::Array<double> boxvec, const double shear)
      : CheckOverlap<pele::leesedwards_distance<ndim>>(
            hs_radii,
            std::make_shared<pele::leesedwards_distance<ndim>>(boxvec, shear)) {
  }
};

template <size_t ndim>
class CheckOverlapLeesEdwardsFrozen
    : public ConfTestFrozenWrapper<CheckOverlapLeesEdwards<ndim>> {
public:
  CheckOverlapLeesEdwardsFrozen(pele::Array<double> hs_radii,
                                pele::Array<double> boxvec,
                                pele::Array<double> &reference_coords,
                                pele::Array<size_t> &frozen_dof,
                                const double shear)
      : ConfTestFrozenWrapper<CheckOverlapLeesEdwards<ndim>>(
            std::make_shared<CheckOverlapLeesEdwards<ndim>>(hs_radii, boxvec,
                                                            shear),
            reference_coords.copy(), frozen_dof.copy()) {}
};

} // namespace bv

#endif // #ifndef _BV_CHECK_OVERLAP_H
