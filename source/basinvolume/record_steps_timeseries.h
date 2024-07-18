#ifndef _BV_RECORD_STEPS_TIMESERIES_H__
#define _BV_RECORD_STEPS_TIMESERIES_H__

#include <memory>
#include <vector>

#include "pele/array.hpp"
#include "pele/distance.hpp"

#include "mcpele/mc.h"
#include "mcpele/record_scalar_timeseries.h"

namespace bv {

template <size_t ndim>
class RecordStepsTimeseries : public mcpele::RecordScalarTimeseries {
protected:
  inline pele::Array<double>
  m_align_coords(pele::Array<double> coords); // aligns structure to old coords
  inline double
  m_get_d(pele::Array<double> coords); // measures distance from old coords
  std::shared_ptr<pele::DistanceInterface> m_dist_policy;
  pele::Array<double> m_origin, m_old_coords, m_rattlers, m_distance;
  size_t m_inoratt, m_nparticles;

public:
  RecordStepsTimeseries(pele::Array<double> origin,
                        pele::Array<double> rattlers, const size_t niter,
                        const size_t record_every)
      : RecordScalarTimeseries(niter, record_every),
        m_dist_policy(std::make_shared<pele::CartesianDistanceWrapper<ndim>>()),
        m_origin(origin.copy()), m_old_coords(origin.copy()),
        m_rattlers(rattlers.copy()), m_distance(origin.size(), 0),
        m_nparticles(origin.size() / ndim) {
    for (size_t i = 0; i < m_old_coords.size(); i += ndim) {
      if (m_rattlers[i] != 0) {
        m_inoratt = i / ndim;
        break;
      }
    }
  }

  virtual ~RecordStepsTimeseries() {}
  virtual double get_recorded_scalar(pele::Array<double> &coords,
                                     const double energy, const bool accepted,
                                     mcpele::MCBase *mc) {
    if (!accepted) {
      return 0.;
    }
    double d = this->m_get_d(coords.copy());
    m_old_coords.assign(coords);
    return d;
  }
};

/*compute distance from origin after aligning two particles
this ignores the rattlers completely and returns msd*/
template <size_t ndim>
double RecordStepsTimeseries<ndim>::m_get_d(pele::Array<double> coords) {
  pele::Array<double> dr(ndim);
  pele::Array<double> aligned_coords =
      this->m_align_coords(coords); // this line modifies coords

  // compute distance between aligned structures
  for (size_t i = 0; i < m_nparticles; ++i) {
    const size_t i1 = i * ndim;
    m_dist_policy->get_rij(dr.data(), &aligned_coords[i1], &m_old_coords[i1]);
    for (size_t j = 0; j < ndim; ++j) {
      m_distance[i1 + j] =
          dr[j]; // * m_rattlers[i1 + j] if want to ignore rattlers
    }
  }

  return pele::norm(m_distance);
}

/**
 * aligns structure new coords to old coords
 */
template <size_t ndim>
pele::Array<double>
RecordStepsTimeseries<ndim>::m_align_coords(pele::Array<double> coords) {
  /*assert(coords.size() == _origin.size());
  assert(coords.size() == _ndim * _nparticles);*/
  pele::Array<double> dr(ndim);

  // measure distance between two non rattlers
  m_dist_policy->get_rij(dr.data(), &coords[m_inoratt * ndim],
                         &m_old_coords[m_inoratt * ndim]);

  // align structures
  for (size_t i = 0; i < m_nparticles; ++i) {
    const size_t i1 = i * ndim;
    for (size_t j = 0; j < ndim; ++j) {
      coords[i1 + j] -= dr[j];
    }
  }
  return coords;
}

} // namespace bv

#endif // #ifndef _MCPELE_RECORD_ENERGY_TIMESERIES_H__
