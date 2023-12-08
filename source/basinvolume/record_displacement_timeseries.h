#ifndef _BV_RECORD_DISPLACEMENT_TIMESERIES_H
#define _BV_RECORD_DISPLACEMENT_TIMESERIES_H

#include <vector>

#include "pele/array.hpp"
#include "pele/distance.hpp"

#include "mcpele/mc.h"
#include "mcpele/record_scalar_timeseries.h"

namespace bv {

/*
 * Record displacement time series, measuring every __record_every-th step.
 */

class RecordDisplacementTimeseries : public mcpele::RecordScalarTimeseries {
private:
  void m_get_vec_distance(const pele::Array<double> &x);
  pele::Array<double> m_origin;
  pele::Array<double> m_distance;
  const size_t m_ndim;
  const size_t m_nparticles;
  const bool m_fix_com;

public:
  RecordDisplacementTimeseries(pele::Array<double> origin, const size_t ndim,
                               const size_t niter, const size_t record_every,
                               const bool fix_com = true);
  virtual ~RecordDisplacementTimeseries() {}
  virtual double get_recorded_scalar(pele::Array<double> &coords,
                                     const double energy, const bool accepted,
                                     mcpele::MC *mc);
};

} // namespace bv

#endif // #ifndef _BV_RECORD_DISPLACEMENT_TIMESERIES_H
