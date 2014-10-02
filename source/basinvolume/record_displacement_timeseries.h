#ifndef _BV_RECORD_DISPLACEMENT_TIMESERIES_H
#define _BV_RECORD_DISPLACEMENT_TIMESERIES_H

#include <vector>

#include "pele/array.h"
#include "pele/distance.h"

#include "mcpele/mc.h"

namespace bv{

/*
 * Record displacement time series, measuring every __record_every-th step.
 */

class RecordDisplacementTimeseries : public mcpele::RecordScalarTimeseries{
    private:
        void m_get_vec_distance(const pele::Array<double>& x);
        pele::Array<double> m_origin, m_distance;
        const size_t m_ndim, m_nparticles;
    public:
        RecordDisplacementTimeseries(pele::Array<double> origin, const size_t ndim, const size_t niter, const size_t record_every);
        virtual ~RecordDisplacementTimeseries(){}
        virtual double get_recorded_scalar(pele::Array<double> &coords, const double energy, const bool accepted, mcpele::MC* mc);
};

}//namespace bv

#endif//#ifndef _BV_RECORD_DISPLACEMENT_TIMESERIES_H
