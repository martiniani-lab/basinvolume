#ifndef _BV_RECORD_DISPLACEMENT_TIMESERIES_H
#define _BV_RECORD_DISPLACEMENT_TIMESERIES_H

#include <vector>

#include "pele/array.h"
#include "pele/distance.h"

#include "mcpele/actions.h"

namespace bv{

/*
 * Record displacement time series, measuring every __record_every-th step.
 */

class RecordDisplacementTimeseries : public mcpele::Action{
    private:
        void _record_displacement_value(const double dx);
        void _get_vec_distance(const pele::Array<double>& x);
        pele::Array<double> _origin, _distance;
        const size_t _ndim, _nparticles, _ts_niter, _record_every;
        std::vector<double> _time_series;
    public:
        RecordDisplacementTimeseries(pele::Array<double> origin, const size_t ndim, const size_t niter, const size_t record_every);
        virtual ~RecordDisplacementTimeseries(){}
        virtual void action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc);
        pele::Array<double> get_time_series();
        void clear(){_time_series.clear();}
};

}//namespace bv

#endif//#ifndef _BV_RECORD_DISPLACEMENT_TIMESERIES_H
