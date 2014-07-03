#ifndef _BV_RECORD_DISP2_HISTOGRAM_H
#define _BV_RECORD_DISP2_HISTOGRAM_H

#include "pele/array.h"

#include "mcpele/histogram.h"
#include "mcpele/actions.h"

namespace bv{

/*
 * Record displacement square histogram
*/

class RecordDisp2Histogram : public mcpele::RecordEnergyHistogram {
protected:
    inline void _get_vec_distance(const pele::Array<double>& x);
    pele::Array<double> _origin, _rattlers, _distance;
    size_t _N, _ndim, _nparticles;
public:
    RecordDisp2Histogram(pele::Array<double> origin, pele::Array<double> rattlers, size_t ndim, double min,
            double max, double bin, size_t eqsteps):
        RecordEnergyHistogram(min, max, bin, eqsteps),
        _origin(origin.copy()),_rattlers(rattlers.copy()),_distance(origin.size()),
        _N(origin.size()), _ndim(ndim), _nparticles(_N/_ndim){}
    virtual ~RecordDisp2Histogram(){};
    virtual void action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc);
};

}//namespace bv

#endif//#ifndef _BV_RECORD_DISP2_HISTOGRAM_H

