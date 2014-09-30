#ifndef _BV_RECORD_DISP2_HISTOGRAM_H
#define _BV_RECORD_DISP2_HISTOGRAM_H

#include "pele/array.h"

#include "mcpele/histogram.h"
#include "mcpele/record_energy_histogram.h"

namespace bv{

/*
 * Record displacement square histogram
*/

class RecordDisp2Histogram : public mcpele::RecordEnergyHistogram {
protected:
    inline void m_get_vec_distance(const pele::Array<double>& x);
    pele::Array<double> m_origin;
    pele::Array<double> m_rattlers;
    pele::Array<double> m_distance;
    const size_t m_N;
    const size_t m_ndim;
    const size_t m_nparticles;
public:
    RecordDisp2Histogram(pele::Array<double> origin, pele::Array<double> rattlers, size_t ndim, double min,
            double max, double bin, size_t eqsteps)
        : RecordEnergyHistogram(min, max, bin, eqsteps),
          m_origin(origin.copy()),
          m_rattlers(rattlers.copy()),
          m_distance(origin.size()),
          m_N(origin.size()),
          m_ndim(ndim),
          m_nparticles(m_N/m_ndim)
    {}
    virtual ~RecordDisp2Histogram(){};
    virtual void action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc);
};

}//namespace bv

#endif//#ifndef _BV_RECORD_DISP2_HISTOGRAM_H

