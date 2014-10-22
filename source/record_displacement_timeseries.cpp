#include "basinvolume/record_displacement_timeseries.h"

namespace bv{

RecordDisplacementTimeseries::RecordDisplacementTimeseries(pele::Array<double> origin, const size_t ndim,
        const size_t ts_niter, const size_t record_every)
    :RecordScalarTimeseries(ts_niter, record_every),
     m_origin(origin.copy()),
     m_distance(m_origin.size()),
     m_ndim(ndim),
     m_nparticles(m_origin.size()/m_ndim)
    {}

inline void RecordDisplacementTimeseries::m_get_vec_distance(const pele::Array<double>& x){
        pele::Array<double> delta_com(m_ndim,0);

        for(size_t i=0;i<m_nparticles;++i)
        {
            size_t i1 = i*m_ndim;
            for(size_t j=0;j<m_ndim;++j){
                double d = (x[i1+j] - m_origin[i1+j]);
                m_distance[i1+j] = d;
                delta_com[j] += d;
            }
        }

        delta_com /= m_nparticles;

        for(size_t i=0;i<m_nparticles;++i)
        {
            size_t i1 = i*m_ndim;
            for(size_t j=0;j<m_ndim;++j)
                m_distance[i1+j] -= delta_com[j];
        }
    }

double RecordDisplacementTimeseries::get_recorded_scalar(pele::Array<double> &coords, const double energy, const bool accepted, mcpele::MC* mc){
    this->m_get_vec_distance(coords);
    return norm(m_distance); //norm
}

}//namespace bv
