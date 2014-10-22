#include "basinvolume/record_disp2_histogram.h"

namespace bv{

void RecordDisp2Histogram::m_get_vec_distance(const pele::Array<double>& x){
        pele::Array<double> delta_com(m_ndim,0);

        for(size_t i = 0 ; i < m_nparticles; ++i)
        {
            size_t i1 = i * m_ndim;
            for(size_t j = 0; j < m_ndim; ++j) {
                double d = (x[i1+j] - m_origin[i1+j]);
                m_distance[i1+j] = d;
                delta_com[j] += d;
            }
        }

        delta_com /= m_nparticles;

        for(size_t i=0; i < m_nparticles; ++i)
        {
            size_t i1 = i * m_ndim;
            for(size_t j = 0; j < m_ndim; ++j) {
                m_distance[i1+j] -= delta_com[j];
            }
        }
    }

void RecordDisp2Histogram::action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc) {
        if (mc->get_iterations_count() > get_eqsteps())
        {
            //compute distances subtracting the origin's coordinates
            this->m_get_vec_distance(coords);

            //compute square displacement from origin
            double norm2 = dot(m_distance, m_distance);
            m_hist.add_entry(norm2);
        }
}

}//namespace bv
