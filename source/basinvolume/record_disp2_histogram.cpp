#include "record_disp2_histogram.h"

namespace bv{

void RecordDisp2Histogram::_get_vec_distance(const pele::Array<double>& x){
        pele::Array<double> delta_com(_ndim,0);

        for(size_t i=0;i<_nparticles;++i)
        {
            size_t i1 = i*_ndim;
            for(size_t j=0;j<_ndim;++j){
                double d = (x[i1+j] - _origin[i1+j]);
                _distance[i1+j] = d;
                delta_com[j] += d;
            }
        }

        delta_com /= _nparticles;

        for(size_t i=0;i<_nparticles;++i)
        {
            size_t i1 = i*_ndim;
            for(size_t j=0;j<_ndim;++j)
                _distance[i1+j] -= delta_com[j];
        }
    }

void RecordDisp2Histogram::action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc) {
        if (mc->get_iterations_count() > get_eqsteps())
        {
            //compute distances subtracting the origin's coordinates
            this->_get_vec_distance(coords);

            //compute square displacement from origin
            double norm2 = dot(_distance,_distance);
            _hist.add_entry(norm2);
        }
}

}//namespace bv
