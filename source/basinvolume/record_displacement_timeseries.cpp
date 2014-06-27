#include "record_displacement_timeseries.h"

namespace bv{

RecordDisplacementTimeseries::RecordDisplacementTimeseries(pele::Array<double> origin, const size_t ndim,
        const size_t ts_niter, const size_t record_every)
    :_origin(origin.copy()), _distance(_origin.size()), _ndim(ndim), _nparticles(_origin.size()/_ndim),
     _ts_niter(ts_niter),_record_every(record_every)
    {
        _time_series.reserve(_ts_niter);
        if (record_every==0) throw std::runtime_error("RecordDisplacementTimeseries: __record_every expected to be at least 1");
    }

inline void RecordDisplacementTimeseries::_get_vec_distance(const pele::Array<double>& x){
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

inline void RecordDisplacementTimeseries::_record_displacement_value(const double dx){
    _time_series.push_back(dx);
}

void RecordDisplacementTimeseries::action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc){
    size_t counter = mc->get_iterations_count();
    if (counter % _record_every == 0){
        this->_get_vec_distance(coords);
        double dx = norm(_distance);
        this->_record_displacement_value(dx);
    }
}

pele::Array<double> RecordDisplacementTimeseries::get_time_series(){
    _time_series.shrink_to_fit();
    return pele::Array<double>(_time_series).copy();
}

}//namespace bv
