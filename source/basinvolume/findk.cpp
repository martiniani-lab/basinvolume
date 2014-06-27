#include <cmath>
#include <algorithm>

#include "findk.h"

namespace bv{

Findk::Findk(pele::Array<double> origin, pele::Array<double> rattlers, size_t ndim, size_t avg_count, double target,
        double factor, size_t navg, double tol, double min, double max, double bin):
	    RecordEnergyHistogram(min, max, bin, 42), //we do not specify equlilibration steps; recoding starts when kmax search converged
            _origin(origin.copy()), _rattlers(rattlers.copy()),_distance(origin.size()),
            _target(target),_factor(factor),_acceptedf(0), _k(1), _tol(tol),
            _old_acceptedf(0), _ndim(ndim), _nparticles(_origin.size()/_ndim),
            _avg_count(avg_count), _navg(navg), _naccepted(0), _nrejected(0), _start(0), _converged(false){}

void Findk::_get_vec_distance(const pele::Array<double>& x){
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

void Findk::action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc){

    size_t mc_count = mc->get_iterations_count();

    if (accepted == true)
        ++_naccepted;
    else
        ++_nrejected;

    //if (_converged&&accepted)
    if (_converged)
    {
        //compute distances subtracting the origin's coordinates
        //this->_get_vec_distance(coords); //update distance in any case, also if new configuration is illegal
        if (accepted) this->_get_vec_distance(coords); //update distance only if new configuration is legal

        //compute square displacement from origin
        double norm2 = dot(_distance,_distance);

        //if search for kmax has converged, push displacement into histogram
        _hist.add_entry(norm2);
        //RecordEnergyHistogram::action(coords,energy,accepted,mc);

        //this will trigger premature exit from the MC run loop
        if (_hist.entries() >= _avg_count){
            mc->_niter = std::numeric_limits<size_t>::max();
        }
    }
    else if(mc_count % _navg == 0)
    {
        _old_acceptedf = _acceptedf;
        _acceptedf = (double) _naccepted / (_naccepted + _nrejected);

        //adjust step if last two step oscillated around the target, uses a lower bound
        double d = (_target - _old_acceptedf) * (_target - _acceptedf);
        if (d < 0){
            _factor = std::min<double>(_factor*(2.0-_factor),0.99);
        }

        double ik = mc->_stepsize;
        _k = 1/(ik*ik);

        std::cout<<"_acceptedf "<<_acceptedf<<std::endl; //debug
        std::cout<<"_k "<<_k<<std::endl; //debug
        std::cout<<"_factor"<<_factor<<std::endl;//debug

        if (std::abs(_target - _acceptedf) <= _tol)
            _converged = true;
        else if (_acceptedf < _target)
            _k /= _factor;
        else
            _k *= _factor;

        //adjust the standard deviation of the normal distribution
        mc->_stepsize = std::sqrt(1.0/_k);
        std::cout << "mc->_stepsize: " << mc->_stepsize << std::endl;

        //now reset to zero memory of acceptance and rejection
        _naccepted = 0;
        _nrejected = 0;
    }

    //reset coordinates to origin
    coords.assign(_origin);
}

}//namespace bv
