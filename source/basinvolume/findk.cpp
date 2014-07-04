#include <cmath>
#include <algorithm>

#include "findk.h"

namespace bv{

Findk::Findk(pele::Array<double> origin, pele::Array<double> rattlers, size_t ndim, size_t avg_count, double target,
        double factor, size_t navg, double tol, double min, double max, double bin)
	:_origin(origin.copy())
	,_rattlers(rattlers.copy())
	,_distance(origin.size())
	,_target(target)
	,_factor(factor)
	,_acceptedf(1)
	,_k(1)
	,_tol(tol)
	,_ndim(ndim)
	,_nparticles(_origin.size()/_ndim)
	,_avg_count(avg_count)
	,_navg(navg)
	,_naccepted(0)
	,_nrejected(0)
	,_start(0)
	,_converged(false)
    ,_hist(min, max, bin)
	{}

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
        double norm2 = dot(_distance, _distance);

        //if search for kmax has converged, push displacement into histogram
        _hist.add_entry(norm2);
        //RecordEnergyHistogram::action(coords,energy,accepted,mc);

        //this will trigger premature exit from the MC run loop
        if (static_cast<size_t>(_hist.entries()) >= _avg_count){
            mc->_niter = std::numeric_limits<size_t>::max();
        }
    }
    else if(mc_count % _navg == 0)
    {
        //_acceptedf = static_cast<double>(_naccepted) / static_cast<double>(_naccepted + _nrejected);
        _acceptedf = static_cast<double>(_naccepted) / (static_cast<double>(_naccepted)+static_cast<double>(_nrejected));

        //adjust step if last two step oscillated around the target, uses a lower bound
        adjust_k(mc_count/_navg, mc);

        //adjust the standard deviation of the normal distribution
        mc->_stepsize = std::sqrt(1.0/_k);
        //std::cout << "mc->_stepsize: " << mc->_stepsize << std::endl;//debug

        //now reset to zero memory of acceptance and rejection
        _naccepted = 0;
        _nrejected = 0;
    }

    //reset coordinates to origin
    coords.assign(_origin);
}

void Findk::adjust_k(const size_t iterations, mcpele::MC* mc){
    // parameter: can be adapted for better convergence
    const size_t period = 3;
    //get k
    const double ik = mc->_stepsize;
    _k = 1/(ik*ik);
    //debug output
    std::cout<<"_acceptedf "<<_acceptedf<<std::endl; //debug
    std::cout<<"_k "<<_k<<std::endl; //debug
    std::cout<<"iterations "<< iterations << std::endl;//debug
    //check for convergence
    if (fabs(_target - _acceptedf) < _tol){
        _converged = true;
        return;
    }
    //adapt k size
    const double tmp1 = 1.0/(iterations%period+1);
    const double tmp2 = 1 + (_target-_acceptedf)/(_target+_acceptedf);
    const double tmp = (1-tmp1) + tmp1*tmp2;
    _k *= tmp*tmp;
}

}//namespace bv
