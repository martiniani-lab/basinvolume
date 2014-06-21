#ifndef _BV_ACTIONS_H
#define _BV_ACTIONS_H

#include <cmath>
#include <algorithm>
#include <list>
#include <vector>
#include "pele/array.h"
#include "pele/distance.h"
#include "mcpele/mc.h"
#include "mcpele/histogram.h"
#include "mcpele/actions.h"
#include "mcpele/takestep.h"
#include "pele/harmonic.h"

using std::runtime_error;
using pele::Array;
using mcpele::MC;
using std::sqrt;
using mcpele::Action;

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
	virtual void action(Array<double> &coords, double energy, bool accepted, MC* mc);
};

inline void RecordDisp2Histogram::_get_vec_distance(const pele::Array<double>& x){
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

void RecordDisp2Histogram::action(Array<double> &coords, double energy, bool accepted, MC* mc) {
		_count = mc->get_iterations_count();

		if (_count > _eqsteps)
		{
			//compute distances subtracting the origin's coordinates
			this->_get_vec_distance(coords);

			//compute square displacement from origin
			double norm2 = dot(_distance,_distance);
			_hist.add_entry(norm2);
			double count = (double) _count - _eqsteps + 1;
			_mean = (_mean*(count-1)+norm2)/count;
			_mean2 = (_mean2*(count-1)+(norm2*norm2))/count;
		}
}

/*
 * Findk accept test, THIS IS A FICTIOUS ACTION (see note)
 * find k for an harmonic potential such that the acceptance is within some range
 * navg number of steps over which acceptance fraction is averaged
 * factor has to be in (0,1)
 * get_prob returns the probability (_acceptedf) associated with kmax
 * avg_count is the number of steps over which the displacement squared is averaged
 *
 *note: this class does some hacky things to exploit the behaviour of MC to get it to do something
 *that it wasn't originally entirely designed for. Weird things:
 * the potential is entirely fictitious, so _k is adjusted through the stepsize
 * set MC->_niter to the largest unsigned inter so that the calculation must terminate
 * */

//template<size_t bdim>
class Findk : public Action {
protected:
    inline void _get_vec_distance(const pele::Array<double>& x);
    pele::Array<double> _origin, _rattlers, _distance;
    double _target, _factor, _acceptedf, _k, _tol, _old_acceptedf, _mean, _mean2;
    size_t _ndim, _nparticles, _avg_count, _navg, _count, _naccepted, _nrejected, _start;
    bool _converged;
public:
    Findk(Array<double> origin, Array<double> rattlers, size_t ndim, size_t avg_count, double target, double factor, size_t navg, double tol);
    virtual ~Findk() {}
    virtual void action(Array<double> &coords, double energy, bool accepted, MC* mc);
    double get_prob(){return _acceptedf;}
    double get_mean(){return _mean;};
    double get_variance(){return (_mean2 - _mean*_mean);};
};

Findk::Findk(Array<double> origin, Array<double> rattlers, size_t ndim, size_t avg_count, double target,
        double factor, size_t navg, double tol):
            _origin(origin.copy()), _rattlers(rattlers.copy()),_distance(origin.size()),
            _target(target),_factor(factor),_acceptedf(0), _k(1), _tol(tol),
            _old_acceptedf(0), _mean(0), _mean2(0), _ndim(ndim), _nparticles(_origin.size()/_ndim),
            _avg_count(avg_count), _navg(navg), _count(0), _naccepted(0), _nrejected(0), _start(0), _converged(false){}

inline void Findk::_get_vec_distance(const pele::Array<double>& x){
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

void Findk::action(Array<double> &coords, double energy, bool accepted, MC* mc){

    size_t mc_count = mc->get_iterations_count();

    if (accepted == true)
        ++_naccepted;
    else
        ++_nrejected;

    if (_converged)
    {
        //increase averaging count
        ++_count;
        //compute distances subtracting the origin's coordinates
        this->_get_vec_distance(coords);

        //compute square displacement from origin
        double norm2 = dot(_distance,_distance);

        _mean = (_mean*(_count-1)+norm2)/_count;
        _mean2 = (_mean2*(_count-1)+(norm2*norm2))/_count;

        //this will trigger premature exit from the MC run loop
        if (_count >= _avg_count)
            mc->_niter = std::numeric_limits<size_t>::max();
    }
    else if(mc_count % _navg == 0)
    {
        _old_acceptedf = _acceptedf;
        _acceptedf = (double) _naccepted / (_naccepted + _nrejected);

        //adjust step if last two step oscillated around the target, uses a lower bound
        double d = (_target - _old_acceptedf) * (_target - _acceptedf);
        if (d < 0){
            _factor = std::min(_factor*(2.0-_factor),0.99);
        }

        double ik = mc->_stepsize;
        _k = 1/(ik*ik);

//        std::cout<<"_acceptedf "<<_acceptedf<<std::endl; //debug
//        std::cout<<"_k "<<_k<<std::endl; //debug

        if (std::abs(_target - _acceptedf) <= _tol)
            _converged = true;
        else if (_acceptedf < _target)
            _k /= _factor;
        else
            _k *= _factor;

        //adjust the standard deviation of the normal distribution
        mc->_stepsize = sqrt(1.0/_k);

        //now reset to zero memory of acceptance and rejection
        _naccepted = 0;
        _nrejected = 0;
    }

    //reset coordinates to origin
    coords.assign(_origin);
}

/*
 * Record displacement time series, measuring every __record_every-th step.
 */

class RecordDisplacementTimeseries : public Action{
    private:
        inline void _record_displacement_value(const double dx);
        inline void _get_vec_distance(const pele::Array<double>& x);
        pele::Array<double> _origin, _distance;
        const size_t _ndim, _nparticles, _ts_niter, _record_every;
        std::vector<double> _time_series;
    public:
        RecordDisplacementTimeseries(pele::Array<double> origin, const size_t ndim, const size_t niter, const size_t record_every);
        virtual ~RecordDisplacementTimeseries(){}
        virtual void action(Array<double> &coords, double energy, bool accepted, MC* mc);
        pele::Array<double> get_time_series();
        void clear(){_time_series.clear();}
};

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

void RecordDisplacementTimeseries::action(Array<double> &coords, double energy, bool accepted, MC* mc){
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

}
#endif
