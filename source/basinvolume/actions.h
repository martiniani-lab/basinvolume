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
	pele::Array<double> _origin, _rattlers, _distance;
	size_t _N, _Nnoratt, _nparticles;
public:
	RecordDisp2Histogram(pele::Array<double> origin, pele::Array<double> rattlers, double min,
	        double max, double bin, size_t eqsteps):
	    RecordEnergyHistogram(min, max, bin, eqsteps),
	    _origin(origin.copy()),_rattlers(rattlers.copy()),_distance(origin.size()),
	    _N(origin.size()), _Nnoratt(0), _nparticles(_N/3){
	        for(size_t i=0;i<_N;i+=3)
	            _Nnoratt += _rattlers[i];
	    }
	virtual ~RecordDisp2Histogram() {delete _hist;}
	virtual void action(Array<double> &coords, double energy, bool accepted, MC* mc);
	virtual void inline get_vec_distance(pele::Array<double> x);
};

inline void RecordDisp2Histogram::get_vec_distance(pele::Array<double> x){
        pele::Array<double> delta_com(3,0);

        for(size_t i=0;i<_nparticles;++i)
        {
            size_t i1 = i*3;
            for(size_t j=0;j<3;++j){
                double d = (x[i1+j] - _origin[i1+j]);
                _distance[i1+j] = d;
                delta_com[j] += d;
            }
        }

        delta_com /= _nparticles;

        for(size_t i=0;i<_nparticles;++i)
        {
            size_t i1 = i*3;
            for(size_t j=0;j<3;++j)
                _distance[i1+j] -= delta_com[j];
        }
    }

void RecordDisp2Histogram::action(Array<double> &coords, double energy, bool accepted, MC* mc) {
		_count = mc->get_iterations_count();
		if (_count >= _eqsteps)
		{
			//compute distances subtracting the origin's coordinates
			this->get_vec_distance(coords);

			//compute square displacement from origin
			double norm2 = 0;
			for (size_t i=0;i<_N;++i)
			    norm2 += _distance[i]*_distance[i];
			_hist->add_entry(norm2);
			_mean = (_mean*(_count-1)+norm2)/_count;
			//std::cout<<"mean "<<_mean<<std::endl;
		}
}

/*
 * Findk accept test, THIS IS A FICTIOUS ACTION (see note)
 * find k for an harmonic potential such that the acceptance is within some range
 * navg number of steps over which acceptance fraction is averaged
 * factor has to be in (0,1)
 *
 *note: this class does some hacky things to exploit the behaviour of MC to get it to do something
 *that it wasn't originally entirely designed for. Weird things:
  * need to keep a shared pointer of harmonicperiodic and then need to cast the potential to this type, to call get_k()
  * set MC->_niter to the largest unsigned inter so that the calculation must terminate
 * */

class Findk : public Action {
protected:
    pele::Array<double> _origin;
    double _target, _factor, _acceptedf, _k, _tol;
    size_t _navg, _count, _naccepted, _nrejected, _start;
public:
    Findk(Array<double> origin, double target, double factor, size_t navg, double tol);
    virtual ~Findk() {}
    virtual void action(Array<double> &coords, double energy, bool accepted, MC* mc);
};

Findk::Findk(Array<double> origin, double target, double factor, size_t navg, double tol):
            _origin(origin.copy()),_target(target),_factor(factor),_acceptedf(0),
            _k(0), _tol(tol), _navg(navg),_count(0),
            _naccepted(0), _nrejected(0), _start(0){}


void Findk::action(Array<double> &coords, double energy, bool accepted, MC* mc){

    _count = mc->get_iterations_count();

    if (accepted == true)
        ++_naccepted;
    else
        ++_nrejected;

    if(_count % _navg == 0)
    {
        _acceptedf = (double) _naccepted / (_naccepted + _nrejected);

        double ik = mc->_stepsize;
        _k = 1/(ik*ik);

        if (std::abs(_target - _acceptedf) <= _tol){
            //std::cout<<"k found: "<<_k<<std::endl; //debug
            //this will trigger premature exit from the MC run loop
            mc->_niter = std::numeric_limits<size_t>::max();
        }
        else if (_acceptedf < _target)
            _k /= _factor;
        else
            _k *= _factor;

        std::cout<<"_acceptedf "<<_acceptedf<<std::endl; //debug
        std::cout<<"_k "<<_k<<std::endl; //debug

        //adjust the standard deviation of the normal distribution
        mc->_stepsize = sqrt(1.0/_k);

        //now reset to zero memory of acceptance and rejection
        _naccepted = 0;
        _nrejected = 0;
    }

    //reset coordinates to origin
    coords.assign(_origin);
}

}
#endif
