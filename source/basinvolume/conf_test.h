#ifndef _BV_CONF_TEST_H__
#define _BV_CONF_TEST_H__

#include <iostream>
#include <math.h>
#include <algorithm>
#include <random>
#include <chrono>
#include "pele/array.h"
#include "pele/optimizer.h"
#include "pele/distance.h"
#include "mcpele/mc.h"
#include "mcpele/conf_test.h"

using std::runtime_error;
using pele::Array;
using mcpele::MC;

namespace bv{

class CheckHyperSphericalContainer:public mcpele::ConfTest{
protected:
    pele::Array<double> _origin, _distance;
    double _radius2;
    size_t _N;
public:
    CheckHyperSphericalContainer(pele::Array<double> origin, double radius);
    virtual bool test(Array<double> &trial_coords, MC * mc);
    virtual ~CheckHyperSphericalContainer(){};
    virtual void inline get_vec_distance(pele::Array<double> coords);
};

CheckHyperSphericalContainer::CheckHyperSphericalContainer(pele::Array<double> origin, double radius):
        _origin(origin.copy()),_distance(origin.size(),0),_radius2(radius*radius),_N((origin.size()/3)){}

void inline CheckHyperSphericalContainer::get_vec_distance(pele::Array<double> coords){
        pele::Array<double> delta_com(3,0);

        for(size_t i=0;i<_N;++i)
        {
            size_t i1 = i*3;
            for(size_t j=0;j<3;++j){
                double d = (coords[i1+j] - _origin[i1+j]);
                _distance[i1+j] = d;
                delta_com[j] += d;
            }
        }

        delta_com /= _N;

        for(size_t i=0;i<_N;++i)
        {
            size_t i1 = i*3;
            for(size_t j=0;j<3;++j)
                _distance[i1+j] -= delta_com[j];
        }
    }

bool CheckHyperSphericalContainer::test(Array<double> &trial_coords, MC * mc)
{
  /*for(size_t i=0;i<_origin.size();++i)
  {
      double r = trial_coords[i] - _origin[i];
      r2 += r*r;
  }*/

  this->get_vec_distance(trial_coords);

  double r = norm(_distance);
  double r2 = r*r;
  if (r2 > _radius2)
      return false;

  return true;
}

/*check same minimum class
 * _optimizer: pointer to object of class GradientOptimizer performing minimisation according to some potential
 * 				passed to the object during its construction
 * _origin: coordinates to which the quenched structure is compared to
 * _rattlers: array of 1s or 0s: if not indicates a rattler,0 -> rattler
 * 															1 -> jammed particle
 * 			this convention removes if statements in the for loop and replaces them with
 * 			arithmetic operation (distance[i] *= rattlers[i]), distance is set artificially to
 * 			zero if the particle is a rattler. note _rattlers.size() = coords.size()
 * _distance: array containing the Euclidean distance between trial_coords and origin
 * _d: norm of distance
 * _rms: root mean square displacement from origin
 * _E = energy of the quenched state
 * _dtol: tolerance on distances
 * _Etol: tolerance on energies (a minimum should be whithin this value from _Emin)
 * _Eor: energy of the origin (must pass it because CheckSameMinimum knows nothing about the potential used by the optimiser)
 * _Nnoratt: total number of non rattlers degrees of freedom
 * */

class CheckSameMinimum:public mcpele::ConfTest{
protected:
	pele::periodic_distance _periodic_dist;
	pele::GradientOptimizer * _optimizer;
	Array<double> _origin, _hs_radii, _rattlers, _distance;
	double _dtol, _d, _rms;
	size_t _N, _Nnoratt, _nparticles;
public:
	CheckSameMinimum(pele::GradientOptimizer * optimizer, Array<double> origin, Array<double> hs_radii, Array<double> boxvec,
			Array<double> rattlers, double dtol);
	virtual bool test(Array<double> &trial_coords, MC * mc);
	virtual ~CheckSameMinimum(){
		if (_optimizer != NULL)
			delete _optimizer;
	}
	double get_distance(){return _d;}
	Array<double> get_distance_array(){
		Array<double> x(_distance.copy());
		return x;
	}
	inline bool check_overlap(Array<double> &trial_coords);
	inline void get_vec_distance(Array<double> quenched_coords);
};

CheckSameMinimum::CheckSameMinimum(pele::GradientOptimizer * optimizer, Array<double> origin, Array<double> hs_radii,
		Array<double> boxvec, Array<double> rattlers, double dtol):
		_periodic_dist(boxvec[0], boxvec[1], boxvec[2]),
		_optimizer(optimizer), _origin(origin.copy()), _hs_radii(hs_radii.copy()),
		_rattlers(rattlers.copy()), _distance(origin.size(),0),_dtol(dtol),_d(0),
		_rms(0),_N(origin.size()),_Nnoratt(0), _nparticles(_N/3){
			for(size_t i=0;i<_N;i+=3)
			    _Nnoratt += _rattlers[i];
		}

//compute distance from origin after aligning the centre of mass
//this ignores the rattlers completely

inline void CheckSameMinimum::get_vec_distance(pele::Array<double> quenched_coords){
        pele::Array<double> delta_com(3,0);

        for(size_t i=0;i<_nparticles;++i)
        {
            size_t i1 = i*3;
            for(size_t j=0;j<3;++j){
                double d = (quenched_coords[i1+j] - _origin[i1+j])*_rattlers[i1];
                _distance[i1+j] = d;
                delta_com[j] += d;
            }
        }

        delta_com /= _Nnoratt;

        for(size_t i=0;i<_nparticles;++i)
        {
            size_t i1 = i*3;
            for(size_t j=0;j<3;++j)
                _distance[i1+j] -= delta_com[j]*_rattlers[i1];
        }
    }

inline bool CheckSameMinimum::check_overlap(Array<double> &trial_coords){
	size_t i,j, i1, j1;
	double dr[3];
	double dij2,dij;

	for(i=0;i<_nparticles;++i){
		i1 = 3*i;
		for(j=0;j<_nparticles;++j){
			if(i != j){
				j1 = 3*j;
				_periodic_dist.get_rij(dr, &trial_coords[i1], &trial_coords[j1]);
				dij2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
				dij = sqrt(dij2);
				dij -= (_hs_radii[i] + _hs_radii[j]);
				if (dij <= 0)
					return false;
				}
			}
		}

	return true;
}

bool CheckSameMinimum::test(Array<double> &trial_coords, MC * mc)
{
	bool quench_success, no_overlap;
	size_t nfev;

	no_overlap = this->check_overlap(trial_coords);

	if (! no_overlap)
		return false;

	_optimizer->reset(trial_coords);
	_optimizer->run();

	//add number of energy evaluations to mc eval count
	nfev = _optimizer->get_niter();
	mc->_neval += nfev;

	//first test: minimisation must have converged
	quench_success = _optimizer->success();
	if (! quench_success)
		return false;

	//compute distance between quenched coords and origin
	//distance for rattlers is set to 0
	this->get_vec_distance(_optimizer->get_x());

	//compute rms displacement from origin
	_d = norm(_distance);
	_rms = _d / sqrt(_Nnoratt);
	if (_rms > _dtol)
		return false;
	else
		return true;
}

}


#endif
