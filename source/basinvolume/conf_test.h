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
#include <memory>

using std::runtime_error;
using pele::Array;
using mcpele::MC;

namespace bv{

class CheckHyperSphericalContainer:public mcpele::ConfTest{
protected:
    pele::Array<double> _origin, _distance;
    double _radius2;
    size_t _ndim,_N;
public:
    CheckHyperSphericalContainer(pele::Array<double> origin, double radius, size_t ndim);
    virtual bool test(Array<double> &trial_coords, MC * mc);
    virtual ~CheckHyperSphericalContainer(){};
    void inline get_vec_distance(pele::Array<double> coords);
};

CheckHyperSphericalContainer::CheckHyperSphericalContainer(pele::Array<double> origin, double radius, size_t ndim):
        _origin(origin.copy()),_distance(origin.size(),0),_radius2(radius*radius),_ndim(ndim), _N((origin.size()/ndim)){}

void inline CheckHyperSphericalContainer::get_vec_distance(pele::Array<double> coords){
        pele::Array<double> delta_com(_ndim,0);

        for(size_t i=0;i<_N;++i)
        {
            size_t i1 = i*_ndim;
            for(size_t j=0;j<_ndim;++j){
                double d = (coords[i1+j] - _origin[i1+j]);
                _distance[i1+j] = d;
                delta_com[j] += d;
            }
        }

        delta_com /= _N;

        for(size_t i=0;i<_N;++i)
        {
            size_t i1 = i*_ndim;
            for(size_t j=0;j<_ndim;++j)
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

/*CHECK OVERLAP*/
template<typename distance_policy>
class CheckOverlap:public mcpele::ConfTest{
protected:
    Array<double> _hs_radii;
    size_t _ndim, _nparticles;
    std::shared_ptr<distance_policy> _periodic_dist;
public:
    CheckOverlap(Array<double> hs_radii, std::shared_ptr<distance_policy> dist=NULL);
    virtual bool test(Array<double> &trial_coords, MC * mc);
    virtual ~CheckOverlap(){};
};

template<typename distance_policy>
CheckOverlap<distance_policy>::CheckOverlap(Array<double> hs_radii, std::shared_ptr<distance_policy> dist):
        _hs_radii(hs_radii.copy()), _periodic_dist(dist)
{
    if (_periodic_dist == NULL)
        throw std::runtime_error("CheckOverlap::periodic distance uninitialised");
    _ndim = dist->get_ndim();
    _nparticles = _hs_radii.size()/_ndim;
}

template<typename distance_policy>
inline bool CheckOverlap<distance_policy>::test(Array<double> &trial_coords, MC * mc){
    size_t i,j, i1, j1;
    std::vector<double> dr(_ndim);
    double dij;

    for(i=0;i<_nparticles;++i){
        i1 = _ndim*i;
        for(j=0;j<_nparticles;++j){
            if(i != j){
                j1 = _ndim*j;
                _periodic_dist->get_rij(dr, &trial_coords[i1], &trial_coords[j1]);
                double dij2 = 0;
                for(size_t k =0;k<_ndim;++k){dij2 += dr[k]*dr[k];}
                dij = sqrt(dij2);
                dij -= (_hs_radii[i] + _hs_radii[j]);
                if (dij <= 0){
                    std::cout<<"rejected"<<std::endl;
                    return false;
                }
                }
            }
        }

    return true;
}

class CheckOverlap2D:public CheckOverlap<pele::periodic_distance2D>{
public:
    CheckOverlap2D(Array<double> hs_radii, Array<double> boxvec):
        CheckOverlap< pele::periodic_distance2D >(hs_radii,
                std::make_shared<pele::periodic_distance2D>(boxvec[0], boxvec[1]))
        {}
};

class CheckOverlap3D:public CheckOverlap<pele::periodic_distance>{
public:
    CheckOverlap3D(Array<double> hs_radii, Array<double> boxvec):
        CheckOverlap< pele::periodic_distance>(hs_radii,
                  std::make_shared<pele::periodic_distance>(boxvec[0], boxvec[1], boxvec[2]))
                  {}
};

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

template<typename distance_policy>
class CheckSameMinimum:public mcpele::ConfTest{
protected:
	pele::GradientOptimizer * _optimizer;
	Array<double> _origin, _hs_radii, _rattlers, _distance;
	double _dtol, _d, _rms;
	size_t _ndim, _N, _Nnoratt, _nparticles;
	std::shared_ptr<distance_policy> _periodic_dist;
public:
	CheckSameMinimum(pele::GradientOptimizer * optimizer, Array<double> origin, Array<double> hs_radii, Array<double> rattlers, double dtol,
	        std::shared_ptr<distance_policy> dist=NULL);
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

template<typename distance_policy>
CheckSameMinimum<distance_policy>::CheckSameMinimum(pele::GradientOptimizer * optimizer, Array<double> origin, Array<double> hs_radii,
		Array<double> rattlers, double dtol, std::shared_ptr<distance_policy> dist):
		_optimizer(optimizer), _origin(origin.copy()), _hs_radii(hs_radii.copy()),
		_rattlers(rattlers.copy()), _distance(origin.size(),0),_dtol(dtol),_d(0),
		_rms(0),_N(origin.size()),_Nnoratt(0),_periodic_dist(dist)
        {
            if (_periodic_dist == NULL)
                throw std::runtime_error("CheckSameMinimum::periodic distance uninitialised");
            _ndim = dist->get_ndim();
            _nparticles = _hs_radii.size()/_ndim;

            for(size_t i=0;i<_N;i+=_ndim)
			    _Nnoratt += _rattlers[i];
		}

//compute distance from origin after aligning the centre of mass
//this ignores the rattlers completely

template<typename distance_policy>
inline void CheckSameMinimum<distance_policy>::get_vec_distance(pele::Array<double> quenched_coords){
        pele::Array<double> delta_com(_ndim,0);

        for(size_t i=0;i<_nparticles;++i)
        {
            size_t i1 = i*_ndim;
            for(size_t j=0;j<_ndim;++j){
                double d = (quenched_coords[i1+j] - _origin[i1+j])*_rattlers[i1];
                _distance[i1+j] = d;
                delta_com[j] += d;
            }
        }

        delta_com /= _Nnoratt;

        for(size_t i=0;i<_nparticles;++i)
        {
            size_t i1 = i*_ndim;
            for(size_t j=0;j<_ndim;++j)
                _distance[i1+j] -= delta_com[j]*_rattlers[i1];
        }
    }

template<typename distance_policy>
inline bool CheckSameMinimum<distance_policy>::check_overlap(Array<double> &trial_coords){
	size_t i,j, i1, j1;
	std::vector<double> dr(_ndim);
	double dij;

	for(i=0;i<_nparticles;++i){
		i1 = _ndim*i;
		for(j=0;j<_nparticles;++j){
			if(i != j){
				j1 = _ndim*j;
				_periodic_dist->get_rij(dr, &trial_coords[i1], &trial_coords[j1]);
				double dij2 = 0;
				for(size_t k =0;k<_ndim;++k){dij2 += dr[k]*dr[k];}
				dij = sqrt(dij2);
				dij -= (_hs_radii[i] + _hs_radii[j]);
				if (dij <= 0)
					return false;
				}
			}
		}

	return true;
}

template<typename distance_policy>
bool CheckSameMinimum<distance_policy>::test(Array<double> &trial_coords, MC * mc)
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

class CheckSameMinimum2D:public CheckSameMinimum<pele::periodic_distance2D>{
public:
    CheckSameMinimum2D(pele::GradientOptimizer * optimizer, Array<double> origin, Array<double> hs_radii,
                        Array<double> boxvec, Array<double> rattlers, double dtol):
        CheckSameMinimum<pele::periodic_distance2D>(optimizer, origin, hs_radii, rattlers, dtol,
                std::make_shared<pele::periodic_distance2D>(boxvec[0], boxvec[1]))
        {}
};

class CheckSameMinimum3D:public CheckSameMinimum<pele::periodic_distance>{
public:
    CheckSameMinimum3D(pele::GradientOptimizer * optimizer, Array<double> origin, Array<double> hs_radii,
                            Array<double> boxvec, Array<double> rattlers, double dtol):
            CheckSameMinimum<pele::periodic_distance>(optimizer, origin, hs_radii, rattlers, dtol,
                    std::make_shared<pele::periodic_distance>(boxvec[0], boxvec[1], boxvec[2]))
        {}
};

}


#endif
