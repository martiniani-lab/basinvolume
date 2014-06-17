#ifndef _BV_CONF_TEST_H__
#define _BV_CONF_TEST_H__

#include <iostream>
#include <math.h>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>

#include "pele/array.h"
#include "pele/optimizer.h"
#include "pele/distance.h"
#include "pele/harmonic.h" //debug
#include "pele/lbfgs.h"
#include "pele/lowest_eig_potential.h"
#include "mcpele/mc.h"
#include "mcpele/conf_test.h"
#include "minima_list.h"

using std::runtime_error;
using pele::Array;
using mcpele::MC;

namespace bv{

class CheckHyperSphericalContainer:public mcpele::ConfTest{
protected:
    inline void _get_vec_distance(const pele::Array<double>& coords);
    pele::Array<double> _origin, _distance;
    double _radius2;
    size_t _ndim,_N;
public:
    CheckHyperSphericalContainer(pele::Array<double> origin, double radius, size_t ndim);
    virtual bool test(Array<double> &trial_coords, MC * mc);
    virtual ~CheckHyperSphericalContainer(){};
};

CheckHyperSphericalContainer::CheckHyperSphericalContainer(pele::Array<double> origin, double radius, size_t ndim):
        _origin(origin.copy()),_distance(origin.size(),0),_radius2(radius*radius),_ndim(ndim), _N((origin.size()/ndim)){}

inline void CheckHyperSphericalContainer::_get_vec_distance(const pele::Array<double>& coords){
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
    /*
    //debug
    std::shared_ptr<pele::BaseHarmonic> potential;
    potential = std::dynamic_pointer_cast<pele::BaseHarmonic>(mc->_potential);
    double k = potential->get_k();
    std::cout<<"k "<<k<<std::endl;
    ////
    */
  this->_get_vec_distance(trial_coords);

  double r2 = dot(_distance,_distance);
  if (r2 > _radius2)
      return false;

  return true;
}

/*CHECK OVERLAP*/
template<typename distance_policy>
class CheckOverlap:public mcpele::ConfTest{
protected:
    const static size_t _ndim = distance_policy::_ndim;
    Array<double> _hs_radii;
    size_t _nparticles;
    std::shared_ptr<distance_policy> _periodic_dist;
public:
    CheckOverlap(Array<double> hs_radii, std::shared_ptr<distance_policy> dist=NULL);
    virtual bool test(Array<double> &trial_coords, MC * mc);
    virtual ~CheckOverlap(){};
};

template<typename distance_policy>
CheckOverlap<distance_policy>::CheckOverlap(Array<double> hs_radii, std::shared_ptr<distance_policy> dist):
        _hs_radii(hs_radii.copy()), _nparticles(_hs_radii.size()), _periodic_dist(dist)
        {
            if (_periodic_dist == NULL)
                throw std::runtime_error("CheckOverlap::periodic distance uninitialised");
        }

template<typename distance_policy>
inline bool CheckOverlap<distance_policy>::test(Array<double> &trial_coords, MC * mc){
    size_t i,j, i1, j1;
    double dr[_ndim];
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
                    //std::cout<<"rejected"<<std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

class CheckOverlap2D:public CheckOverlap<pele::periodic_distance<2>>{
public:
    CheckOverlap2D(Array<double> hs_radii, double const *boxvec):
        CheckOverlap< pele::periodic_distance<2> >(hs_radii,
                std::make_shared<pele::periodic_distance<2>>(boxvec))
        {}
};

class CheckOverlap3D:public CheckOverlap<pele::periodic_distance<3>>{
public:
    CheckOverlap3D(Array<double> hs_radii, double const *boxvec):
        CheckOverlap< pele::periodic_distance<3>>(hs_radii,
                  std::make_shared<pele::periodic_distance<3>>(boxvec))
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
 * _inoratt: index of first non-rattler
 * _Nnoratt: number of non-rattlers
 * use the flag -D RECORD_MINIMA_LIST to record information about the minima that we fall into
 * */

/*class CheckSameMinimumInterface: public mcpele::ConfTest{
    virtual inline void _get_vec_distance(pele::Array<double> quenched_coords) = 0;
    virtual ~CheckSameMinimumInterface();
}*/


class CheckSameMinimum{
protected:
    //protected functions
    inline void _get_vec_distance(pele::Array<double> quenched_coords);
    inline double _check_convergence(pele::Array<double> quenched_coords);
    inline void _record_lowesteig_ts(double eig, MC * mc);
    inline void _record_minimum();

    //protected member variables checksameminimum
    size_t _ndim;
    pele::GradientOptimizer * _optimizer;
    pele::BasePotential * _potential;
    Array<double> _origin, _hs_radii, _rattlers, _distance;
    double _dtol, _d, _rms;
    size_t _nparticles;
    std::shared_ptr<pele::DistanceInterface> _dist_policy;
    size_t _Nnoratt, _inoratt;

    //convergence test classes
    bool _perform_convergence_test;
    double _lbfgstol, _lbfgsM, _lbfgsniter, _lbfgsmaxstep, _lowtol, _hightol, _eigtol, _H0;
    size_t _ts_niter, _record_every;
    Array<double> _ranvec;
    std::vector<double> _lowesteig_ts;

    //minima listing
    bool _record_minimum_list;
    double _tol_delta_x, _tol_energy, _tol_delta_x_element;
    Array<double> _aligned_quenched_coords;
    //MinimaList<distance_policy> _minima_list;
public:
    CheckSameMinimum(pele::GradientOptimizer * optimizer, pele::BasePotential * potential, Array<double> origin, Array<double> hs_radii,
            Array<double> rattlers, double dtol, size_t ndim, std::shared_ptr<pele::DistanceInterface> dist=NULL);
    virtual bool test(Array<double> &trial_coords, MC * mc);
    virtual ~CheckSameMinimum(){}
    double get_distance(){return _d;}
    Array<double> get_distance_array(){
        return _distance.copy();
    }
    pele::Array<double> get_lowesteig_ts(){
        _lowesteig_ts.shrink_to_fit();
        return pele::Array<double>(_lowesteig_ts).copy();
    }
    void lowesteig_ts_clear(){_lowesteig_ts.clear();}
};

CheckSameMinimum::CheckSameMinimum(pele::GradientOptimizer * optimizer, pele::BasePotential * potential, Array<double> origin,
        Array<double> hs_radii, Array<double> rattlers, double dtol, size_t ndim, std::shared_ptr<pele::DistanceInterface> dist):
        _ndim(ndim), _optimizer(optimizer), _potential(potential), _origin(origin.copy()), _hs_radii(hs_radii.copy()),
        _rattlers(rattlers.copy()), _distance(origin.size(),0),_dtol(dtol),_d(0),
        _rms(0),_nparticles(_hs_radii.size()), _dist_policy(dist),_Nnoratt(0),
        //convergence test
        _perform_convergence_test(true), _lbfgstol(1e-3), _lbfgsM(5), _lbfgsniter(100),
        _lbfgsmaxstep(0.3), _lowtol(1e-10), _hightol(_optimizer->get_tol()), _eigtol(0.1), _H0(1), _ts_niter(10000), _record_every(10),
        _ranvec(_origin.copy())
        /*//mimina listing
        _record_minimum_list(false), _tol_delta_x(1e-10), _tol_energy(1e-10), _tol_delta_x_element(_tol_delta_x*origin.size()),
        _minima_list(_tol_delta_x, _tol_energy, _tol_delta_x_element, _dist_policy)*/
        {
            if (_dist_policy == NULL || _ndim==0)
                throw std::runtime_error("CheckSameMinimum::CheckSameMinimum distance policy uninitialised");

            for(size_t i=0;i<_origin.size();i+=_ndim)
                if (_rattlers[i] != 0){
                    _inoratt = i/_ndim;
                    break;
                }

            for(size_t i=0;i<_origin.size();i+=_ndim)
                _Nnoratt += _rattlers[i];

            if (_perform_convergence_test)
                _ranvec/=norm(_ranvec);
                _lowesteig_ts.reserve(_ts_niter);

            /*if (_record_minimum_list)
                _aligned_quenched_coords.resize(_origin.size());*/
        }

//compute distance from origin after aligning the centre of mass
//this ignores the rattlers completely

inline void CheckSameMinimum::_get_vec_distance(pele::Array<double> quenched_coords){
        pele::Array<double> dr(_ndim);

        //measure distance between two non rattlers
        _dist_policy->get_rij(dr.data(), &quenched_coords[_inoratt], &_origin[_inoratt]);

        //align structures
        for(size_t i=0;i<_nparticles;++i)
        {
            size_t i1 = i*_ndim;
            for(size_t j=0;j<_ndim;++j){
                quenched_coords[i1+j] -= dr[j];
            }
        }
        if (_record_minimum_list)
            _aligned_quenched_coords = quenched_coords.copy();

        //compute distance between aligned structures
        for(size_t i=0;i<_nparticles;++i)
        {
            size_t i1 = i*_ndim;
            _dist_policy->get_rij(dr.data(), &quenched_coords[i1], &_origin[i1]);

            for(size_t j=0;j<_ndim;++j){
                _distance[i1+j] = dr[j] * _rattlers[i1+j];
            }
        }
    }

inline void CheckSameMinimum::_record_lowesteig_ts(double eig, MC * mc)
{
    size_t counter = mc->get_iterations_count();
        if (counter % _record_every == 0){
            _lowesteig_ts.push_back(eig);
        }
}

inline double CheckSameMinimum::_check_convergence(pele::Array<double> quenched_coords)
{
    bool minimum = false;
    double lowesteig;
    size_t l = 0;
    while (minimum == false && l < 10){
        minimum = true;
        pele::LowestEigPotential lowesteigpot(_potential, quenched_coords, _ndim);
        pele::LBFGS lbfgs(&lowesteigpot, _ranvec.copy(), _lbfgstol, _lbfgsM);
        lbfgs.set_maxstep(_lbfgsmaxstep);
        lbfgs.set_H0(_H0);
        lbfgs.set_use_relative_f(1);
        lbfgs.run(_lbfgsniter);
        _H0 = lbfgs.get_H0();
        lowesteig = lbfgs.get_f();
        if ( lowesteig < _eigtol){
            minimum = false;
            _optimizer->set_tol(_lowtol);
            _optimizer->run();
            std::cout<<"NOT A MINIMUM"<<std::endl;
        }
        ++l;
    }
    _optimizer->set_tol(_hightol);
    return lowesteig;
}

bool CheckSameMinimum::test(Array<double> &trial_coords, MC * mc)
{
    _optimizer->reset(trial_coords);
    _optimizer->run();

    if (_perform_convergence_test){
        double eig = this->_check_convergence(_optimizer->get_x());
        this->_record_lowesteig_ts(eig, mc);
    }

    //add number of energy evaluations to mc eval count
    size_t nfev = _optimizer->get_nfev();
    mc->_neval += nfev;

    //first test: minimisation must have converged
    bool quench_success = _optimizer->success();
    if (! quench_success)
        return false;


    //compute distance between quenched coords and origin
    //distance for rattlers is set to 0
    this->_get_vec_distance(_optimizer->get_x());

    //compute rms displacement from origin
    _d = norm(_distance);
    _rms = _d / sqrt(_Nnoratt);
    if (_rms > _dtol){
	if (_record_minimum_list)
	    this->_record_minimum();
        //std::cout<<"failed quench rms "<<_rms<<std::endl;
        return false;
    }
    else{
        //std::cout<<"successfull quench rms "<<_rms<<std::endl;
        return true;
    }
}

/*inline void CheckSameMinimum::_record_minimum()
{
    _minima_list.check_new_minimum(_d, _optimizer->get_f(), _aligned_quenched_coords, _rattlers);
}*/

class CheckSameMinimum2D:public CheckSameMinimum{
public:
    CheckSameMinimum2D(pele::GradientOptimizer * optimizer, pele::BasePotential * potential, Array<double> origin, Array<double> hs_radii,
                        Array<double> rattlers, double dtol):
        CheckSameMinimum(optimizer, potential, origin, hs_radii, rattlers, dtol, 2,
                std::make_shared<pele::CartesianDistanceWrapper<2>>())
        {}
};

class CheckSameMinimum3D:public CheckSameMinimum{
public:
    CheckSameMinimum3D(pele::GradientOptimizer * optimizer, pele::BasePotential * potential, Array<double> origin, Array<double> hs_radii,
                        Array<double> rattlers, double dtol):
        CheckSameMinimum(optimizer, potential, origin, hs_radii, rattlers, dtol, 3,
                std::make_shared<pele::CartesianDistanceWrapper<3>>())
        {}
};

class CheckSameMinimumPeriodic2D:public CheckSameMinimum{
public:
    CheckSameMinimumPeriodic2D(pele::GradientOptimizer * optimizer, pele::BasePotential * potential, Array<double> origin, Array<double> hs_radii,
                        double const *boxvec, Array<double> rattlers, double dtol):
        CheckSameMinimum(optimizer, potential, origin, hs_radii, rattlers, dtol, 2,
                std::make_shared<pele::PeriodicDistanceWrapper<2>>(boxvec))
        {}
};

class CheckSameMinimumPeriodic3D:public CheckSameMinimum{
public:
    CheckSameMinimumPeriodic3D(pele::GradientOptimizer * optimizer, pele::BasePotential * potential, Array<double> origin, Array<double> hs_radii,
                        double const *boxvec, Array<double> rattlers, double dtol):
            CheckSameMinimum(optimizer, potential, origin, hs_radii, rattlers, dtol, 3,
                    std::make_shared<pele::PeriodicDistanceWrapper<3> >(boxvec))
        {}
};

}


#endif
