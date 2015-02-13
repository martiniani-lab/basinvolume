#ifndef _BV_CHECK_SAME_MINIMUM_H
#define _BV_CHECK_SAME_MINIMUM_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "pele/distance.h"
#include "pele/optimizer.h"

#include "mcpele/mc.h"
#include "mcpele/histogram.h"

#include "convergence_test.h"
#include "minima_list.h"
#include "bv_cg_descent.h"


namespace bv {

/*check same minimum class
 * _optimizer: pointer to object of class GradientOptimizer performing minimisation according to some potential
 *                 passed to the object during its construction
 * _origin: coordinates to which the quenched structure is compared to
 * _rattlers: array of 1s or 0s: if not indicates a rattler,0 -> rattler
 *                                                             1 -> jammed particle
 *             this convention removes if statements in the for loop and replaces them with
 *             arithmetic operation (distance[i] *= rattlers[i]), distance is set artificially to
 *             zero if the particle is a rattler. note _rattlers.size() = coords.size()
 * _distance: array containing the Euclidean distance between trial_coords and origin
 * _d: norm of distance
 * _rms: root mean square displacement from origin
 * _E = energy of the quenched state
 * _dtol: tolerance on distances
 * _Etol: tolerance on energies (a minimum should be whithin this value from _Emin)
 * _Eor: energy of the origin (must pass it because CheckSameMinimum knows nothing about the potential used by the optimiser)
 * _inoratt: index of first non-rattler
 * _Nnoratt: number of non-rattlers
 * */

template <class OPT_T=pele::GradientOptimizer>
class CheckSameMinimum : public mcpele::ConfTest {
protected:
    inline pele::Array<double> _align_coords(pele::Array<double> coords);
    inline double _get_d2(pele::Array<double> coords);
    inline void _check_convergence(pele::Array<double> quenched_coords);
    bool _quench(pele::Array<double> &trial_coords);
    size_t _ndim;
    std::shared_ptr<OPT_T> _optimizer;
    std::shared_ptr<pele::BasePotential> _potential;
    pele::Array<double> _origin;
    pele::Array<double> _rattlers;
    pele::Array<double> _distance;
    pele::Array<double> _new_minimum;
    double _dtol;
    double _d;
    double _rms;
    size_t _nparticles;
    std::shared_ptr<pele::DistanceInterface> _dist_policy;
    size_t _Nnoratt;
    size_t _inoratt;
    mcpele::Moments m_failed_quench_frac;
    //convergence test classes
    bool _perform_convergence_test;
    convergence_test _conv_test;
    //minima list
    size_t m_eqsteps;
    bool _collect_minima_list;
    MinimaList _minima_list;
public:
    CheckSameMinimum(std::shared_ptr<OPT_T> optimizer,
            std::shared_ptr<pele::BasePotential> potential,
            pele::Array<double> origin, pele::Array<double> rattlers, double dtol,
            size_t ndim, const size_t eqsteps=0, std::shared_ptr<pele::DistanceInterface> dist=NULL,
            const bool perform_convergence_test=false, 
            const bool collect_minima_list=false);
    virtual bool conf_test(pele::Array<double> &trial_coords, mcpele::MC * mc);
    virtual ~CheckSameMinimum() {}
    double get_distance() const { return _d; }
    pele::Array<double> get_distance_array() const { return _distance.copy(); }
    bool perform_convergence_test() const { return _perform_convergence_test; }
    bool collect_minima_list() const { return _collect_minima_list; }
    //forwarding minima database information to the outside
    size_t ml_nr_distinct_minima() const { return _minima_list.nr_distinct_minima(); }
    double get_failed_quench_frac() const { return m_failed_quench_frac.mean(); }
    /**
     * return and Array of the minima we've found
     *
     * This is primarily for easy access in cython.  C++ code should probably
     * use the iterator syntax
     */
    pele::Array<Minimum*> get_array_of_minima();
};

template <class OPT_T>
CheckSameMinimum<OPT_T>::CheckSameMinimum(std::shared_ptr<OPT_T> optimizer,
        std::shared_ptr<pele::BasePotential> potential, pele::Array<double> origin,
        pele::Array<double> rattlers, double dtol, size_t ndim,
        const size_t eqsteps, std::shared_ptr<pele::DistanceInterface> dist,
        const bool perform_convergence_test, const bool collect_minima_list)
    : _ndim(ndim),
      _optimizer(optimizer),
      _potential(potential),
      _origin(origin.copy()),
      _rattlers(rattlers.copy()),
      _distance(origin.size(), 0),
      _new_minimum(origin.size()),
      _dtol(dtol),
      _d(0),
      _rms(0),
      _nparticles(origin.size() / ndim),
      _dist_policy(dist),
      _Nnoratt(0),
      _perform_convergence_test(perform_convergence_test),
      _conv_test(30, 1e-10, _optimizer->get_tol(), 0.1, _origin, potential, ndim),
      m_eqsteps(eqsteps),
      _collect_minima_list(collect_minima_list),
      _minima_list(_dtol * sqrt(origin.size()), _optimizer->get_tol(), _dtol)
{
    if (_dist_policy == NULL) {
        throw std::runtime_error("CheckSameMinimum::CheckSameMinimum: distance policy uninitialised");
    }
    if (_origin.size() != _rattlers.size()) {
        throw std::runtime_error("CheckSameMinimum::CheckSameMinimum: illegal input: origin vs rattlers");
    }
    if (_origin.size() % _ndim) {
        throw std::runtime_error("CheckSameMinimum::CheckSameMinimum: illegal input: origin vs boxdimension");
    }
    for (size_t i = 0; i < _origin.size(); i +=_ndim) {
        if (_rattlers[i] != 0){
            _inoratt = i / _ndim;
            break;
        }
    }
    for (size_t i = 0; i < _origin.size(); i +=_ndim) {
        _Nnoratt += _rattlers[i];
    }
}

template <class OPT_T>
pele::Array<Minimum*> CheckSameMinimum<OPT_T>::get_array_of_minima()
{
    pele::Array<Minimum*> minima(_minima_list.nr_distinct_minima());
    size_t i = 0;
    for (auto & m : _minima_list) {
        minima[i++] = &m;
    }
    return minima;
}

template <class OPT_T>
void CheckSameMinimum<OPT_T>::_check_convergence(pele::Array<double> quenched_coords)
{
    _conv_test.check_convergence(quenched_coords, _optimizer);
}

template <>
void CheckSameMinimum<BvCGDescent>::_check_convergence(pele::Array<double> quenched_coords)
{
    //do nothing (conv test assumes that optimizer is of type pele::GradientDescent)
    //in principle check_convergence can be templated easily, I have not done it because
    //this function is currently not being used
}

/**
 * aligns structures
 */
template <class OPT_T>
pele::Array<double> CheckSameMinimum<OPT_T>::_align_coords(pele::Array<double> coords)
{
    /*assert(coords.size() == _origin.size());
    assert(coords.size() == _ndim * _nparticles);*/
    pele::Array<double> dr(_ndim);

    //measure distance between two non rattlers
    _dist_policy->get_rij(dr.data(), &coords[_inoratt], &_origin[_inoratt]);

    //align structures
    for (size_t i = 0; i < _nparticles; ++i) {
        const size_t i1 = i * _ndim;
        for (size_t j = 0; j < _ndim; ++j) {
            coords[i1+j] -= dr[j];
        }
    }

    return coords;
}

/*compute distance from origin after aligning two particles
this ignores the rattlers completely and returns rmsd squared*/
template <class OPT_T>
double CheckSameMinimum<OPT_T>::_get_d2(pele::Array<double> coords)
{
    pele::Array<double> dr(_ndim);
    pele::Array<double> aligned_coords = this->_align_coords(coords);

    //compute distance between aligned structures
    for (size_t i = 0; i < _nparticles; ++i) {
        const size_t i1 = i * _ndim;
        _dist_policy->get_rij(dr.data(), &aligned_coords[i1], &_origin[i1]);
        for (size_t j = 0; j < _ndim; ++j) {
            _distance[i1 + j] = dr[j] * _rattlers[i1 + j];
        }
    }

    //avoid taking square roots by return squared quantities
    return dot(_distance,_distance);
}

/*quench configuration and add minimum to new minimum list*/

template <class OPT_T>
bool CheckSameMinimum<OPT_T>::_quench(pele::Array<double> &trial_coords)
{
    _optimizer->reset(trial_coords);

    bool success = true;
    double d2 = this->_get_d2(_optimizer->get_x());
    double rmsd2 = d2 / _Nnoratt;
    double dtol2 = _dtol * _dtol;
    const size_t opt_maxiter = _optimizer->get_maxiter();

    //this might become an infinite loop
    //optimizer stop-criterion needs to be checked before calling one_iteration
    while (rmsd2 > dtol2 && static_cast<size_t>(_optimizer->get_niter()) < opt_maxiter) {
        if (_optimizer->stop_criterion_satisfied()) {
            //minimisation converged before satisfying distance criterion,
            //save minimum and return false
            success = false;
            break;
        }
        _optimizer->one_iteration();
        d2 = this->_get_d2(_optimizer->get_x());
        rmsd2 = d2 / _Nnoratt;
    }

    //assign attributes for rms displacement from origin
    _d = sqrt(d2);
    _rms = sqrt(rmsd2);

    return success;
}


//template specialization when using cg_descent
template <>
bool CheckSameMinimum<BvCGDescent>::_quench(pele::Array<double> &trial_coords){
    _optimizer->reset(trial_coords);
    _optimizer->run();
    _d = sqrt(_optimizer->get_d2());
    _rms = sqrt(_optimizer->get_rmsd2());
    bool success = _optimizer->success();
    return success;
}

template <class OPT_T>
bool CheckSameMinimum<OPT_T>::conf_test(pele::Array<double> &trial_coords, mcpele::MC * mc)
{
    bool quench_success = true;
    bool same_minimum = this->_quench(trial_coords);

    //add number of energy evaluations to mc eval count
    const size_t nfev = _optimizer->get_nfev();
    mc->m_neval += nfev;

    if (_perform_convergence_test) {
        this->_check_convergence(_optimizer->get_x());
    }

    //check if minimisation has converged
    //if exited loop with rmsd>dtol2 and success == true
    //then the quench has failed in the given no. of steps
    if (_rms > _dtol && same_minimum) {
        quench_success = false;
    }
    m_failed_quench_frac.update(!quench_success);
    if (!quench_success) {
        return false;
    }

    if (!same_minimum) {
        //if quench has converged to different minimum then one might want to
        //save the new minimum
        //std::cout<<"failed quench rms "<<_rms<<"dtol"<<_dtol<<std::endl;
        if (_collect_minima_list && mc->get_iterations_count() > m_eqsteps) {
            _new_minimum.assign(this->_align_coords(_optimizer->get_x()));
            _minima_list.insert_minimum(_d, _optimizer->get_f(), _new_minimum, _rattlers);
        }
        return false;
    }
    else {
        return true;
    }
}

template<size_t ndim>
class CheckSameMinimumCartesian : public CheckSameMinimum<> {
public:
    CheckSameMinimumCartesian(std::shared_ptr<pele::GradientOptimizer> optimizer,
            std::shared_ptr<pele::BasePotential> potential,
            pele::Array<double> origin, pele::Array<double> rattlers, double dtol,
            size_t eqsteps=0, bool perform_convergence_test=false,
            bool collect_minima_list=false)
        : CheckSameMinimum<>(optimizer, potential, origin, rattlers,
                dtol, ndim, eqsteps, std::make_shared<pele::CartesianDistanceWrapper<ndim> >(),
                perform_convergence_test, collect_minima_list)
    {}
};

template<size_t ndim>
class CheckSameMinimumPeriodic : public CheckSameMinimum<> {
public:
    CheckSameMinimumPeriodic(std::shared_ptr<pele::GradientOptimizer> optimizer,
            std::shared_ptr<pele::BasePotential> potential,
            pele::Array<double> origin, pele::Array<double> boxvec,
            pele::Array<double> rattlers, double dtol,
            size_t eqsteps=0, bool perform_convergence_test=false,
            bool collect_minima_list=false)
        : CheckSameMinimum<>(optimizer, potential, origin, rattlers,
                dtol, ndim, eqsteps,
                std::make_shared<pele::PeriodicDistanceWrapper<ndim> >(boxvec),
                perform_convergence_test, collect_minima_list)
    {}
};

template<size_t ndim>
class CheckSameMinimumCGDCartesian : public CheckSameMinimum<BvCGDescent> {
public:
    CheckSameMinimumCGDCartesian(
            std::shared_ptr<pele::BasePotential> potential,
            pele::Array<double> origin, pele::Array<double> rattlers,
            double etol, double dtol,
            size_t opt_maxiter, size_t opt_PrintLevel, size_t eqsteps=0,
            bool perform_convergence_test=false,
            bool collect_minima_list=false)
        : CheckSameMinimum<BvCGDescent>(std::make_shared<BvCGDescent>(potential, origin, origin,
                rattlers, ndim, std::make_shared<pele::CartesianDistanceWrapper<ndim> >(),
                etol, dtol, opt_maxiter, opt_PrintLevel),
                potential, origin, rattlers,
                dtol, ndim, eqsteps,
                std::make_shared<pele::CartesianDistanceWrapper<ndim> >(),
                perform_convergence_test, collect_minima_list)
    {}
};

template<size_t ndim>
class CheckSameMinimumCGDPeriodic : public CheckSameMinimum<BvCGDescent> {
public:
    CheckSameMinimumCGDPeriodic(
            std::shared_ptr<pele::BasePotential> potential,
            pele::Array<double> origin, pele::Array<double> boxvec,
            pele::Array<double> rattlers, double etol, double dtol,
            size_t opt_maxiter, size_t opt_PrintLevel, size_t eqsteps=0,
            bool perform_convergence_test=false,
            bool collect_minima_list=false)
        : CheckSameMinimum<BvCGDescent>(std::make_shared<BvCGDescent>(potential, origin, origin,
                rattlers, ndim, std::make_shared<pele::PeriodicDistanceWrapper<ndim> >(boxvec),
                etol, dtol, opt_maxiter, opt_PrintLevel),
                potential, origin, rattlers,
                dtol, ndim, eqsteps,
                std::make_shared<pele::PeriodicDistanceWrapper<ndim> >(boxvec),
                perform_convergence_test, collect_minima_list)
    {}
};

} // namespace bv

#endif // #ifndef _BV_CHECK_SAME_MINIMUM_H
