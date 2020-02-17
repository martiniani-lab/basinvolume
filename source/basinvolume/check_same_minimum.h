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
#include "pele/vecn.h"

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
 *             zero if the particle is a rattler.
 * _d: norm of distance
 * _dmax: maximum displacement from origin for any particle
 * _E = energy of the quenched state
 * _dtol: tolerance on distances
 * _Etol: tolerance on energies (a minimum should be whithin this value from _Emin)
 * _Eor: energy of the origin (must pass it because CheckSameMinimum knows nothing about the potential used by the optimiser)
 * _inoratt: index of first non-rattler
 * _Nnoratt: number of non-rattlers
 * */

class CheckSameMinimumInterface : public mcpele::ConfTest{
public:
    virtual ~CheckSameMinimumInterface(){};
    virtual bool conf_test(pele::Array<double> &trial_coords, mcpele::MC * mc)=0;
    virtual size_t ml_nr_distinct_minima() const =0;
    virtual pele::Array<Minimum*> get_array_of_minima() =0;
    virtual double get_failed_quench_frac() const =0;
};

template <typename distance_policy, class OPT_T=pele::GradientOptimizer>
class CheckSameMinimum : public CheckSameMinimumInterface {
protected:
    static const size_t _ndim = distance_policy::_ndim;
    inline void _align_coords(pele::Array<double> & coords);
    inline double _get_d2(pele::Array<double> const & coords);
    inline double _get_d2_max(pele::Array<double> const & coords);
    inline void _check_convergence(pele::Array<double> const & quenched_coords);
    bool _quench(pele::Array<double> &trial_coords);
    std::shared_ptr<OPT_T> _optimizer;
    std::shared_ptr<pele::BasePotential> _potential;
    pele::Array<double> _origin;
    pele::Array<double> _rattlers;
    pele::Array<double> _new_minimum;
    double _dtol, _dtol_barriercheck;
    double _d;
    double _dmax;
    size_t _nbarrierchecks;
    double _tol_barrier;
    size_t _nparticles;
    const std::shared_ptr<distance_policy> _dist_policy;
    size_t _Nnoratt;
    size_t _inoratt;
    mcpele::Moments m_failed_quench_frac;
    //convergence test classes
    bool _perform_convergence_test;
    convergence_test<OPT_T> _conv_test;
    //minima list
    size_t m_eqsteps;
    bool _collect_minima_list;
    MinimaList _minima_list;
public:
    CheckSameMinimum(std::shared_ptr<OPT_T> optimizer,
            std::shared_ptr<pele::BasePotential> potential,
            pele::Array<double> & origin, pele::Array<double> & rattlers, double dtol,
            const size_t eqsteps, std::shared_ptr<distance_policy> const & dist,
            const bool perform_convergence_test,
            const bool collect_minima_list);
    virtual bool conf_test(pele::Array<double> &trial_coords, mcpele::MC * mc);
    virtual ~CheckSameMinimum() {}
    double get_distance() const { return _d; }
    bool perform_convergence_test() const { return _perform_convergence_test; }
    bool collect_minima_list() const { return _collect_minima_list; }
    //forwarding minima database information to the outside
    virtual size_t ml_nr_distinct_minima() const { return _minima_list.nr_distinct_minima(); }
    virtual double get_failed_quench_frac() const { return m_failed_quench_frac.mean(); }
    /**
     * return and Array of the minima we've found
     *
     * This is primarily for easy access in cython.  C++ code should probably
     * use the iterator syntax
     */
    virtual pele::Array<Minimum*> get_array_of_minima();
};

template <typename distance_policy, class OPT_T>
CheckSameMinimum<distance_policy, OPT_T>::CheckSameMinimum(std::shared_ptr<OPT_T> optimizer,
        std::shared_ptr<pele::BasePotential> potential, pele::Array<double> & origin,
        pele::Array<double> & rattlers, double dtol,
        const size_t eqsteps, std::shared_ptr<distance_policy> const & dist,
        const bool perform_convergence_test, const bool collect_minima_list)
    : _optimizer(optimizer),
      _potential(potential),
      _origin(origin.copy()),
      _rattlers(rattlers.size() / _ndim),
      _new_minimum(origin.size()),
      _dtol(dtol),
      _dtol_barriercheck(100 * _dtol),
      _tol_barrier(1e-5),
      _nbarrierchecks(20),
      _d(0),
      _dmax(0),
      _nparticles(origin.size() / _ndim),
      _dist_policy(dist),
      _Nnoratt(0),
      _perform_convergence_test(perform_convergence_test),
      _conv_test(30, 1e-10, _optimizer->get_tol(), 0.1, _origin, potential, _ndim),
      m_eqsteps(eqsteps),
      _collect_minima_list(collect_minima_list),
      _minima_list(_dtol * sqrt(origin.size()), _optimizer->get_tol(), _dtol)
{
    if (_dist_policy == NULL) {
        throw std::runtime_error("CheckSameMinimum::CheckSameMinimum: distance policy uninitialised");
    }
    if (_origin.size() != rattlers.size()) {
        throw std::runtime_error("CheckSameMinimum::CheckSameMinimum: illegal input: origin vs rattlers");
    }
    if (_origin.size() % _ndim) {
        throw std::runtime_error("CheckSameMinimum::CheckSameMinimum: illegal input: origin vs boxdimension");
    }
    bool no_stable_yet = true;
    for (size_t i = 0; i < _rattlers.size(); ++i) {
        _rattlers[i] = rattlers[i*_ndim];
        if (no_stable_yet && _rattlers[i] != 0) {
            _inoratt = i;
            no_stable_yet = false;
        }
        _Nnoratt += _rattlers[i];
    }
}

template <typename distance_policy, class OPT_T>
pele::Array<Minimum*> CheckSameMinimum<distance_policy, OPT_T>::get_array_of_minima()
{
    pele::Array<Minimum*> minima(_minima_list.nr_distinct_minima());
    size_t i = 0;
    for (auto & m : _minima_list) {
        minima[i++] = &m;
    }
    return minima;
}

template <typename distance_policy, class OPT_T>
void CheckSameMinimum<distance_policy, OPT_T>::_check_convergence(pele::Array<double> const & quenched_coords)
{
    _conv_test.check_convergence(quenched_coords, _optimizer);
}

/**
 * aligns structures
 */
template <typename distance_policy, class OPT_T>
void CheckSameMinimum<distance_policy, OPT_T>::_align_coords(pele::Array<double> & coords)
{
    /*assert(coords.size() == _origin.size());
    assert(coords.size() == _ndim * _nparticles);*/
    pele::VecN<_ndim, double> dr;

    //measure distance between a non-rattler and its origin
    _dist_policy->get_rij(dr.data(), &coords[_inoratt*_ndim], &_origin[_inoratt*_ndim]);

    //align structures
    #pragma simd
    for (size_t i = 0; i < _nparticles; ++i) {
        const size_t i1 = i * _ndim;
        #pragma unroll
        for (size_t j = 0; j < _ndim; ++j) {
            coords[i1 + j] -= dr[j];
        }
    }
}

/*compute the squared distance from origin after aligning one particle with its origin
this ignores the rattlers completely*/
template <typename distance_policy, class OPT_T>
double CheckSameMinimum<distance_policy, OPT_T>::_get_d2(pele::Array<double> const & coords)
{
    double distance2 = 0;
    pele::VecN<_ndim, double> dr_align;

    //measure distance between a non-rattler and its origin
    _dist_policy->get_rij(dr_align.data(), &coords[_inoratt*_ndim], &_origin[_inoratt*_ndim]);

    //compute distance between aligned structures
    for (size_t i = 0; i < _nparticles; ++i) {
        const size_t i1 = i * _ndim;
        pele::VecN<_ndim, double> dr, x_aligned;
        #pragma unroll
        for (size_t j = 0; j < _ndim; ++j) {
            x_aligned[j] = coords[i1 + j] - dr_align[j];
        }
        _dist_policy->get_rij(dr.data(), x_aligned.data(), &_origin[i1]);
        #pragma unroll
        for (size_t j = 0; j < _ndim; ++j) {
            distance2 += dr[j] * dr[j] * _rattlers[i];
        }
    }

    //avoid taking square roots by returning squared quantities
    return distance2;
}

/*compute maximum of squared distances between a particle and its origin after aligning one particle with its origin
this ignores the rattlers completely*/
template <typename distance_policy, class OPT_T>
double CheckSameMinimum<distance_policy, OPT_T>::_get_d2_max(pele::Array<double> const & coords)
{
    double distance2 = 0;
    pele::VecN<_ndim, double> dr_align;

    //measure distance between a non-rattler and its origin
    _dist_policy->get_rij(dr_align.data(), &coords[_inoratt*_ndim], &_origin[_inoratt*_ndim]);

    //compute distance between aligned structures
    for (size_t i = 0; i < _nparticles; ++i) {
        const size_t i1 = i * _ndim;
        pele::VecN<_ndim, double> dr, x_aligned;
        #pragma unroll
        for (size_t j = 0; j < _ndim; ++j) {
            x_aligned[j] = coords[i1 + j] - dr_align[j];
        }
        _dist_policy->get_rij(dr.data(), x_aligned.data(), &_origin[i1]);
        double current_distance2 = 0;
        #pragma unroll
        for (size_t j = 0; j < _ndim; ++j) {
            current_distance2 += dr[j] * dr[j];
        }
        if (_rattlers[i] && current_distance2 > distance2) {
            distance2 = current_distance2;
        }
    }

    //avoid taking square roots by returning squared quantities
    return distance2;
}

/*quench configuration and add minimum to new minimum list*/
template <typename distance_policy, class OPT_T>
bool CheckSameMinimum<distance_policy, OPT_T>::_quench(pele::Array<double> &trial_coords)
{
    _optimizer->reset(trial_coords);

    bool success = true;
    double d2max = this->_get_d2_max(_optimizer->get_x());
    double dtol2 = _dtol * _dtol;
    const size_t opt_maxiter = _optimizer->get_maxiter();

    //this might become an infinite loop
    //optimizer stop-criterion needs to be checked before calling one_iteration
    while (d2max > dtol2 && static_cast<size_t>(_optimizer->get_niter()) < opt_maxiter) {
        if (_optimizer->stop_criterion_satisfied()) {
            //minimisation converged before satisfying distance criterion,
            //save minimum and return false
            success = false;
            break;
        }
        _optimizer->one_iteration();
        d2max = this->_get_d2_max(_optimizer->get_x());
    }

    //assign attributes for rms displacement from origin
    _dmax = sqrt(d2max);
    _d = sqrt(this->_get_d2(_optimizer->get_x()));

    return success;
}

/* template specialization when using cg_descent
 * Since a partially specialized class member is impossible, this needs to be
 * done for each distance_policy when using CG
 */
template <>
bool CheckSameMinimum<pele::cartesian_distance<2UL>, BvCGDescent<pele::cartesian_distance<2UL>>>::_quench(pele::Array<double> &trial_coords){
    _optimizer->reset(trial_coords);
    _optimizer->run();
    _dmax = sqrt(_optimizer->get_d2_max());
    _d = sqrt(_optimizer->get_d2());
    return _optimizer->success();
}
template <>
bool CheckSameMinimum<pele::cartesian_distance<3UL>, BvCGDescent<pele::cartesian_distance<3UL>>>::_quench(pele::Array<double> &trial_coords){
    _optimizer->reset(trial_coords);
    _optimizer->run();
    _dmax = sqrt(_optimizer->get_d2_max());
    _d = sqrt(_optimizer->get_d2());
    return _optimizer->success();
}
// linetest
template <typename distance_policy, class OPT_T>
bool CheckSameMinimum<distance_policy, OPT_T>::conf_test(pele::Array<double> &trial_coords, mcpele::MC * mc)
{
    bool quench_success;
    bool same_minimum = this->_quench(trial_coords);

    // Check if this is indeed a different minimum by searching for a barrier between origin and coordinates
    if (!same_minimum && _dmax <= _dtol_barriercheck)
    {
        quench_success = true;
        double energy_minima = std::max(_potential->get_energy(_origin), _optimizer->get_f());
        bool barrier_found = false;
        for (int icheck = 1; icheck <= _nbarrierchecks; ++icheck) {
            double stepratio = icheck / (_nbarrierchecks + 1);
            #pragma simd
            for (int i = 0; i < _origin.size(); ++i) {
                _new_minimum[i] = (1 - stepratio) * _origin[i] + stepratio * _optimizer->get_x()[i];
            }
            if(_potential->get_energy(_new_minimum) > energy_minima + _tol_barrier) {
                barrier_found = true;
                break;
            }
        }
        same_minimum = !barrier_found;
    } else {
        // check if minimisation has converged
        // if exited loop with dmax > dtol and success == true
        // then the quench has failed in the given no. of steps
        quench_success = _dmax <= _dtol || !same_minimum;
    }

    //add number of energy evaluations to mc eval count
    const size_t nfev = _optimizer->get_nfev();
    mc->m_neval += nfev;

    if (_perform_convergence_test) {
        this->_check_convergence(_optimizer->get_x());
    }

    m_failed_quench_frac.update(!quench_success);
    if (!quench_success) {
        return false;
    }

    if (!same_minimum) {
        //if quench has converged to different minimum then one might want to
        //save the new minimum
        // std::cout <<" failed quench dmax " << _dmax << ", dtol " << _dtol << std::endl;
        if (_collect_minima_list && mc->get_iterations_count() > m_eqsteps) {
            _new_minimum.assign(_optimizer->get_x());
            this->_align_coords(_new_minimum);
            _minima_list.insert_minimum(_d, _optimizer->get_f(), _new_minimum, _rattlers);
        }
        return false;
    }
    else {
        return true;
    }
}

template<size_t ndim>
class CheckSameMinimumCartesian : public CheckSameMinimum<pele::cartesian_distance<ndim>> {
public:
    CheckSameMinimumCartesian(std::shared_ptr<pele::GradientOptimizer> optimizer,
            std::shared_ptr<pele::BasePotential> potential,
            pele::Array<double> origin, pele::Array<double> rattlers, double dtol,
            size_t eqsteps=0, bool perform_convergence_test=false,
            bool collect_minima_list=false)
        : CheckSameMinimum<pele::cartesian_distance<ndim>>(optimizer, potential, origin, rattlers,
                dtol, eqsteps, std::make_shared<pele::cartesian_distance<ndim> >(),
                perform_convergence_test, collect_minima_list)
    {}
};

template<size_t ndim>
class CheckSameMinimumPeriodic : public CheckSameMinimum<pele::periodic_distance<ndim>> {
public:
    CheckSameMinimumPeriodic(std::shared_ptr<pele::GradientOptimizer> optimizer,
            std::shared_ptr<pele::BasePotential> potential,
            pele::Array<double> origin, pele::Array<double> boxvec,
            pele::Array<double> rattlers, double dtol,
            size_t eqsteps=0, bool perform_convergence_test=false,
            bool collect_minima_list=false)
        : CheckSameMinimum<pele::periodic_distance<ndim>>(optimizer, potential, origin, rattlers,
                dtol, eqsteps,
                std::make_shared<pele::periodic_distance<ndim> >(boxvec),
                perform_convergence_test, collect_minima_list)
    {}
};

template<size_t ndim>
class CheckSameMinimumCGDCartesian : public CheckSameMinimum<pele::cartesian_distance<ndim>, BvCGDescent<pele::cartesian_distance<ndim>>> {
public:
    CheckSameMinimumCGDCartesian(
            std::shared_ptr<pele::BasePotential> potential,
            pele::Array<double> origin, pele::Array<double> rattlers,
            double tol, double dtol,
            size_t opt_maxiter, size_t opt_PrintLevel, size_t eqsteps=0,
            bool perform_convergence_test=false,
            bool collect_minima_list=false)
        : CheckSameMinimum<pele::cartesian_distance<ndim>, BvCGDescent<pele::cartesian_distance<ndim>>>(std::make_shared<BvCGDescent<pele::cartesian_distance<ndim>>>(potential, origin, origin,
                rattlers, std::make_shared<pele::cartesian_distance<ndim>>(),
                tol, dtol, opt_maxiter, opt_PrintLevel),
                potential, origin, rattlers,
                dtol, eqsteps,
                std::make_shared<pele::cartesian_distance<ndim> >(),
                perform_convergence_test, collect_minima_list)
    {}
};

template<size_t ndim>
class CheckSameMinimumCGDPeriodic : public CheckSameMinimum<pele::periodic_distance<ndim>, BvCGDescent<pele::periodic_distance<ndim>>> {
public:
    CheckSameMinimumCGDPeriodic(
            std::shared_ptr<pele::BasePotential> potential,
            pele::Array<double> origin, pele::Array<double> boxvec,
            pele::Array<double> rattlers, double tol, double dtol,
            size_t opt_maxiter, size_t opt_PrintLevel, size_t eqsteps=0,
            bool perform_convergence_test=false,
            bool collect_minima_list=false)
        : CheckSameMinimum<pele::periodic_distance<ndim>, BvCGDescent<pele::periodic_distance<ndim>>>(std::make_shared<BvCGDescent<pele::periodic_distance<ndim>>>(potential, origin, origin,
                rattlers, std::make_shared<pele::periodic_distance<ndim>>(boxvec),
                tol, dtol, opt_maxiter, opt_PrintLevel),
                potential, origin, rattlers,
                dtol, eqsteps,
                std::make_shared<pele::periodic_distance<ndim> >(boxvec),
                perform_convergence_test, collect_minima_list)
    {}
};

} // namespace bv

#endif // #ifndef _BV_CHECK_SAME_MINIMUM_H
