#ifndef _BV_CHECK_SAME_MINIMUM_H
#define _BV_CHECK_SAME_MINIMUM_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>

#include "pele/array.h"
#include "pele/optimizer.h"
#include "pele/distance.h"
#include "pele/harmonic.h" //debug

#include "mcpele/mc.h"
#include "mcpele/conf_test.h"

#include "convergence_test.h"
#include "minima_list.h"

namespace bv{


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

class CheckSameMinimum:public mcpele::ConfTest{
protected:
    inline void _get_vec_distance(pele::Array<double> quenched_coords);
    inline void _check_convergence(pele::Array<double> quenched_coords);
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
    convergence_test _conv_test;
    //minima list
    bool _collect_minima_list;
    MinimaList _minima_list;
public:
    CheckSameMinimum(pele::GradientOptimizer * optimizer, 
            pele::BasePotential * potential, Array<double> origin,
            Array<double> hs_radii, Array<double> rattlers, double dtol, 
            size_t ndim, std::shared_ptr<pele::DistanceInterface> dist=NULL,
            const bool perform_convergence_test=false, 
            const bool collect_minima_list=false);
    virtual bool test(Array<double> &trial_coords, mcpele::MC * mc);
    virtual ~CheckSameMinimum() {}

    double get_distance() { return _d; }
    Array<double> get_distance_array()
    {
        return _distance.copy();
    }
    bool perform_convergence_test() const { return _perform_convergence_test; }
    bool collect_minima_list() const { return _collect_minima_list; }
    //forwarding minima database information to the outside
    size_t ml_nr_distinct_minima() const { return _minima_list.nr_distinct_minima(); }

    /**
     * return and Array of the minima we've found
     *
     * This is primarily for easy access in cython.  C++ code should probably
     * use the iterator syntax
     */
    pele::Array<Minimum *> get_array_of_minima()
    {
        pele::Array<Minimum *> minima(_minima_list.nr_distinct_minima());
        size_t i = 0;
        for (auto & m : _minima_list) {
            minima[i++] = &m;
        }
        return minima;
    }
};

template<size_t ndim>
class CheckSameMinimumCartesian:public CheckSameMinimum{
public:
    CheckSameMinimumCartesian(pele::GradientOptimizer * optimizer,
            pele::BasePotential * potential, Array<double> origin,
            Array<double> hs_radii, Array<double> rattlers, double dtol, bool
            perform_convergence_test=false, bool collect_minima_list=false)
        : CheckSameMinimum(optimizer, potential, origin, hs_radii, rattlers,
                dtol, ndim, std::make_shared<pele::CartesianDistanceWrapper<ndim> >(),
                perform_convergence_test, collect_minima_list)
    {}
};

template<size_t ndim>
class CheckSameMinimumPeriodic:public CheckSameMinimum{
public:
    CheckSameMinimumPeriodic(pele::GradientOptimizer * optimizer,
            pele::BasePotential * potential, Array<double> origin,
            Array<double> hs_radii, double const *boxvec, 
            Array<double> rattlers, double dtol, 
            bool perform_convergence_test=false, 
            bool collect_minima_list=false)
        : CheckSameMinimum(optimizer, potential, origin, hs_radii, rattlers,
                dtol, ndim,
                std::make_shared<pele::PeriodicDistanceWrapper<ndim> >(boxvec),
                perform_convergence_test, collect_minima_list)
    {}
};

}//namespace bv

#endif//#ifndef _BV_CHECK_SAME_MINIMUM_H
