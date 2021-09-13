#ifndef _BV_CONVERGENCE_TEST_H
#define _BV_CONVERGENCE_TEST_H

#include <memory>

#include "pele/array.hpp"

#include "mcpele/lowest_eigenvalue.h"

namespace bv{


template<class OPT_T=pele::GradientOptimizer>
class convergence_test{
private:
    const double _lowtol;
    const double _hightol;
    const double _eigtol;
    mcpele::FindLowestEigenvalue _ev_finder;
public:
    convergence_test(const size_t _lbfgsniter_, const double _lowtol_,
            const double _hightol_, const double _eigtol_,
            pele::Array<double> _ranvec_, std::shared_ptr<pele::BasePotential> landscape_potential,
            const size_t boxdimension)
        : _lowtol(_lowtol_),
          _hightol(_hightol_),
          _eigtol(_eigtol_),
          _ev_finder(landscape_potential, boxdimension, _ranvec_, _lbfgsniter_)
    {}

    void check_convergence(pele::Array<double> const & quenched_coords,
            std::shared_ptr<OPT_T> _optimizer);
};

/**
 * compute the lowest eigenvalue to ensure that it is positive
 */
template<class OPT_T>
void convergence_test<OPT_T>::check_convergence(pele::Array<double> const & quenched_coords,
        std::shared_ptr<OPT_T> _optimizer)
{
    bool minimum = false;
    size_t l = 0;
    while (minimum == false && l < 10){
        minimum = true;
        const double lowesteig = _ev_finder.compute_lowest_eigenvalue(quenched_coords);
        if (lowesteig < _eigtol){
            minimum = false;
            _optimizer->set_tol(_lowtol);
            _optimizer->run();
            std::cout<<"NOT A MINIMUM"<<std::endl;
        }
        ++l;
    }
    _optimizer->set_tol(_hightol);
}


}//namespace bv

#endif//#ifndef _BV_CONVERGENCE_TEST_H
