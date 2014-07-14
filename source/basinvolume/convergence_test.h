#ifndef _BV_CONVERGENCE_TEST_H
#define _BV_CONVERGENCE_TEST_H

#include <memory>

#include "pele/array.h"

#include "mcpele/lowest_eigenvalue.h"

namespace bv{


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
    void check_convergence(pele::Array<double> quenched_coords,
            std::shared_ptr<pele::GradientOptimizer> _optimizer);
};


}//namespace bv

#endif//#ifndef _BV_CONVERGENCE_TEST_H
