#ifndef _BV_CONVERGENCE_TEST_H
#define _BV_CONVERGENCE_TEST_H

#include <memory>
#include "pele/array.h"

namespace bv{


class convergence_test{
private:
    double _lbfgstol, _lbfgsM, _lbfgsniter, _lbfgsmaxstep, _lowtol, _hightol, _eigtol, _H0;
    pele::Array<double> _ranvec;
public:
    convergence_test(const double _lbfgstol_, const double _lbfgsM_, const
            double _lbfgsniter_, const double _lbfgsmaxstep_, const double
            _lowtol_, const double _hightol_, const double _eigtol_, const
            double _H0_, pele::Array<double> _ranvec_)
        : _lbfgstol(_lbfgstol_), _lbfgsM(_lbfgsM_), _lbfgsniter(_lbfgsniter_),
          _lbfgsmaxstep(_lbfgsmaxstep_), _lowtol(_lowtol_), _hightol(_hightol_),
          _eigtol(_eigtol_), _H0(_H0_), _ranvec(_ranvec_.copy())
    {
        _ranvec /= norm(_ranvec);
    }
    void check_convergence(pele::Array<double> quenched_coords,
            std::shared_ptr<pele::BasePotential> _potential, const size_t _ndim,
            std::shared_ptr<pele::GradientOptimizer> _optimizer);
};


}//namespace bv

#endif//#ifndef _BV_CONVERGENCE_TEST_H
