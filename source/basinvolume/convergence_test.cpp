#include "pele/lbfgs.h"
#include "pele/lowest_eig_potential.h"

#include "convergence_test.h"

namespace bv{


void convergence_test::check_convergence(pele::Array<double> quenched_coords,
        pele::BasePotential * _potential, const size_t _ndim,
        pele::GradientOptimizer * _optimizer)
{
    //std::cout << "convergence_test::check_convergence" << std::endl;
    bool minimum = false;
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
        double lowesteig = lbfgs.get_f();
        if ( lowesteig < _eigtol){
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
