#include <memory>
#include "pele/lbfgs.h"
#include "pele/lowest_eig_potential.h"

#include "basinvolume/convergence_test.h"

namespace bv{

/**
 * compute the lowest eigenvalue to ensure that it is positive
 */
void convergence_test::check_convergence(pele::Array<double> quenched_coords,
        std::shared_ptr<pele::GradientOptimizer> _optimizer)
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
