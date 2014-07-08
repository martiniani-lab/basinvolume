#include <cmath>

#include "check_same_minimum.h"

using pele::Array;
using mcpele::MC;

namespace bv{


CheckSameMinimum::CheckSameMinimum(pele::GradientOptimizer * optimizer,
        pele::BasePotential * potential, Array<double> origin, Array<double>
        hs_radii, Array<double> rattlers, double dtol, size_t ndim,
        std::shared_ptr<pele::DistanceInterface> dist, const bool
        perform_convergence_test, const bool collect_minima_list)
    : _ndim(ndim), 
    _optimizer(optimizer), 
    _potential(potential),
    _origin(origin.copy()), 
    _hs_radii(hs_radii.copy()),
    _rattlers(rattlers.copy()),
    _distance(origin.size(), 0),
    _new_minimum(origin.size()),
    _dtol(dtol), _d(0),
    _rms(0), 
    _nparticles(_hs_radii.size()), 
    _dist_policy(dist),_Nnoratt(0),
    _perform_convergence_test(perform_convergence_test), 
    _conv_test(1e-2, 5, 30, 0.3, 1e-10, _optimizer->get_tol(), 0.1, 1, _origin),
    _collect_minima_list(collect_minima_list),
    _minima_list(_dtol*sqrt(origin.size()), _optimizer->get_tol(), _dtol) //MinimaList(tol_delta_x_, tol_energy_, tol_delta_x_element_)
{
    if (_dist_policy == NULL)
        throw std::runtime_error("CheckSameMinimum::CheckSameMinimum distance policy uninitialised");

    for (size_t i=0;i<_origin.size();i+=_ndim) {
        if (_rattlers[i] != 0){
            _inoratt = i/_ndim;
            break;
        }
    }

    for (size_t i=0;i<_origin.size();i+=_ndim){
        _Nnoratt += _rattlers[i];
    }

}

//compute distance from origin after aligning the centre of mass
//this ignores the rattlers completely

void CheckSameMinimum::_get_vec_distance(pele::Array<double> quenched_coords)
{
    pele::Array<double> dr(_ndim);

    //measure distance between two non rattlers
    _dist_policy->get_rij(dr.data(), &quenched_coords[_inoratt], &_origin[_inoratt]);

    //align structures
    for(size_t i=0;i<_nparticles;++i) {
        size_t i1 = i*_ndim;
        for(size_t j=0;j<_ndim;++j){
            quenched_coords[i1+j] -= dr[j];
        }
    }

    if (_collect_minima_list){
        _new_minimum.assign(quenched_coords);
    }

    //compute distance between aligned structures
    for(size_t i=0;i<_nparticles;++i) {
        size_t i1 = i*_ndim;
        _dist_policy->get_rij(dr.data(), &quenched_coords[i1], &_origin[i1]);

        for(size_t j=0;j<_ndim;++j){
            _distance[i1+j] = dr[j] * _rattlers[i1+j];
        }
    }
}

void CheckSameMinimum::_check_convergence(pele::Array<double> quenched_coords)
{
    _conv_test.check_convergence(quenched_coords, _potential, _ndim, _optimizer);
}

bool CheckSameMinimum::test(Array<double> &trial_coords, MC * mc)
{
    _optimizer->reset(trial_coords);
    _optimizer->run();

    if (_perform_convergence_test){
        this->_check_convergence(_optimizer->get_x());
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
        //std::cout<<"failed quench rms "<<_rms<<std::endl;
        if (_collect_minima_list){
            _minima_list.insert_minimum(_d, _optimizer->get_f(), _new_minimum, _rattlers);
        }
        return false;
    }
    else{
        //std::cout<<"successfull quench rms "<<_rms<<std::endl;
        return true;
    }
}


}//namespace bv
