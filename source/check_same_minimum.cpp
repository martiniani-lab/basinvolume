#include <cmath>

#include "basinvolume/check_same_minimum.h"

using pele::Array;
using mcpele::MC;

namespace bv{


CheckSameMinimum::CheckSameMinimum(std::shared_ptr<pele::GradientOptimizer> optimizer,
        std::shared_ptr<pele::BasePotential> potential, Array<double> origin, Array<double>
        hs_radii, Array<double> rattlers, double dtol, size_t ndim,
        const size_t eqsteps, std::shared_ptr<pele::DistanceInterface> dist,
        const bool perform_convergence_test, const bool collect_minima_list)
    : _ndim(ndim),
      _optimizer(optimizer),
      _potential(potential),
      _origin(origin.copy()),
      _hs_radii(hs_radii.copy()),
      _rattlers(rattlers.copy()),
      _distance(origin.size(), 0),
      _new_minimum(origin.size()),
      _dtol(dtol),
      _d(0),
      _rms(0),
      _nparticles(_hs_radii.size()),
      _dist_policy(dist),
      _Nnoratt(0),
      _perform_convergence_test(perform_convergence_test),
      _conv_test(30, 1e-10, _optimizer->get_tol(), 0.1, _origin, potential, ndim),
      m_eqsteps(eqsteps),
      _collect_minima_list(collect_minima_list),
      _minima_list(_dtol*sqrt(origin.size()), _optimizer->get_tol(), _dtol)
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

void CheckSameMinimum::_check_convergence(pele::Array<double> quenched_coords)
{
    _conv_test.check_convergence(quenched_coords, _optimizer);
}


/*aligns structures*/
pele::Array<double> CheckSameMinimum::_align_coords(pele::Array<double> coords){
    /*assert(coords.size() == _origin.size());
    assert(coords.size() == _ndim * _nparticles);*/
    pele::Array<double> dr(_ndim);

    //measure distance between two non rattlers
    _dist_policy->get_rij(dr.data(), &coords[_inoratt], &_origin[_inoratt]);

    //align structures
    for(size_t i=0;i<_nparticles;++i) {
        size_t i1 = i*_ndim;
        for(size_t j=0;j<_ndim;++j){
            coords[i1+j] -= dr[j];
        }
    }

    return coords;
}

/*compute distance from origin after aligning two particles
this ignores the rattlers completely and returns rmsd squared*/
double CheckSameMinimum::_get_d2(pele::Array<double> coords)
{
    pele::Array<double> dr(_ndim);
    pele::Array<double> aligned_coords = this->_align_coords(coords);

    //compute distance between aligned structures
    for(size_t i=0;i<_nparticles;++i) {
        size_t i1 = i*_ndim;
        _dist_policy->get_rij(dr.data(), &aligned_coords[i1], &_origin[i1]);

        for(size_t j=0;j<_ndim;++j){
            _distance[i1+j] = dr[j] * _rattlers[i1+j];
        }
    }

    //avoid taking square roots by return squared quantities
    return dot(_distance,_distance);
}

/*quench configuration and add minimum to new minimum list*/
bool CheckSameMinimum::_quench(pele::Array<double> &trial_coords){
    _optimizer->reset(trial_coords);

    bool success = true;
    double d2 = this->_get_d2(_optimizer->get_x());
    double rmsd2 = d2/_Nnoratt;
    double dtol2 = _dtol*_dtol;
    const size_t opt_maxiter = _optimizer->get_maxiter();

    //this might become an infinite loop
    //optimizer stop-criterion needs to be checked before calling one_iteration
    while(rmsd2 > dtol2 && _optimizer->get_niter() < opt_maxiter){
        if (_optimizer->stop_criterion_satisfied()){
            //minimisation converged before satisfying distance criterion,
            //save minimum and return false
            success = false;
            break;
        }
        _optimizer->one_iteration();
        d2 = this->_get_d2(_optimizer->get_x());
        rmsd2 = d2/_Nnoratt;
    }

    //assign attributes for rms displacement from origin
    _d = sqrt(d2);
    _rms = sqrt(rmsd2);

    return success;
}

bool CheckSameMinimum::conf_test(Array<double> &trial_coords, MC * mc)
{
    bool quench_success = true;
    bool same_minimum = this->_quench(trial_coords);

    //add number of energy evaluations to mc eval count
    const size_t nfev = _optimizer->get_nfev();
    mc->m_neval += nfev;

    if (_perform_convergence_test){
        this->_check_convergence(_optimizer->get_x());
    }

    //check if minimisation has converged
    //if exited loop with rmsd>dtol2 and success == true
    //then the quench has failed in the given no. of steps
    if(_rms > _dtol && same_minimum){
        quench_success = false;
    }
    m_failed_quench_frac.update(!quench_success);
    if(!quench_success){
        return false;
    }

    if (!same_minimum){
        //if quench has converged to different minimum then one might want to
        //save the new minimum
        //std::cout<<"failed quench rms "<<_rms<<"dtol"<<_dtol<<std::endl;
        if (_collect_minima_list && mc->get_iterations_count() > m_eqsteps){
            _new_minimum.assign(this->_align_coords(_optimizer->get_x()));
            _minima_list.insert_minimum(_d, _optimizer->get_f(), _new_minimum, _rattlers);
        }
        return false;
    }
    else{
        return true;
    }
}


}//namespace bv
