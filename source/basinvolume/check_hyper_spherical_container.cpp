#include "check_hyper_spherical_container.h"

namespace bv{

CheckHyperSphericalContainer::CheckHyperSphericalContainer(pele::Array<double> origin, double radius, size_t ndim):
        _origin(origin.copy()),_distance(origin.size(),0),_radius2(radius*radius),_ndim(ndim), _N((origin.size()/ndim)){}

void CheckHyperSphericalContainer::_get_vec_distance(const pele::Array<double>& coords){
        pele::Array<double> delta_com(_ndim,0);

        for(size_t i=0;i<_N;++i)
        {
            size_t i1 = i*_ndim;
            for(size_t j=0;j<_ndim;++j){
                double d = (coords[i1+j] - _origin[i1+j]);
                _distance[i1+j] = d;
                delta_com[j] += d;
            }
        }

        delta_com /= _N;

        for(size_t i=0;i<_N;++i)
        {
            size_t i1 = i*_ndim;
            for(size_t j=0;j<_ndim;++j)
                _distance[i1+j] -= delta_com[j];
        }
    }

bool CheckHyperSphericalContainer::test(Array<double> &trial_coords, mcpele::MC * mc)
{
    /*
    //debug
    std::shared_ptr<pele::BaseHarmonic> potential;
    potential = std::dynamic_pointer_cast<pele::BaseHarmonic>(mc->_potential);
    double k = potential->get_k();
    std::cout<<"k "<<k<<std::endl;
    ////
    */
  this->_get_vec_distance(trial_coords);

  double r2 = dot(_distance,_distance);
  if (r2 > _radius2)
      return false;

  return true;
}

}//namespace bv
