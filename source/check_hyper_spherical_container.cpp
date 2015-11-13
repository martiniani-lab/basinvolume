#include "basinvolume/check_hyper_spherical_container.h"

using pele::Array;

namespace bv{

CheckHyperSphericalContainer::CheckHyperSphericalContainer(pele::Array<double> origin, double radius, size_t ndim)
    : m_origin(origin.copy()),
      m_distance(origin.size(), 0),
      m_radius2(radius*radius),
      m_ndim(ndim),
      m_N((origin.size()/ndim))
{}

void CheckHyperSphericalContainer::_get_vec_distance(const pele::Array<double>& coords)
{
    pele::Array<double> delta_com(m_ndim, 0);

    for(size_t i=0;i<m_N;++i) {
        size_t i1 = i*m_ndim;
        for(size_t j=0;j<m_ndim;++j){
            double d = (coords[i1+j] - m_origin[i1+j]);
            m_distance[i1+j] = d;
            delta_com[j] += d;
        }
    }

    delta_com /= m_N;

    for(size_t i=0;i<m_N;++i) {
        size_t i1 = i*m_ndim;
        for(size_t j=0;j<m_ndim;++j){
            m_distance[i1+j] -= delta_com[j];
        }
    }
}

bool CheckHyperSphericalContainer::conf_test(Array<double> &trial_coords, mcpele::MC * mc)
{
    /*
    //debug
    std::shared_ptr<pele::BaseHarmonic> potential;
    potential = std::dynamic_pointer_cast<pele::BaseHarmonic>(mc->_potential);
    double k = potential->get_k();
    std::cout<<"k "<<k<<std::endl;
    ////
    */
    //this->_get_vec_distance(trial_coords);
    m_distance.assign(m_origin);
    m_distance -= trial_coords;

    double r2 = dot(m_distance, m_distance);
    if (r2 > m_radius2)
        return false;

    return true;
}

}//namespace bv
