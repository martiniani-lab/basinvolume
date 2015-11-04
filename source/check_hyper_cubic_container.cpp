#include "basinvolume/check_hyper_cubic_container.h"

using pele::Array;

namespace bv{

CheckHyperCubicContainer::CheckHyperCubicContainer(pele::Array<double> origin, double sidelength, size_t ndim)
    : m_origin(origin.copy()),
    m_distance(origin.size(), 0),
    m_halfside(sidelength/2.0),
    m_ndim(ndim),
    m_N((origin.size()/ndim))
{}

void CheckHyperCubicContainer::_get_vec_distance(const pele::Array<double>& coords)
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

bool CheckHyperCubicContainer::conf_test(Array<double> &trial_coords, mcpele::MC * mc)
{
    this->_get_vec_distance(trial_coords);

    for(size_t i=0; i<m_distance.size(); ++i){
        double l = m_distance[i];
        bool inside = l >= -m_halfside && l <= m_halfside;
        if (not inside) return false;
    }

    return true;
}

}//namespace bv
