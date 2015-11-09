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

bool CheckHyperCubicContainer::conf_test(Array<double> &trial_coords, mcpele::MC * mc)
{
    m_distance.assign(m_origin);
    m_distance -= trial_coords;

    for(size_t i=0; i<m_distance.size(); ++i){
        double l = m_distance[i];
        bool inside = std::fabs(l) <= m_halfside;
        if (not inside) return false;
    }

    return true;
}

}//namespace bv
