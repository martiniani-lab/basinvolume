#include "basinvolume/bv_cg_descent.h"

/* this is an implementation of the cg_descent algorithm
 * with an additional termination condition on the distance
 * from the origin
*/

namespace bv {

BvCGDescent::BvCGDescent(std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> x0, pele::Array<double> origin,
            pele::Array<double> rattlers, size_t ndim, std::shared_ptr<pele::DistanceInterface> dist, double tol,
            double dtol, size_t maxiter, size_t PrintLevel)
    : pycgd::CGDescent(potential, x0, tol, PrintLevel),
      m_origin(origin.copy()),
      m_rattlers(rattlers.copy()),
      m_distance(origin.size()),
      m_dtol2(dtol*dtol),
      m_d2(0),
      m_rmsd2(0),
      m_rmsgtol(tol),
      m_ndim(ndim),
      m_nparticles(origin.size() / ndim),
      m_maxiter(maxiter),
      m_Nnoratt(0),
      m_dist_policy(dist)
{
            if (m_dist_policy == NULL) {
                throw std::runtime_error("CheckSameMinimum::CheckSameMinimum: distance policy uninitialised");
            }
            if (m_origin.size() != m_rattlers.size()) {
                throw std::runtime_error("CheckSameMinimum::CheckSameMinimum: illegal input: origin vs rattlers");
            }
            if (m_origin.size() % m_ndim) {
                throw std::runtime_error("CheckSameMinimum::CheckSameMinimum: illegal input: origin vs boxdimension");
            }
            for (size_t i = 0; i < m_origin.size(); i += m_ndim) {
                if (m_rattlers[i] != 0) {
                    m_inoratt = i / m_ndim;
                    break;
                }
            }
            for (size_t i = 0; i < m_origin.size(); i += m_ndim) {
                m_Nnoratt += m_rattlers[i];
            }
            this->set_maxit(m_maxiter);
            this->set_memory(0); //guarantees that memory is set to 0 irrespective of default settings
}

pele::Array<double> BvCGDescent::m_align_coords(pele::Array<double> coords)
{
    /*assert(coords.size() == _origin.size());
    assert(coords.size() == _ndim * _nparticles);*/
    pele::Array<double> dr(m_ndim);

    //measure distance between two non rattlers
    m_dist_policy->get_rij(dr.data(), &coords[m_inoratt], &m_origin[m_inoratt]);

    //align structures
    for (size_t i = 0; i < m_nparticles; ++i) {
        const size_t i1 = i * m_ndim;
        for (size_t j = 0; j < m_ndim; ++j) {
            coords[i1+j] -= dr[j];
        }
    }

    return coords;
}

double BvCGDescent::m_get_d2(pele::Array<double> coords)
{
    pele::Array<double> dr(m_ndim);
    pele::Array<double> aligned_coords = this->m_align_coords(coords);

    //compute distance between aligned structures
    for (size_t i = 0; i < m_nparticles; ++i) {
        const size_t i1 = i * m_ndim;
        m_dist_policy->get_rij(dr.data(), &aligned_coords[i1], &m_origin[i1]);
        for (size_t j = 0; j < m_ndim; ++j) {
            m_distance[i1 + j] = dr[j] * m_rattlers[i1 + j];
        }
    }

    //avoid taking square roots by return squared quantities
    return dot(m_distance,m_distance);
}

bool BvCGDescent::test_convergence(double energy, pele::Array<double> x, pele::Array<double> g){
        m_d2 = this->m_get_d2(x);
        m_rmsd2 = m_d2 / m_Nnoratt;
        return (m_rmsd2 < m_dtol2);
    }
} // namespace bv
