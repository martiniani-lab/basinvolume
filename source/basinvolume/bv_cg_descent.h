#ifndef _BV_CGD_H__
#define _BV_CGD_H__

#include "pele/distance.h"
#include "PyCG_DESCENT/cg_descent_wrapper.hpp"

/* this is an implementation of the cg_descent algorithm
 * with an additional termination condition on the distance
 * from the origin
*/

namespace bv {

template <typename distance_policy>
class BvCGDescent : public pycgd::CGDescent {
protected:
    static const size_t m_ndim = distance_policy::_ndim;
    pele::Array<double> m_origin;
    pele::Array<double> m_rattlers;
    pele::Array<double> m_distance;
    pele::Array<double> m_aligned_coords; //!< Coordinates after alignment, used in m_get_d2
    double m_dtol2, m_d2, m_rmsd2, m_rmsgtol;
    size_t m_nparticles, m_maxiter, m_inoratt, m_Nnoratt;
    const std::shared_ptr<distance_policy> m_dist_policy;
    void m_align_coords(pele::Array<double> & coords);
    double m_get_d2(pele::Array<double> const & coords);
public:
    BvCGDescent(std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> & x0,
                pele::Array<double> & origin, pele::Array<double> & rattlers,
                std::shared_ptr<distance_policy> const & dist, double tol=1e-4,
                double dtol=1e-4, size_t maxiter=1e6, size_t PrintLevel=0);

    virtual ~BvCGDescent() {}
    virtual bool test_convergence(double energy, pele::Array<double> const & x);
    inline double get_d2() { return m_d2; }
    inline double get_rmsd2() { return m_rmsd2; }
};

template <typename distance_policy>
BvCGDescent<distance_policy>::BvCGDescent(std::shared_ptr<pele::BasePotential> potential,
                         const pele::Array<double> & x0, pele::Array<double> & origin,
                         pele::Array<double> & rattlers,
                         std::shared_ptr<distance_policy> const & dist, double tol,
                         double dtol, size_t maxiter, size_t PrintLevel)
    : pycgd::CGDescent(potential, x0, tol, PrintLevel),
      m_origin(origin.copy()),
      m_rattlers(rattlers.copy()),
      m_distance(origin.size()),
      m_aligned_coords(origin.size()),
      m_dtol2(dtol*dtol),
      m_d2(0),
      m_rmsd2(0),
      m_rmsgtol(tol),
      m_nparticles(origin.size() / m_ndim),
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

template <typename distance_policy>
void BvCGDescent<distance_policy>::m_align_coords(pele::Array<double> & coords)
{
    /*assert(coords.size() == _origin.size());
    assert(coords.size() == _ndim * _nparticles);*/
    pele::VecN<m_ndim, double> dr;

    //measure distance between two non rattlers
    m_dist_policy->get_rij(dr.data(), &coords[m_inoratt], &m_origin[m_inoratt]);

    //align structures
    #pragma simd
    for (size_t i = 0; i < m_nparticles; ++i) {
        const size_t i1 = i * m_ndim;
        #pragma unroll
        for (size_t j = 0; j < m_ndim; ++j) {
            coords[i1 + j] -= dr[j];
        }
    }
}

template <typename distance_policy>
double BvCGDescent<distance_policy>::m_get_d2(pele::Array<double> const & coords)
{
    double distance2 = 0;
    m_aligned_coords.assign(coords);
    this->m_align_coords(m_aligned_coords);

    //compute distance between aligned structures
    #pragma simd reduction( + : distance2)
    for (size_t i = 0; i < m_nparticles; ++i) {
        const size_t i1 = i * m_ndim;
        pele::VecN<m_ndim, double> dr;
        m_dist_policy->get_rij(dr.data(), &m_aligned_coords[i1], &m_origin[i1]);
        #pragma unroll
        for (size_t j = 0; j < m_ndim; ++j) {
            const double current_distance = dr[j] * m_rattlers[i1 + j];
            distance2 += current_distance * current_distance;
        }
    }

    //avoid taking square roots by returning squared quantities
    return distance2;
}

template <typename distance_policy>
bool BvCGDescent<distance_policy>::test_convergence(double energy, pele::Array<double> const & x)
{
        m_d2 = this->m_get_d2(x);
        m_rmsd2 = m_d2 / m_Nnoratt;
        return (m_rmsd2 < m_dtol2);
}

} // namespace bv

#endif // #ifndef _BV_CGD_H__
