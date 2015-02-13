#ifndef _BV_CGD_H__
#define _BV_CGD_H__

#include "pele/distance.h"
#include "PyCG_DESCENT/cg_descent_wrapper.hpp"

/* this is an implementation of the cg_descent algorithm
 * with an additional termination condition on the distance
 * from the origin
*/

namespace bv{

class BvCGDescent : public pycgd::CGDescent {
protected:
    pele::Array<double> m_origin;
    pele::Array<double> m_rattlers;
    pele::Array<double> m_distance;
    double m_dtol2, m_d2, m_rmsd2;
    size_t m_ndim, m_nparticles, m_maxiter, m_inoratt, m_Nnoratt;
    std::shared_ptr<pele::DistanceInterface> m_dist_policy;
    pele::Array<double> m_align_coords(pele::Array<double> coords);
    double m_get_d2(pele::Array<double> coords);
public:
    BvCGDescent(std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> x0, pele::Array<double> origin,
            pele::Array<double> rattlers, size_t ndim, std::shared_ptr<pele::DistanceInterface> dist, double tol=1e-4,
            double dtol=1e-4, size_t maxiter=1e6, size_t PrintLevel=0);

    virtual ~BvCGDescent(){}
    virtual bool test_convergence(double energy, pele::Array<double> x, pele::Array<double> g);
    inline double get_d2(){return m_d2;};
    inline double get_rmsd2(){return m_rmsd2;};
};

}

#endif
