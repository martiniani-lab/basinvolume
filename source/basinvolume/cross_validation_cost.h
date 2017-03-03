#ifndef _BV_CROSS_VALIDATION_H
#define _BV_CROSS_VALIDATION_H

#include <cmath>
#include <algorithm>
#include <functional>
#include "pele/base_potential.h"

namespace bv {

class CrossValidationCost : public pele::BasePotential {
protected:
    pele::Array<double> m_data;
    const size_t m_ndim;
    double m_h;
    static constexpr double pi = M_PI;
    CrossValidationCost(const pele::Array<double> data)
        : m_data(data.copy()),
          m_ndim(m_data.size()),
          m_h(2.0)
    {}
public:
    virtual ~CrossValidationCost() {}
    /*virtual double inline get_energy(pele::Array<double> const & x) =0;
    virtual double get_energy_gradient(pele::Array<double> const & x, pele::Array<double> & grad)
    {
        double energy = this->get_energy(x);
        this->numerical_gradient(x, grad);
        return energy;
    }*/
    double get_h() { return m_h; }
};

class GaussianCrossValidationCost : public CrossValidationCost {
protected:
    double m_nd(double x, double h2);
public:
    GaussianCrossValidationCost(const pele::Array<double> data)
            : CrossValidationCost(data)
    {}
    virtual ~GaussianCrossValidationCost() {}
    virtual double inline get_energy(pele::Array<double> const & x);
};

inline double GaussianCrossValidationCost::m_nd(double x, double h2){
    return std::exp(-0.5 * x*x / h2) / std::sqrt(2 * pi * h2);
}

inline double GaussianCrossValidationCost::get_energy(pele::Array<double> const & x){
    m_h = x[0];
    double termA = this->m_nd(0.0, 2 * m_h*m_h);
    double termB = 0.0;
    double termC = 0.0;
    for (size_t i=0; i<m_ndim; ++i){
        for (size_t j=i+1; j < m_ndim; ++j){
            double dij = m_data[i] - m_data[j];
            termB += 2 * this->m_nd(dij, 2 * m_h*m_h);
            termC += 2 * this->m_nd(dij, m_h*m_h);
        }
    }
    return 1 / (m_ndim - 1) * termA + (m_ndim - 2) / (m_ndim * (m_ndim - 1) * (m_ndim - 1)) * termB - 2 / (m_ndim * (m_ndim - 1)) * termC;
}

}

#endif
