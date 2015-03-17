#ifndef _BV_CHECK_SAME_MINIMUM_CONFIG_H
#define _BV_CHECK_SAME_MINIMUM_CONFIG_H

#include "mcpele/mc.h"

namespace bv {

class CheckSameMinimumConfig : public mcpele::ConfTest {
public:
    CheckSameMinimumConfig(std::shared_ptr<pele::GradientOptimizer> optimizer,
                           std::shared_ptr<pele::BasePotential> potential,
                           pele::Array<double> origin,
                           double dtol)
        : m_optimizer(optimizer),
          m_potential(potential),
          m_origin(origin),
          m_dtol(dtol)
    {}
    bool conf_test(pele::Array<double>& trial_coords, mcpele::MC* mc)
    {
        double dist_orig_2;
        const bool quench_success = quench(trial_coords, dist_orig_2);
        if (!quench_success) {
            return false;
        }
        if (dist_orig_2 > m_dtol * m_dtol) {
            return false;
        }
        return true;
    }
    bool quench(pele::Array<double>& trial_coords, double& dist_orig_2)
    {
        m_optimizer->reset(trial_coords);
        m_optimizer->run();
        pele::Array<double> delta_orig = m_optimizer->get_x();
        delta_orig -= m_origin;
        dist_orig_2 = pele::dot(delta_orig, delta_orig);
        return m_optimizer->success();
    }
private:    
    std::shared_ptr<pele::GradientOptimizer> m_optimizer;
    std::shared_ptr<pele::BasePotential> m_potential;
    pele::Array<double> m_origin;
    const double m_dtol;
};

} // namespace bv

#endif // #ifndef _BV_CHECK_SAME_MINIMUM_CONFIG_H
