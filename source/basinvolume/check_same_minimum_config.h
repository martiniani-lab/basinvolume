#ifndef _BV_CHECK_SAME_MINIMUM_CONFIG_H
#define _BV_CHECK_SAME_MINIMUM_CONFIG_H


#include "pele/optimizer.hpp"

#include "mcpele/mc.h"


namespace bv {

class CheckSameMinimumConfig : public mcpele::ConfTest {
protected:
    pele::Array<double> m_delta_orig;
public:
    CheckSameMinimumConfig(std::shared_ptr<pele::GradientOptimizer> optimizer,
                           std::shared_ptr<pele::BasePotential> potential,
                           pele::Array<double> origin,
                           double dtol)
        : m_optimizer(optimizer),
          m_potential(potential),
          m_origin(origin.copy()),
          m_dtol(dtol),
          m_nfev(0),
          m_nr_failed_quenches(0),
          m_nr_total_quenches(0),
          m_delta_orig(origin.size())
    {}
    bool conf_test(pele::Array<double>& trial_coords, mcpele::MC* mc)
    {
        double dist_orig_2;
        const bool quench_success = quench(trial_coords, dist_orig_2);
        ++m_nr_total_quenches;
        if (!quench_success) {
            ++m_nr_failed_quenches;
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
        m_nfev += m_optimizer->get_nfev();
        m_delta_orig.assign(m_optimizer->get_x());
        m_delta_orig -= m_origin;
        dist_orig_2 = pele::dot(m_delta_orig, m_delta_orig);
        return m_optimizer->success();
    }
    size_t get_nfev() const
    {
        return m_nfev;
    }
    double get_failed_quench_fraction() const
    {
        return static_cast<double>(m_nr_failed_quenches) / static_cast<double>(m_nr_total_quenches);
    }
    pele::Array<double> get_origin() const
    {
        return m_origin;
    }
private:
    std::shared_ptr<pele::GradientOptimizer> m_optimizer;
    std::shared_ptr<pele::BasePotential> m_potential;
    pele::Array<double> m_origin;
    const double m_dtol;
    size_t m_nfev;
    size_t m_nr_failed_quenches;
    size_t m_nr_total_quenches;
};

} // namespace bv

#endif // #ifndef _BV_CHECK_SAME_MINIMUM_CONFIG_H
