#ifndef _BV_CHECK_EXPONENTIALLY_DECAYING_PROFILE_H_
#define _BV_CHECK_EXPONENTIALLY_DECAYING_PROFILE_H_

#include "mcpele/mc.h"

namespace bv {
class CheckExponentiallyDecayingProfile : public mcpele::ConfTest {
    const double m_unity_radius;
    const double m_unity_radius2;
    const double m_decay_length;
    const pele::Array<double> m_origin;
    std::mt19937_64 m_generator;
public:
    CheckExponentiallyDecayingProfile(pele::Array<double> origin, const double unity_radius, const double decay_length, const size_t seed=42)
        : m_unity_radius(unity_radius),
          m_unity_radius2(unity_radius * unity_radius),
          m_decay_length(decay_length),
          m_origin(origin.copy()),
          m_generator(seed)
    {}
    bool conf_test(pele::Array<double>& trial_coords, mcpele::MC* mc)
    {
        pele::Array<double> tmp = trial_coords.copy();
        tmp -= m_origin;
        const double r2 = pele::dot(tmp, tmp);
        return std::uniform_real_distribution<double>{0, 1}(m_generator) < get_profile_value(r2);
    }
private:
    double get_profile_value(const double r2) const
    {
        if (r2 < m_unity_radius2) {
            return 1;
        }
        return std::exp(-(std::sqrt(r2) - m_unity_radius) / m_decay_length);
    }        
};

} // namespace bv

#endif // #ifndef _BV_CHECK_EXPONENTIALLY_DECAYING_PROFILE_H_
