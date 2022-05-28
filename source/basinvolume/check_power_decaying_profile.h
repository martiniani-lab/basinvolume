#ifndef _BV_CHECK_POWER_DECAYING_PROFILE_H_
#define _BV_CHECK_POWER_DECAYING_PROFILE_H_

#include "check_exponentially_decaying_profile.h"

namespace bv {
    
class CheckPowerDecayingProfile : public CheckExponentiallyDecayingProfile {
    const double m_exponent;
public:
    CheckPowerDecayingProfile(pele::Array<double> origin, const double unity_radius, const double exponent,
    const size_t seed=42)
        : CheckExponentiallyDecayingProfile(origin, unity_radius, 1, false, seed),
          m_exponent(exponent)
    {}
    double get_profile_value(const double r2) const
    {
        if (r2 < m_unity_radius2) {
            return 1;
        }
        return std::pow(m_unity_radius2 / r2, 0.5 * m_exponent);
    }
};

} // namespace bv

#endif // #ifndef _BV_CHECK_POWER_DECAYING_PROFILE_H_
