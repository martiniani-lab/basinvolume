#ifndef _BV_CHECK_BARRIER_CROSSING_H_
#define _BV_CHECK_BARRIER_CROSSING_H_

#include "mcpele/mc.h"

namespace bv {

class CheckBarrierCrossing : public mcpele::ConfTest {
    const double m_ss;
    const double m_prefactor;
    const double m_barrier_height, m_min_energy;
    std::mt19937_64 m_generator;
public:
    CheckBarrierCrossing(const double ss=1, const double prefactor=1, const double min_energy=0,
    const size_t seed=44)
        : m_ss(ss),
          m_prefactor(prefactor),
          m_barrier_height(get_potential_energy(0)),
          m_min_energy(min_energy),
          m_generator(seed)
    {}
    bool conf_test(pele::Array<double>& trial_coords, mcpele::MC* mc)
    {
        const double r2 = pele::dot(trial_coords, trial_coords);
        const double pot_energy = get_potential_energy(r2);
        if (pot_energy > m_min_energy && get_kinetic_energy() > m_barrier_height - pot_energy) {
            return true;
        }
        return false;
    }
    double get_potential_energy(const double r2) const
    {
        // this is a gaussian centred at 0 with height m_prefactor and variance ss
//        return m_prefactor * std::exp(-0.5 * r2 / m_ss) / std::sqrt(2 * M_PI * m_ss);
        // the following has correct kT units, sqrt(m_ss) is the unit of length, m_prefactor is dimensionless
        // we multiply a prefactor in units of kT times a dimensionless standard normal
        return m_ss * m_prefactor * std::exp(-0.5 * r2 / m_ss) / std::sqrt(2 * M_PI);
    }
    double get_kinetic_energy() 
    {
        // http://www.cplusplus.com/reference/random/chi_squared_distribution/
        // https://en.wikipedia.org/wiki/Chi_distribution
        // https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution
        // http://www.math.uah.edu/stat/special/Maxwell.html
        // prefactor is scale parameter kT/m
        double prefactor = m_ss;
        const double v2 = prefactor * std::chi_squared_distribution<double>(1)(m_generator);
        return 0.5 * v2;
    }
};
    
} // namespace bv

#endif // #ifndef _BV_CHECK_BARRIER_CROSSING_H_
