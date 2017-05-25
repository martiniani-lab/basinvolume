#ifndef _BV_INDEPENDENCE_SAMPLING_H
#define _BV_INDEPENDENCE_SAMPLING_H

#include <math.h>
#include <random>

#include "pele/array.h"

namespace bv {

class IndependenceSampling {
protected:
    size_t m_seed;
    std::mt19937_64 m_generator;
    std::uniform_int_distribution<int> m_index_dist;
    std::uniform_real_distribution<double> m_real_dist;

public:
    IndependenceSampling(size_t seed)
        : m_seed(seed),
          m_real_dist(0.0, 1.0)
    {
        m_generator.seed(m_seed);
    }

    int exchange (pele::Array<int> exchange_pattern,
                  pele::Array<double> dxs,
                  pele::Array<double> betas, int nexchanges)
    {
        m_index_dist = std::uniform_int_distribution<int>(0, exchange_pattern.size() - 1);

        std::vector<double> energies(dxs.size());
        for (int i = 0; i < exchange_pattern.size(); i++) {
            exchange_pattern[i] = i;
            energies[i] = 0.5 * dxs[i] * dxs[i];
        }

        int naccept = 0;
        for (int i = 0; i < nexchanges; i++) {

            int i = m_index_dist(m_generator);
            int j = m_index_dist(m_generator);
            while(i == j) {
                j = m_index_dist(m_generator);
            }

            int inow = exchange_pattern[i];
            int jnow = exchange_pattern[j];
            double w = exp((energies[inow]-energies[jnow])
                           * (betas[i]-betas[j]));

            double rnd = m_real_dist(m_generator);
            if (w > rnd) {
                exchange_pattern[i] = jnow;
                exchange_pattern[j] = inow;
                naccept += 1;
            }
        }
        return naccept;
    }
};

} // namespace bv

#endif // #ifndef _BV_INDEPENDENCE_SAMPLING_H
