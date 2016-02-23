#ifndef _BV_CHECK_MINIMUM_IS_HCP_H
#define _BV_CHECK_MINIMUM_IS_HCP_H

#include "pele/distance.h"
#include "pele/optimizer.h"

#include "mcpele/mc.h"

#include "basinvolume/Y4m.h"

namespace bv {

class CheckMinimumIsHCP : public mcpele::ConfTest {
private:
    std::shared_ptr<pele::GradientOptimizer> m_optimizer;
    const double m_Q4tol;
    const size_t m_boxdim;
    const double m_Q4hcp;
    pele::periodic_distance<3> m_dist;
    const double m_rcut2;
    bool m_verbose;
public:
    CheckMinimumIsHCP(std::shared_ptr<pele::GradientOptimizer> optimizer=NULL, const double Q4tol=1e-10, const pele::Array<double>& boxvec={50, 50, 50}, const double rcut=2.1)
        : m_optimizer(optimizer),
          m_Q4tol(Q4tol),
          m_boxdim(3),
          m_Q4hcp(7./72.),
          m_dist(boxvec),
          m_rcut2(rcut * rcut),
          m_verbose(false)
    {}
    bool conf_test(pele::Array<double>& trial_coords, mcpele::MC* mc)
    {
        m_optimizer->reset(trial_coords);
        m_optimizer->run();
        if (!m_optimizer->success()) {
            return false;
        }
        const pele::Array<double> minimum = m_optimizer->get_x();
        const size_t nr_particles = minimum.size() / m_boxdim;
        for (size_t particle_index = 0; particle_index < nr_particles; ++particle_index) {
            if (Q4_deviates_from_HCP(minimum, particle_index)) {
                return false;
            }
        }
        return true;
    }
    void set_verbose()
    {
        m_verbose = true;
    }
    bool Q4_deviates_from_HCP(const pele::Array<double>& x, const size_t particle_index) const
    {
        const double tmp = get_Q4(x, particle_index);
        if (m_verbose) {
            std::cout << particle_index << "---\n";
            std::cout << tmp << "\t" << tmp / m_Q4hcp << "\n";
        }
        return std::fabs(m_Q4hcp - tmp) / m_Q4hcp > m_Q4tol;
    }
    double get_Q4(const pele::Array<double>& x, const size_t particle_index) const
    {
        double abs2_qlm_sum = 0;
        for (int m = -4; m <= 4; ++m) {
            abs2_qlm_sum += get_abs2_qlm(x, particle_index, m);
        }
        return std::sqrt(4. * M_PI / 9. * abs2_qlm_sum);
    }
    double get_abs2_qlm(const pele::Array<double>& x, const size_t particle_index, const int m) const
    {
        const size_t nr_particles = x.size() / 3;
        size_t nr_neighbours = 0;
        std::complex<double> sum = 0;
        for (size_t i = 0; i < nr_particles; ++i) {
            if (i != particle_index) {
                pele::Array<double> rij(3);
                m_dist.get_rij(rij.data(), x.data() + particle_index * 3, x.data() + i * 3);
                const double r2 = pele::dot(rij, rij);
                if (r2 < m_rcut2) {
                    ++nr_neighbours;
                    sum += Y4M(m, rij[0], rij[1], rij[2]);
                }
            }
        }
        if (nr_neighbours == 0) {
            return 0;
        }
        const double tmp = std::abs(sum);
        return (tmp * tmp) / (nr_neighbours * nr_neighbours);
    }
};

} // namespace bv

#endif //#ifndef _BV_CHECK_MINIMUM_IS_HCP_H
