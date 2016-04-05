#ifndef _BV_CHECK_MINIMUM_IS_HCP_H
#define _BV_CHECK_MINIMUM_IS_HCP_H

#include "pele/distance.h"
#include "pele/optimizer.h"

#include "mcpele/histogram.h"
#include "mcpele/mc.h"

#include "basinvolume/simple_solid_angle_neighbors.h"
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
    const bool m_fixed_distance_cutoff;
    const bool m_record_q4_histogram;
    mcpele::Histogram m_q4_histogram;
public:
    CheckMinimumIsHCP(std::shared_ptr<pele::GradientOptimizer> optimizer=NULL, const double Q4tol=1e-10, const pele::Array<double>& boxvec={50, 50, 50}, const double rcut=2.1, const bool fixed_distance_cutoff=false, const bool record_q4_histogram=false, const size_t nr_bins=14)
        : m_optimizer(optimizer),
          m_Q4tol(Q4tol),
          m_boxdim(3),
          m_Q4hcp(7. / 72.),
          m_dist(boxvec),
          m_rcut2(rcut * rcut),
          m_verbose(false),
          m_fixed_distance_cutoff(fixed_distance_cutoff),
          m_record_q4_histogram(record_q4_histogram),
          m_q4_histogram(0, 0.4, 0.4 / nr_bins)
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
        bool is_hcp = true;
        for (size_t particle_index = 0; particle_index < nr_particles; ++particle_index) {
            if (!m_record_q4_histogram) {
                if (Q4_deviates_from_HCP(minimum, particle_index)) {
                    return false;
                }
            }
            else {
                if (Q4_deviates_from_HCP(minimum, particle_index)) {
                    is_hcp = false;
                }
            }
        }
        if (m_record_q4_histogram) {
            return is_hcp;
        }
        return true;
    }
    void set_verbose()
    {
        m_verbose = true;
    }
    bool Q4_deviates_from_HCP(const pele::Array<double>& x, const size_t particle_index)
    {
        const double tmp = get_Q4(x, particle_index);
        if (m_verbose) {
            std::cout << particle_index << "---\n";
            std::cout << tmp << "\t" << tmp / m_Q4hcp << "\n";
        }
        if (m_record_q4_histogram) {
            m_q4_histogram.add_entry(tmp);
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
        std::vector<size_t> neighbours;
        std::vector<double> weights;
        get_neighbours(x, particle_index, neighbours, weights);
        const size_t nr_neighbours = neighbours.size();
        std::complex<double> sum = 0;
        for (size_t k = 0; k < nr_neighbours; ++k) {
            const size_t i = neighbours.at(k);
            pele::Array<double> rij(m_boxdim);
            m_dist.get_rij(rij.data(), x.data() + particle_index * m_boxdim, x.data() + i * m_boxdim);
            sum += Y4M(m, rij[0], rij[1], rij[2]) * weights.at(k);
        }
        if (nr_neighbours == 0) {
            return 0;
        }
        const double tmp = std::abs(sum) / std::accumulate(weights.begin(), weights.end(), double(0));
        return tmp * tmp;
    }
    void get_neighbours(const pele::Array<double>& x, const size_t centre, std::vector<size_t>& neighbours, std::vector<double>& weights) const
    {
        if (m_fixed_distance_cutoff) {
            get_fixed_distance_cutoff_neighbours(x, centre, neighbours);
            weights.assign(neighbours.size(), 1);
        }
        else {
            get_sann_neighbours(x, centre, neighbours, weights);
        }
    }
    void get_fixed_distance_cutoff_neighbours(const pele::Array<double>& x, const size_t centre, std::vector<size_t>& neighbours) const
    {
        const size_t nr_particles = x.size() / m_boxdim;
        for (size_t i = 0; i < nr_particles; ++i) {
            if (i != centre) {
                pele::Array<double> rij(m_boxdim);
                m_dist.get_rij(rij.data(), x.data() + centre * m_boxdim, x.data() + i * m_boxdim);
                const double r2 = pele::dot(rij, rij);
                if (r2 < m_rcut2) {
                    neighbours.push_back(i);
                }
            }
        }
    }
    void get_sann_neighbours(const pele::Array<double>& x, const size_t center, std::vector<size_t>& neighbors, std::vector<double>& weights) const
    {
        SimpleSolidAngleNeighbors<pele::periodic_distance<3> > sann(x, x.size() / m_boxdim, m_dist);
        sann.compute_neighbors_weights(center, neighbors, weights);
    }
    pele::Array<double> get_hist_x() const
    {
        std::vector<double> vectics(m_q4_histogram.get_vectics());
        pele::Array<double> tmp(vectics);
        return tmp.copy();
    }
    pele::Array<double> get_hist_y() const
    {
        std::vector<double> vecdata(m_q4_histogram.get_vecdata_normalized());
        pele::Array<double> tmp(vecdata);
        return tmp.copy();
    }
    pele::Array<double> get_hist_ey() const
    {
        std::vector<double> v(m_q4_histogram.get_vecdata_error());
        pele::Array<double> tmp(v);
        return tmp.copy();
    }
};

} // namespace bv

#endif //#ifndef _BV_CHECK_MINIMUM_IS_HCP_H
