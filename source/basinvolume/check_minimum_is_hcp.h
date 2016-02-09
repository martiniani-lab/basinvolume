#ifndef _BV_CHECK_MINIMUM_IS_HCP_H
#define _BV_CHECK_MINIMUM_IS_HCP_H

#import "Y4m.h"

/*
 
:<math>Y_{4}^{-4}(\theta,\varphi)={3\over 16}\sqrt{35\over 2\pi}\cdot e^{-4i\varphi}\cdot\sin^{4}\theta
= \frac{3}{16} \sqrt{\frac{35}{2 \pi}} \cdot \frac{(x - i y)^4}{r^4}</math>
:<math>Y_{4}^{-3}(\theta,\varphi)={3\over 8}\sqrt{35\over \pi}\cdot e^{-3i\varphi}\cdot\sin^{3}\theta\cdot\cos\theta
= \frac{3}{8} \sqrt{\frac{35}{\pi}} \cdot \frac{(x - i y)^3 z}{r^4}</math>
:<math>Y_{4}^{-2}(\theta,\varphi)={3\over 8}\sqrt{5\over 2\pi}\cdot e^{-2i\varphi}\cdot\sin^{2}\theta\cdot(7\cos^{2}\theta-1)
= \frac{3}{8} \sqrt{\frac{5}{2 \pi}} \cdot \frac{(x - i y)^2 \cdot (7 z^2 - r^2)}{r^4}</math>
:<math>Y_{4}^{-1}(\theta,\varphi)={3\over 8}\sqrt{5\over \pi}\cdot e^{-i\varphi}\cdot\sin\theta\cdot(7\cos^{3}\theta-3\cos\theta)
= \frac{3}{8} \sqrt{\frac{5}{\pi}} \cdot \frac{(x - i y) \cdot z \cdot (7 z^2 - 3 r^2)}{r^4}</math>
:<math>Y_{4}^{0}(\theta,\varphi)={3\over 16}\sqrt{1\over \pi}\cdot(35\cos^{4}\theta-30\cos^{2}\theta+3)
= \frac{3}{16} \sqrt{\frac{1}{\pi}} \cdot \frac{(35 z^4 - 30 z^2 r^2 + 3 r^4)}{r^4}</math>
:<math>Y_{4}^{1}(\theta,\varphi)={-3\over 8}\sqrt{5\over \pi}\cdot e^{i\varphi}\cdot\sin\theta\cdot(7\cos^{3}\theta-3\cos\theta)
= \frac{- 3}{8} \sqrt{\frac{5}{\pi}} \cdot \frac{(x + i y) \cdot z \cdot (7 z^2 - 3 r^2)}{r^4}</math>
:<math>Y_{4}^{2}(\theta,\varphi)={3\over 8}\sqrt{5\over 2\pi}\cdot e^{2i\varphi}\cdot\sin^{2}\theta\cdot(7\cos^{2}\theta-1)
= \frac{3}{8} \sqrt{\frac{5}{2 \pi}} \cdot \frac{(x + i y)^2 \cdot (7 z^2 - r^2)}{r^4}</math>
:<math>Y_{4}^{3}(\theta,\varphi)={-3\over 8}\sqrt{35\over \pi}\cdot e^{3i\varphi}\cdot\sin^{3}\theta\cdot\cos\theta
= \frac{- 3}{8} \sqrt{\frac{35}{\pi}} \cdot \frac{(x + i y)^3 z}{r^4}</math>
:<math>Y_{4}^{4}(\theta,\varphi)={3\over 16}\sqrt{35\over 2\pi}\cdot e^{4i\varphi}\cdot\sin^{4}\theta
= \frac{3}{16} \sqrt{\frac{35}{2 \pi}} \cdot \frac{(x + i y)^4}{r^4}</math>


 * */

namespace bv {
    
class CheckMinimumIsHCP : public mcpele::ConfTest {
private:
    std::shared_ptr<pele::GradientOptimizer> m_optimizer;
public:
    CheckMinimumIsHCP(std::shared_ptr<pele::GradientOptimizer> optimizer, const double Q4tol)
        : m_optimizer(optimizer),
          m_Q4tol(Q4tol),
          m_boxdim(3),
          m_Q4hcp()
    {}
    bool conf_test(pele::Array<double>& trial_coords, mcpele::MC* mc)
    {
        m_optimizer->rese(trial_coords);
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
    bool Q4_deviates_from_HCP(const pele::Array<double>& x, const size_t particle_index) const
    {
        return std::fabs(m_Q4hcp - get_Q4(x, particle_index)) > m_Q4tol;
    }
    double get_Q4(const pele::Array<double>& x, const size_t particle_index) const
    {
        double abs2_qlm_sum = 0;
        for (int l = -4; l <= 4; ++l) {
            abs2_qlm_sum += get_abs2_qlm(x, particle_index, l);
        }
        return std::sqrt(4 * M_PI / 9 * abs2_qlm_sum);
    }
    double get_abs2_qlm(const pele::Array<double>& x, const size_t particle_index, const int l) const
    {
        NearestNeighborList nn_info(x, particle_index, distance_cutoff);
    }
};

} // namespace bv

#endif //#ifndef _BV_CHECK_MINIMUM_IS_HCP_H
