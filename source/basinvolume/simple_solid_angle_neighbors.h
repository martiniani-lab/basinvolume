#ifndef SIMPLE_SOLID_ANGLE_NEIGHBORS_H
#define SIMPLE_SOLID_ANGLE_NEIGHBORS_H

#include <map>
#include <type_traits>

namespace bv {

template<class T>
class SimpleSolidAngleNeighbors {
public:
    typedef T distance_t;
private:
    const pele::Array<double>& m_coords;
    const size_t m_nparticles;
    const distance_t& m_distance;
public:
    SimpleSolidAngleNeighbors(const pele::Array<double>& coords, const size_t nparticles, const distance_t& distance)
        : m_coords(coords),
          m_nparticles(nparticles),
          m_distance(distance)
    {
        static_assert(distance_t::_ndim == 3, "illegal box dimension");
    }
    void compute_neighbors_weights(const size_t center, std::vector<size_t>& neighbors, std::vector<double>& weights)
    {
        // This follows exactly: ftp://ftp.aip.org/epaps/journ_chem_phys/E-JCPSA6-136-022224/sann.c
        size_t count = m_nparticles - 1;
        if (count < 3) {
            throw std::runtime_error("SimpleSolidAngleNeighbors: too few particles");
        }
        std::multimap<double, size_t> d;
        for (size_t k = 0; k < m_nparticles; ++k) {
            if (k != center) {
                d.insert(std::make_pair(get_distance(k, center), k));
            }
        }
        std::multimap<double, size_t>::const_iterator s = d.begin();
        double distance_sum = 0;
        for (size_t k = 0; k < 3; ++k, ++s) {
            distance_sum += s->first;
            neighbors.push_back(s->second);
        }
        double radius = distance_sum;
        size_t i = 3;
        while (i < count && radius > s->first) {
            distance_sum += s->first;
            radius = distance_sum / (i - 2);
            neighbors.push_back(s->second);
            ++i;
            ++s;
        }
        if (i == count) {
            throw std::runtime_error("SimpleSolidAngleNeighbors: too few particles");
        }
        s = d.begin();
        for (size_t k = 0; k < neighbors.size(); ++k, ++s) {
            weights.push_back(1 - s->first / radius);
        }
    }
    double get_distance(const size_t k, const size_t center) const
    {
        pele::Array<double> rij(distance_t::_ndim);
        m_distance.get_rij(rij.data(), m_coords.data() + center * distance_t::_ndim, m_coords.data() + k * distance_t::_ndim);
        return pele::norm(rij);
    }
};    
    
} // namespace bv

#endif // #ifndef SIMPLE_SOLID_ANGLE_NEIGHBORS_H
