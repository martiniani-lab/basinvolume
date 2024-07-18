#ifndef _BV_RECORD_ACCEPTANCE_HISTOGRAM_H_
#define _BV_RECORD_ACCEPTANCE_HISTOGRAM_H_

#include "mcpele/histogram.h"
#include "mcpele/mc.h"

namespace bv {

class MomentAccArray {
  std::vector<std::shared_ptr<mcpele::Moments>> m_acc;
  const double m_xmin;
  const double m_xmax;
  const double m_delta;

public:
  MomentAccArray(const double xmin, const double xmax, const size_t nbins)
      : m_xmin(xmin), m_xmax(xmax), m_delta((xmax - xmin) / nbins) {
    for (size_t i = 0; i < nbins; ++i) {
      m_acc.push_back(std::make_shared<mcpele::Moments>());
    }
  }
  void add(const double x, const double inp) {
    if (x < m_xmin || x > m_xmax) {
      return;
    }
    m_acc.at(get_index(x))->update(inp);
  }
  size_t get_index(const double x) const {
    return std::min<size_t>(static_cast<size_t>((x - m_xmin) / m_delta),
                            m_acc.size() - 1);
  }
  double get_x(const size_t index) const {
    return m_xmin + (index + 0.5) * m_delta;
  }
  double get_mean(const size_t index) const { return m_acc.at(index)->mean(); }
  size_t size() const { return m_acc.size(); }
};

class RecordAcceptanceHistogram : public mcpele::Action {
  const pele::Array<double> m_origin;
  MomentAccArray m_dist_acc;
  const size_t m_eqsteps;

public:
  RecordAcceptanceHistogram(pele::Array<double> origin, const double rmin,
                            const double rmax, const size_t nbins,
                            const size_t eqsteps)
      : m_origin(origin.copy()), m_dist_acc(rmin, rmax, nbins),
        m_eqsteps(eqsteps) {}
  void action(pele::Array<double> &coords, double energy, bool accepted,
              mcpele::MCBase *mc) {
    if (mc->get_iterations_count() > m_eqsteps) {
      pele::Array<double> tmp = coords.copy();
      tmp -= m_origin;
      const double r = pele::norm(tmp);
      m_dist_acc.add(r, accepted);
    }
  }
  pele::Array<double> get_acceptance_distance_values() const {
    pele::Array<double> result(m_dist_acc.size());
    for (size_t i = 0; i < result.size(); ++i) {
      result[i] = m_dist_acc.get_x(i);
    }
    return result.copy();
  }
  pele::Array<double> get_acceptance_fraction_values() const {
    pele::Array<double> result(m_dist_acc.size());
    for (size_t i = 0; i < result.size(); ++i) {
      result[i] = m_dist_acc.get_mean(i);
    }
    return result.copy();
  }
};

} // namespace bv

#endif // #ifndef _BV_RECORD_ACCEPTANCE_HISTOGRAM_H_
