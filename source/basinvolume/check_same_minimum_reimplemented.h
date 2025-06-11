#ifndef _BV_CHECK_SAME_MINIMUM_REIMPLEMENTED_H
#define _BV_CHECK_SAME_MINIMUM_REIMPLEMENTED_H

#include <memory>

#include "pele/distance.hpp"
#include "pele/optimizer.hpp"

#include "mcpele/mc.h"

#include "check_same_minimum.h"
#include "inside_basin_test.h"
#include "minima_list.h"

namespace bv {

template <typename distance_policy, class OPT_T = pele::GradientOptimizer>
class CheckSameMinimumReimplemented : public CheckSameMinimumInterface {
private:
  std::unique_ptr<PairPotentialBasin<distance_policy>> _basin;
  std::unique_ptr<InsideBasinTest> _inside_basin_test;

public:
  CheckSameMinimumReimplemented(
      std::shared_ptr<OPT_T> optimizer,
      std::shared_ptr<pele::BasePotential> potential,
      pele::Array<double> &origin, pele::Array<double> &rattlers, double dtol,
      const size_t eqsteps, std::shared_ptr<distance_policy> const &dist,
      const bool collect_minima_list) {
    _basin = std::make_unique<PairPotentialBasin<distance_policy>>(
        potential, optimizer, origin, rattlers, dtol, eqsteps,
        collect_minima_list);
    _inside_basin_test = std::make_unique<InsideBasinTest>(_basin.get());
  }

  virtual ~CheckSameMinimumReimplemented() = default;

  virtual bool conf_test(pele::Array<double> &trial_coords,
                         mcpele::MC *mc) override {
    return _inside_basin_test->conf_test(trial_coords, mc);
  }

  virtual size_t ml_nr_distinct_minima() const override {
    return _basin->get_minima_list().nr_distinct_minima();
  }

  virtual pele::Array<Minimum *> get_array_of_minima() override {
    pele::Array<Minimum *> minima(_basin->get_minima_list().nr_distinct_minima());
    size_t i = 0;
    for (auto &m : _basin->get_minima_list()) {
      minima[i++] = &m;
    }
    return minima;
  }

  virtual double get_failed_quench_frac() const override {
    return _inside_basin_test->get_failed_quench_frac();
  }
};

} // namespace bv

#endif // _BV_CHECK_SAME_MINIMUM_REIMPLEMENTED_H 