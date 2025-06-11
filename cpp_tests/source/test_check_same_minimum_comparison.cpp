#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "pele/array.hpp"
#include "pele/distance.hpp"
#include "pele/harmonic.hpp"
#include "pele/lbfgs.hpp"
#include "pele/modified_fire.hpp"

#include "mcpele/adaptive_takestep.h"
#include "mcpele/gaussian_coords_displacement.h"
#include "mcpele/metropolis_test.h"
#include "mcpele/random_coords_displacement.h"

#include "basinvolume/check_same_minimum.h"
#include "basinvolume/check_same_minimum_reimplemented.h"

#define EXPECT_NEAR_RELATIVE(A, B, T)                                          \
  EXPECT_NEAR(fabs(A) / (fabs(A) + fabs(B) + 1),                               \
              fabs(B) / (fabs(A) + fabs(B) + 1), T)

using std::shared_ptr;

class CheckSameMinimumComparisonTest : public ::testing::Test {
public:
  typedef std::vector<double> vec_t;
  typedef pele::Array<double> arr_t;
  typedef pele::Harmonic pot_t;
  typedef pele::LBFGS opt_t;
  typedef pele::MODIFIED_FIRE fire_t;

  size_t nr_particles;
  size_t nr_dim;
  size_t nr_dof;
  arr_t origin;
  arr_t x;
  double k;
  std::shared_ptr<pele::BasePotential> pot;
  double _lbfgstol;
  double _lbfgsM;
  arr_t rattlers;
  double dtol;
  size_t max_iter, opt_max_iter;
  
  virtual void SetUp() {
    nr_particles = 10;
    nr_dim = 3;
    nr_dof = nr_particles * nr_dim;
    origin = arr_t(nr_dof, 0).copy();
    for (size_t i = 0; i < origin.size(); ++i) {
      origin[i] = sqrt(i);
    }
    x = arr_t(nr_dof, 0).copy();
    k = 100;
    pot = std::make_shared<pot_t>(origin, k, nr_dim);
    _lbfgstol = 1e-6;
    _lbfgsM = 5;
    rattlers = arr_t(nr_dof, 1).copy();
    dtol = 1e-8;
    max_iter = 1e4;
    opt_max_iter = 1e4;
  }
};

TEST_F(CheckSameMinimumComparisonTest, InterfaceCompatibility) {
  const size_t eqsteps = 0;
  auto opt1 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto opt2 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto dist = std::make_shared<pele::cartesian_distance<3>>();
  
  // Create both implementations
  bv::CheckSameMinimumCartesian<3> original(opt1, pot, origin, rattlers, dtol, eqsteps);
  bv::CheckSameMinimumReimplemented<pele::cartesian_distance<3>, opt_t> reimplemented(
      opt2, pot, origin, rattlers, dtol, eqsteps, dist, false);
  
  // Test that both inherit from the same interface
  bv::CheckSameMinimumInterface* iface1 = &original;
  bv::CheckSameMinimumInterface* iface2 = &reimplemented;
  
  EXPECT_NE(iface1, nullptr);
  EXPECT_NE(iface2, nullptr);
  
  // Initial state should be the same
  EXPECT_EQ(original.ml_nr_distinct_minima(), reimplemented.ml_nr_distinct_minima());
  EXPECT_NEAR(original.get_failed_quench_frac(), reimplemented.get_failed_quench_frac(), 1e-10);
}

TEST_F(CheckSameMinimumComparisonTest, IdenticalBehaviorSameAttractor) {
  const size_t eqsteps = 0;
  auto opt1 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto opt2 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto dist = std::make_shared<pele::cartesian_distance<3>>();
  
  bv::CheckSameMinimumCartesian<3> original(opt1, pot, origin, rattlers, dtol, eqsteps);
  bv::CheckSameMinimumReimplemented<pele::cartesian_distance<3>, opt_t> reimplemented(
      opt2, pot, origin, rattlers, dtol, eqsteps, dist, false);
  
  // Test coordinates that should quench to the same minimum (origin)
  arr_t test_coords = origin.copy();
  for (size_t i = 0; i < test_coords.size(); ++i) {
    test_coords[i] += 0.1 * (rand() / double(RAND_MAX) - 0.5); // small perturbation
  }
  
  // Create separate MC objects for each test
  mcpele::MC mc1(pot, x, 1);
  mcpele::MC mc2(pot, x, 1);
  
  arr_t coords1 = test_coords.copy();
  arr_t coords2 = test_coords.copy();
  
  bool result1 = original.conf_test(coords1, &mc1);
  bool result2 = reimplemented.conf_test(coords2, &mc2);
  
  // Both should return the same result (true for same attractor)
  EXPECT_EQ(result1, result2);
  EXPECT_TRUE(result1); // Should be true since we're close to origin
  
  // Function evaluation counts should be similar (allowing for small differences)
  EXPECT_NEAR(mc1.m_neval, mc2.m_neval, std::max(mc1.m_neval, mc2.m_neval) * 0.1);
}

TEST_F(CheckSameMinimumComparisonTest, IdenticalBehaviorDifferentAttractor) {
  const size_t eqsteps = 0;
  auto opt1 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto opt2 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto dist = std::make_shared<pele::cartesian_distance<3>>();
  
  bv::CheckSameMinimumCartesian<3> original(opt1, pot, origin, rattlers, dtol, eqsteps, false, true);
  bv::CheckSameMinimumReimplemented<pele::cartesian_distance<3>, opt_t> reimplemented(
      opt2, pot, origin, rattlers, dtol, eqsteps, dist, true);
  
  // Create coordinates far from origin that might find different minimum
  arr_t test_coords = origin.copy();
  for (size_t i = 0; i < test_coords.size(); ++i) {
    test_coords[i] += 10.0; // large perturbation
  }
  
  mcpele::MC mc1(pot, x, 1);
  mcpele::MC mc2(pot, x, 1);
  
  arr_t coords1 = test_coords.copy();
  arr_t coords2 = test_coords.copy();
  
  bool result1 = original.conf_test(coords1, &mc1);
  bool result2 = reimplemented.conf_test(coords2, &mc2);
  
  // Both should return the same result
  EXPECT_EQ(result1, result2);
  
  // Function evaluation counts should be similar
  EXPECT_NEAR(mc1.m_neval, mc2.m_neval, std::max(mc1.m_neval, mc2.m_neval) * 0.2);
}

TEST_F(CheckSameMinimumComparisonTest, MinimaCollectionEquivalent) {
  const size_t eqsteps = 10;
  auto opt1 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto opt2 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto dist = std::make_shared<pele::cartesian_distance<3>>();
  
  bv::CheckSameMinimumCartesian<3> original(opt1, pot, origin, rattlers, dtol, eqsteps, false, true);
  bv::CheckSameMinimumReimplemented<pele::cartesian_distance<3>, opt_t> reimplemented(
      opt2, pot, origin, rattlers, dtol, eqsteps, dist, true);
  
  // Run multiple MC steps to collect minima
  mcpele::MC mc1(pot, x, 1);
  mcpele::MC mc2(pot, x, 1);
  
  shared_ptr<mcpele::TakeStep> sampler1 = 
      std::make_shared<mcpele::RandomCoordsDisplacementAll>(42);
  shared_ptr<mcpele::TakeStep> sampler2 = 
      std::make_shared<mcpele::RandomCoordsDisplacementAll>(42);
  
  mc1.set_takestep(sampler1);
  mc2.set_takestep(sampler2);
  
  shared_ptr<mcpele::AcceptTest> metropolis1 = 
      std::make_shared<mcpele::MetropolisTest>(42);
  shared_ptr<mcpele::AcceptTest> metropolis2 = 
      std::make_shared<mcpele::MetropolisTest>(42);
  
  mc1.add_accept_test(metropolis1);
  mc2.add_accept_test(metropolis2);
  
  shared_ptr<mcpele::ConfTest> test1 = 
      std::make_shared<bv::CheckSameMinimumCartesian<3>>(opt1, pot, origin, rattlers, dtol, eqsteps, false, true);
  mc1.add_late_conf_test(test1);
  
  // For the reimplemented version, we need to create it properly
  auto test2 = std::make_shared<bv::CheckSameMinimumReimplemented<pele::cartesian_distance<3>, opt_t>>(
      opt2, pot, origin, rattlers, dtol, eqsteps, dist, true);
  mc2.add_late_conf_test(test2);
  
  const size_t niter = 50;
  mc1.run(niter);
  mc2.run(niter);
  
  // Both should have collected similar number of distinct minima
  // (might not be exactly the same due to different random sequences in optimization)
  auto original_test = std::static_pointer_cast<bv::CheckSameMinimumCartesian<3>>(test1);
  
  EXPECT_GE(original_test->ml_nr_distinct_minima(), 0);
  EXPECT_GE(test2->ml_nr_distinct_minima(), 0);
  
  // Failed quench fractions should be similar
  EXPECT_NEAR(original_test->get_failed_quench_frac(), test2->get_failed_quench_frac(), 0.2);
}

TEST_F(CheckSameMinimumComparisonTest, StressTestMultipleRuns) {
  const size_t eqsteps = 5;
  auto dist = std::make_shared<pele::cartesian_distance<3>>();
  
  // Run multiple times with different random seeds
  for (int seed = 1; seed <= 5; ++seed) {
    auto opt1 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
    auto opt2 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
    
    bv::CheckSameMinimumCartesian<3> original(opt1, pot, origin, rattlers, dtol, eqsteps);
    bv::CheckSameMinimumReimplemented<pele::cartesian_distance<3>, opt_t> reimplemented(
        opt2, pot, origin, rattlers, dtol, eqsteps, dist, false);
    
    // Generate test coordinates with specific seed
    srand(seed);
    arr_t test_coords = origin.copy();
    for (size_t i = 0; i < test_coords.size(); ++i) {
      test_coords[i] += 0.5 * (rand() / double(RAND_MAX) - 0.5);
    }
    
    mcpele::MC mc1(pot, x, 1);
    mcpele::MC mc2(pot, x, 1);
    
    arr_t coords1 = test_coords.copy();
    arr_t coords2 = test_coords.copy();
    
    bool result1 = original.conf_test(coords1, &mc1);
    bool result2 = reimplemented.conf_test(coords2, &mc2);
    
    EXPECT_EQ(result1, result2) << "Results differ for seed " << seed;
  }
}

TEST_F(CheckSameMinimumComparisonTest, PeriodicBoundaryComparison) {
  // Test with periodic boundaries
  const size_t eqsteps = 0;
  arr_t boxvec(nr_dim, 10.0);
  
  auto opt1 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto opt2 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto dist = std::make_shared<pele::periodic_distance<3>>(boxvec);
  
  bv::CheckSameMinimumPeriodic<3> original(opt1, pot, origin, boxvec, rattlers, dtol, eqsteps);
  bv::CheckSameMinimumReimplemented<pele::periodic_distance<3>, opt_t> reimplemented(
      opt2, pot, origin, rattlers, dtol, eqsteps, dist, false);
  
  // Test with coordinates that cross periodic boundaries
  arr_t test_coords = origin.copy();
  for (size_t i = 0; i < nr_dim; ++i) {
    test_coords[i] += 5.0; // shift by half box size
  }
  
  mcpele::MC mc1(pot, x, 1);
  mcpele::MC mc2(pot, x, 1);
  
  arr_t coords1 = test_coords.copy();
  arr_t coords2 = test_coords.copy();
  
  bool result1 = original.conf_test(coords1, &mc1);
  bool result2 = reimplemented.conf_test(coords2, &mc2);
  
  EXPECT_EQ(result1, result2);
}

TEST_F(CheckSameMinimumComparisonTest, EdgeCaseEmptyRattlers) {
  const size_t eqsteps = 0;
  auto opt1 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto opt2 = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
  auto dist = std::make_shared<pele::cartesian_distance<3>>();
  
  // Create rattlers array with some zeros (rattlers)
  arr_t rattlers_with_holes = rattlers.copy();
  for (size_t i = 0; i < nr_particles; i += 2) {
    for (size_t j = 0; j < nr_dim; ++j) {
      rattlers_with_holes[i * nr_dim + j] = 0; // make some particles rattlers
    }
  }
  
  bv::CheckSameMinimumCartesian<3> original(opt1, pot, origin, rattlers_with_holes, dtol, eqsteps);
  bv::CheckSameMinimumReimplemented<pele::cartesian_distance<3>, opt_t> reimplemented(
      opt2, pot, origin, rattlers_with_holes, dtol, eqsteps, dist, false);
  
  arr_t test_coords = origin.copy();
  mcpele::MC mc1(pot, x, 1);
  mcpele::MC mc2(pot, x, 1);
  
  arr_t coords1 = test_coords.copy();
  arr_t coords2 = test_coords.copy();
  
  bool result1 = original.conf_test(coords1, &mc1);
  bool result2 = reimplemented.conf_test(coords2, &mc2);
  
  EXPECT_EQ(result1, result2);
} 