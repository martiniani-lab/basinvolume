#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>

#include <gtest/gtest.h>

#include "pele/array.h"
#include "pele/distance.h"
#include "pele/harmonic.h"
#include "pele/lbfgs.h"
#include "pele/modified_fire.h"

#include "mcpele/mc.h"
#include "mcpele/takestep.h"
#include "mcpele/accept_test.h"
#include "mcpele/actions.h"
#include "mcpele/conf_test.h"

#include "basinvolume/check_same_minimum.h"
#include "basinvolume/findk.h"

#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(fabs(A)/(fabs(A)+fabs(B)+1), fabs(B)/(fabs(A)+fabs(B)+1), T)

using std::shared_ptr;

class CheckSameMinimumTest: public ::testing::Test{
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
    arr_t shifted_origin;
    arr_t x;
    double k;
    std::shared_ptr<pele::BasePotential> pot;
    double _lbfgstol;
    double _lbfgsM;
    arr_t hs_radii;
    arr_t rattlers;
    double dtol;
    double stepsize;
    size_t max_iter;
    virtual void SetUp()
    {
        nr_particles = 42;
        nr_dim = 3;
        nr_dof = nr_particles*nr_dim;
        origin = arr_t(nr_dof,0).copy();
        shifted_origin = arr_t(nr_dof,11).copy();
        x = arr_t(nr_dof,0).copy();
        k = 4242;
        pot = std::make_shared<pot_t>(origin, k, nr_dim);
        _lbfgstol = 1e-2;
        _lbfgsM = 5;
        hs_radii = arr_t(nr_particles,1).copy();
        rattlers = arr_t(nr_dof,1).copy();
        dtol = 1e-7;
        stepsize = 1e-2;
        max_iter = 1e5;
    }
};

TEST_F(CheckSameMinimumTest, BasicFunctionality){
    auto opt = std::make_shared<opt_t>(pot, origin, _lbfgstol, _lbfgsM);
    bv::CheckSameMinimumCartesian<3>  check_basic(opt, pot, origin, hs_radii, rattlers, dtol);
    EXPECT_TRUE(check_basic.perform_convergence_test()==false);
    EXPECT_TRUE(check_basic.collect_minima_list()==false);
    bv::CheckSameMinimumCartesian<3>  check_eigenvalues(opt, pot, origin, hs_radii, rattlers, dtol, true, false);
    EXPECT_TRUE(check_eigenvalues.perform_convergence_test()==true);
    EXPECT_TRUE(check_eigenvalues.collect_minima_list()==false);
    bv::CheckSameMinimumCartesian<3>  check_minima(opt, pot, origin, hs_radii, rattlers, dtol, false, true);
    EXPECT_TRUE(check_minima.perform_convergence_test()==false);
    EXPECT_TRUE(check_minima.collect_minima_list()==true);
    bv::CheckSameMinimumCartesian<3>  check_both(opt, pot, origin, hs_radii, rattlers, dtol, true, true);
    EXPECT_TRUE(check_both.perform_convergence_test()==true);
    EXPECT_TRUE(check_both.collect_minima_list()==true);
}

TEST_F(CheckSameMinimumTest, MCInteraction){
    auto opt = std::make_shared<fire_t>(pot, origin, 1e-2, 1, 1);
    mcpele::MC mc(pot, x, 1, stepsize);
    shared_ptr<mcpele::TakeStep> sampler_uniform = std::make_shared<mcpele::RandomCoordsDisplacement>();
    mc.set_takestep(sampler_uniform);
    shared_ptr<mcpele::AcceptTest> metropolis = std::make_shared<mcpele::MetropolisTest>(42);
    mc.add_accept_test(metropolis);
    const size_t adj_iter(max_iter/1e1);
    shared_ptr<mcpele::Action> adjust_step = std::make_shared<mcpele::AdjustStep>(0.2, 0.5, adj_iter, adj_iter/1e1);
    mc.add_action(adjust_step);
    //add conf tests, check same minimum
    shared_ptr<mcpele::ConfTest> check_basic = std::make_shared<bv::CheckSameMinimumCartesian<3> >(opt, pot, origin, hs_radii, rattlers, dtol);
    shared_ptr<mcpele::ConfTest> check_eigenvalues = std::make_shared<bv::CheckSameMinimumCartesian<3> >(opt, pot, origin, hs_radii, rattlers, dtol, true, false);
    shared_ptr<mcpele::ConfTest> check_minima = std::make_shared<bv::CheckSameMinimumCartesian<3> >(opt, pot, origin, hs_radii, rattlers, dtol, false, true);
    shared_ptr<mcpele::ConfTest> check_both = std::make_shared<bv::CheckSameMinimumCartesian<3> >(opt, pot, origin, hs_radii, rattlers, dtol, true, true);
    mc.add_conf_test(check_basic);
    mc.add_conf_test(check_eigenvalues);
    mc.add_late_conf_test(check_minima);
    mc.add_late_conf_test(check_both);
    //run mc
    //mc.set_print_progress();
    const size_t niter = 1e2;
    mc.run(niter);
    //check output
    EXPECT_TRUE(mc.get_iterations_count()==niter);
    EXPECT_NEAR(mc.get_conf_rejection_fraction(), 0, 1e-10); //there is only one minimum, so there should be no rejection due to check same minimum
}

TEST_F(CheckSameMinimumTest, FindkTestSingleBasin){
    mcpele::MC mc(pot, x, 1, stepsize);
    shared_ptr<mcpele::TakeStep> sampler_uniform = std::make_shared<mcpele::RandomCoordsDisplacement>();
    mc.set_takestep(sampler_uniform);
    //add action findk
    const size_t findk__avg_count = 1e3;
    const double findk__target = 0.85;
    const size_t findk__navg = 1e3;
    const double findk__tol = 0.05;
    const double findk__min = 0;
    const double findk__max = 10;
    const double findk__bin = 0.2;
    shared_ptr<mcpele::Action> findk = std::make_shared<bv::Findk>(origin, rattlers, nr_dim, findk__avg_count, findk__target, 0.424242, findk__navg, findk__tol, findk__min, findk__max, findk__bin);
    mc.add_action(findk);
    //run mc
    //mc.set_print_progress();
    const size_t niter = 1e5;
    mc.run(niter);
    //check output
    EXPECT_TRUE(mc.get_iterations_count()==niter);
    EXPECT_NEAR(mc.get_conf_rejection_fraction(), 0, 1e-10); //there is only one minimum, so there should be no rejection due to check same minimum
    //since there is only one basin, and no rejection, k should decrease to zero
    //the precise final value depends on the inital value, the iteration, etc.
    EXPECT_NEAR(std::static_pointer_cast<bv::Findk>(findk)->get_k(), 0, 1);
    //check that stepsize of mc is correctly adapted to k as adjusted in findk
    EXPECT_NEAR_RELATIVE(mc._stepsize, 1/sqrt( std::static_pointer_cast<bv::Findk>(findk)->get_k() ), 1e-15);
}

