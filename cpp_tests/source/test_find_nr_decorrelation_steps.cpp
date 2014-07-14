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

#include "basinvolume/find_nr_decorrelation_steps.h"

#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(fabs(A)/(fabs(A)+fabs(B)+1), fabs(B)/(fabs(A)+fabs(B)+1), T)

using std::shared_ptr;

class FindNrDecorrelationSepsTest: public ::testing::Test{
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

TEST_F(FindNrDecorrelationSepsTest, BasicWorks){
    mcpele::MC mc(pot, x, 1, stepsize);
    shared_ptr<mcpele::TakeStep> sampler_uniform = std::make_shared<mcpele::RandomCoordsDisplacement>(42);
    mc.set_takestep(sampler_uniform);
    //add things to mc
    const size_t niter = 1e5;
    const double desired_mean_rsm_displ_ = 3*stepsize;
    const size_t nr_iterations_start_ = niter/10;
    const size_t nr_samples_avergage_ = 3;
    bv::FindNrDecorrelationSteps* finder = new bv::FindNrDecorrelationSteps(desired_mean_rsm_displ_,
            nr_iterations_start_, nr_samples_avergage_, x, nr_dim);
    mc.add_action(std::shared_ptr<mcpele::Action>(finder));
    EXPECT_EQ(finder->done(), false);
    //run mc
    //mc.set_print_progress();
    mc.run(niter);
    //check output
    EXPECT_EQ(mc.get_iterations_count(), niter);
    EXPECT_EQ(finder->done(), true);
    //std::cout << "finder->get_nr_decorrelation_steps(): " << std::endl;
    //std::cout << finder->get_nr_decorrelation_steps() << std::endl;
}

TEST_F(FindNrDecorrelationSepsTest, NullTargetWorks){
    mcpele::MC mc(pot, x, 1, stepsize);
    shared_ptr<mcpele::TakeStep> sampler_uniform = std::make_shared<mcpele::RandomCoordsDisplacement>(42);
    mc.set_takestep(sampler_uniform);
    //add things to mc
    const size_t niter = 1e5;
    const double desired_mean_rsm_displ_ = 0;
    const size_t nr_iterations_start_ = niter/10;
    const size_t nr_samples_avergage_ = 3;
    bv::FindNrDecorrelationSteps* finder = new bv::FindNrDecorrelationSteps(desired_mean_rsm_displ_,
            nr_iterations_start_, nr_samples_avergage_, x, nr_dim);
    mc.add_action(std::shared_ptr<mcpele::Action>(finder));
    EXPECT_EQ(finder->done(), false);
    //run mc
    //mc.set_print_progress();
    mc.run(niter);
    //check output
    EXPECT_EQ(mc.get_iterations_count(), niter);
    EXPECT_EQ(finder->done(), true);
    EXPECT_EQ(finder->get_nr_decorrelation_steps(), 0u);
}

TEST_F(FindNrDecorrelationSepsTest, MoreNeedsMoreWorks){
    mcpele::MC mc(pot, x, 1, stepsize);
    mcpele::MC mc2(pot, x, 1, stepsize);
    shared_ptr<mcpele::TakeStep> sampler_uniform = std::make_shared<mcpele::RandomCoordsDisplacement>(42);
    shared_ptr<mcpele::TakeStep> sampler_uniform2 = std::make_shared<mcpele::RandomCoordsDisplacement>(42);
    mc.set_takestep(sampler_uniform);
    mc2.set_takestep(sampler_uniform2);
    //add things to mc
    const size_t niter = 1e5;
    const double desired_mean_rsm_displ_ = 1*stepsize;
    const double desired_mean_rsm_displ_2 = 2*stepsize;
    const size_t nr_iterations_start_ = niter/10;
    const size_t nr_samples_avergage_ = 3;
    bv::FindNrDecorrelationSteps* finder = new bv::FindNrDecorrelationSteps(desired_mean_rsm_displ_,
            nr_iterations_start_, nr_samples_avergage_, x, nr_dim);
    bv::FindNrDecorrelationSteps* finder2 = new bv::FindNrDecorrelationSteps(desired_mean_rsm_displ_2,
                nr_iterations_start_, nr_samples_avergage_, x, nr_dim);
    mc.add_action(std::shared_ptr<mcpele::Action>(finder));
    mc.add_action(std::shared_ptr<mcpele::Action>(finder2));
    EXPECT_EQ(finder->done(), false);
    EXPECT_EQ(finder2->done(), false);
    //run mc
    //mc.set_print_progress();
    mc.run(niter);
    mc2.run(niter);
    //check output
    EXPECT_EQ(mc.get_iterations_count(), niter);
    EXPECT_EQ(mc2.get_iterations_count(), niter);
    EXPECT_EQ(finder->done(), true);
    EXPECT_EQ(finder2->done(), true);
    EXPECT_TRUE(finder->get_nr_decorrelation_steps() > 0);
    EXPECT_TRUE(finder->get_nr_decorrelation_steps() < finder2->get_nr_decorrelation_steps());
}

