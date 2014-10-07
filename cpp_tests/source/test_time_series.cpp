#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <memory>
#include <gtest/gtest.h>

#include "pele/array.h"
#include "pele/harmonic.h"

#include "mcpele/metropolis_test.h"

#include "basinvolume/record_displacement_timeseries.h"

#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(fabs(A)/(fabs(A)+fabs(B)+1), fabs(B)/(fabs(A)+fabs(B)+1), T)

struct TrivialTakestep : public mcpele::TakeStep{
    size_t call_count;
    TrivialTakestep()
        : call_count(0)
    {}
    virtual void takestep(pele::Array<double> &coords, double stepsize, mcpele::MC * mc=NULL)
    {
        call_count++;
    }
};

TEST(TimeSeriesMoments, Works){
    const size_t boxdim = 3;
    const size_t nparticles = 100;
    const size_t ndof = nparticles * boxdim;
    const size_t niter = 40000;
    const size_t record_every = 1;
    const double stepsize = 1e-1;
    const size_t eq_steps = (niter/record_every)/4;
    pele::Array<double> coords(ndof, 0);
    pele::Array<double> origin(ndof, 0);

    const double k = 20;
    std::shared_ptr<pele::Harmonic> potential = std::make_shared<pele::Harmonic>(origin, k, boxdim);
    std::shared_ptr<mcpele::MC> mc = std::make_shared<mcpele::MC>(potential, coords, 1, stepsize);

    mcpele::MetropolisTest* metropolis = new mcpele::MetropolisTest(43);
    //mcpele::AdjustStep* adjust_step = new mcpele::AdjustStep(0.2, 0.9, eq_steps, eq_steps/10);
    bv::RecordDisplacementTimeseries* ts = new bv::RecordDisplacementTimeseries(origin, boxdim, niter, record_every);

    mc->add_accept_test(std::shared_ptr<mcpele::MetropolisTest>(metropolis));
    mc->set_takestep(std::make_shared<mcpele::RandomCoordsDisplacement>(42));
    //mc->add_action(std::shared_ptr<mcpele::AdjustStep>(adjust_step));
    mc->add_action(std::shared_ptr<bv::RecordDisplacementTimeseries>(ts));
    //mc->set_takestep(std::make_shared<TrivialTakestep>());

    mc->run(niter);
    EXPECT_EQ(mc->get_iterations_count(), niter);

    pele::Array<double> series = ts->get_time_series();
    EXPECT_EQ(series.size(), niter / record_every);

    const double mean_true = ndof / k;
    const double var_true = 2*ndof / (k*k);

    mcpele::Moments moments;
    for(size_t i=eq_steps; i<series.size();++i){
        moments(series[i]*series[i]);
    }

    const double mean = moments.mean();
    const double var = moments.variance();
    EXPECT_NEAR_RELATIVE(mean_true, mean, 1e-2);
    EXPECT_NEAR_RELATIVE(var_true, var, 1e-2);
    std::cout<<"accept_f "<<mc->get_accepted_fraction()<<std::endl;
    std::cout<<"stepsize "<<mc->get_stepsize()<<std::endl;
    std::cout<<"mean_true "<<mean_true<<" mean "<<mean<<std::endl;
    std::cout<<"var_true "<<var_true<<" var "<<var<<std::endl;

}

TEST(MovingAverageTest, Works) {
    const size_t boxdim = 3;
    const size_t nparticles = 100;
    const size_t ndof = nparticles * boxdim;
    const size_t niter = 20000;
    const size_t record_every = 10;
    const size_t eq_steps = (niter/record_every)/2;
    const double stepsize = 1e-1;
    pele::Array<double> coords(ndof, 0);
    pele::Array<double> origin(ndof, 0);

    const double k = 20;
    std::shared_ptr<pele::Harmonic> potential = std::make_shared<pele::Harmonic>(origin, k, boxdim);
    std::shared_ptr<mcpele::MC> mc = std::make_shared<mcpele::MC>(potential, coords, 1, stepsize);

    mcpele::MetropolisTest* metropolis = new mcpele::MetropolisTest(43);
    mcpele::AdjustStep* adjust_step = new mcpele::AdjustStep(0.2, 0.9, eq_steps, eq_steps/10);
    bv::RecordDisplacementTimeseries* ts = new bv::RecordDisplacementTimeseries(origin, boxdim, niter, record_every);

    mc->add_accept_test(std::shared_ptr<mcpele::MetropolisTest>(metropolis));
    mc->set_takestep(std::make_shared<mcpele::RandomCoordsDisplacement>(42));
    mc->add_action(std::shared_ptr<mcpele::AdjustStep>(adjust_step));
    mc->add_action(std::shared_ptr<bv::RecordDisplacementTimeseries>(ts));
    //mc->set_takestep(std::make_shared<TrivialTakestep>());

    mc->run(niter);
    EXPECT_EQ(mc->get_iterations_count(), niter);

    pele::Array<double> series = ts->get_time_series();
    EXPECT_EQ(series.size(), niter / record_every);

    const size_t nr_steps_to_check = (niter - eq_steps) / record_every;

    const double mean = (ts->get_moving_average_mean(nr_steps_to_check)).first;
    const double var = (ts->get_moving_average_variance(nr_steps_to_check)).first;
    std::cout<<"accept_f "<<mc->get_accepted_fraction()<<std::endl;
    std::cout<<"stepsize "<<mc->get_stepsize()<<std::endl;
    std::cout<<" mean "<<mean<<std::endl;
    std::cout<<" var "<<var<<std::endl;

    const double rel_std_threshold = 0.5;
    EXPECT_TRUE(ts->moving_average_is_stable(nr_steps_to_check, rel_std_threshold));
    EXPECT_TRUE(ts->moving_average_is_stable(nr_steps_to_check, 5e-2));
    EXPECT_FALSE(ts->moving_average_is_stable(nr_steps_to_check, 1e-5));
    EXPECT_FALSE(ts->moving_average_is_stable(nr_steps_to_check, 1e-10));
    EXPECT_FALSE(ts->moving_average_is_stable(nr_steps_to_check, 1e-20));
    EXPECT_FALSE(ts->moving_average_is_stable(nr_steps_to_check, 0));
}
