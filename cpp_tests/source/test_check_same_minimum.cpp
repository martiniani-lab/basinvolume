#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "pele/array.h"
#include "pele/distance.h"
#include "pele/harmonic.h"
#include "pele/lbfgs.h"

#include "mcpele/mc.h"

#include "basinvolume/conf_test.h"

#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(fabs(A)/(fabs(A)+fabs(B)+1), fabs(B)/(fabs(A)+fabs(B)+1), T)

class CheckSameMinimumTest: public ::testing::Test{
public:
typedef typename pele::cartesian_distance<3> dist_t;
    typedef std::vector<double> vec_t;
    typedef pele::Array<double> arr_t;
    typedef pele::Harmonic pot_t;
    typedef pele::LBFGS opt_t;
    size_t nr_particles;
    size_t nr_dim;
    size_t nr_dof;
    arr_t origin;
    double k;
    pele::BasePotential* pot;
    double _lbfgstol;
    double _lbfgsM;
    arr_t hs_radii;
    arr_t rattlers;
    double dtol;
    virtual void SetUp(){
	nr_particles = 42;
	nr_dim = 3;
	nr_dof = nr_particles*nr_dim;
	origin = arr_t(nr_dof,0).copy();
	k = 42;
	pot = new pot_t(origin, 42, nr_dim);
	_lbfgstol = 1e-2;
	_lbfgsM = 5;
	hs_radii = arr_t(nr_dof,1).copy();
	rattlers = arr_t(nr_dof,1).copy();
	dtol = 1e-6;
    }
    virtual void TearDown(){
	delete pot;
    }
};

TEST_F(CheckSameMinimumTest, BasicFunctionality){
    // TODO: adapt to new layout of check same minimum class
    //pele::GradientOptimizer* opt = new opt_t(pot, origin, _lbfgstol, _lbfgsM);
    //bv::CheckSameMinimum<dist_t> check(opt, pot, origin, hs_radii, rattlers, dtol, std::make_shared<dist_t>());
    //delete opt;
}
