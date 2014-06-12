#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <gtest/gtest.h>

#include "pele/array.h"
#include "pele/distance.h"
#include "basinvolume/minima_list.h"

#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(fabs(A)/(fabs(A)+fabs(B)+1), fabs(B)/(fabs(A)+fabs(B)+1), T)
//EXPECT_NEAR(a, a, 1e-10);
//EXPECT_NEAR_RELATIVE(a, a, 1e-10);

class MinimaListTest: public ::testing::Test{
public:
    double tol_delta_x;
    double tol_energy;
    double tol_delta_x_element;
    typedef typename pele::cartesian_distance<3> dist_t;
    virtual void SetUp(){
	tol_delta_x = 0.1;
	tol_energy = 0.1;
	tol_delta_x_element = 0.1;
    }
    virtual void TearDown() {

    }
};

TEST_F(MinimaListTest, CountingAndCoords){
    bv::MinimaList<dist_t> ml(tol_delta_x, tol_energy, tol_delta_x_element, std::make_shared<dist_t>());
    bv::MinimaList<dist_t> mlr(tol_delta_x, tol_energy, tol_delta_x_element, std::make_shared<dist_t>());
    const size_t nr_particles = 20;
    const size_t dim = 3;
    std::vector<double> xv(nr_particles*dim,0);
    for (size_t i = 0; i < xv.size(); ++i){xv.at(i) = i;}
    pele::Array<double> x(xv);
    std::vector<double> rattlerav(xv.size(),1);
    std::vector<double> rattlerbv(xv.size(),1);
    const size_t ratpos = 2;
    rattlerbv.at(ratpos) = 0;
    std::vector<double> xvr(xv);
    xvr.at(ratpos) = -424242;
    pele::Array<double> xr(xvr);
    pele::Array<double> rattlera(rattlerav);
    pele::Array<double> rattlerb(rattlerbv);
    const size_t nr_insertions = 10;
    for (size_t i = 0; i < nr_insertions; ++i){
	const bool status = ml.check_new_minimum(42, 44, x, rattlera);
	bool statusr;
	if (i!=3) statusr= mlr.check_new_minimum(42, 44, xr, rattlerb);
	else statusr= mlr.check_new_minimum(42, 44, x, rattlerb);
	EXPECT_TRUE( status==false ^ i==0 );
	EXPECT_TRUE( statusr==false ^ i==0 );
    }
    EXPECT_TRUE(ml.nr_distinct_minima()==1);
    EXPECT_TRUE(mlr.nr_distinct_minima()==1);
    EXPECT_TRUE(ml.nr_minimum_visits(0)==nr_insertions);
    EXPECT_TRUE(mlr.nr_minimum_visits(0)==nr_insertions);
}

// DeltaXScalarTest
// EnergyTest
// CountingDifferentMinima
