#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <gtest/gtest.h>

#include "pele/array.h"
#include "pele/distance.h"
#include "basinvolume/minima_list.h"

#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(fabs(A)/(fabs(A)+fabs(B)+1), fabs(B)/(fabs(A)+fabs(B)+1), T)

class MinimaListTest: public ::testing::Test{
public:
    typedef typename pele::cartesian_distance<3> dist_t;
    typedef std::vector<double> vec_t;
    typedef pele::Array<double> arr_t;
    double tol_delta_x;
    double tol_energy;
    double tol_delta_x_element;
    size_t nr_particles;
    size_t dim;
    vec_t xv;
    arr_t x;
    vec_t rattlerav;
    vec_t rattlerbv;
    size_t ratpos;
    vec_t xvr;
    arr_t xr;
    arr_t rattlera;
    arr_t rattlerb;
    size_t nr_insertions;
    virtual void SetUp(){
	tol_delta_x = 0.1;
	tol_energy = 0.1;
	tol_delta_x_element = 0.1;
	nr_particles = 20;
	dim = 3;
	xv.resize(nr_particles*dim);
	x = arr_t(xv);
	for (size_t i = 0; i < xv.size(); ++i){xv.at(i) = i;}
	rattlerav = vec_t(xv.size(),1);
	rattlerbv = vec_t(xv.size(),1);
	ratpos = 2;
	rattlerbv.at(ratpos) = 0;
	xvr = xv;
	xvr.at(ratpos) = -424242;
	xr = arr_t(xvr);
	rattlera = arr_t(rattlerav);
	rattlerb = arr_t(rattlerbv);
	nr_insertions = 10;
    }
    virtual void TearDown() {

    }
};

TEST_F(MinimaListTest, CountingAndCoords){
    bv::MinimaList<dist_t> ml(tol_delta_x, tol_energy, tol_delta_x_element, std::make_shared<dist_t>());
    bv::MinimaList<dist_t> mlr(tol_delta_x, tol_energy, tol_delta_x_element, std::make_shared<dist_t>());
    for (size_t i = 0; i < nr_insertions; ++i){
	const bool status = ml.check_new_minimum(42, 44, x, rattlera);
	bool statusr;
	if (i!=3) statusr= mlr.check_new_minimum(42, 44, xr, rattlerb);
	else statusr= mlr.check_new_minimum(42, 44, x, rattlerb);
	EXPECT_TRUE( (status==false) ^ (i==0) );
	EXPECT_TRUE( (statusr==false) ^ (i==0) );
    }
    EXPECT_TRUE(ml.nr_distinct_minima()==1);
    EXPECT_TRUE(mlr.nr_distinct_minima()==1);
    EXPECT_TRUE(ml.nr_minimum_visits(0)==nr_insertions);
    EXPECT_TRUE(mlr.nr_minimum_visits(0)==nr_insertions);
}

TEST_F(MinimaListTest, DeltaXScalarTest){
    bv::MinimaList<dist_t> ml(tol_delta_x, tol_energy, tol_delta_x_element, std::make_shared<dist_t>());
    bool status;
    status = ml.check_new_minimum(42, 44, x, rattlera);
    EXPECT_TRUE(status);
    status = ml.check_new_minimum(42+0.5*tol_delta_x, 44, x, rattlera);
    EXPECT_TRUE(status==false);
    status = ml.check_new_minimum(42+1.5*tol_delta_x, 44, x, rattlera);
    EXPECT_TRUE(status);
    status = ml.check_new_minimum(42-1.5*tol_delta_x, 44, x, rattlera);
    EXPECT_TRUE(status);
    EXPECT_TRUE(ml.nr_distinct_minima()==3);
    EXPECT_TRUE(ml.nr_minimum_visits(0)==2);
    EXPECT_TRUE(ml.nr_minimum_visits(1)==1);
    EXPECT_TRUE(ml.nr_minimum_visits(2)==1);
}

TEST_F(MinimaListTest, EnergyTest){
    bv::MinimaList<dist_t> ml(tol_delta_x, tol_energy, tol_delta_x_element, std::make_shared<dist_t>());
    bool status;
    status = ml.check_new_minimum(42, 44, x, rattlera);
    EXPECT_TRUE(status);
    EXPECT_TRUE(ml.nr_distinct_minima()==1);
    EXPECT_TRUE(ml.nr_minimum_visits(0)==1);
    status = ml.check_new_minimum(42, 44+0.5*tol_energy, x, rattlera);
    EXPECT_TRUE(status==false);
    EXPECT_TRUE(ml.nr_distinct_minima()==1);
    EXPECT_TRUE(ml.nr_minimum_visits(0)==2);
    status = ml.check_new_minimum(42, 44+1.5*tol_energy, x, rattlera);
    EXPECT_TRUE(status);
    status = ml.check_new_minimum(42, 44-1.5*tol_energy, x, rattlera);
    EXPECT_TRUE(status);
    EXPECT_TRUE(ml.nr_distinct_minima()==3);
    EXPECT_TRUE(ml.nr_minimum_visits(0)==2);
    EXPECT_TRUE(ml.nr_minimum_visits(1)==1);
    EXPECT_TRUE(ml.nr_minimum_visits(2)==1);
    EXPECT_NEAR(44, ml.get_energy(0), 1e-15);
    EXPECT_NEAR(44+1.5*tol_energy, ml.get_energy(1), 1e-15);
    EXPECT_NEAR(44-1.5*tol_energy, ml.get_energy(2), 1e-15);
}

TEST_F(MinimaListTest, CoordinateTest2){
    bv::MinimaList<dist_t> ml(tol_delta_x, tol_energy, tol_delta_x_element, std::make_shared<dist_t>());
    bool status;
    status = ml.check_new_minimum(42, 44, x, rattlera);
    EXPECT_TRUE(status);
    arr_t y(x.copy());
    y[4] += 0.5*tol_delta_x_element;
    status = ml.check_new_minimum(42, 44, y, rattlera);
    EXPECT_TRUE(status==false);
    y[4] += tol_delta_x_element;
    status = ml.check_new_minimum(42, 44, y, rattlera);
    EXPECT_TRUE(status);
    status = ml.check_new_minimum(42, 44, y, rattlera);
    EXPECT_TRUE(status==false);
}

TEST_F(MinimaListTest, EuclideanConnectionTest){
    bv::MinimaList<dist_t> ml(tol_delta_x, tol_energy, tol_delta_x_element, std::make_shared<dist_t>());
    arr_t y(x.copy());
    for (size_t i=0; i < y.size(); ++i){y[i] += sqrt(i+2);}
    ml.check_new_minimum(0.3,0.4,x,rattlera);
    ml.check_new_minimum(0.3,0.4,y,rattlera);
    EXPECT_TRUE(ml.nr_distinct_minima()==2);
    const size_t nr_points(100);
    // compute distance directly, consider arrays x, y, where x corresponds to i and y corresponds to j
    arr_t direct_displ_vector;
    direct_displ_vector.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i){
	direct_displ_vector[i] = (x[i]-y[i])/double(nr_points-1);
    }
    // compute displacement vector buy minimum class
    arr_t displ_vector_A = ml.euclidean_displacement_vector(0,1,nr_points);
    arr_t displ_vector_B = ml.euclidean_displacement_vector(0,y,nr_points);
    // compare
    for (size_t i = 0; i < x.size(); ++i){
	EXPECT_NEAR(direct_displ_vector[i], displ_vector_A[i], 1e-15);
	EXPECT_NEAR(displ_vector_A[i], displ_vector_B[i], 1e-15);
    }
    // test iteration direction
    for (size_t i = 0; i < x.size(); ++i){
	EXPECT_NEAR(y[i]+(nr_points-1)*displ_vector_A[i], x[i], 1e-15);
    }
}
