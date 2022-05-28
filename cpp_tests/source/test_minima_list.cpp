#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <gtest/gtest.h>

#include "pele/array.h"

#include "basinvolume/minima_list.h"

#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(fabs(A)/(fabs(A)+fabs(B)+1), fabs(B)/(fabs(A)+fabs(B)+1), T)

class MinimaListTest: public ::testing::Test{
public:
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

std::vector<size_t> nr_minima_visits(bv::MinimaList & ml)
{
    std::vector<size_t> nvec(ml.nr_distinct_minima());
    size_t i = 0;
    for (auto const & m : ml){
        nvec[i++] = m.count();
    }
    return nvec;
}

/** return the vector of energies to each minimum */
std::vector<double> energies(bv::MinimaList & ml)
{
    std::vector<double> evec(ml.nr_distinct_minima());
    size_t i = 0;
    for (auto const & m : ml){
        evec[i] = m.energy();
        ++i;
    }
    return evec;
}


TEST_F(MinimaListTest, CountingAndCoords){
    bv::MinimaList ml(tol_delta_x, tol_energy, tol_delta_x_element);
    bv::MinimaList mlr(tol_delta_x, tol_energy, tol_delta_x_element);
    for (size_t i = 0; i < nr_insertions; ++i){
	const bool status = ml.insert_minimum(42, 44, x, rattlera);
	bool statusr;
	if (i!=3) statusr= mlr.insert_minimum(42, 44, xr, rattlerb);
	else statusr= mlr.insert_minimum(42, 44, x, rattlerb);
	EXPECT_TRUE( (status==false) ^ (i==0) );
	EXPECT_TRUE( (statusr==false) ^ (i==0) );
    }
    EXPECT_TRUE(ml.nr_distinct_minima()==1);
    EXPECT_TRUE(mlr.nr_distinct_minima()==1);
    EXPECT_EQ(nr_minima_visits(ml).at(0), nr_insertions);
    EXPECT_EQ(nr_minima_visits(mlr).at(0), nr_insertions);
}

TEST_F(MinimaListTest, DeltaXScalarTest){
    bv::MinimaList ml(tol_delta_x, tol_energy, tol_delta_x_element);
    bool status;
    status = ml.insert_minimum(42, 44, x, rattlera);
    EXPECT_TRUE(status);
    status = ml.insert_minimum(42+0.5*tol_delta_x, 44, x, rattlera);
    EXPECT_TRUE(status==false);
    status = ml.insert_minimum(42+1.5*tol_delta_x, 44, x, rattlera);
    EXPECT_TRUE(status);
    status = ml.insert_minimum(42-1.5*tol_delta_x, 44, x, rattlera);
    EXPECT_TRUE(status);
    EXPECT_TRUE(ml.nr_distinct_minima()==3);
    EXPECT_TRUE(nr_minima_visits(ml).at(0)==2);
    EXPECT_TRUE(nr_minima_visits(ml).at(1)==1);
    EXPECT_TRUE(nr_minima_visits(ml).at(2)==1);
}

TEST_F(MinimaListTest, EnergyTest){
    bv::MinimaList ml(tol_delta_x, tol_energy, tol_delta_x_element);
    bool status;
    status = ml.insert_minimum(42, 44, x, rattlera);
    EXPECT_TRUE(status);
    EXPECT_EQ(ml.nr_distinct_minima(), 1u);
    EXPECT_EQ(nr_minima_visits(ml).at(0), 1u);
    status = ml.insert_minimum(42, 44+0.5*tol_energy, x, rattlera);
    EXPECT_FALSE(status);
    EXPECT_EQ(ml.nr_distinct_minima(), 1u);
    EXPECT_EQ(nr_minima_visits(ml).at(0), 2u);
    status = ml.insert_minimum(42, 44+1.5*tol_energy, x, rattlera);
    EXPECT_TRUE(status);
    status = ml.insert_minimum(42, 44-1.5*tol_energy, x, rattlera);
    EXPECT_TRUE(status);
    EXPECT_TRUE(ml.nr_distinct_minima()==3);
    EXPECT_TRUE(nr_minima_visits(ml).at(0)==2);
    EXPECT_TRUE(nr_minima_visits(ml).at(1)==1);
    EXPECT_TRUE(nr_minima_visits(ml).at(2)==1);
    EXPECT_NEAR(44, energies(ml).at(0), 1e-15);
    EXPECT_NEAR(44+1.5*tol_energy, energies(ml).at(1), 1e-15);
    EXPECT_NEAR(44-1.5*tol_energy, energies(ml).at(2), 1e-15);
}

TEST_F(MinimaListTest, CoordinateTest2){
    bv::MinimaList ml(tol_delta_x, tol_energy, tol_delta_x_element);
    bool status;
    status = ml.insert_minimum(42, 44, x, rattlera);
    EXPECT_TRUE(status);
    arr_t y(x.copy());
    y[4] += 0.5*tol_delta_x_element;
    status = ml.insert_minimum(42, 44, y, rattlera);
    EXPECT_TRUE(status==false);
    y[4] += tol_delta_x_element;
    status = ml.insert_minimum(42, 44, y, rattlera);
    EXPECT_TRUE(status);
    status = ml.insert_minimum(42, 44, y, rattlera);
    EXPECT_TRUE(status==false);
}
