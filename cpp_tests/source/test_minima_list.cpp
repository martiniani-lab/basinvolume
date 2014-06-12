//#include "pele/array.h"
//#include "pele/frozen_atoms.h"
//#include "pele/hs_wca.h"
#include "basinvolume/minima_list.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <gtest/gtest.h>

/*
using pele::Array;
using pele::HS_WCAFrozen;
using pele::HS_WCAPeriodicFrozen;
using pele::HS_WCA;
using pele::HS_WCAPeriodic;
using pele::HS_WCA2D;
using pele::HS_WCA2DFrozen;
using pele::HS_WCAPeriodic2D;
using pele::HS_WCAPeriodic2DFrozen;
*/

#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(fabs(A)/(fabs(A)+fabs(B)+1), fabs(B)/(fabs(A)+fabs(B)+1), T)

class MinimaListTest: public ::testing::Test{
public:
    double a;
    virtual void SetUp(){
        a=12;
    }
    virtual void TearDown() {
        ;
    }
};

TEST_F(MinimaListTest, BasicTest){
    EXPECT_NEAR(a, a, 1e-10);
    EXPECT_NEAR_RELATIVE(a, a, 1e-10);
}
