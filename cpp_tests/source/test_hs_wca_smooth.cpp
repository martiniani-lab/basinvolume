#include "pele/array.h"
#include "basinvolume/hs_wca_smooth.h"

#include <iostream>
#include <stdexcept>
#include <gtest/gtest.h>

using pele::Array;
using bv::HS_WCA_Smooth;


/*
 * HS_WCA tests
 */

class HS_WCA_SmoothTest :  public ::testing::Test
{
public:
    double eps, sca, etrue;
    Array<double> x, g, gnum, radii;
    virtual void SetUp(){
        eps = 2.1;
        sca = 1.4;
        x = Array<double>(9);
        x[0] = 0.1;
        x[1] = 0.2;
        x[2] = 0.3;
        x[3] = 0.44;
        x[4] = 0.55;
        x[5] = 1.66;
        x[6] = 0.88;
        x[7] = 1.1;
        x[8] = 3.32;
        radii = Array<double>(3);
        double f = .35;
        radii[0] = .91 * f;
        radii[1] = 1.1 * f;
        radii[2] = 1.13 * f;
        etrue = 42.956261377570286;
        g = Array<double>(x.size());
        gnum = Array<double>(x.size());
    }
};

TEST_F(HS_WCA_SmoothTest, Energy_Works){
    HS_WCA_Smooth pot(eps, sca, radii);
    double e = pot.get_energy(x);
    ASSERT_NEAR(e, etrue, 1e-10);
}

TEST_F(HS_WCA_SmoothTest, EnergyGradient_AgreesWithNumerical){
    HS_WCA_Smooth pot(eps, sca, radii);
    double e = pot.get_energy_gradient(x, g);
    double ecomp = pot.get_energy(x);
    ASSERT_NEAR(e, ecomp, 1e-10);
    pot.numerical_gradient(x, gnum, 1e-6);
    for (size_t k=0; k<g.size(); ++k){
        ASSERT_NEAR(g[k], gnum[k], 1e-6);
    }
}

TEST_F(HS_WCA_SmoothTest, EnergyGradientHessian_AgreesWithNumerical){
    HS_WCA_Smooth pot(eps, sca, radii);
    Array<double> h(x.size()*x.size());
    Array<double> hnum(h.size());
    double e = pot.get_energy_gradient_hessian(x, g, h);
    double ecomp = pot.get_energy(x);
    pot.numerical_gradient(x, gnum);
    pot.numerical_hessian(x, hnum);

    EXPECT_NEAR(e, ecomp, 1e-10);
    for (size_t i=0; i<g.size(); ++i){
        ASSERT_NEAR(g[i], gnum[i], 1e-6);
    }
    for (size_t i=0; i<h.size(); ++i){
        ASSERT_NEAR(h[i], hnum[i], 1e-3);
    }
}
