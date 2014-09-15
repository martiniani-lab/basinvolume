#include <gtest/gtest.h>

#include "pele/array.h"

#include "basinvolume/check_overlap.h"

class CheckOverlapTest: public ::testing::Test{
public:
    static const size_t nr_dim = 3;
    size_t nr_particles;
    //size_t nr_dim;
    size_t nr_dof;
    pele::Array<double> x;
    pele::Array<double> x_overlap;
    pele::Array<double> hs_radii;
    pele::Array<double> boxvec;
    virtual void SetUp()
    {
        nr_particles = 42;
        //nr_dim = 3;
        nr_dof = nr_particles * nr_dim;
        x = pele::Array<double>(nr_dof, 0);
        x_overlap = pele::Array<double>(nr_dof, 0);
        hs_radii = pele::Array<double>(nr_particles, 1);
        for (size_t i = 0; i < nr_particles; ++i) {
            x[i * nr_dim] = 2.1 * i;
            x_overlap[i * nr_dim] = 1.9 * i;
        }
        boxvec = pele::Array<double>(nr_dim, nr_particles * nr_dof);
    }
};

TEST_F(CheckOverlapTest, Works){
    bv::CheckOverlapPeriodic<nr_dim> check_overlap(hs_radii, boxvec);
    EXPECT_TRUE(check_overlap.conf_test(x, NULL));
    EXPECT_FALSE(check_overlap.conf_test(x_overlap, NULL));
}
