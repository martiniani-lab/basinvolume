#include <random>

#include <gtest/gtest.h>

#include "pele/hs_wca.h"
#include "pele/lbfgs.h"
#include "pele/harmonic.h"

#include "mcpele/mc.h"
#include "mcpele/gaussian_coords_displacement.h"

#include "basinvolume/check_overlap.h"
#include "basinvolume/check_overlap_cell_lists.h"

class CheckOverlapTest : public ::testing::Test {
public:
    static const size_t nr_dim = 3;
    size_t nr_particles;
    size_t nr_dof;
    pele::Array<double> x;
    pele::Array<double> x_overlap;
    pele::Array<double> hs_radii;
    pele::Array<double> boxvec;
    double k;
    std::shared_ptr<pele::BasePotential> pot;
    virtual void SetUp()
    {
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif
        nr_particles = 12;
        nr_dof = nr_particles * nr_dim;
        x = pele::Array<double>(nr_dof, 0);
        x_overlap = pele::Array<double>(nr_dof, 0);
        hs_radii = pele::Array<double>(nr_particles, 1);
        for (size_t i = 0; i < nr_particles; ++i) {
            x[i * nr_dim] = 2.1 * i;
            x_overlap[i * nr_dim] = 1.9 * i;
        }
        boxvec = pele::Array<double>(nr_dim, 2 * hs_radii.get_max() + std::max<double>(x.get_max(), x_overlap.get_max()));
        k = 4242;
        pot = std::make_shared<pele::Harmonic>(x, k, nr_dim);
    }
};

TEST_F(CheckOverlapTest, Works)
{
    mcpele::MC mc(pot, x, 1);
    std::shared_ptr<mcpele::TakeStep> sampler_uniform = std::make_shared<mcpele::SampleGaussian>(42, 1, x);
    mc.set_takestep(sampler_uniform);
    bv::CheckOverlapPeriodic<nr_dim> check_overlap(hs_radii, boxvec);
    EXPECT_TRUE(check_overlap.conf_test(x, &mc));
    EXPECT_FALSE(check_overlap.conf_test(x_overlap, &mc));
    bv::CheckOverlapCartesian<nr_dim> check_overlap_non_periodic(hs_radii);
    EXPECT_TRUE(check_overlap_non_periodic.conf_test(x, &mc));
    EXPECT_FALSE(check_overlap_non_periodic.conf_test(x_overlap, &mc));
}

TEST_F(CheckOverlapTest, CellLists_Works)
{
    mcpele::MC mc(pot, x, 1);
    std::shared_ptr<mcpele::TakeStep> sampler_uniform = std::make_shared<mcpele::SampleGaussian>(42, 1, x);
    mc.set_takestep(sampler_uniform);
    bv::CheckOverlapPeriodicCellLists<nr_dim> check_overlap(hs_radii.copy(), boxvec.copy(), false);
    EXPECT_TRUE(check_overlap.conf_test(x, &mc));
    EXPECT_FALSE(check_overlap.conf_test(x_overlap, &mc));
    bv::CheckOverlapCartesianCellLists<nr_dim> check_overlap_non_periodic(hs_radii, boxvec, false);
    EXPECT_TRUE(check_overlap_non_periodic.conf_test(x, &mc));
    EXPECT_FALSE(check_overlap_non_periodic.conf_test(x_overlap, &mc));
}

class CheckOverlapManyParticlesTest : public ::testing::Test {
public:
    static const size_t nr_dim = 2;
    size_t nr_particles;
    size_t nr_dof;
    pele::Array<double> x_initial;
    pele::Array<double> x_minimized;
    pele::Array<double> hs_radii;
    pele::Array<double> hs_radii_inflated;
    pele::Array<double> boxvec;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniL;
    std::uniform_real_distribution<double> uniR;
    virtual void SetUp()
    {
        nr_particles = 200;
        nr_dof = nr_particles * nr_dim;
        x_initial = pele::Array<double>(nr_dof, 0);
        x_minimized = pele::Array<double>(nr_dof, 0);
        hs_radii = pele::Array<double>(nr_particles, 0);
        hs_radii_inflated = pele::Array<double>(nr_particles, 0);
        rng.seed(42);
        const double L = 3 * std::sqrt(nr_particles);
        boxvec = pele::Array<double>(nr_dim, L);
        uniL = std::uniform_real_distribution<double>(0, L);
        uniR = std::uniform_real_distribution<double>(0.9, 1.1);
        for (size_t i = 0; i < nr_dof; ++i) {
            x_initial[i] = uniL(rng);
        }
        const double scale = 1.8;
        for (size_t i = 0; i < nr_particles; ++i) {
            hs_radii[i] = uniR(rng);
            hs_radii_inflated[i] = scale * hs_radii[i];
        }
    }
};

TEST_F(CheckOverlapManyParticlesTest, CellListsOverlap_Works)
{
    const double eps = 1;
    const double sca = 0.2;
    std::shared_ptr<pele::HS_WCAPeriodicCellLists<nr_dim> > potential = std::make_shared<pele::HS_WCAPeriodicCellLists<nr_dim> >(eps, sca, hs_radii, boxvec);
    mcpele::MC mc(potential, x_initial, 1);
    std::shared_ptr<mcpele::TakeStep> sampler_uniform = std::make_shared<mcpele::SampleGaussian>(42, 1, x_initial);
    mc.set_takestep(sampler_uniform);
    pele::LBFGS optimizer(potential, x_initial);
    optimizer.run();
    x_minimized = optimizer.get_x();
    std::cout << "energy before: " << potential->get_energy(x_initial) << "\n";
    std::cout << "energy after: " << potential->get_energy(x_minimized) << std::endl;
    EXPECT_FALSE(bv::CheckOverlapPeriodicCellLists<nr_dim>(hs_radii, boxvec, false).conf_test(x_initial, &mc));
    EXPECT_TRUE(bv::CheckOverlapPeriodicCellLists<nr_dim>(hs_radii, boxvec, false).conf_test(x_minimized, &mc));
    EXPECT_FALSE(bv::CheckOverlapPeriodicCellLists<nr_dim>(hs_radii_inflated, boxvec, false).conf_test(x_minimized, &mc));
}
