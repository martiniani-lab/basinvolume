#include <random>

#include <gtest/gtest.h>

#include "pele/inversepower_stillinger.h"
#include "pele/lbfgs.h"
#include "pele/lj_cut.h"
#include "pele/modified_fire.h"

#include "basinvolume/check_minimum_is_hcp.h"

TEST(BasicY4, Works)
{
    std::vector<std::complex<double> > r_true =
    {std::complex<double>(-0.01580473901589224, + 0.05418767662591624),
    std::complex<double>(-0.2107406047941103, + 0.0383164735989292),
    std::complex<double>(-0.2508924538339834, - 0.3345232717786446), 
    std::complex<double>(0.1520637903896079,- 0.3041275807792157),
    std::complex<double>(-0.1926808175955507, 0),
    std::complex<double>(-0.1520637903896079, - 0.3041275807792157),
    std::complex<double>(-0.2508924538339834, + 0.3345232717786446),
    std::complex<double>(0.2107406047941103, + 0.0383164735989292),
    std::complex<double>(-0.01580473901589224, - 0.05418767662591624)};
    for (int m = -4; m <= 4; ++m) {
        const std::complex<double> r = bv::Y4M(m, 1, 2, 3);
        EXPECT_NEAR(r_true.at(m + 4).real(), r.real(), 1e-15);
        EXPECT_NEAR(r_true.at(m + 4).imag(), r.imag(), 1e-15);
    }
}

TEST(Q4HCP, Works)
{
    const double q4_true = 7./72.;
    bv::CheckMinimumIsHCP c;
    pele::Array<double> x(3 * 64);
    size_t n = 0;
    for (size_t k = 0; k < 4; ++k) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t i = 0; i < 4; ++i) {
                x[n * 3] = 2 * i + (j + k) % 2;
                x[n * 3 + 1] = std::sqrt(3) * (j + (k % 2) / 3.);
                x[n * 3 + 2] = 2 * std::sqrt(6) / 3 * k;
                ++n;
            }
        }
    }
    const double q4_comp = c.get_Q4(x, 22);
    EXPECT_DOUBLE_EQ(q4_true, q4_comp);
}

TEST(Q4HCPConfTest, Works)
{
    const size_t a = 4;
    pele::Array<double> x(3 * std::pow(a, 3));
    size_t n = 0;
    for (size_t k = 0; k < a; ++k) {
        for (size_t j = 0; j < a; ++j) {
            for (size_t i = 0; i < a; ++i) {
                x[n * 3] = 2 * i + (j + k) % 2;
                x[n * 3 + 1] = std::sqrt(3) * (j + (k % 2) / 3.);
                x[n * 3 + 2] = 2 * std::sqrt(6) / 3 * k;
                ++n;
            }
        }
    }
    pele::Array<double> bv = {a * (2), a * (std::sqrt(3)), a * (sqrt(6) / 3 * 2)};
    std::shared_ptr<pele::BasePotential> pot = std::make_shared<pele::LJCutPeriodic>(4., 4., 2.5, bv);
    std::shared_ptr<pele::GradientOptimizer> optimizer = std::make_shared<pele::MODIFIED_FIRE>(pot, x, 1, 1, 1);
    optimizer->set_max_iter(1e7);
    optimizer->set_tol(1e-10);
    bv::CheckMinimumIsHCP c(optimizer, 1e-10, bv);
    EXPECT_TRUE(c.conf_test(x, NULL));
}
