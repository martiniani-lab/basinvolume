#include <gtest/gtest.h>

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
