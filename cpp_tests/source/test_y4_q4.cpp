#include <gtest/gtest.h>

#include "basinvolume/Y4m.h"

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
