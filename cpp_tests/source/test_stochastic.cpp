#include <gtest/gtest.h>

#include "basinvolume/check_exponentially_decaying_profile.h"
#include "basinvolume/record_acceptance_histogram.h"

TEST(StochasticTest, ProbabilityProfileWorks)
{
    const size_t nr_samples_per_bin = 10000;
    pele::Array<double> origin = {1.2, 2.1, 3.9, 4.8, 5.7, 6.6, 7.5, 8.4, 9.3, 9.2, 9.1};
    const double unity_radius = 3.3;
    const double decay_length = 2.2;
    bv::CheckExponentiallyDecayingProfile test(origin, unity_radius, decay_length);
    bv::CheckExponentiallyDecayingProfile test_cubic(origin, unity_radius, decay_length, true);
    const double rmin = 0;
    const double rmax = 20;
    const size_t nbins = 100;
    const double delta = (rmax - rmin) / nbins;
    bv::MomentAccArray acc(rmin, rmax, nbins);
    bv::MomentAccArray acc_cubic(rmin, rmax, nbins);
    for (size_t i = 0; i < nbins; ++i) {
        const double r = rmin + (0.5 + i) * delta;
        pele::Array<double> x = origin.copy();
        x[i % x.size()] += r;
        for (size_t k = 0; k < nr_samples_per_bin; ++k) {
            acc.add(r, test.conf_test(x, NULL));
            acc_cubic.add(r, test_cubic.conf_test(x, NULL));
        }
        EXPECT_NEAR(acc.get_mean(i), test.get_profile_value(r * r), 2 / std::sqrt(nr_samples_per_bin));
        EXPECT_NEAR(acc_cubic.get_mean(i), test.get_profile_value(r * r), 2 / std::sqrt(nr_samples_per_bin));
        if (r < unity_radius) {
            EXPECT_DOUBLE_EQ(acc.get_mean(i), 1);
        }
    }
}
