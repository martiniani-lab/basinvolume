#include <gtest/gtest.h>

#include "basinvolume/check_power_decaying_profile.h"
#include "basinvolume/record_acceptance_histogram.h"

double get_radial_exp_profile(const double u, const double d, const double r2)
{
    if (r2 < u * u) {
        return 1;
    }
    return std::exp(-(std::sqrt(r2) - u) / d);
}

double get_radial_power_profile(const double u, const double e, const double r2)
{
    if (r2 < u * u) {
        return 1;
    }
    return std::pow(u / std::sqrt(r2), e);
}

TEST(StochasticTest, ProbabilityProfileWorks)
{
    const size_t nr_samples_per_bin = 10000;
    pele::Array<double> origin = {1.2, 2.1, 3.9, 4.8, 5.7, 6.6, 7.5, 8.4, 9.3, 9.2, 9.1};
    const double unity_radius = 3.3;
    const double decay_length = 2.2;
    const double exponent = 4;
    std::vector<std::shared_ptr<bv::CheckExponentiallyDecayingProfile> > test = {std::make_shared<bv::CheckExponentiallyDecayingProfile>(origin, unity_radius, decay_length),
            std::make_shared<bv::CheckExponentiallyDecayingProfile>(origin, unity_radius, decay_length, true),
            std::make_shared<bv::CheckPowerDecayingProfile>(origin, unity_radius, exponent)};
    const double rmin = 0;
    const double rmax = 20;
    const size_t nbins = 100;
    const double delta = (rmax - rmin) / nbins;
    std::vector<std::shared_ptr<bv::MomentAccArray> > acc;
    for (size_t i = 0; i < 3; ++i) {
        acc.push_back(std::make_shared<bv::MomentAccArray>(rmin, rmax, nbins));
    }
    for (size_t i = 0; i < nbins; ++i) {
        const double r = rmin + (0.5 + i) * delta;
        pele::Array<double> x = origin.copy();
        x[i % x.size()] += r;
        for (size_t k = 0; k < nr_samples_per_bin; ++k) {
            for (size_t s = 0; s < 3; ++s) {
                acc.at(s)->add(r, test.at(s)->conf_test(x, NULL));
            }
        }
        for (size_t s = 0; s < 3; ++s) {
            EXPECT_NEAR(acc.at(s)->get_mean(i), test.at(s)->get_profile_value(r * r), 2 / std::sqrt(nr_samples_per_bin));
            EXPECT_DOUBLE_EQ(test.at(0)->get_profile_value(r * r), get_radial_exp_profile(unity_radius, decay_length, r * r));
            EXPECT_DOUBLE_EQ(test.at(2)->get_profile_value(r * r), get_radial_power_profile(unity_radius, exponent, r * r));
        }
        if (r < unity_radius) {
            EXPECT_DOUBLE_EQ(acc.at(0)->get_mean(i), 1);
            EXPECT_DOUBLE_EQ(acc.at(2)->get_mean(i), 1);
        }
    }
}
