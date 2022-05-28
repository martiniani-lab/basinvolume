#ifndef _BV_Y4M_H_
#define _BV_Y4M_H_

#include <cmath>
#include <complex>
#include <stdexcept>

#include "pele/meta_pow.h"

namespace bv {

template<int N>
std::complex<double> cp(const double re, const double im)
{
    return pele::pos_int_pow<N>(std::complex<double>(re, im));
}
    
std::complex<double> Y4M(const short m, const double x, const double y, const double z)
{
    const double r2 = x*x + y*y + z*z;
    const double r4 = r2 * r2;
    if (m == -4) {
        return 3./16. * std::sqrt(35./(2 * M_PI)) * cp<4>(x, -y) / r4;
    }
    if (m == -3) {
        return 3./8. * std::sqrt(35./M_PI) * cp<3>(x, -y) * z / r4;
    }
    if (m == -2) {
        return 3./8. * std::sqrt(5./(2 * M_PI)) * cp<2>(x, -y) * (7*z*z - r2) / r4;
    }
    if (m == -1) {
        return 3./8. * std::sqrt(5./M_PI) * cp<1>(x, -y) * z * (7*z*z - 3*r2) / r4;
    }
    if (m == 0) {
        return 3./16. * std::sqrt(1./M_PI) * (35*pele::pos_int_pow<4>(z) - 30*z*z*r2 + 3*r4) / r4;
    }
    if (m == 1) {
        return -3./8. * std::sqrt(5./M_PI) * cp<1>(x, y) * z * (7*z*z - 3*r2) / r4;
    }
    if (m == 2) {
        return 3./8. * std::sqrt(5./(2 * M_PI)) * cp<2>(x, y) * (7*z*z - r2) / r4;
    }
    if (m == 3) {
        return -3./8. * std::sqrt(35./M_PI) * cp<3>(x, y) * z / r4;
    }
    if (m == 4) {
        return 3./16. * std::sqrt(35./(2 * M_PI)) * cp<4>(x, y) / r4;
    }
    throw std::runtime_error("Y4M: illegal input, M needs to be integer between -4 and 4");
}
    
} // namespace bv

#endif // #ifndef _BV_Y4M_H_
