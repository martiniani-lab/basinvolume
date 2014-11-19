#ifndef _BV_UTILS_H
#define _BV_UTILS_H

#include <cmath>
#include <algorithm>
#include <list>
#include <vector>
#include "pele/array.h"
#include "pele/distance.h"
#include <fstream>

using std::runtime_error;
using pele::Array;
using std::sqrt;

namespace bv{


/**
 * return the distance between coords and origin after subtracting the center
 * of mass
 */
inline double get_distance_com(const pele::Array<double>& coords, 
        const pele::Array<double>& origin, const size_t ndim)
{
    pele::Array<double> delta_com(ndim, 0);
    pele::Array<double> distance(coords.size());
    size_t nparticles = coords.size() / ndim;

    for(size_t i=0; i<nparticles; ++i) {
        size_t const i1 = i*ndim;
        for(size_t j=0; j<ndim; ++j) {
            double const d = (coords[i1+j] - origin[i1+j]);
            distance[i1+j] = d;
            delta_com[j] += d;
        }
    }

    delta_com /= nparticles;

    for(size_t i=0;i < nparticles; ++i) {
        size_t const i1 = i*ndim;
        for(size_t j=0; j<ndim; ++j) {
            distance[i1+j] -= delta_com[j];
        }
    }

    double d = norm(distance);
    return d;
}

Array<double> cread_txt(const std::string fname){
    std::ifstream input(fname,  std::ifstream::in);
    std::vector<double> data;
    double x;
    while(input >> x){data.push_back(x);}
    return Array<double>(data).copy();
}

}

#endif
