#ifndef _BV_UTILS_H
#define _BV_UTILS_H

#include <cmath>
#include <algorithm>
#include <list>
#include <vector>
#include "pele/array.h"
#include "pele/distance.h"

using std::runtime_error;
using pele::Array;
using std::sqrt;

namespace bv{

inline double get_distance_com(const pele::Array<double>& coords, const pele::Array<double>& origin, const size_t ndim){
    pele::Array<double> delta_com(ndim,0);
    pele::Array<double> distance(coords.size());
    size_t nparticles = coords.size()/ndim;

    for(size_t i=0;i<nparticles;++i)
    {
        size_t i1 = i*ndim;
        for(size_t j=0;j<ndim;++j){
            double d = (coords[i1+j] - origin[i1+j]);
            distance[i1+j] = d;
            delta_com[j] += d;
        }
    }

    delta_com /= nparticles;

    for(size_t i=0;i<nparticles;++i)
    {
        size_t i1 = i*ndim;
        for(size_t j=0;j<ndim;++j)
            distance[i1+j] -= delta_com[j];
    }

    double d = norm(distance);
    return d;
}

}

#endif
