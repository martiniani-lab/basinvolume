#ifndef _BV_UTILS_H
#define _BV_UTILS_H

#include <cmath>
#include <algorithm>
#include <numeric>
#include <list>
#include <vector>
#include "pele/array.h"
#include "pele/distance.h"
#include <fstream>
#include <exception>
#include <limits>

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

inline double statistical_inefficiency(const pele::Array<double>& tsA, const pele::Array<double>& tsB,
        const bool fast, const size_t mintime=10)
{
    // Create copies of input arguments.
    pele::Array<double> A_n(tsA.copy());
    pele::Array<double> B_n(tsB.copy());
    // Be sure A_n and B_n have the same dimensions.
    assert(A_n.size() == B_n.size());
    // Get the length of the timeseries.
    size_t N = A_n.size();
    // Initialize statistical inefficiency estimate with uncorrelated value.
    double g = 1.0;
    // Compute mean of each timeseries.
    double meanA = std::accumulate(A_n.begin(), A_n.end(), 0.0) / static_cast<double>(N);
    double meanB = std::accumulate(B_n.begin(), B_n.end(), 0.0) / static_cast<double>(N);
    // Use temporary copies to make arrays of fluctuation from mean (wrap, do not copy so we're just renaming the arrays).
    A_n -= meanA;
    B_n -= meanB;
    //Compute estimator of covariance of (A,B) using estimator that will ensure C(0) = 1.
    //pele::Array<double> dAB(A_n.copy());
    //dAB *= B_n;
    double sigma2_AB = dot(A_n, B_n) / static_cast<double>(N); //standard estimator to ensure C(0) = 1

    // Trap the case where this covariance is zero, and we cannot proceed.
    double macheps2 = 2 * std::numeric_limits<double>::epsilon();
    if(std::fabs(sigma2_AB) <= macheps2){//sigma2_AB==0.0
        throw std::runtime_error("Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency");
    }
    /*
     Accumulate the integrated correlation time by computing the normalized correlation time at
     increasing values of t.  Stop accumulating if the correlation function goes negative, since
     this is unlikely to occur unless the correlation function has decayed to the point where it
     is dominated by noise and indistinguishable from zero.
     */
    size_t t = 1;
    size_t increment = 1;
    while(t < N-1){
        double C = 0.0;
        for (size_t i=0; i<N-t; ++i){
            C += A_n[i] * B_n[i+t] + B_n[i] * A_n[i+t];
        }
        C /= (2.0 * static_cast<double>(N - t) * sigma2_AB);

        /*Terminate if the correlation function has crossed zero and we've computed the correlation
        function at least up to 'mintime'.*/

        if (C <= 0.0 && t > mintime){
            break;
        }

        //Accumulate contribution to the statistical inefficiency.
        g += 2.0 * C * (1.0 - static_cast<double>(t) / static_cast<double>(N)) * static_cast<double>(increment);

        //Increment t and the amount by which we increment t.
        t += increment;

        //Increase the interval if "fast mode" is on.
        if (fast){
            increment += 1;
        }
    }

    //g must be at least unity
    if (g < 1.0){
        g = 1.0;
    }

    return g;
}

inline pele::Array<double> detect_equilibration(const pele::Array<double>& tsA, const bool fast, const size_t nskip=1)
{
    pele::Array<double> tsA_n = tsA.copy();
    std::vector<double> A(tsA_n.begin(), tsA_n.end());
    size_t T = A.size();
    double meanA = std::accumulate(A.begin(), A.end(), 0.0) / static_cast<double>(T);
    double varA = 0.0;
    for (size_t j=0; j<A.size(); ++j){varA += (A[j]-meanA)*(A[j]-meanA);}
    varA /= static_cast<double>(T);

    // Special case if timeseries is constant.
    if (varA == 0.0){
        std::vector<double> v;
        v.push_back(0);
        v.push_back(1);
        v.push_back(T); //{0, 1, T}
        return pele::Array<double>(v).copy();
    }

    pele::Array<double> g_t(T-1, 1.0);
    pele::Array<double> Neff_t(T-1, 1.0);

    double macheps2 = 2 * std::numeric_limits<double>::epsilon();
    for(size_t i=0; i<T-1; i+=nskip)
    {
        std::vector<double> At(A.begin()+i, A.end());
        double meanAt = std::accumulate(At.begin(), At.end(), 0.0) / static_cast<double>(At.size());
        double varAt = 0.0;
        for (size_t j=0; j<At.size(); ++j){varAt += (At[j]-meanAt)*(At[j]-meanAt);}
        varAt /= static_cast<double>(At.size());
        //if timeseries segment is constant set statistical efficiency to 1
        if (std::fabs(varAt) <= macheps2){//varAt==0.0
            g_t[i] = 1.0;
        }
        else{
            try{
                g_t[i] = statistical_inefficiency(pele::Array<double>(At), pele::Array<double>(At), fast);
            }
            catch (std::exception& e){
                std::cout << "Standard exception: " << e.what() << std::endl;
                std::cout << "varAt: " << varAt << std::endl;
                std::cout << "At: " << pele::Array<double>(At) << std::endl;
                //this should not be necessary as the previous statement should have catched this
                g_t[i] = 1.0;
            }
        }
        Neff_t[i] = (T - i + 1) / g_t[i];
    }

    double Neff_max = Neff_t.get_max();
    size_t t = std::max_element(Neff_t.begin(),Neff_t.end()) - Neff_t.begin();
    double g = g_t[t];

    std::vector<double> v;
    v.push_back((double) t);
    v.push_back(g);
    v.push_back(Neff_max);
    return pele::Array<double>(v).copy();
}





}
#endif
