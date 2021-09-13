#ifndef _BV_FINDK_H
#define _BV_FINDK_H

#include <list>
#include <vector>

#include "pele/array.hpp"
#include "pele/distance.hpp"
#include "pele/harmonic.hpp"

#include "mcpele/histogram.h"
#include "mcpele/mc.h"

namespace bv {

/*
 * Findk accept test, THIS IS A FICTIOUS ACTION (see note)
 * find k for an harmonic potential such that the acceptance is within some range
 * navg number of steps over which acceptance fraction is averaged
 * get_prob returns the probability (_acceptedf) associated with kmax
 *
 *note: this class does some hacky things to exploit the behaviour of MC to get it to do something
 *that it wasn't originally entirely designed for. Weird things:
 * the potential is entirely fictitious, so _k is adjusted through the stepsize
 * set MC->_niter to the largest unsigned inter so that the calculation must terminate
 * */

class Findk : public mcpele::Action {

protected:
    void _get_vec_distance(const pele::Array<double>& x);
    void adjust_k(const size_t, mcpele::MC*);
    pele::Array<double> _origin;
    pele::Array<double>_rattlers;
    pele::Array<double>_distance;
    const double _target;
    double _acceptedf;
    double _k;
    const double _tol;
    const size_t _ndim;
    const size_t _nparticles;
    const size_t _navg;
    size_t _naccepted;
    size_t _nrejected;
    size_t _start;
    bool _converged;
    const bool m_fix_com;

private:
    mcpele::Histogram _hist;

public:
    Findk(pele::Array<double> origin, pele::Array<double> rattlers,
            size_t ndim, double target,
            size_t navg, double tol, double min, double max, double bin, const bool fix_com=true);
    virtual ~Findk() {}
    virtual void action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc);
  double get_prob() const { return static_cast<double>(_naccepted) / (static_cast<double>(_naccepted) + static_cast<double>(_nrejected)); }
  double get_k() const { return _k; }
  int get_entries() const { return _hist.get_count(); }
  double get_mean() const { return _hist.get_mean(); }
  double get_variance() const { return _hist.get_variance(); }
  pele::Array<double> get_histogram() const
  { std::cout << "this \n" ;
    std::vector<double> vecdata(_hist.get_vecdata());
    pele::Array<double> histogram(vecdata);
    return histogram.copy();
  }
};

} // namespace bv

#endif // #ifndef _BV_FINDK_H
