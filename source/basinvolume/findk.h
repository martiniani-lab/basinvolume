#ifndef _BV_FINDK_H
#define _BV_FINDK_H

#include <list>
#include <vector>
#include "pele/array.h"
#include "pele/distance.h"
#include "mcpele/mc.h"
#include "mcpele/histogram.h"
#include "mcpele/actions.h"
#include "mcpele/takestep.h"
#include "pele/harmonic.h"

namespace bv{

/*
 * Findk accept test, THIS IS A FICTIOUS ACTION (see note)
 * find k for an harmonic potential such that the acceptance is within some range
 * navg number of steps over which acceptance fraction is averaged
 * factor has to be in (0,1)
 * get_prob returns the probability (_acceptedf) associated with kmax
 * avg_count is the number of steps over which the displacement squared is averaged
 *
 *note: this class does some hacky things to exploit the behaviour of MC to get it to do something
 *that it wasn't originally entirely designed for. Weird things:
 * the potential is entirely fictitious, so _k is adjusted through the stepsize
 * set MC->_niter to the largest unsigned inter so that the calculation must terminate
 * */

class Findk : public mcpele::Action{

protected:
    void _get_vec_distance(const pele::Array<double>& x);
    void adjust_k(const size_t, mcpele::MC*);
    pele::Array<double> _origin;
    pele::Array<double>_rattlers;
    pele::Array<double>_distance;
    double _target;
    double _factor;
    double _acceptedf;
    double _k;
    double _tol;
    size_t _ndim;
    size_t _nparticles;
    size_t _avg_count;
    size_t _navg;
    size_t _naccepted;
    size_t _nrejected;
    size_t _start;
    bool _converged;

private:
    mcpele::Histogram _hist;

public:
    Findk(pele::Array<double> origin, pele::Array<double> rattlers, size_t ndim, size_t avg_count, double target, double factor,
            size_t navg, double tol, double min, double max, double bin);
    virtual ~Findk() {}
    virtual void action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc);
    double get_prob(){return _acceptedf;}
    double get_k(){return _k;}
    int get_entries() const {return _hist.entries();}
    double get_mean() const {return _hist.get_mean();}
    double get_variance() const {return _hist.get_variance();}
    pele::Array<double> get_histogram() const {
        std::vector<double> vecdata(_hist.get_vecdata());
        pele::Array<double> histogram(vecdata);
        return histogram.copy();
    }
};

}//namespace bv

#endif//#ifndef _BV_FINDK_H
