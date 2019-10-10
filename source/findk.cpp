#include <algorithm>
#include <cmath>
#include <chrono>
#include <ctime>

#include "mcpele/gaussian_coords_displacement.h"

#include "basinvolume/findk.h"

namespace bv {

Findk::Findk(pele::Array<double> origin, pele::Array<double> rattlers, size_t ndim, double target,
        size_t navg, double tol, double min, double max, double bin, const bool fix_com)
	: _origin(origin.copy()),
      _rattlers(rattlers.copy()),
      _distance(origin.size()),
      _target(target),
      _acceptedf(1),
      _k(42424242),
      _tol(tol),
      _ndim(ndim),
      _nparticles(_origin.size() / _ndim),
      _navg(navg),
      _naccepted(0),
      _nrejected(0),
      _start(0),
      _converged(false),
      m_fix_com(fix_com),
      _hist(min, max, bin)
{}

void Findk::_get_vec_distance(const pele::Array<double>& x)
{
    if (m_fix_com) {
        pele::Array<double> delta_com(_ndim, 0);
        for (size_t i = 0; i < _nparticles; ++i) {
            const size_t i1 = i * _ndim;
            for (size_t j = 0; j < _ndim; ++j) {
                const double d = (x[i1 + j] - _origin[i1 + j]);
                _distance[i1 + j] = d;
                delta_com[j] += d;
            }
        }
        delta_com /= _nparticles;
        for (size_t i = 0; i < _nparticles; ++i) {
            const size_t i1 = i * _ndim;
            for (size_t j = 0; j < _ndim; ++j)
                _distance[i1 + j] -= delta_com[j];
        }
    }
    else {
        for (size_t i = 0; i < _nparticles; ++i) {
            const size_t i1 = i * _ndim;
            for (size_t j = 0; j < _ndim; ++j) {
                const double d = (x[i1 + j] - _origin[i1 + j]);
                _distance[i1 + j] = d;
            }
        }
    }
}

void Findk::action(pele::Array<double> &coords, double energy, bool accepted, mcpele::MC* mc)
{

    const size_t mc_count = mc->get_iterations_count();

    if (accepted) {
        ++_naccepted;
    }
    else {
        ++_nrejected;
    }

    if (_converged) { // kmax search converged
        mc->m_niter = std::numeric_limits<size_t>::max(); // can use terminate() when that is merged, leave for now
    }
    else if (mc_count % _navg == 0) { // kmax seach not yet converged
        _acceptedf = static_cast<double>(_naccepted) / (static_cast<double>(_naccepted) + static_cast<double>(_nrejected));
        //adjust step if last two step oscillated around the target, uses a lower bound
        adjust_k(mc_count / _navg, mc);
        //adjust the standard deviation of the normal distribution
        static_cast<mcpele::SampleGaussian*>(mc->get_takestep().get())->set_stepsize(std::sqrt(1.0 / _k));
        //now reset to zero memory of acceptance and rejection
        _naccepted = 0;
        _nrejected = 0;
    }
    //reset coordinates to origin although this obsolete because the new SampleGaussian ignores the new coordinates
    //and resamples from the origin
    //coords.assign(_origin);
}

void Findk::adjust_k(const size_t iterations, mcpele::MC* mc)
{
    // parameter: can be adapted for better convergence
    const size_t period = 3;
    //get k
    const double ik = static_cast<mcpele::SampleGaussian*>(mc->get_takestep().get())->get_stepsize();
    _k = 1 / (ik * ik);
    //debug output

    // Get current time
    auto tp = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(tp);
    std::tm * ptm = std::localtime(&tt);
    char time_str[32];
    std::strftime(time_str, 32, "%d/%m/%Y %H:%M:%S", ptm);

    std::cout << time_str << ": Iteration " << iterations << std::endl; //debug
    std::cout << "_acceptedf " << _acceptedf << std::endl; //debug
    std::cout << "_k " << _k << std::endl; //debug

    //check for convergence
    if (fabs(_target - _acceptedf) < _tol) {
        _converged = true;
        return;
    }
    //adapt k size
    const double tmp1 = 1.0 / (iterations % period + 1);
    const double tmp2 = 1 + (_target - _acceptedf) / (_target + _acceptedf);
    const double tmp = (1 - tmp1) + tmp1 * tmp2;
    _k *= tmp * tmp;
}

} // namespace bv
