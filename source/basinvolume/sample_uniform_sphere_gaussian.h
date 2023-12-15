#ifndef _BV_SAMPLE_UNIFORM_SPHERE_GAUSSIAN_H__
#define _BV_SAMPLE_UNIFORM_SPHERE_GAUSSIAN_H__

#include "mcpele/gaussian_coords_displacement.h"
#include "mcpele/mc.h"
#include <random>

namespace bv {

/**
 * Sample a simple Gaussian distribution N(coords, stepsize)
 * this step samples first from the standard normal N(0, 1) and outputs a
 * random variate sampled from N(0, stepsize)
 */

class SampleUniformSphereGaussian : public mcpele::GaussianTakeStep {
protected:
  pele::Array<double> m_origin;

public:
  SampleUniformSphereGaussian(const size_t rseed, const double stepsize,
                              const pele::Array<double> origin);
  virtual ~SampleUniformSphereGaussian() {}
  virtual void displace(pele::Array<double> &coords, mcpele::MC *mc);
};

} // namespace bv

#endif // #ifndef _BV_SAMPLE_UNIFORM_SPHERE_GAUSSIAN_H__
