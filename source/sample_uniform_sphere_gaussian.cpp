#include "basinvolume/sample_uniform_sphere_gaussian.h"

namespace bv {

SampleUniformSphereGaussian::SampleUniformSphereGaussian(
    const size_t rseed, const double stepsize, const pele::Array<double> origin)
    : GaussianTakeStep(rseed, stepsize, origin.size()),
      m_origin(origin.copy()) {}

/*see https://en.wikipedia.org/wiki/Multivariate_normal_distribution
 * SampleGaussian::displace ignore the coords passed to it and it sample
 * with mean centered at the origin and stdev defined by the stepsize
 * */
void SampleUniformSphereGaussian::displace(pele::Array<double> &coords,
                                           mcpele::MC *mc) {
  // assert(coords.size() == m_ndim);
  this->m_sample_normal_vec();
  m_normal_vec /= norm(m_normal_vec);
  double randz = m_distribution(m_generator); // this is sample from N(0,1)
  double ldisplacement =
      fabs(randz) * m_stepsize; // this corresponds to N(0,stepsize)
  for (size_t i = 0; i < m_ndim; ++i) {
    coords[i] =
        m_origin[i] +
        m_normal_vec[i] *
            ldisplacement; // here the stepsize plays the same role as the
                           // stdev. This is sampled from N(0,stepsize)
  }
  ++m_count;
}

} // namespace bv
