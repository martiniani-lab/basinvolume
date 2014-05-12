from __future__ import division
from gauss_lobatto import *
import numpy as np

def spring_constants_variable_transform(nr_points, k_max, displ_k0, nr_particles, dimension):
    """
    Given the number of points n,
    the maximum spring constant k_max,
    the average displacement squared at k=0,
    the number of particles,
    and the Ecuclidean dimension of the box;
    computes the k values to use in PT.
    Reference: Daniel A. Asenjo-Andrews, PhD thesis,  p 34
    """
    t = Gauss_Lobatto_abscissas(nr_points)()
    kappa = nr_particles*dimension/displ_k0
    k = [((1.0+k_max/kappa)**((ti+1.0)/2.0)-1.0)*kappa for ti in t]
    return k

def test_variable_transform(k, displ_k0, nr_particles, dimension):
    """
    Computes the GL abscissas from the given kvalues.
    Consistency check only.
    """
    kappa = dimension*nr_particles/displ_k0
    kmax = max(k)
    t = [2*np.log(1+ki/kappa)/np.log(1+kmax/kappa)-1 for ki in k]
    return t

def calculate_GL_integral_with_transform(u_sq_k, k_max, nr_particles, dimension):
    """
    Input: Squared displacements, measured at the spring constant values given by
    spring_constants_variable_transform(nr_points, k_max, displ_k0, nr_particles, dimension).
    Applies the variable transform to the integrand (multiplies by Jacobian).
    Output: Integral over squared displacements from zero to k_max (maximum spring constant).
    """
    nr_points = len(u_sq_k)
    displ_k0 = u_sq_k[0]
    kappa = nr_particles*dimension/displ_k0
    k = spring_constants_variable_transform(nr_points, k_max, displ_k0, nr_particles, dimension)
    f = [u_sq_ki*0.5*(ki+kappa)*np.log(1.0+k_max/kappa) for (u_sq_ki,ki) in zip(u_sq_k,k)]
    return calculate_GL_integral(f)

if __name__ == "__main__":
    nr_points = 6
    k_max = 1042
    displ_k0 = 22
    nr_particles = 128
    dimension = 3
    k = spring_constants_variable_transform(nr_points, k_max, displ_k0, nr_particles, dimension)
    print k
    t = test_variable_transform(k, displ_k0, nr_particles, dimension)
    print t
    print Gauss_Lobatto_abscissas(nr_points)()
    