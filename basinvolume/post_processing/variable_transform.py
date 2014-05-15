from __future__ import division
from gauss_lobatto import Gauss_Lobatto_abscissas, Gauss_Lobatto_weights, calculate_GL_integral
import numpy as np

def spring_constants_variable_transform(nr_points, k_max, displ_k_min, nr_particles, dimension, k_min=0.0):
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
    kappa = nr_particles*dimension/displ_k_min
    k = [k_min - kappa + kappa*(1.0+(k_max-k_min)/kappa)**((1.0+ti)/2.0) for ti in t]
    return k

def test_variable_transform(k, displ_k_min, nr_particles, dimension):
    """
    Computes the GL abscissas from the given kvalues.
    Consistency check only.
    """
    kappa = dimension*nr_particles/displ_k_min
    kmax = max(k)
    kmin = min(k)
    t = [2*np.log(1+(ki-kmin)/kappa)/np.log(1+(kmax-kmin)/kappa)-1 for ki in k]
    return t

def calculate_GL_integral_with_transform(u_sq_k, k_max, nr_particles, dimension, k_min=0.0):
    """
    Input: Squared displacements, measured at the spring constant values given by
    spring_constants_variable_transform(nr_points, k_max, displ_k0, nr_particles, dimension).
    Applies the variable transform to the integrand (multiplies by Jacobian).
    Output: Integral over squared displacements from zero to k_max (maximum spring constant).
    """
    nr_points = len(u_sq_k)
    displ_k_min = u_sq_k[0]
    kappa = nr_particles*dimension/displ_k_min
    k = spring_constants_variable_transform(nr_points, k_max, displ_k_min, nr_particles, dimension, k_min)
    f = [u_sq_ki*0.5*(ki-k_min+kappa)*np.log(1.0+(k_max-k_min)/kappa) for (u_sq_ki,ki) in zip(u_sq_k,k)]
    return calculate_GL_integral(f)

def calculate_GL_integral_with_transform_get_error(u_sq_k, u_sq_var_k, k_max, nr_particles, dimension, k_min=0.0):
    """
    Estimates the statistical error of above integral from statistical errors of the squared displacements.
    The error estimate of the integral is sqrt( sum( w_i**2 * var_i ) ), where w_i is the GL integration weight,
    and var_i is the variance of the integrand (see Daniel's thesis, p 72).
    """
    nr_points = len(u_sq_var_k)
    if nr_points != len(u_sq_k):
        raise Exception("calculate_GL_integral_with_transform_get_error: squared displacements and variances have different lengths")
    displ_k_min = u_sq_k[0]
    kappa = nr_particles*dimension/displ_k_min
    k = spring_constants_variable_transform(nr_points, k_max, displ_k_min, nr_particles, dimension, k_min)
    weights = Gauss_Lobatto_weights(Gauss_Lobatto_abscissas(nr_points)())()
    var_integrand = [u_sq_var_ki*0.5*(ki-k_min+kappa)*np.log(1.0+(k_max-k_min)/kappa) for (u_sq_var_ki,ki) in zip(u_sq_var_k,k)]
    sum_sq_weights_vars = sum( wi*wi*vari for (wi,vari) in zip(weights,var_integrand) )
    return np.sqrt(sum_sq_weights_vars)

if __name__ == "__main__":
    nr_points = 6
    k_max = 1042
    displ_k0 = 22
    nr_particles = 128
    dimension = 3
    k_min = 100
    k = spring_constants_variable_transform(nr_points, k_max, displ_k0, nr_particles, dimension, k_min)
    print k
    t = test_variable_transform(k, displ_k0, nr_particles, dimension)
    print t
    print Gauss_Lobatto_abscissas(nr_points)()
    