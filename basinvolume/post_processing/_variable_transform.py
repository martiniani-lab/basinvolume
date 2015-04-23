from __future__ import division
from basinvolume.post_processing import Gauss_Lobatto_abscissas, Gauss_Lobatto_weights, calculate_GL_integral
import numpy as np
try:
    from scipy.integrate import simps
except:
    print("import error")

def spring_constants_variable_transform(nr_points, k_max, displ_k_min, nr_particles, dimension, k_min=0.0, kappa_const=1.0):
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
    kappa = nr_particles*dimension/displ_k_min*kappa_const
    k = [k_min - kappa + kappa*(1.0+(k_max-k_min)/kappa)**((1.0+ti)/2.0) for ti in t]
    return k

def test_variable_transform(k, displ_k_min, nr_particles, dimension, kappa_const=1.0):
    """
    Computes the GL abscissas from the given kvalues.
    Consistency check only.
    """
    kappa = dimension*nr_particles/displ_k_min*kappa_const
    kmax = max(k)
    kmin = min(k)
    t = [2*np.log(1+(ki-kmin)/kappa)/np.log(1+(kmax-kmin)/kappa)-1 for ki in k]
    return t

def calculate_GL_integral_with_transform(u_sq_k, k_max, nr_particles, dimension, k_min=0.0, kappa_const=1.0, displ_k_min_trafo=None):
    """
    Input: Squared displacements, measured at the spring constant values given by
    spring_constants_variable_transform(nr_points, k_max, displ_k0, nr_particles, dimension).
    Applies the variable transform to the integrand (multiplies by Jacobian).
    Output: Integral over squared displacements from zero to k_max (maximum spring constant).
    """
    nr_points = len(u_sq_k)
    displ_k_min = 0
    if displ_k_min_trafo is None:
        displ_k_min = u_sq_k[0]
    else:
        displ_k_min = displ_k_min_trafo
    kappa = nr_particles*dimension/displ_k_min*kappa_const
    k = spring_constants_variable_transform(nr_points, k_max, displ_k_min, nr_particles, dimension, k_min=k_min, kappa_const=kappa_const)
    f = np.array([u_sq_ki*0.5*(ki-k_min+kappa)*np.log(1.0+(k_max-k_min)/kappa) for (u_sq_ki,ki) in zip(u_sq_k,k)])
    return calculate_GL_integral(f), f

def calculate_simple_integral(u_sq_k, k_max, nr_particles, dimension, k_min=0.0, kappa_const=1.0, displ_k_min_trafo=None):
    nr_points = len(u_sq_k)
    displ_k_min = 0
    if displ_k_min_trafo is None:
        displ_k_min = u_sq_k[0]
    else:
        displ_k_min = displ_k_min_trafo
    k = spring_constants_variable_transform(nr_points, k_max, displ_k_min, nr_particles, dimension, k_min=k_min, kappa_const=kappa_const)
    return simps(u_sq_k, x=k), u_sq_k


def calculate_GL_integral_with_transform_get_error(u_sq_k, u_sq_var_k, k_max, nr_particles, dimension, k_min=0.0, kappa_const=1.0, displ_k_min_trafo=None):
    """
    Estimates the statistical error of above integral from statistical errors of the squared displacements.
    The error estimate of the integral is sqrt( sum( w_i**2 * var_i ) ), where w_i is the GL integration weight,
    and var_i is the variance of the integrand (see Daniel's thesis, p 72).
    Note: By providing displ_k_min_trafo, one can use a displacement at k_0 (estimated before PT runs) for the variable transform, different from the PT k=0 result. 
    """
    nr_points = len(u_sq_var_k)
    if nr_points != len(u_sq_k):
        raise Exception("calculate_GL_integral_with_transform_get_error: squared displacements and variances have different lengths")
    displ_k_min = 0
    if displ_k_min_trafo is None:
        displ_k_min = u_sq_k[0]
    else:
        displ_k_min = displ_k_min_trafo
    kappa = nr_particles*dimension/displ_k_min*kappa_const
    k = spring_constants_variable_transform(nr_points, k_max, displ_k_min, nr_particles, dimension, k_min=k_min, kappa_const=kappa_const)
    weights = Gauss_Lobatto_weights(Gauss_Lobatto_abscissas(nr_points)())()
    var_integrand = np.array([u_sq_var_ki*0.5*(ki-k_min+kappa)*np.log(1.0+(k_max-k_min)/kappa) for (u_sq_var_ki,ki) in zip(u_sq_var_k,k)])
    sum_sq_weights_vars = sum( wi*wi*vari for (wi,vari) in zip(weights,var_integrand) )
    return np.sqrt(sum_sq_weights_vars), np.sqrt(var_integrand)
    
def calculate_simple_integral_get_error(u_sq_k, u_sq_var_k, k_max, nr_particles, dimension, k_min=0.0, kappa_const=1.0, displ_k_min_trafo=None):
    nr_points = len(u_sq_var_k)
    if nr_points != len(u_sq_k):
        raise Exception("calculate_GL_integral_with_transform_get_error: squared displacements and variances have different lengths")
    displ_k_min = 0
    if displ_k_min_trafo is None:
        displ_k_min = u_sq_k[0]
    else:
        displ_k_min = displ_k_min_trafo
    kappa = nr_particles*dimension/displ_k_min*kappa_const
    k = spring_constants_variable_transform(nr_points, k_max, displ_k_min, nr_particles, dimension, k_min=k_min, kappa_const=kappa_const)
    weights = np.ones(nr_points)
    var_integrand = u_sq_var_k
    sum_sq_weights_vars = sum(wi * wi * vari for (wi, vari) in zip(weights, var_integrand))
    return np.sqrt(sum_sq_weights_vars), np.sqrt(var_integrand)

if __name__ == "__main__":
    nr_points = 6
    k_max = 300000
    displ_k0 = 22
    nr_particles = 64
    dimension = 2
    k_min = 0
    kappa_const = 1
    k = spring_constants_variable_transform(nr_points, k_max, displ_k0, nr_particles, dimension, k_min, kappa_const)
    print k
    delta_k = [k[i + 1] - k[i] for i in xrange(len(k) - 1)]
    print delta_k
    t = test_variable_transform(k, displ_k0, nr_particles, dimension, kappa_const)
    print t
    print Gauss_Lobatto_abscissas(nr_points)()
    import matplotlib.pyplot as plt
    plt.plot(k, k, "o")
    plt.show()
    
