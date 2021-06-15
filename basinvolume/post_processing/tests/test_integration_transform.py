from __future__ import division
from __future__ import print_function
from basinvolume.post_processing import *
from scipy.integrate import quad, fixed_quad
import numpy as np

# def integrand(x):
#     #return (x+10)**-2
#     #return (3*128)/(x+10)
#     #return x+42
#     #this is N*d/(k+kappa), the expected form of the integrand
#     #(for best performance of the variable transform)
#     return (3*128)/(x+3*128/22)


def test_integration_with_transform(nr_points, k_max, displ_k0, nr_particles, dimension, kappa_const, displ_k_min_trafo):
    """
    Computes integral with variable transform and directly for consistency check.
    """
    k_old = spring_constants_variable_transform(nr_points, k_max, displ_k0, nr_particles, dimension, kappa_const=kappa_const)
    k = spring_constants_variable_transform(nr_points, k_max, displ_k_min_trafo, nr_particles, dimension, kappa_const=kappa_const)
    print("k_old: ")
    print(k_old)
    print("k: ")
    print(k)
    samples = [integrand(x) for x in k]
    print("integral by variable transform (reference):")
    print(calculate_GL_integral_with_transform(samples, k_max, nr_particles, dimension, kappa_const=kappa_const, displ_k_min_trafo=displ_k_min_trafo))
    print("error on integral by variable transform with parameter:")
    print(calculate_GL_integral_with_transform_get_error(samples, samples, k_max, nr_particles, dimension, k_min=0.0, kappa_const=kappa_const, displ_k_min_trafo=samples[0]))
    print("integral by quad:")
    print(quad(integrand, k[0], k[-1]))
    print("integral by fixed_quad:")
    print(fixed_quad(integrand, k[0], k[-1], n=nr_points))
    print("integral by plain GL integration:")
    print(calculate_GL_integral_range(integrand, k[0], k[-1], nr_points))

    
# if __name__ == "__main__":
#     nr_points = 6
#     k_max = 100
#     displ_k0 = 22
#     nr_particles = 128
#     dimension = 3
#     kappa_const=1
#     displ_k_min_trafo = 22
#     test_integration_with_transform(nr_points, k_max, displ_k0, nr_particles, dimension, kappa_const, displ_k_min_trafo)