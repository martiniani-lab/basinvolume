from __future__ import division
from variable_transform import *
from scipy.integrate import quad, fixed_quad

def integrand(x):
    #return (x+10)**-2
    #return (3*128)/(x+10)
    #return x+42
    #this is N*d/(k+kappa), the expected form of the integrand
    #(for best performance of the variable transform)
    return (3*128)/(x+3*128/22)

def test_integration_with_transform(nr_points, k_max, displ_k0, nr_particles, dimension):
    """
    Computes integral with variable transform and directly for consistency check.
    """
    k = spring_constants_variable_transform(nr_points, k_max, displ_k0, nr_particles, dimension)
    samples = [integrand(x) for x in k]
    print "integral by variable transform:"
    print calculate_GL_integral_with_transform(samples, k_max, nr_particles, dimension)
    print "integral by quad:"
    print quad(integrand, k[0], k[-1])
    print "integral by fixed_quad:"
    print fixed_quad(integrand, k[0], k[-1], n=nr_points)
    print "integral by plain GL integration:"
    print calculate_GL_integral_range(integrand, k[0], k[-1], nr_points)
    
if __name__ == "__main__":
    nr_points = 6
    k_max = 100
    displ_k0 = 22
    nr_particles = 128
    dimension = 3
    test_integration_with_transform(nr_points, k_max, displ_k0, nr_particles, dimension)