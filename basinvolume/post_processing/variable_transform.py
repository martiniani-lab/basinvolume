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
    