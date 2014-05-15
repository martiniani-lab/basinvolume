from __future__ import division
import numpy as np
from scipy.integrate import quad, romb
from math import log, pi
from gauss_lobatto import calculate_GL_integral_range
from variable_transform import spring_constants_variable_transform, calculate_GL_integral_with_transform
from get_volume import F_Basin_From_MC_Data_Free_COM, F_Basin_From_MC_Data

def F_HO(N,d,k):
    return -0.5*N*d*log(2*pi/k)

if __name__ == "__main__":
    """
    Test of integration.
    """
    nr_particles = 10
    dimension = 3
    k1 = 22
    k2 = 44
    ref_F1 = F_HO(nr_particles,dimension,k1)
    ref_F2 = F_HO(nr_particles,dimension,k2)
    int_F1 = []
    int_F1.append( ref_F2 - 0.5*nr_particles*dimension*log(k2/k1) )
    int_F1.append( ref_F2 - 0.5*quad(lambda x: nr_particles*dimension/x, k1, k2)[0] )
    #computation by GL without variable transform, order 4
    int_F1.append( ref_F2 - 0.5*calculate_GL_integral_range(lambda x: nr_particles*dimension/x, k1, k2, 4) )
    #computation by GL with variable transform, order 4
    k_order4 = spring_constants_variable_transform(4, k2, nr_particles*dimension/k1, nr_particles, dimension, k1)
    int_F1.append( ref_F2 - 0.5*calculate_GL_integral_with_transform([nr_particles*dimension/ki for ki in k_order4], k2, nr_particles, dimension, k1) )
    print 'reference:'
    print ref_F1
    print 'integrated:'
    for res in int_F1:
        print res  
    """
    Test of volume computation.
    """  
    L = 1000
    nr_points = 6
    k_max = 100
    prob = 1
    displ_k0 = (L/2)**2
    nr_particles = 1
    dimension = 1
    k = spring_constants_variable_transform(nr_points, k_max, displ_k0, nr_particles, dimension)
    usq = np.zeros(nr_points)
    usq[0] = displ_k0
    for i in xrange(1,nr_points):
        usq[i] = 1.0/k[i]
    vol = F_Basin_From_MC_Data_Free_COM(dimension, nr_particles, k, usq, prob)
    F0 = vol.get_free_energy_F0()
    F0_, F0_error = vol.get_free_energy_F0(np.ones(nr_points))
    print "F0, free: ", F0
    print "F0_, free: ", F0_
    print "error F0_: ", F0_error
    
    