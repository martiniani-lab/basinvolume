from __future__ import division
from scipy.integrate import quad
from math import log, pi, sqrt
#import scipy.stats.histogram as hist
#TODO: For a given set of displacement2 histograms (or, better: means and variances as function of k), read in, compute volume, generate 1 histogram of volumes
#parameters

class Poly_HS_Fluid(object):
    """
    Returns -ln(V_acc), for the accessible parent system (fluid) volume.
    See: http://dx.doi.org/10.1103/PhysRevLett.112.098002
    """
    def get_fex(self, phiHD):
        return quad(lambda phi: (self.Z(phi)-1.0)/phi, 0.0, phiHD)[0]
    def get_F_acc(self, phiHD, V_box, nr_particles):
        return -nr_particles*log(V_box)+nr_particles*self.get_fex(phiHD)

class Poly_HS_Fluid_2d(Poly_HS_Fluid):
    """
    Uses equation of state for 2d fluid.
    See eqs 6 and 8 of: http://dx.doi.org/10.1080/00268979909482932
    m1 and m2 are the 1st and 2nd moment of the diameter size distribution.
    """
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
        self.phiCP = pi/sqrt(12) #max phi in 2d
    def Z(self, phi):
        return self.ZOne(phi)*(self.m1**2/self.m2) + (1.0-self.m1**2.0/self.m2)/(1.0-phi)
    def ZOne(self, phi):
        return 1.0 / ( 1.0 - 2.0*phi + (2.0*self.phiCP-1.0)*phi**2/self.phiCP**2 )

class Poly_HS_Fluid_3d(Poly_HS_Fluid):
    """
    Uses equation of state for 3d fluid.
    See eqs 12, 13, 14 of: http://dx.doi.org/10.1080/00268979909482932
    The equation of state for the HS fluid in ZOne is by Carnahan and Starling, ref 1 of paper cited above.
    m1, m2, m3 are the 1st, 2nd, 3rd moments of the diameter size distribution. 
    """
    def __init__(self, m1, m2, m3):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
    def Z(self, phi):
        return 1.0 + (self.ZOne(phi)-1.0)*self.m2/(2.0*self.m3**2)*(self.m2**2+self.m1*self.m3) + phi/(1.0-phi)*( 1.0-self.m2/self.m3**2*(2.0*self.m2**2-self.m1*self.m3) )
    def ZOne(self, phi):
        return (1.0+phi+phi**2-phi**3)/(1.0-phi)**3

def F_acc_Gaussian_Poly_HS_Fluid(phiHD, V_box, nr_particles, box_dimension, diameter_mean, diameter_variance):
    """
    From the mean and variance of the Gaussian diameter distribution (fluid particles), computes the moments and eventually -ln(V_acc).
    http://mathworld.wolfram.com/NormalDistribution.html
    """
    m1 = diameter_mean
    m2 = diameter_mean**2 + diameter_variance
    m3 = diameter_mean*(diameter_mean**2+3.0*diameter_variance)
    if box_dimension == 2:
        return Poly_HS_Fluid_2d(m1,m2).get_F_acc(phiHD, V_box, nr_particles)
    elif box_dimension == 3:
        return Poly_HS_Fluid_3d(m1,m2,m3).get_F_acc(phiHD, V_box, nr_particles)
    else:
        raise Exception("F_acc_Gaussian_Poly_HS_Fluid: illegal box_dimension")
    
"""
class Free_Energy_Histogram(object):
    self.F0 = [1,1,1,1]
    hist(self.F0)
"""

if __name__ == "__main__":
    d2 = Poly_HS_Fluid_2d(2,3)
    print d2.get_fex(1.0/2.0)
    d3 = Poly_HS_Fluid_3d(1,2,3)
    print d3.get_fex(1.0/2.0)
    print "F_acc_Gaussian_Poly_HS_Fluid:"
    phiHD = 0.2
    L_box = 5 
    nr_particles = 10
    diameter_mean = 1
    diameter_variance = 0.1
    print F_acc_Gaussian_Poly_HS_Fluid(phiHD, L_box**2, nr_particles, 2, diameter_mean, diameter_variance)
    print F_acc_Gaussian_Poly_HS_Fluid(phiHD, L_box**3, nr_particles, 3, diameter_mean, diameter_variance)
    
    