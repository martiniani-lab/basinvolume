from __future__ import division
from __future__ import print_function
from builtins import object
from scipy.integrate import quad
from math import log, pi, sqrt

class Poly_HS_Fluid(object):
    """
    Returns -ln(V_acc), for the accessible parent system (fluid) volume.
    See: http://dx.doi.org/10.1103/PhysRevLett.112.098002
    """
    def get_fex(self, phiHD):
        return quad(lambda phi: (self.Z(phi) - 1.0) / phi, 0.0, phiHD)[0]
    def get_F_acc(self, phiHD, V_box, nr_particles):
        return -nr_particles * log(V_box) + nr_particles * self.get_fex(phiHD)

class Poly_HS_Fluid_2d(Poly_HS_Fluid):
    """
    Uses equation of state for 2d fluid.
    See eqs 6 and 8 of: http://dx.doi.org/10.1080/00268979909482932
    m1 and m2 are the 1st and 2nd moment of the diameter size distribution.
    """
    def __init__(self, m1, m2, name="Santos"):
        self.m1 = m1
        self.m2 = m2
        self.name = name
        self.phiCP = pi / sqrt(12) #max phi in 2d
    def Z(self, phi):
        return self.ZOne(phi) * (self.m1**2 / self.m2) + (1.0 - self.m1**2.0 / self.m2) / (1.0 - phi)
    def ZOne(self, phi):
        if self.name == "Santos": #ref: http://dx.doi.org/10.1080/00268979909482932
            return 1.0 / ( 1.0 - 2.0 * phi + (2.0 * self.phiCP - 1.0) * phi**2 / self.phiCP**2)
        elif self.name == "Kolafa": #ref: http://dx.doi.org/10.1080/00268970600967963
            rho_max = 0.9
            phi_max = rho_max * 0.25 * pi
            if phi > phi_max or phi < 0:
                raise Exception("illegal volume fraction, phi = {} > phi_max = {} or phi < 0".format(phi, phi_max))
            x = phi / (1 - phi)
            return 1 + 2 * x + 1.12801775 * x**2 + 0.00181895291 * x**3 - 0.0526134737 * x**4 \
                   + 0.0504960168 * x**5 - 0.0325537792 * x**6 + 0.0134578632 * x**7 + 0.00140888182 * x**8 \
                   - 0.00834273601 * x**9 + 0.00694127367 * x**10 - 0.00262254723 * x**11 + 0.000355746352 * x**12 \
                   - 5.24672938e-9 * x**22 + 5.88054639e-23 * x**57
        else:
            raise Exception("illegal eos name: " + self.name)

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
        return 1.0 + (self.ZOne(phi) - 1.0) * self.m2 / (2.0 * self.m3**2) * (self.m2**2 + self.m1 * self.m3) + phi / (1.0 - phi) * (1.0 - self.m2 / self.m3** 2 * (2.0 * self.m2**2 - self.m1 * self.m3))
    def ZOne(self, phi):
        """
        Use different monnodisperse eos for phi >=, < 0.5: see e.g. first year report, page 41 and
        ref: http://dx.doi.org/10.1080/00268979909482932
        """
        if phi >= 0.5:
            return (1.0 + phi + phi**2 - phi**3) / (1.0 - phi)**3
        else:
            return (1 + phi + phi**2 - 2 * phi**3 * (1 + phi) / 3) / (1 - phi)**3

def F_acc_Gaussian_Poly_HS_Fluid(phiHD, V_box, nr_particles, box_dimension, diameter_mean, diameter_variance):
    """
    From the mean and variance of the Gaussian diameter distribution (fluid particles), computes the moments and eventually -ln(V_acc).
    http://mathworld.wolfram.com/NormalDistribution.html
    """
    m1 = diameter_mean
    m2 = diameter_mean**2 + diameter_variance
    m3 = diameter_mean * (diameter_mean**2 + 3.0 * diameter_variance)
    print("nr_particles",nr_particles,"m1",diameter_mean,"m2",m2,"m3",m3)
    if box_dimension == 2:
        return Poly_HS_Fluid_2d(m1,m2).get_F_acc(phiHD, V_box, nr_particles)
    elif box_dimension == 3:
        return Poly_HS_Fluid_3d(m1,m2,m3).get_F_acc(phiHD, V_box, nr_particles)
    else:
        raise Exception("F_acc_Gaussian_Poly_HS_Fluid: illegal box_dimension")

def test_HS_fluids():
    d2 = Poly_HS_Fluid_2d(2, 3)
    print(d2.get_fex(1.0 / 2.0))
    d3 = Poly_HS_Fluid_3d(1, 2, 3)
    print(d3.get_fex(1.0 / 2.0))
    print("F_acc_Gaussian_Poly_HS_Fluid:")
    phiHD = 0.2
    L_box = 5 
    nr_particles = 10
    diameter_mean = 1
    diameter_variance = 0.1
    print(F_acc_Gaussian_Poly_HS_Fluid(phiHD, L_box**2, nr_particles, 2, diameter_mean, diameter_variance))
    print(F_acc_Gaussian_Poly_HS_Fluid(phiHD, L_box**3, nr_particles, 3, diameter_mean, diameter_variance))
    
if __name__ == "__main__":
    test_HS_fluids()
    
