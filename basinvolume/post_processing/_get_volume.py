from __future__ import division
from math import pi, log
from basinvolume.post_processing import calculate_GL_integral_with_transform, calculate_GL_integral_with_transform_get_error
import numpy as np
#import argparse
#import os
#import sys

class Base_Compute_Integral(object):
    def __init__(self, dimension, nr_particles, k_values, displacements, kappa_const=1.0, displ_k_min_trafo=None):
        """
        Compute the integral needed for the th. integration.
        This only computes in the integral over the squared displacements and does not apply corrections nor does it know about the Einstein crystal part.
        dimension: Euclidean dimension of box
        nr_particles: number of particles
        k_values: array of spring constants, in inceasing order
        displacements: array of displacements corresponding to the array of spring constants
        kappa_const: A constant that can be set to change the variable transform in the integral
        displ_k_min_trafo: if provided, this means that the variable transform is computed from a displacement at k=0 different from
                            the one used to actually compute the integral
        """
        self.dimension = dimension
        self.nr_particles = nr_particles
        self.k_values = k_values
        self.displacements = displacements
        self.kappa_const = kappa_const
        self.displ_k_min_trafo = displ_k_min_trafo
        self.k_max = self.k_values[-1]
        if self.k_max !=  max(self.k_values):
            raise Exception("Base_Compute_Integral: label mismatch")
        self.nr_points = len(self.k_values)
        if self.nr_points != len(self.displacements):
            raise Exception("Base_Compute_Integral: illegal input")
        self.integral_over_displacements, self.f = calculate_GL_integral_with_transform(self.displacements, self.k_max, 
                                                                                        self.nr_particles, self.dimension, k_min=self.k_values[0], kappa_const=self.kappa_const, displ_k_min_trafo=self.displ_k_min_trafo)
    def _calculate_error_F0(self, displacements_variance):
        """
        calculate_GL_integral_with_transform_get_error(u_sq_k, u_sq_var_k, k_max, nr_particles, dimension, k_min=0.0, kappa_const=1.0)
        """
        sigF0, sigIntegrand = calculate_GL_integral_with_transform_get_error(self.displacements, displacements_variance, self.k_max, 
                                                                  self.nr_particles, self.dimension, k_min=self.k_values[0], kappa_const=self.kappa_const, displ_k_min_trafo=self.displ_k_min_trafo) 
        return 0.5*sigF0, sigIntegrand
    
class F_Basin_From_MC_Data_Free_COM(Base_Compute_Integral):
    """
    Computes the free energy F(0) = -log(v).
    Here there is no correction for the fixed c.o.m.
    """
    def __init__(self, dimension, nr_particles, k_values, displacements, prob, kappa_const=1.0, displ_k_min_trafo=None):
        super(F_Basin_From_MC_Data_Free_COM,self).__init__(dimension, nr_particles, k_values, displacements, kappa_const=kappa_const, displ_k_min_trafo=displ_k_min_trafo)
        self.prob = prob
        
    def get_free_energy_F0(self, displacements_variance):
        F0 = -log(self.prob) - (self.nr_particles*self.dimension/2.0)*log(2.0*pi/self.k_max) - 0.5*self.integral_over_displacements
        sigF0, sigf = self._calculate_error_F0(displacements_variance)
        
        return F0, sigF0, self.f, sigf

class F_Basin_From_MC_Data(Base_Compute_Integral):
    """
    Calculate the basin volume from the MC output via
    thermodynamic integration.
    *nr_particles: number of particles in packing
    *k_values: spring constants as used for PT and k_max
    *displacements: mean squared-displacements as obtained from MC, for the corresponding k_values
    *k_max: maximum spring constant used in MC
    *probability to stay in basin with given k_max
    *Euclidean dimension of system
    Reference: Xu et al., PRL 106, 245502 (2011)
    Daniel A. Asenjo-Andrews, PhD thesis
    """
    def __init__(self, dimension, nr_particles, k_values, displacements, box_volume, prob, kappa_const=1.0, displ_k_min_trafo=None):
        super(F_Basin_From_MC_Data,self).__init__(dimension, nr_particles, k_values, displacements, kappa_const=kappa_const, displ_k_min_trafo=displ_k_min_trafo)
        self.box_volume = box_volume
        self.prob = prob
         
    def get_free_energy_F0(self, displacements_variance):
        """
        Computes the free energy F(0) = -log(v).
        Reference: Daniel A. Asenjo-Andrews, PhD thesis, p 115
        (First term: We do not have box_volume==1)
        we added a +log(self.nr_particles) term
        """
        ##############################INCL KINETIC TERM#########################################
        #This way of correcting for fixed c.o.m. includes a term from the kinetic part of the Einstein crystal partition function.
        #F0 = -np.log(self.box_volume) - np.log(self.prob) - (self.nr_particles*self.dimension/2.0)*np.log(2.0*pi/self.k_max) + \
        #(self.dimension/2.0)*np.log(2.0*pi/(self.nr_particles*self.k_max)) - 0.5*self.integral_over_displacements
        
        ##############################NO KINETIC TERM#########################################
        F0 = -0.5 * self.integral_over_displacements - np.log(self.box_volume) - ((self.nr_particles - 1.0) * self.dimension / 2.0) * np.log(2.0 * pi / self.k_max) - np.log(self.prob) 
        
        sigF0, sigf = self._calculate_error_F0(displacements_variance)
        
        return F0, sigF0, self.f, sigf
    
def F_Basin_From_MC_Data__get_free_energy_F0_approx_kmax_displ0(displ2_k0, kmax, box_volume, nr_particles, dimension, prob_kmax):
    """
    Compute free energy F(0) = -log(v), by assuming that the approximation
    used to make the integrand flat is the true integrand behavior.
    """
    approx_integral_over_displacements = None
    xi = (nr_particles - 1) * dimension / displ2_k0
    def _approx(k):
        return (nr_particles - 1) * dimension / (k + xi)
    from scipy.integrate import quad
    approx_integral_over_displacements, err = quad(_approx, 0, kmax)
    return -0.5 * approx_integral_over_displacements - np.log(box_volume) - ((nr_particles - 1.0) * dimension / 2.0) * np.log(2.0 * pi / kmax) - np.log(prob_kmax)
    
if __name__ == "__main__":
    """
    Read in MC data for spring constants, displacements, probability to be in basin;
    plus parameters: nr particles, Euclidean dimension, Vbox
    """
    #TODO: Read in parameters and output from MC simulation
    #TODO: Compute F(0) and print output in reasonable way
    #F_Basin_From_MC_Data...
    ###################################  
