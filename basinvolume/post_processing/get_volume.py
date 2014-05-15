from __future__ import division
from math import pi, log
from variable_transform import calculate_GL_integral_with_transform
#import numpy as np
#import argparse
#import os
#import sys

class F_Basin_From_MC_Data_Free_COM(object):
    def __init__(self, dimension, nr_particles, k_values, displacements, prob):
        self.dimension = dimension
        self.nr_particles = nr_particles
        self.k_values = k_values
        self.displacements = displacements
        self.prob = prob
        self.k_max = self.k_values[-1]
        self.integral_over_displacements, self.f = calculate_GL_integral_with_transform(self.displacements, self.k_max, self.nr_particles, self.dimension)
    
    def get_free_energy_F0(self):
        """
        Computes the free energy F(0) = -log(v).
        Here there is no correction for the fixed c.o.m.
        """
        F0 = -log(self.prob) - (self.nr_particles*self.dimension/2.0)*log(2.0*pi/self.k_max) - 0.5*self.integral_over_displacements
        return F0, self.f

class F_Basin_From_MC_Data(object):
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
    def __init__(self, dimension, nr_particles, k_values, displacements, box_volume, prob):
        self.dimension = dimension
        self.nr_particles = nr_particles
        self.k_values = k_values
        self.displacements = displacements
        self.box_volume = box_volume
        self.prob = prob
        self.k_max = self.k_values[-1]
        if self.k_max !=  max(self.k_values):
            raise Exception("F_Basin_From_MC_Data: label mismatch")
        self.nr_points = len(self.k_values)
        if self.nr_points != len(self.displacements):
            raise Exception("F_Basin_From_MC_Data: illegal input")
        self._calculate_integral()
        
    def _calculate_integral(self):
        """
        Calculates the integral over the squared displacements from 0 to k_max
        """
        self.integral_over_displacements, self.f = calculate_GL_integral_with_transform(self.displacements, self.k_max, self.nr_particles, self.dimension)
        
    def get_free_energy_F0(self):
        """
        Computes the free energy F(0) = -log(v).
        Reference: Daniel A. Asenjo-Andrews, PhD thesis, p 115
        (First term: We do not have box_volume==1)
        """
        F0 = -log(self.box_volume) - log(self.prob) - (self.nr_particles*self.dimension/2.0)*log(2.0*pi/self.k_max) \
        + (self.dimension/2.0)*log(2.0*pi/(self.nr_particles*self.k_max)) - 0.5*self.integral_over_displacements
        
        return F0, self.f
    
class F_Basin_Th_Integration(object):
    """
    Read in MC data for spring constants, displacements, probability to be in basin;
    plus parameters: nr particles, Euclidean dimension
    """
    #TODO: Read in parameters and output from MC simulation
    #TODO: Compute F(0) and print output in reasonable way
    ###################################
