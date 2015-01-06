from __future__ import division
import numpy as np

class EqFluidSnapshots(object):
    """
    Runs the equilibrium fluid corresponding to a given sample of radii
    and initial conditions.
    This should also print snapshots at equal and specified intervals.
    """
    def __init__(self, radii, coordinates, boxvec):
        self.radii = radii
        self.coordinates = coordinates
        self.boxvec = boxvec
        #
        
        
