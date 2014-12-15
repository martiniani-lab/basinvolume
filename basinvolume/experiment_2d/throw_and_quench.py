from __future__ import division
import numpy as np

class ThrowAndQuench(object):
    """
    Throws particles of defined HS radii into a box uniformly at random
    and quenches to find a legal initial configuration.
    """
    def __init__(self):
        # 4 throw particles with these radii into a box uniformly and minimize with LBFGS
