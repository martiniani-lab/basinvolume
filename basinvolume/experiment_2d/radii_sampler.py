from __future__ import division
import numpy as np

class RadiiSampler(object):
    """
    Reads in experimental radii distribution and uses that to sample
    radii for the reference fluid.
    """
    def __init__(self):
        # 1 read radii etc
        # 2 do kernel density estimate of radii distribution, optimize bandwidth
        # 3 sample radii from that distribution
