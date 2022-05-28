from __future__ import division
from cross_validation_bandwidth_selection import get_bandwidth_estimate
from cross_validation_bandwidth_selection import sample_from_pdf

class ExpRadiiDistribution(object):
    """
    Takes the experimental radii and leanrs their distribution.
    Then it samples from this distribution the requested number of new
    radii for the equilibrium reference packing.
    """
    def __init__(self, input_radii):
        self.input_radii = input_radii
    def sample_radii(self, nr_particles, random_state=None):
        self.bandwidth = get_bandwidth_estimate(self.input_radii[::20])
        print("bandwidth:", self.bandwidth)
        return sample_from_pdf(self.input_radii, nr_particles, self.bandwidth, random_state=random_state)
