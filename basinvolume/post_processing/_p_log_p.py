from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.special import gamma
    from basinvolume.utils import to_string, save_pdf, log_factorial
    from basinvolume.utils import ResultsFile, OutlierDetection
    from basinvolume.utils import MomentsAcc, CDFAccumulator
    from scipy import integrate
except ImportError as err:
    print err

class F0MeanError(object):
    """
    The mean is not weithed by the error because we want to have the mean as
    sampled with bias, see APFEntropy below.
    """
    def __init__(self, F0):
        F0 = np.array(F0)
        self.mean = np.mean(F0)
        self.sample_variance_error = np.sqrt(np.var(F0) / len(F0))

class APFEntropy(object):
    def __init__(self, F0, volume_sanity_check):
        self.F0 = F0
        self.F0_acc = volume_sanity_check.F0_acc
        self.V_acc = volume_sanity_check.V_acc
        self.nr_particles = volume_sanity_check.nr_particles
    def compute_and_write_entropy(self, entropy_file_path):
        """
        Granular entropy according to Asenjo14: 10.1103/PhysRevLett.112.098002
        S_\text{APF}^* = \langle F \rangle_\text{biased} + \log(V_\text{acc})
        S_\text{APF} = S_\text{APF}^* - \log(N!)
        """
        F0_stat = F0MeanError(self.F0)
        self.S_star = F0_stat.mean - self.F0_acc
        if self.S_star < 0:
            raise Exception("APFEntropy: compute_and_write_entropy: entropy computation failed")
        self.S = self.S_star - log_factorial(self.nr_particles)
        self.error_S_star = F0_stat.sample_variance_error
        self.error_S = self.error_S_star
        print "Granular entropy according to APF:"
        print "S_star:", self.S_star, "+/-", self.error_S_star 
        print "S:", self.S, "+/-", self.error_S
        self.write_to_file(entropy_file_path)
    def write_to_file(self, entropy_file_path):
        f = ResultsFile(entropy_file_path)
        f.set_heading("ENTROPY_APF")
        f.to_file("S_star", self.S_star)
        f.to_file("error_S_star", self.error_S_star)
        f.to_file("S", self.S)
        f.to_file("error_S", self.error_S)
        f.close()
