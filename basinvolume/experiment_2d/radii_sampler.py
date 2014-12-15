from __future__ import division
import numpy as np
from exp_file_handler import ExpFileHandler

class RadiiSampler(object):
    """
    Reads in experimental radii distribution and uses that to sample
    radii for the reference fluid.
    """
    def __init__(self, exp_data_set_index, exp_data_set_name_begin, data_dir, nr_particles):
        self.exp_data_set_index = exp_data_set_index
        self.exp_data_set_name_begin = exp_data_set_name_begin
        self.data_dir = data_dir
        self.nr_particles = nr_particles
        #
        self.data_file_name = self.exp_data_set_name_begin + str(self.data_set_index) + ".dat"
        self.data_dir = os.path.abspath(self.data_dir)
        self.data_file_path = os.path.join(self.data_dir, self.data_file_name)
        self.exp_data = ExpFileHandler(self.data_file_path)
        self.full_exp_radii = self.exp_data.radii
        self.exp_distribution = ExpRadiiDistribution(self.full_exp_radii)
        self.exp_distribution.learn_distribution()
        self.radii = self.exp_distribution.sample_radii(self.nr_particles)
        # 1 read radii etc
        # 2 do kernel density estimate of radii distribution, optimize bandwidth
        # 3 sample radii from that distribution
