from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt
from exp_file_handler import ExpFileHandler
from exp_radii_distribution import ExpRadiiDistribution
from cross_validation_bandwidth_selection import get_pdf

class RadiiSampler(object):
    """
    Reads in experimental radii distribution and uses that to sample
    radii for the reference fluid.
    
    Procedure:
    1) Read radii from file.
    2) Learn radii distribution.
    3) Sample radii from that distribution.
    """
    def __init__(self, exp_data_set_index, exp_data_set_name_begin, data_dir, nr_particles, show_distribution=False):
        print("radii sampler")
        self.exp_data_set_index = exp_data_set_index
        self.exp_data_set_name_begin = exp_data_set_name_begin
        self.data_dir = data_dir
        self.nr_particles = nr_particles
        self.show_distribution = show_distribution
        #
        self.data_file_name = self.exp_data_set_name_begin + str(self.exp_data_set_index) + ".dat"
        self.data_dir = os.path.abspath(self.data_dir)
        self.data_file_path = os.path.join(self.data_dir, self.data_file_name)
        print("self.data_file_path")
        print(self.data_file_path)
        self.exp_data = ExpFileHandler(self.data_file_path)
        self.full_exp_radii = self.exp_data.radii
        self.exp_distribution = ExpRadiiDistribution(self.full_exp_radii)
        self.radii = self.exp_distribution.sample_radii(self.nr_particles)
        if self.show_distribution:
            self.show_radii_distribution()
    def show_radii_distribution(self):
        plt.hist(self.full_exp_radii, bins=14, normed=True, label="Experimental")
        plt.xlabel(r"Radius $r$")
        plt.ylabel(r"PDF($r$)")
        nr_points = 1000
        k_pdf_x = np.linspace(np.amin(self.full_exp_radii), np.amax(self.full_exp_radii), nr_points)
        k_pdf_y = get_pdf(self.full_exp_radii, k_pdf_x, bandwidth=self.exp_distribution.bandwidth)
        plt.plot(k_pdf_x, k_pdf_y, label="KDE")
        plt.plot(self.radii, np.zeros(len(self.radii)), "o", label="Sampled")
        plt.legend(loc=1)
        plt.show()
