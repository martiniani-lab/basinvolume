from __future__ import division

import numpy as np
import os

from basinvolume.utils import trymakedir

from multi_gauss_wrap import MultiGaussWrap

class ComputerCommon(object):
    """
    Common functionality of volume computers for benchmark purposes.
    
    Basic structure for computing basin voulme as function of number of
    function calls and writing it to disk.
    """
    def __init__(self, results_path):
        self.results_path = results_path
        self.iterations = []
        self.evaluations = []
        self.volume = []
        
    def compute_volume(self, pot, large_small_flag):
        for i in xrange(self.max_iterations):
            self.iterations.append(i + 1)
            evaluations, volume = self.volume_iteration()
            self.evaluations.append(evaluations)
            self.volume.append(volume)
            
    def print_results(self, nr_gaussians, nr_dimensions, pot_index):
        """
        Print 3 arrays, iterations, evaluations, volume.
        
        For all methods, volume should agree for large number of
        iterations.
        Evaluations should be different in general.
        Then arrays can also be analysed to find for each method the
        smallest number of evaluations, such that volume is close to
        asymptotic volume within, say, 2%.
        """
        trymakedir(self.this_path(nr_gaussians, nr_dimensions, pot_index))
        np.savetxt(self.this_iterations_path(nr_gaussians, nr_dimensions, pot_index), self.iterations)
        np.savetxt(self.this_evaluations_path(nr_gaussians, nr_dimensions, pot_index), self.evaluations)
        np.savetxt(self.this_volume_path(nr_gaussians, nr_dimensions, pot_index), self.volume)
    
    def this_path(self, nr_gaussians, nr_dimensions, pot_index):
        return os.path.join(self.results_path, str(nr_gaussians), str(nr_dimensions), str(pot_index))
        
    def this_iterations_path(self, nr_gaussians, nr_dimensions, pot_index):
        return os.path.join(self.this_path(nr_gaussians, nr_dimensions, pot_index), self.get_method_label() + "_iterations.txt")
        
    def this_evaluations_path(self, nr_gaussians, nr_dimensions, pot_index):
        return os.path.join(self.this_path(nr_gaussians, nr_dimensions, pot_index), self.get_method_label() + "_evaluations.txt")
        
    def this_volume_path(self, nr_gaussians, nr_dimensions, pot_index):
        return os.path.join(self.this_path(nr_gaussians, nr_dimensions, pot_index), self.get_method_label() + "_volume.txt")


def run_computer(potential_dir, results_dir, large_or_small_flag, nr_gaussians, nr_dimensions, nr_samples, ComputerMethod):
    """
    Run ComputerMethod volume computation on gaussian landscapes.
    
    Computes volume by ComputerMethod as function of number of
    potential energy function calls.
    
    Parameters
    ----------
    
    potential_dir : string
        Directory from which the potentials shall be read.
        The file format is supposed to be as follows:
        potentials/nr_gaussians/nr_dimensions/pot_index/pot.txt
        potentials/nr_gaussians/nr_dimensions/pot_index/large_basin_index.txt
        potentials/nr_gaussians/nr_dimensions/pot_index/small_basin_index.txt
        
    results_dir : string
        Directory where the results shall be written to.
        The file format is supposed to be as follows:
        results_dir/nr_gaussians/nr_dimensions/pot_index/label(ComputerMethod)_iterations.txt
        results_dir/nr_gaussians/nr_dimensions/pot_index/label(ComputerMethod)_evaluations.txt
        results_dir/nr_gaussians/nr_dimensions/pot_index/label(ComputerMethod)_volume.txt
        
    large_or_small_flag : string
        Flag to indicate if the large or the small basin should be computed.
        Large means sampled uniformly at random (done before).
        Small means with fixed index (fixed to 0 for now).
        
    nr_gaussians : integer
        Number of gaussians (i.e., number of minima) in the potential
        energy surface.
        
    nr_dimensions : integer
        Euclidean dimension of the space mapping on the potential energy
        surface, i.e., number of degrees of freedom.
        
    nr_samples : integer
        Number of different potential energy landscapes sampled at each
        (nr_gaussians, nr_dimensions).
    """
    pot_wrapper = MultiGaussWrap(potential_dir)
    for pot_index in xrange(nr_samples):
        pot = pot_wrapper.get_pot(nr_gaussians, nr_dimensions, pot_index)
        computer = ComputerMethod(results_dir)
        computer.compute_volume(pot, large_or_small_flag)
        computer.print_results(nr_gaussians, nr_dimensions, pot_index)
