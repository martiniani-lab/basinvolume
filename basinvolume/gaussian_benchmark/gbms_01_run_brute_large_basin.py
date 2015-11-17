from __future__ import division

import numpy as np

from computer_common import ComputerCommon

class BruteComputer(ComputerCommon):
    """
    Compute volume of a basin in the gaussian landscape by brute force,
    as function of the number of function calls, and print it to the
    disk.
    """
    def __init__(self, results_dir):
        super(BruteComputer, self).__init__(results_dir)
        self.iterations = []
        self.evaluations = []
        self.volume = []
        
    def compute_volume(self, pot, large_small_flag):
        for i in xrange(self.max_iterations):
            self.iterations.append(i + 1)
            evaluations_, volume_ = self.volume_iteration()
            self.evaluations.append(evaluations_)
            self.volume.append(volume_)
            
    def print_results(self):
        """
        Print 3 arrays, iterations, evaluations, volume.
        
        For all methods, volume should agree for large number of
        iterations.
        Evaluations should be different in general.
        Then arrays can also be analysed to find for each method the
        smallest number of evaluations, such that volume is close to
        asymptotic volume within, say, 2%.
        """

def run_brute(potential_dir, results_dir, large_or_small_flag, nr_gaussians, nr_dimensions, nr_samples):
    """
    Run brute force volume computation on gaussian landscapes.
    
    Computes volume by brute force rejection as function of number of
    potential energy function calls.
    
    Parameters
    ----------
    
    potential_dir : string
        Directory where the potentials shall be written to.
        The file format is supposed to be as follows:
        potentials/nr_gaussians/nr_dimensions/pot_index/pot.txt
        potentials/nr_gaussians/nr_dimensions/pot_index/large_basin_index.txt
        potentials/nr_gaussians/nr_dimensions/pot_index/small_basin_index.txt
        
    results_dir : string
        Directory where the results shall be written to.
        The file format is supposed to be as follows:
        results_dir/nr_gaussians/nr_dimensions/pot_index/brute.txt
        
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
        computer = BruteComputer(results_dir)
        computer.compute_volume(pot, large_or_small_flag)

if __name__ == "__main__":
    nr_samples = 20
    potential_dir = os.path.join(os.getcwd(), "potentials")
    large_basin_results_dir = os.path.join(os.getcwd(), "large_basin_results")
    for nr_gaussians in [5]:
        for nr_dimensions in [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]:
            run_brute(potential_dir, large_basin_results_dir, "large", nr_gaussians, nr_dimensions, nr_samples)
