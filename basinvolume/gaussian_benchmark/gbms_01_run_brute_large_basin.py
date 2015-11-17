from __future__ import division

import numpy as np

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
        Number of gaussians (i.e., number of minima) in the potential energy surface.
        
    nr_dimensions : integer
        Euclidean dimension of the space mapping on the potential energy surface, i.e., number of degrees of freedom.
        
    nr_samples : integer
        Number of different potential energy landscapes sampled at each (nr_gaussians, nr_dimensions)
    """
    

if __name__ == "__main__":
    nr_samples = 20
    potential_dir = os.path.join(os.getcwd(), "potentials")
    large_basin_results_dir = os.path.join(os.getcwd(), "large_basin_results")
    for nr_gaussians in [5]:
        for nr_dimensions in [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]:
            run_brute(potential_dir, large_basin_results_dir, "large", nr_gaussians, nr_dimensions, nr_samples)
