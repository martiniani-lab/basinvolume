from __future__ import division

import os

from multi_gauss_wrap import MultiGaussWrap

def generate_potentials(potential_dir, nr_gaussians, nr_dimensions, nr_samples):
    """
    Generate multi-gaussian potentials for the volume computation benchmark.
    
    Parameters of potential are written to file such that they are readable with current stitcher-reader.
    Selecting benchmark basins stores the index of a basin sampled uniformly at random ("large basin") and of a basin of fixed index (e.g., zero), "small basin".
    
    Parameters
    ----------
    
    potential_dir : string
        Directory where the potentials shall be written to.
        The file format is supposed to be as follows:
        potentials/nr_gaussians/nr_dimensions/pot_index/pot.txt
        potentials/nr_gaussians/nr_dimensions/pot_index/large_basin_index.txt
        potentials/nr_gaussians/nr_dimensions/pot_index/small_basin_index.txt
        
    nr_gaussians : integer
        Number of gaussians (i.e., number of minima) in the potential energy surface.
        
    nr_dimensions : integer
        Euclidean dimension of the space mapping on the potential energy surface, i.e., number of degrees of freedom.
        
    nr_samples : integer
        Number of different potential energy landscapes sampled at each (nr_gaussians, nr_dimensions)
    """
    for pot_index in xrange(nr_samples):
        pot = MultiGaussWrap(potential_dir)
        if not pot.exists(nr_gaussians, nr_dimensions, pot_index):
            pot.generate(nr_gaussians, nr_dimensions, pot_index)
            pot.select_benchmark_basins(nr_gaussians, nr_dimensions, pot_index)

if __name__ == "__main__":
    nr_samples = 20
    potential_dir = os.path.join(os.getcwd(), "potentials")
    for nr_gaussians in [5]:
        for nr_dimensions in [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]:
            generate_potentials(potential_dir, nr_gaussians, nr_dimensions, nr_samples)
