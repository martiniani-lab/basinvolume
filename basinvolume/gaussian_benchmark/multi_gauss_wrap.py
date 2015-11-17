from __future__ import division

import numpy as np
import os

from pele.potentials import SumGaussianPot
from pele.optimize import ModifiedFireCPP

from basinvolume.utils import get_uniform_in_sphere
from basinvolume.utils import trymakedir

def MinGenerator(nmeans, dimension, System_R, Ref_R, min_sep=None):
    """
    This is taken from Shang's code, in the trajectories project:
    /VolCalcTest/VolCalcTest_MultiGaussian_dimtest.py
    """
    meanlist = []
    if min_sep is None:
        min_sep = 4 * Ref_R

    failures = 0
    while len(meanlist) < nmeans and failures < 2000 * nmeans:
        direction = np.random.uniform(low=-1.0, high=1.0, size=dimension)
        direction /= np.linalg.norm(direction)
        magnitude = np.random.uniform(low=0, high=System_R - Ref_R)

        new_mean = magnitude * direction

        acceptable = True
        i = 0
        while i in xrange(len(meanlist)) and acceptable:  # Not too close to another min
            acceptable = np.linalg.norm(new_mean - meanlist[i]) > min_sep
            i += 1

        if acceptable:
            meanlist.append(new_mean)
        else:
            failures += 1
    return meanlist

class MultiGaussWrap(object):
    """
    Wrapper for Shang's multi-gaussian potential.
    
    Writes and reads multi-gaussian potentials (ensembles) as needed for the gaussian benchmark computaions.
    Allows to use same interface to potentials for each method in the benchmark.
    """
    def __init__(self, potential_path, small_basin_index=0, RefRadius=1, R=10):
        self.potential_path = potential_path
        self.small_basin_index = small_basin_index
        self.RefRadius = RefRadius
        self.R = R
        
    def this_path(self, nr_gaussians, nr_dimensions, index):
        return os.path.join(self.potential_path, str(nr_gaussians), str(nr_dimensions), str(index))
        
    def this_pot_path(self, nr_gaussians, nr_dimensions, index):
        return os.path.join(self.this_path(nr_gaussians, nr_dimensions, index), "pot.txt")
        
    def this_large_index_path(self, nr_gaussians, nr_dimensions, index):
        return os.path.join(self.this_path(nr_gaussians, nr_dimensions, index), "large_basin_index.txt")
        
    def this_small_index_path(self, nr_gaussians, nr_dimensions, index):
        return os.path.join(self.this_path(nr_gaussians, nr_dimensions, index), "small_basin_index.txt")
        
    def exists(self, nr_gaussians, nr_dimensions, index):
        return os.path.exists(self.this_path(nr_gaussians, nr_dimensions, index))
    
    def generate(self, nr_gaussians, nr_dimensions, index):
        means = MinGenerator(nr_gaussians, nr_dimensions, self.R, self.RefRadius, min_sep=5 * self.RefRadius)
        covMatrixDiags = []
        for i in xrange(len(means)):
            covMatrixDiags.append(np.absolute(np.random.normal(loc=4, scale=2)) * np.ones(nr_dimensions))
        trymakedir(self.this_path(nr_gaussians, nr_dimensions, index))
        f = open(self.this_pot_path(nr_gaussians, nr_dimensions, index), "w")
        f.write('\n\nMeans:\t\t\t\tCov:\n')
        for i in xrange(len(means)):
            f.write(str(means[i]) + '\t' + str(covMatrixDiags[i]) + '\n')
        f.close()
        
    def select_benchmark_basins(self, nr_gaussians, nr_dimensions, index):
        # Sample one point in the sphere of radius R uniformly at random
        start = get_uniform_in_sphere(self.R, nr_dimensions)
        # Minimise from there
        end = self.get_local_minimum(nr_gaussians, nr_dimensions, index, start)
        # Determine basin index
        self.large_basin_index = self.get_basin_index(nr_gaussians, nr_dimensions, index, end)
        # Write corresponding large basin index to file
        np.writetxt(self.this_large_index_path(nr_gaussians, nr_dimensions, index), self.large_basin_index)
        # Write default / input small basin index to file
        np.writetxt(self.this_small_index_path(nr_gaussians, nr_dimensions, index), self.small_basin_index)
        
    def get_local_minimum(self, nr_gaussians, nr_dimensions, index, start):
        pot = self.get_pot(nr_gaussians, nr_dimensions, index)
        optimizer = ModifiedFireCPP(start, pot, dtmax=1, maxstep=1e-1, tol=1e-8, nsteps=1e8, verbosity=0)
        optimizer.reset(start)
        result = optimizer.run()
        return result.coords
        
    def get_pot(self, nr_gaussians, nr_dimensions, index):
        means, cov = self.get_mean_cov(nr_gaussians, nr_dimensions, index)
        return SumGaussianPot(means, cov)
        
    def get_mean_cov(self, nr_gaussians, nr_dimensions, index):
        from utils import get_means_cov
        return get_means_cov(self.this_pot_path(nr_gaussians, nr_dimensions, index))
        
    #def get_basin_index(nr_gaussians, nr_dimensions, index, end):
        
