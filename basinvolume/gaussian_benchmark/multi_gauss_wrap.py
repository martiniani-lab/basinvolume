from __future__ import division
from __future__ import absolute_import

from builtins import str
from builtins import range
from builtins import object
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
        while i in range(len(meanlist)) and acceptable:  # Not too close to another min
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
        return os.path.join(
            self.potential_path, str(nr_gaussians), str(nr_dimensions), str(index)
        )

    def this_pot_path(self, nr_gaussians, nr_dimensions, index):
        return os.path.join(
            self.this_path(nr_gaussians, nr_dimensions, index), "pot.txt"
        )

    def this_large_index_path(self, nr_gaussians, nr_dimensions, index):
        return os.path.join(
            self.this_path(nr_gaussians, nr_dimensions, index), "large_basin_index.txt"
        )

    def this_large_origin_path(self, nr_gaussians, nr_dimensions, index):
        return os.path.join(
            self.this_path(nr_gaussians, nr_dimensions, index), "large_basin_origin.txt"
        )

    def this_small_index_path(self, nr_gaussians, nr_dimensions, index):
        return os.path.join(
            self.this_path(nr_gaussians, nr_dimensions, index), "small_basin_index.txt"
        )

    def this_small_origin_path(self, nr_gaussians, nr_dimensions, index):
        return os.path.join(
            self.this_path(nr_gaussians, nr_dimensions, index), "small_basin_origin.txt"
        )

    def exists(self, nr_gaussians, nr_dimensions, index):
        return os.path.exists(self.this_path(nr_gaussians, nr_dimensions, index))

    def generate(self, nr_gaussians, nr_dimensions, index):
        means = MinGenerator(
            nr_gaussians,
            nr_dimensions,
            self.R,
            self.RefRadius,
            min_sep=5 * self.RefRadius,
        )
        covMatrixDiags = []
        for i in range(len(means)):
            covMatrixDiags.append(
                np.absolute(np.random.normal(loc=4, scale=2)) * np.ones(nr_dimensions)
            )
        trymakedir(self.this_path(nr_gaussians, nr_dimensions, index))
        f = open(self.this_pot_path(nr_gaussians, nr_dimensions, index), "w")
        f.write("\n\nMeans:\t\t\t\tCov:\n")
        for i in range(len(means)):
            f.write(str(means[i]) + "\t" + str(covMatrixDiags[i]) + "\n")
        f.close()

    def select_benchmark_basins(self, nr_gaussians, nr_dimensions, index):
        # Sample one point in the sphere of radius R uniformly at random
        start = get_uniform_in_sphere(self.R, nr_dimensions)
        # Minimise from there
        end = self.get_local_minimum(nr_gaussians, nr_dimensions, index, start)
        # Determine basin index
        self.large_basin_index = self.get_basin_index(
            nr_gaussians, nr_dimensions, index, end
        )

        def to_file(name, number):
            f = open(name, "w")
            f.write(str(number) + "\n")
            f.close()

        # Write corresponding large basin index to file
        to_file(
            self.this_large_index_path(nr_gaussians, nr_dimensions, index),
            int(self.large_basin_index),
        )
        # Write corresponding large basin origin to file
        np.savetxt(self.this_large_origin_path(nr_gaussians, nr_dimensions, index), end)
        # Write default / input small basin index to file
        to_file(
            self.this_small_index_path(nr_gaussians, nr_dimensions, index),
            int(self.small_basin_index),
        )
        # Write default / input small basin origin to file
        means, cov = self.get_mean_cov(nr_gaussians, nr_dimensions, index)
        np.savetxt(
            self.this_small_origin_path(nr_gaussians, nr_dimensions, index),
            self.get_local_minimum(
                nr_gaussians, nr_dimensions, index, means[self.small_basin_index]
            ),
        )

    def get_large_basin_origin(self, nr_gaussians, nr_dimensions, index):
        return np.loadtxt(
            self.this_large_origin_path(nr_gaussians, nr_dimensions, index)
        )

    def get_small_basin_origin(self, nr_gaussians, nr_dimensions, index):
        return np.loadtxt(
            self.this_small_origin_path(nr_gaussians, nr_dimensions, index)
        )

    def get_local_minimum(self, nr_gaussians, nr_dimensions, index, start):
        pot = self.get_pot(nr_gaussians, nr_dimensions, index)
        optimizer = ModifiedFireCPP(
            start, pot, dtmax=1, maxstep=1e-1, tol=1e-8, nsteps=1e8, verbosity=0
        )
        optimizer.reset(start)
        result = optimizer.run()
        return result.coords

    def get_pot(self, nr_gaussians, nr_dimensions, index):
        means, cov = self.get_mean_cov(nr_gaussians, nr_dimensions, index)
        return SumGaussianPot(means, cov)

    def get_mean_cov(self, nr_gaussians, nr_dimensions, index):
        from .utils import get_means_cov

        return get_means_cov(self.this_pot_path(nr_gaussians, nr_dimensions, index))

    def get_basin_index(self, nr_gaussians, nr_dimensions, index, end):
        """
        Map coordinates in 'end' to minimum index.

        Return index k, such that the local minimum reached from mean(k)
        has smallest euclidean distance to 'end'.
        """
        means, cov = self.get_mean_cov(nr_gaussians, nr_dimensions, index)
        minima = [
            self.get_local_minimum(nr_gaussians, nr_dimensions, index, means[i][:])
            for i in range(nr_gaussians)
        ]
        distances = [np.linalg.norm(minima[i][:] - end) for i in range(nr_gaussians)]
        basin_index = np.argmin(distances)
        return basin_index

    def get_origin(self, nr_gaussians, nr_dimensions, index, large_small_flag):
        if large_small_flag == "large":
            return self.get_large_basin_origin(nr_gaussians, nr_dimensions, index)
        elif large_small_flag == "small":
            return self.get_small_basin_origin(nr_gaussians, nr_dimensions, index)
        else:
            raise Exception("get_origin: large_small_flag is illegal")
