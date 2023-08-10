from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import object
import argparse
import os
import copy
import subprocess
import shlex
import shutil
import numpy as np
from pele.optimize import ModifiedFireCPP
from pele.potentials import SumGaussianPot
from pele.potentials import Harmonic
from mcpele.monte_carlo import CheckSphericalContainer
from mcpele.monte_carlo import RandomCoordsDisplacement
from mcpele.monte_carlo import MetropolisTest
import basinvolume
from basinvolume.monte_carlo import CheckSameMinimumConfig
from basinvolume.utils import ResultsFile
from basinvolume.utils import to_string
from basinvolume.utils import volume_nball
from basinvolume.utils import trymakedir
from .brute_force_2d import BruteForce2D

try:
    from .utils import *
except Exception as e:
    print(e)


class EvalCounter(object):
    def __init__(self):
        self.count = 0


def compute_volume(minimum_index=None, means=None, cov=None):
    if minimum_index >= means.shape[0] or minimum_index < 0:
        raise Exception("illegal input: index of minimum")
    if not means.shape == cov.shape:
        raise Exception("illegal input: means shape is not cov shape")
    res = []
    config = "config{}.gauss".format(minimum_index)
    # Brute force rejection sampling computation of volume i.
    bf = BruteForce2D(means=means, cov=cov, minimum_index=minimum_index)
    bf.compute_volume()
    # Print all volumes to screen.
    print("---direct MC result---")
    print(("bf.basin_volume", bf.basin_volume))
    print(("bf.error_basin_volume", bf.error_basin_volume))
    # Print all volumes to file.
    fout = ResultsFile(
        os.path.join(
            os.getcwd(), "volume_method_comparison_brute{}".format(minimum_index)
        )
    )
    fout.set_heading("DIRECT REJECTION SAMPLING")
    fout.to_file("bf.basin_volume", bf.basin_volume)
    fout.to_file("bf.error_basin_volume", bf.error_basin_volume)
    fout.to_file("bf.nfev", int(bf.nfev))
    fout.close()


if __name__ == "__main__":
    """
    means = np.asarray([
    [-0.66188835, -4.90248303],
    [-2.50068746,  1.00984605],
    [ 2.79156189,  3.46309925],
    [ 4.7283517, -1.92263871],
    [-7.2936999,  -2.33272689],
    [-5.75772061,  6.56907688],
    [ 1.26304236,  8.80647431],
    [ 7.82900305,  2.89542514],
    [ 5.35866288, -7.17580499],
    [-5.16164399, -7.13075119]
    ])
    cov = np.asarray([
    [ 2.89579414,  2.89579414],
    [ 3.27805735,  3.27805735],
    [ 2.36765046,  2.36765046],
    [ 4.02608698,  4.02608698],
    [ 1.87897462,  1.87897462],
    [ 3.45861227,  3.45861227],
    [ 4.63020449,  4.63020449],
    [ 5.88827858,  5.88827858],
    [ 1.97514666,  1.97514666],
    [ 1.62236091,  1.62236091]
    ])
    """
    parser = argparse.ArgumentParser(
        description="Compute gaussian landscape volumes with TI and rejection sampling to compare to trajectories method"
    )
    parser.add_argument("--gauss_path", type=str, default=os.getcwd())
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    means, cov = get_means_cov(args.gauss_path)
    print(("means", means))
    print(("cov", cov))
    compute_volume(minimum_index=args.index, means=means, cov=cov)
