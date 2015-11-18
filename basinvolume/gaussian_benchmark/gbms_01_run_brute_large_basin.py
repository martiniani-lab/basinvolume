from __future__ import division

import numpy as np
import os

from computer_common import ComputerCommon
from computer_common import run_computer

class BruteComputer(ComputerCommon):
    """
    Compute volume of a basin in the gaussian landscape by brute force,
    as function of the number of function calls, and print it to the
    disk.
    """
    def __init__(self, results_dir, opt_parameters, pes_parameters,
        vol_parameters):
        super(BruteComputer, self).__init__(results_dir, opt_parameters,
            pes_parameters, vol_parameters)
        
    def get_method_label(self):
        return "brute"
        
    def get_evaluations_volume_one_iteration(self):
        evaluations = 42
        volume = 44
        return evaluations, volume
        

if __name__ == "__main__":
    nr_samples = 20
    opt_parameters = dict([("opt_dtmax", 1), ("opt_tol", 1e-8), ("opt_nsteps", 1e8), ("opt_maxstep", 0.1)])
    pes_parameters = dict([("cms_dtol", 1)])
    vol_parameters = dict([("max_iterations", 100)])
    potential_dir = os.path.join(os.getcwd(), "potentials")
    large_basin_results_dir = os.path.join(os.getcwd(), "large_basin_results")
    for nr_gaussians in [5]:
        for nr_dimensions in [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]:
            run_computer(potential_dir, large_basin_results_dir, "large",
                nr_gaussians, nr_dimensions, nr_samples, BruteComputer,
                opt_parameters, pes_parameters, vol_parameters)
