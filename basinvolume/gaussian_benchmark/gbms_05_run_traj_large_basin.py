from __future__ import division

import numpy as np
import os

from computer_common import ComputerCommon
from computer_common import run_computer

def run_traj(ls_basin_label):
    """
    Execute trajectory method basin computation for large or small basin.
    """
    if ls_basin_label is not "large" and ls_basin_label is not "small":
        raise Exception("ls_basin_label: illegal input, can be large or small only")
    nr_samples = 20
    vol_parameters = dict([("max_iterations", 1000)])
    traj_parameters = dict([("nr_samples_increment", 1000)])
    potential_dir = os.path.join(os.getcwd(), "potentials")
    ls_basin_results_dir = os.path.join(os.getcwd(), ls_basin_label + "_basin_results")
    for nr_gaussians in [5]:
        for nr_dimensions in [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]:
            run_computer(potential_dir, ls_basin_results_dir,
                ls_basin_label, nr_gaussians, nr_dimensions, nr_samples,
                TrajComputer, opt_parameters, pes_parameters,
                vol_parameters, traj_parameters)

if __name__ == "__main__":
    run_traj("large")
