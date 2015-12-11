from __future__ import division

import numpy as np
import os

from computer_common import ComputerCommon
from computer_common import run_computer

class TrajEngine(object):
    """
    Engine to do iteration-wise computation of basin volume with traj
    method.
    """
    

class TrajComputer(ComputerCommon):
    """
    Compute volume of basin in the gaussian landscape by trajectory 
    method as function of the number of function calls and print to
    disk.
    """
    def __ini__(self, results_dir, opt_parameters, pes_parameters,
        vol_parameters, method_parameters, pot):
        super(TrajComputer, self).__init__(results_dir, opt_parameters,
            pes_parameters, vol_parameters, method_parameters, pot)
        self.traj_engine = TrajEngine(self.method_parameters,
            self.pes_parameters, self.pot, self.opt_parameters)
            
    def get_method_label(self):
        return "traj"
        
    def get_evaluations_volume_one_iteration(self):
        self.traj_engine.one_iteration()
        evaluations = self.traj_engine.evaluations
        volume = self.traj_engine.volume
        return evaluations, volume

def run_traj(ls_basin_label):
    """
    Execute trajectory method basin computation for large or small basin.
    """
    if ls_basin_label is not "large" and ls_basin_label is not "small":
        raise Exception("ls_basin_label: illegal input, can be large or small only")
    nr_samples = 20
    opt_parameters = None
    vol_parameters = dict([("max_iterations", 1000)])
    traj_parameters = dict([("nr_samples_increment", 1000)])
    potential_dir = os.path.join(os.getcwd(), "potentials")
    ls_basin_results_dir = os.path.join(os.getcwd(), ls_basin_label + "_basin_results")
    for nr_gaussians in [5]:
        for nr_dimensions in [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]:
            pes_parameters = dict([("csm_dtol", 1),
                ("nr_dimensions", nr_dimensions),
                ("radius_container", 10)])
            run_computer(potential_dir, ls_basin_results_dir,
                ls_basin_label, nr_gaussians, nr_dimensions, nr_samples,
                TrajComputer, opt_parameters, pes_parameters,
                vol_parameters, traj_parameters)

if __name__ == "__main__":
    run_traj("large")
