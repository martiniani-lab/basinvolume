from __future__ import division

import numpy as np
import os

from computer_common import ComputerCommon
from computer_common import run_computer

try:
    import trajectories.volcalc_onthefly_no_store_2 as vc
except Exception as e:
    print(e)

class TrajEngine(object):
    """
    Engine to do iteration-wise computation of basin volume with traj
    method.
    """
    def __init__(self, traj_parameters, pes_parameters, pot, opt_parameters, vol_parameters):
        self.traj_parameters = traj_parameters
        self.pes_parameters = pes_parameters
        self.pot = pot
        self.opt_parameters = opt_parameters
        self.vol_parameters = vol_parameters
        self.walk = vc.Volcalc_onthefly(stepsize=self.traj_parameters["stepsize"],
            primesteps=self.traj_parameters["primesteps"],
            ntraj_inc=self.traj_parameters["nr_samples_increment"],
            ntraj_max=self.traj_parameters["nr_samples_increment"]*self.vol_parameters["max_iterations"])
        
    def compute_volume_traj(self):
        self.walk.run(self.pot, self.pes_parameters["origin"],
            WalkOrigin=None, RefRadius=self.pes_parameters["csm_dtol"],
            MinAtStartCoord=True,
            quench_dt=self.traj_parameters["quench_dt"],
            quench_maxsteps=self.traj_parameters["quench_maxsteps"],
            quench_tol=self.traj_parameters["quench_tol"],
            quench_hessiantol=self.traj_parameters["quench_hessiantol"],
            quench_backtrack=self.traj_parameters["quench_backtrack"],
            targetaccept=self.traj_parameters["targetaccept"],
            check_RefRadius=True,
            fn_call_counter=vc.Counter(),
            convergence_repeats=self.traj_parameters["convergence_repeats"],
            convergence_tol=self.traj_parameters["convergence_tol"],
            convergence_fraction=self.traj_parameters["convergence_fraction"],
            old_backtracker_descent=self.traj_parameters["old_backtracker_descent"])
        self.volume = self.walk.EstimatedVolume
        self.evaluations = self.walk.evaluations
        self.ini_evals = self.walk.ini_evals

class TrajComputer(ComputerCommon):
    """
    Compute volume of basin in the gaussian landscape by trajectory 
    method as function of the number of function calls and print to
    disk.
    """
    def __init__(self, results_dir, opt_parameters, pes_parameters,
        vol_parameters, method_parameters, pot):
        super(TrajComputer, self).__init__(results_dir, opt_parameters,
            pes_parameters, vol_parameters, method_parameters, pot)
        self.traj_engine = TrajEngine(self.method_parameters,
            self.pes_parameters, self.pot, self.opt_parameters,
            self.vol_parameters)
            
    def get_method_label(self):
        return "traj"

def run_traj(ls_basin_label):
    """
    Execute trajectory method basin computation for large or small basin.
    """
    if ls_basin_label is not "large" and ls_basin_label is not "small":
        raise Exception("ls_basin_label: illegal input, can be large or small only")
    nr_samples = 10
    opt_parameters = None
    vol_parameters = dict([("max_iterations", 10000)])
    traj_parameters = dict([("nr_samples_increment", 1),
        ("stepsize", 5), ("primesteps", 3000),
        ("quench_maxsteps", 1e18),
        ("quench_tol", 1e-8),
        ("quench_hessiantol", 0.001),
        ("quench_backtrack", True),
        ("targetaccept", 0.25),
        ("convergence_repeats", 20),
        ("convergence_tol", 0),
        ("convergence_fraction", 0.05),
        ("old_backtracker_descent", False)])
    potential_dir = os.path.join(os.getcwd(), "potentials")
    ls_basin_results_dir = os.path.join(os.getcwd(), ls_basin_label + "_basin_results")
    for nr_gaussians in [5]:
        for nr_dimensions in [2, 3, 4, 5, 10, 15, 20, 25, 30]:
            pes_parameters = dict([("csm_dtol", 1),
                ("nr_dimensions", nr_dimensions),
                ("radius_container", 10)])
            traj_parameters["quench_dt"] = 0.038 * nr_dimensions ** -1.027
            run_computer(potential_dir, ls_basin_results_dir,
                ls_basin_label, nr_gaussians, nr_dimensions, nr_samples,
                TrajComputer, opt_parameters, pes_parameters,
                vol_parameters, traj_parameters, is_traj=True)

if __name__ == "__main__":
    run_traj("large")
