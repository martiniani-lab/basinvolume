from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
from builtins import object
import argparse as ap
import numpy as np
import os

from pele.optimize import ModifiedFireCPP

from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import UniformSphericalSampling
from mcpele.monte_carlo import NullPotential

from basinvolume.monte_carlo import CheckSameMinimumConfig
from basinvolume.utils import log_volume_nball

from .computer_common import ComputerCommon
from .computer_common import run_computer

class MC(_BaseMCRunner):
    def set_control(self, tmp):
        self.set_temperature(tmp)
    def run(self, nr_iterations):
        for _ in range(nr_iterations):
            self.one_iteration()
            
class EngineCommonOpt(object):
    """
    Common parts of volume engines for Brute and TI engines.
    """
    def __init__(self, pes_parameters, potential, opt_parameters):
        self.pes_parameters = pes_parameters
        self.potential = potential
        self.opt_parameters = opt_parameters
        self.optimizer = ModifiedFireCPP(self.pes_parameters["origin"],
            self.potential, dtmax=self.opt_parameters["opt_dtmax"],
            maxstep=self.opt_parameters["opt_maxstep"],
            tol=self.opt_parameters["opt_tol"],
            nsteps=self.opt_parameters["opt_nsteps"],
            verbosity=self.opt_parameters["verbosity"])
        self.conftest_check_same_minimum = CheckSameMinimumConfig(self.potential,
            self.pes_parameters["origin"], self.pes_parameters["csm_dtol"],
            opt=self.optimizer, opt_tol=self.opt_parameters["opt_tol"],
            opt_maxiter=self.opt_parameters["opt_nsteps"])

class BruteEngine(EngineCommonOpt):
    """
    Engine to do iteration-wise computation of basin volume by brute
    force rejection.
    """
    def __init__(self, brute_parameters, pes_parameters, potential, opt_parameters):
        super(BruteEngine, self).__init__(pes_parameters, potential, opt_parameters)
        self.brute_parameters = brute_parameters
        self.mc_parameters = dict([("temperature", 1),
            ("max_nr_samples", 1e14)])
        self.mc_potential = NullPotential()
        self.mc = MC(self.mc_potential, self.pes_parameters["origin"],
            self.mc_parameters["temperature"],
            self.mc_parameters["max_nr_samples"])
        self.step = UniformSphericalSampling(42, self.pes_parameters["radius_container"])
        self.mc.set_takestep(self.step)
        self.mc.set_report_steps(0)
        self.mc.add_conf_test(self.conftest_check_same_minimum)
        self.evaluations = 0
        self.volume = 0
    
    def one_iteration(self):
        self.mc.run(self.brute_parameters["nr_samples_increment"])
        p = self.mc.get_accepted_fraction()
        self.volume = np.exp(np.log(p) +
            log_volume_nball(self.pes_parameters["radius_container"],
            self.pes_parameters["nr_dimensions"]))
        self.evaluations = self.conftest_check_same_minimum.get_nfev()
    

class BruteComputer(ComputerCommon):
    """
    Compute volume of a basin in the gaussian landscape by brute force,
    as function of the number of function calls, and print it to the
    disk.
    """
    def __init__(self, results_dir, opt_parameters, pes_parameters,
        vol_parameters, method_parameters, pot):
        super(BruteComputer, self).__init__(results_dir, opt_parameters,
            pes_parameters, vol_parameters, method_parameters, pot)
        self.brute_engine = BruteEngine(self.method_parameters,
            self.pes_parameters, self.pot, self.opt_parameters)
        self.ini_evals = 0 # Brute force does not need initialisation.
        
    def get_method_label(self):
        return "brute"
        
    def get_evaluations_volume_one_iteration(self):
        self.brute_engine.one_iteration()
        evaluations = self.brute_engine.evaluations
        volume = self.brute_engine.volume
        return evaluations, volume
        

def run_brute(ls_basin_label):
    """
    Parse input args: nr_dimenisons, sample_index, nr_iterations.
    """
    arg = ap.ArgumentParser()
    arg.add_argument("--nr_dimensions", type=int, help="Euclidean dimension of potential landscape")
    arg.add_argument("--sample_index", type=int, help="Index of landscape to measure")
    arg.add_argument("--nr_iterations", type=int, help="Maximum nr of iterations to consider for volume measurement")
    arg = arg.parse_args()
    nr_dimensions = arg.nr_dimensions
    sample_index = arg.sample_index
    nr_iterations = arg.nr_iterations
    print(("arg", arg))
    """
    Execute brute force basin computation for large or small basin.
    
    Parameter
    ---------
    
    ls_basin_label : string
        Needs to be "large" or "small" and indicates size label of basin to be computed.
    """
    
    if ls_basin_label is not "large" and ls_basin_label is not "small":
        raise Exception("ls_basin_label: illegal input, can be large or small only")
    opt_parameters = dict([("opt_dtmax", 1), ("opt_tol", 1e-8),
        ("opt_nsteps", 1e8), ("opt_maxstep", 0.1), ("verbosity", 0)])
    vol_parameters = dict([("max_iterations", nr_iterations)])
    brute_parameters = dict([("nr_samples_increment", 1)])
    potential_dir = os.path.join(os.getcwd(), "potentials")
    ls_basin_results_dir = os.path.join(os.getcwd(), ls_basin_label + "_basin_results")
    for nr_gaussians in [5]:
        pes_parameters = dict([("csm_dtol", 1),
            ("nr_dimensions", nr_dimensions),
            ("radius_container", 10)])
        run_computer(potential_dir, ls_basin_results_dir, ls_basin_label,
            nr_gaussians, nr_dimensions, sample_index, BruteComputer,
            opt_parameters, pes_parameters, vol_parameters,
            brute_parameters)

if __name__ == "__main__":
    run_brute("large")
