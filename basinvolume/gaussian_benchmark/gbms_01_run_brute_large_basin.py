from __future__ import division

import numpy as np
import os

from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import UniformSphericalSampling
from mcpele.monte_carlo import NullPotential

from basinvolume.monte_carlo import CheckSameMinimumConfig
from basinvolume.utils import volume_nball

from computer_common import ComputerCommon
from computer_common import run_computer

class MC(_BaseMCRunner):
    def set_control(self, tmp):
        self.set_temperature(tmp)
    def run(self, nr_iterations):
        for _ in xrange(nr_iterations):
            self.one_iteration()

class BruteEngine(object):
    """
    Engine to do iteration-wise computation of basin volume by brute
    force rejection.
    """
    def __init__(self, brute_parameters, pes_parameters, potential):
        self.brute_parameters = brute_parameters
        self.pes_parameters = pes_parameters
        self.potential = potential
        self.mc_parameters = dict([("temperature", 1),
            ("max_nr_samples", 1e14)])
        self.mc_potential = NullPotential()
        self.mc = MC(self.mc_potential, self.pes_parameters["origin"],
            self.mc_parameters["temperature"],
            self.mc_parameters["max_nr_samples"])
        self.step = UniformSphericalSampling(42, self.pes_parameters["radius_container"])
        self.mc.set_takestep(self.step)
        self.mc.set_report_steps(0)
        self.conftest_check_same_minimum = CheckSameMinimumConfig(self.potential,
            self.pes_parameters["origin"], self.pes_parameters["csm_dtol"],
            opt=self.optimizer, opt_tol=self.opt_parameters["opt_tol"],
            opt_maxiter=self.opt_parameters["opt_nsteps"])
        self.mc.add_conf_test(self.conftest_check_same_minimum)
        self.evaluations = 0
        self.volume = 0
    
    def one_iteration(self):
        self.mc.run(self.brute_parameters["nr_samples_increment"])
        p = self.mc.get_accepted_fraction()
        self.volume = p * volume_nball(self.pes_parameters["radius_container"], self.pes_parameters["nr_dimensions"])
    

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
        self.brute_engine = BruteEngine(self.method_parameters, self.pes_parameters, self.pot)
        
    def get_method_label(self):
        return "brute"
        
    def get_evaluations_volume_one_iteration(self):
        self.brute_engine.one_iteration()
        evaluations = self.brute_engine.evaluations
        volume = self.brute_engine.volume
        return evaluations, volume
        

if __name__ == "__main__":
    nr_samples = 20
    opt_parameters = dict([("opt_dtmax", 1), ("opt_tol", 1e-8), ("opt_nsteps", 1e8), ("opt_maxstep", 0.1)])
    vol_parameters = dict([("max_iterations", 100)])
    brute_parameters = dict([("nr_samples_increment", 1000)])
    potential_dir = os.path.join(os.getcwd(), "potentials")
    large_basin_results_dir = os.path.join(os.getcwd(), "large_basin_results")
    for nr_gaussians in [5]:
        for nr_dimensions in [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]:
            pes_parameters = dict([("csm_dtol", 1),
                ("nr_dimensions", nr_dimensions),
                ("radius_container", 10)])
            run_computer(potential_dir, large_basin_results_dir, "large",
                nr_gaussians, nr_dimensions, nr_samples, BruteComputer,
                opt_parameters, pes_parameters, vol_parameters,
                brute_parameters)
