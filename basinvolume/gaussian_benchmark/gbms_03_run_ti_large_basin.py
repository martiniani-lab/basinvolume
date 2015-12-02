from __future__ import division

import numpy as np
import os

from pele.optimize import ModifiedFireCPP
from pele.potentials import Harmonic

from mcpele.monte_carlo import MetropolisTest
from mcpele.monte_carlo import NullPotential
from mcpele.monte_carlo import RandomCoordsDisplacement

from basinvolume.monte_carlo import CheckHyperSphericalContainer
from basinvolume.monte_carlo import CheckSameMinimumConfig
from basinvolume.monte_carlo import Findk
from basinvolume.monte_carlo import RecordDisp2Histogram

from computer_common import ComputerCommon
from computer_common import run_computer
from gaussian_benchmark_kmax_run import GaussianBenchmarkKmaxRun
from gaussian_benchmark_kmin_run import GaussianBenchmarkKminRun
from gbms_01_run_brute_large_basin import EngineCommonOpt

class TIEngine(EngineCommonOpt):
    """
    Engine to do iteration-wise computation of basin volume by TI method.
    """
    def __init__(self, ti_parameters, pes_parameters, potential, opt_parameters, seeds=None):
        super(TIEngine, self).__init__(pes_parameters, potential, opt_parameters)
        self.ti_parameters = ti_parameters
        self.pes_parameters["rattlers"] = np.ones(self.pes_parameters["origin"].size)
        self.ti_parameters["equilibration_steps"] = self.ti_parameters["adjustf_niter"] + self.ti_parameters["pt_eq_niter"]
        self.setup(seeds)
        
    def setup(self, seeds):
        self.conftest_outer_sphere = CheckHyperSphericalContainer(np.zeros(self.pes_parameters["nr_dimensions"]),
            self.pes_parameters["radius_container"],
            self.pes_parameters["nr_dimensions"])
        self.evaluations = 0
        self.volume = 0
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max),
                    seed_metropolis=np.random.randint(i32max))
        self.seeds = seeds
        self.setup_ti_kmax()
        self.takestep = RandomCoordsDisplacement(self.seeds['seed_takestep'],
            5, report_interval=100, factor=0.9, min_acc_ratio=0.2,
            max_acc_ratio=0.2, single=False,
            bdim=self.pes_parameters["nr_dimensions"])
        self.metropolis = MetropolisTest(self.seeds['seed_metropolis'])
        self.harmonic_potential = Harmonic(self.pes_parameters["origin"],
            42, bdim=self.pes_parameters["nr_dimensions"],
            com=self.ti_parameters["harmonic_com_flag"])
        self.setup_ti_kmin()
        self.setup_ti_pt_walks()
        
    def one_iteration(self):
        self.continue_pt_walks()
        self.volume = self.compute_volume()
        self.evaluations = self.conftest_check_same_minimum.get_nfev()
        
    def setup_ti_kmax(self):
        self.action_findk = Findk(self.pes_parameters["origin"],
                                  self.pes_parameters["rattlers"],
                                  self.pes_parameters["nr_dimensions"],
                                  self.ti_parameters["kmax_avgcount"],
                                  self.ti_parameters["ktarget"],
                                  self.ti_parameters["knavg"],
                                  self.ti_parameters["ktol"],
                                  self.ti_parameters["hmin"],
                                  self.ti_parameters["hmax"],
                                  self.ti_parameters["binsize"],
                                  fix_com=self.ti_parameters["harmonic_com_flag"])
        self.kmax_run = GaussianBenchmarkKmaxRun(NullPotential(),
                                                 self.pes_parameters["origin"],
                                                 1,
                                                 self.ti_parameters["kmax_niter"])
        self.kmax_run.setup(pot_optimizer=self.potential,
                            origin=self.pes_parameters["origin"],
                            seeds=None,
                            conftest_outer_sphere=self.conftest_outer_sphere,
                            conftest_check_same_minimum=self.conftest_check_same_minimum,
                            action_findk=self.action_findk,
                            avgcount=self.ti_parameters["kmax_avgcount"],
                            niter=self.ti_parameters["kmax_niter"])
        self.kmax_run.run()
        self.kmax = self.kmax_run.get_k()
        self.kmax_displ2 = self.kmax_run.get_displ2()
        self.prob_kmax = self.kmax_run.get_prob_kmax()
        self.var_displ_kmax = self.kmax_run.get_var_displ_kmax()
        self.kmax_displ2_nr_samples = self.kmax_run.get_entries()
        
    def setup_ti_kmin(self):
        self.action_record_displ_kmin = RecordDisp2Histogram(self.pes_parameters["origin"],
                                                             self.pes_parameters["rattlers"],
                                                             self.pes_parameters["nr_dimensions"],
                                                             self.ti_parameters["hmin"],
                                                             self.ti_parameters["hmax"],
                                                             self.ti_parameters["binsize"],
                                                             self.ti_parameters["equilibration_steps"],
                                                             fix_com=self.ti_parameters["harmonic_com_flag"])
        self.kmin_run = GaussianBenchmarkKminRun(pot_optimizer=self.potential,
                                                 origin=self.pes_parameters["origin"],
                                                 conftest_outer_sphere=self.conftest_outer_sphere,
                                                 conftest_check_same_minimum=self.conftest_check_same_minimum,
                                                 action_record_displ=self.action_record_displ_kmin,
                                                 adjustf_niter=self.ti_parameters["adjustf_niter"],
                                                 pt_eq_niter=self.ti_parameters["pt_eq_niter"],
                                                 equilibration_steps=self.ti_parameters["equilibration_steps"],
                                                 metropolis=self.metropolis,
                                                 takestep=self.takestep,
                                                 potential=self.harmonic_potential,
                                                 niter=self.ti_parameters["kmin_niter"],
                                                 nparticles=1)
        self.kmin_run.run_kmin()
        self.displ2_kmin_mean, self.displ2_kmin_variance = self.kmin_run.get_displ2_kmin()
        print("displ2_kmin", self.displ2_kmin_mean)
        print("displ2_kmin_variance", self.displ2_kmin_variance)
        assert(False)
        
    #def setup_ti_pt_walks(self):
    
    #def continue_pt_walks(self):
        
    #def compute_volume(self):
        

class TIComputer(ComputerCommon):
    """
    Computer volume of basin in the gaussian landscape by thermodynamic integration
    as function of the number of energy function calls and print that to disk.
    """
    def __init__(self, results_dir, opt_parameters, pes_parameters,
        vol_parameters, method_parameters, pot):
        super(TIComputer, self).__init__(results_dir, opt_parameters,
            pes_parameters, vol_parameters, method_parameters, pot)
        self.ti_engine = TIEngine(self.method_parameters,
            self.pes_parameters, self.pot, self.opt_parameters)
            
    def get_method_labels(self):
        return "ti"
        
    def get_evaluations_volume_one_iteration(self):
        self.ti_engine.one_iteration()
        evaluations = self.ti_engine.evaluations
        volume = self.ti_engine.volume
        return evaluations, volume

def run_ti(ls_basin_label):
    """
    Execute ti basin volume computation for large or small basin.
    
    Parameter
    ---------
    
    ls_basin_label : string
        Needs to be "large" or "small" and indicates size label of basin to be computed.
    """
    if ls_basin_label is not "large" and ls_basin_label is not "small":
        raise Exception("ls_basin_label: illegal input, can be large or small only")
    nr_samples = 20
    opt_parameters = dict([("opt_dtmax", 1), ("opt_tol", 1e-8),
        ("opt_nsteps", 1e8), ("opt_maxstep", 0.1), ("verbosity", 0)])
    vol_parameters = dict([("max_iterations", 100)])
    ti_parameters = dict([("nr_samples_increment", 1000),
        ("ktarget", 0.8), ("knavg", 500), ("ktol", 0.05), ("hmin", 0),
        ("hmax", 1), ("binsize", 0.005), ("harmonic_com_flag", False),
        ("kmax_niter", 1e5), ("kmax_avgcount", 1e4),
        ("adjustf_niter", 1e4), ("pt_eq_niter", 1e5),
        ("kmin_niter", 2e5)])
    potential_dir = os.path.join(os.getcwd(), "potentials")
    ls_basin_results_dir = os.path.join(os.getcwd(), ls_basin_label + "_basin_results")
    for nr_gaussians in [5]:
        for nr_dimensions in [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]:
            pes_parameters = dict([("csm_dtol", 1),
                ("nr_dimensions", nr_dimensions),
                ("radius_container", 10)])
            run_computer(potential_dir, ls_basin_results_dir, ls_basin_label,
                nr_gaussians, nr_dimensions, nr_samples, TIComputer,
                opt_parameters, pes_parameters, vol_parameters,
                ti_parameters)

if __name__ == "__main__":
    run_ti("large")
