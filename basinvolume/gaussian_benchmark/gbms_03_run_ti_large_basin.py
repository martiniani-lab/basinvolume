from __future__ import division

import argparse as ap
import copy
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
from basinvolume.post_processing import F_Basin_From_MC_Data
from basinvolume.post_processing import F_Basin_From_MC_Data_Free_COM
from basinvolume.post_processing import spring_constants_variable_transform

try:
    from computer_common import ComputerCommon
    from computer_common import run_computer
    from gaussian_benchmark_kmax_run import GaussianBenchmarkKmaxRun
    from gaussian_benchmark_kmin_run import GaussianBenchmarkKminRun
    from gbms_01_run_brute_large_basin import EngineCommonOpt
except Exception as e:
    print(e)

class TIVolumeComputer(object):
    """
    Take information obtained in Ti related simulations and return volume.

    Initially this takes the results of the kmax and kmin runs, then,
    after each biased random walk iteration, it takes the updated displ2
    array and computes the volume. The output is intended for the volume
    computation benchmark, where the volume estimate is analysed as
    function of the number of calls to the potential energy function.
    """
    def __init__(self, karray, displ_k_max, bdim, nparticles, prob_kmax,
        displ2_kmin_mean):
        self.karray = copy.deepcopy(karray)
        self.displ_k_max = displ_k_max
        self.bdim = bdim
        self.nparticles = nparticles
        self.prob_kmax = prob_kmax
        self.displ2_kmin_mean = displ2_kmin_mean

    def compute_volume(self, direct_k_u2_means):
        self.u2_array = copy.deepcopy(direct_k_u2_means)
        self.u2_array = np.append(self.u2_array, self.displ_k_max)
        sqared_std_errors = np.ones(len(self.u2_array))
        self.F0unc, self.sigF0unc, self.farrayunc, self.sigfarrayunc = F_Basin_From_MC_Data_Free_COM(self.bdim,
                                                                                                     self.nparticles,
                                                                                                     self.karray,
                                                                                                     self.u2_array,
                                                                                                     self.prob_kmax,
                                                                                                     displ_k_min_trafo=self.displ2_kmin_mean,
                                                                                                     simple_integrator=False).get_free_energy_F0(sqared_std_errors)
        return np.exp(-self.F0unc)

class TIEngine(EngineCommonOpt):
    """
    Engine to do iteration-wise computation of basin volume by TI method.
    """
    def __init__(self, ti_parameters, pes_parameters, potential, opt_parameters, seeds=None, verbose=False):
        super(TIEngine, self).__init__(pes_parameters, potential, opt_parameters)
        self.ti_parameters = ti_parameters
        self.pes_parameters["rattlers"] = np.ones(self.pes_parameters["origin"].size)
        self.ti_parameters["equilibration_steps"] = self.ti_parameters["adjustf_niter"] + self.ti_parameters["pt_eq_niter"]
        self.verbose = verbose
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
            5, report_interval=500, factor=0.95, min_acc_ratio=0.2,
            max_acc_ratio=0.2, single=False,
            bdim=self.pes_parameters["nr_dimensions"])
        self.metropolis = MetropolisTest(self.seeds['seed_metropolis'])
        self.harmonic_potential = Harmonic(self.pes_parameters["origin"],
            42, bdim=self.pes_parameters["nr_dimensions"],
            com=self.ti_parameters["harmonic_com_flag"])
        self.setup_ti_kmin()
        self.setup_ti_pt_walks()
        self.ini_evals = self.conftest_check_same_minimum.get_nfev()

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
        print("TI ko step", self.takestep.get_stepsize())
        if self.verbose:
            print("displ2_kmin", self.displ2_kmin_mean)
            print("displ2_kmin_variance", self.displ2_kmin_variance)
            print("kmin niter:", self.kmin_run.get_iterations_count())

    def setup_ti_pt_walks(self):
        self.all_k_values = spring_constants_variable_transform(self.ti_parameters["nprocs"] + 1,
                                                                self.kmax,
                                                                self.displ2_kmin_mean,
                                                                1,
                                                                self.pes_parameters["nr_dimensions"])
        self.volume_computer = TIVolumeComputer(self.all_k_values,
                                                self.kmax_displ2,
                                                self.pes_parameters["nr_dimensions"],
                                                1,
                                                self.prob_kmax,
                                                self.displ2_kmin_mean)
        self.direct_k_values = self.all_k_values[:-1]
        self.direct_k_u2_means = np.zeros(len(self.direct_k_values))
        self.direct_k_u2_variances = np.zeros(len(self.direct_k_values))
        self.direct_k_walkers = []
        for k_index, k_value in enumerate(self.direct_k_values):
            self.direct_k_walkers.append(DirectKWalker(k_index, k_value,
                self.pes_parameters, self.ti_parameters, self.potential,
                self.conftest_outer_sphere, self.metropolis,
                self.conftest_check_same_minimum, self.seeds))
            self.direct_k_walkers[k_index].step_choice_equilibration()

    def one_iteration(self):
        self.continue_pt_walks()
        self.volume = self.volume_computer.compute_volume(self.direct_k_u2_means)
        self.evaluations = self.conftest_check_same_minimum.get_nfev()

    def continue_pt_walks(self):
        for i in xrange(len(self.direct_k_values)):
            m, v = self.direct_k_walkers[i].one_iteration_walk()
            self.direct_k_u2_means[i] = m
            self.direct_k_u2_variances[i] = v

class DirectKWalker(object):
    """
    Perform biased random walk in basin with spring constant k; setup is
    like for kmin, but with kmin = k > 0.
    """
    def __init__(self, k_index, k_value, pes_parameters, ti_parameters,
        potential, conftest_outer_sphere, metropolis,
        conftest_check_same_minimum, seeds, verbose=False):
        self.k_index = k_index
        self.k_value = k_value
        self.pes_parameters = pes_parameters
        self.ti_parameters = ti_parameters
        self.potential = potential
        self.conftest_outer_sphere = conftest_outer_sphere
        self.metropolis = metropolis
        self.conftest_check_same_minimum = conftest_check_same_minimum
        self.seeds = seeds
        self.verbose = verbose
        self.takestep = RandomCoordsDisplacement(self.seeds['seed_takestep'] + k_index,
            5, report_interval=500, factor=0.95, min_acc_ratio=0.2,
            max_acc_ratio=0.2, single=False,
            bdim=self.pes_parameters["nr_dimensions"])
        self.harmonic_potential = Harmonic(self.pes_parameters["origin"],
            k_value, bdim=self.pes_parameters["nr_dimensions"],
            com=self.ti_parameters["harmonic_com_flag"])
        self.action_record_displ_ki = RecordDisp2Histogram(self.pes_parameters["origin"],
                                                           self.pes_parameters["rattlers"],
                                                           self.pes_parameters["nr_dimensions"],
                                                           self.ti_parameters["hmin"],
                                                           self.ti_parameters["hmax"],
                                                           self.ti_parameters["binsize"],
                                                           self.ti_parameters["equilibration_steps"],
                                                           fix_com=self.ti_parameters["harmonic_com_flag"])
        self.ki_run = GaussianBenchmarkKminRun(pot_optimizer=self.potential,
                                               origin=self.pes_parameters["origin"],
                                               conftest_outer_sphere=self.conftest_outer_sphere,
                                               conftest_check_same_minimum=self.conftest_check_same_minimum,
                                               action_record_displ=self.action_record_displ_ki,
                                               adjustf_niter=self.ti_parameters["adjustf_niter"],
                                               pt_eq_niter=self.ti_parameters["pt_eq_niter"],
                                               equilibration_steps=self.ti_parameters["equilibration_steps"],
                                               metropolis=self.metropolis,
                                               takestep=self.takestep,
                                               potential=self.harmonic_potential,
                                               niter=self.ti_parameters["kmin_niter"],
                                               nparticles=1)
        self.ki_run.set_control(self.k_value)

    def step_choice_equilibration(self):
        """
        Adapt stepsize to obtain around 20% acceptance probability and
        run equilibration steps with that stepsize. This is done here
        such that later each iteration can simply add a fixed number of
        Monte Carlo steps to the biased walk.
        """
        if self.verbose:
            print("step_choice_equilibration: number of MC iterations before:", self.ki_run.get_iterations_count())
        self.ki_run.niter = self.ti_parameters["equilibration_steps"]
        self.ki_run.run()
        if self.verbose:
            print("equilibration_steps", self.ti_parameters["equilibration_steps"])
            print("self.k_value", self.k_value)
            print("self.takestep.get_stepsize()", self.takestep.get_stepsize())
            print("self.ki_run.get_accepted_fraction()", self.ki_run.get_accepted_fraction())
            print("self.action_record_displ_ki.get_count()", self.action_record_displ_ki.get_count())
            print("step_choice_equilibration: number of MC iterations after:", self.ki_run.get_iterations_count())
            print("self.conftest_check_same_minimum.get_nfev()", self.conftest_check_same_minimum.get_nfev())
            print("------")

    def one_iteration_walk(self):
        if self.verbose:
            print("one_iteration_walk: number of MC iterations before:", self.ki_run.get_iterations_count())
        self.ki_run.niter = self.ti_parameters["nr_samples_increment"]
        self.ki_run.run()
        if self.verbose:
            print("self.action_record_displ_ki.get_count()", self.action_record_displ_ki.get_count())
            print("one_iteration_walk: number of MC iterations after:", self.ki_run.get_iterations_count())
            print("self.conftest_check_same_minimum.get_nfev()", self.conftest_check_same_minimum.get_nfev())
        return self.ki_run.get_displ2_kmin()


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
        self.ini_evals = self.ti_engine.ini_evals

    def get_method_label(self):
        return "ti"

    def get_evaluations_volume_one_iteration(self):
        self.ti_engine.one_iteration()
        evaluations = self.ti_engine.evaluations
        volume = self.ti_engine.volume
        return evaluations, volume

def run_ti(ls_basin_label):
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
    """
    Execute ti basin volume computation for large or small basin.

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
    ti_parameters = dict([("nr_samples_increment", 1),
        ("ktarget", 0.9), ("knavg", 100), ("ktol", 0.05), ("hmin", 0),
        ("hmax", 1), ("binsize", 0.005), ("harmonic_com_flag", False),
        ("kmax_niter", 2e4), ("kmax_avgcount", 1e4),
        ("adjustf_niter", 4e4), ("pt_eq_niter", 4e4),
        ("kmin_niter", 1e5), ("nprocs", 7)])
    potential_dir = os.path.join(os.getcwd(), "potentials")
    ls_basin_results_dir = os.path.join(os.getcwd(), ls_basin_label + "_basin_results")
    for nr_gaussians in [5]:
        pes_parameters = dict([("csm_dtol", 1),
            ("nr_dimensions", nr_dimensions),
            ("radius_container", 10)])
        run_computer(potential_dir, ls_basin_results_dir, ls_basin_label,
            nr_gaussians, nr_dimensions, sample_index, TIComputer,
            opt_parameters, pes_parameters, vol_parameters,
            ti_parameters)

if __name__ == "__main__":
    run_ti("large")
