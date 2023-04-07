from __future__ import division, print_function
from builtins import range
import numpy as np
import sys

from pele.potentials import Harmonic
from mcpele.monte_carlo import _BaseMCRunner, NullPotential, UniformSphericalSampling
from basinvolume.monte_carlo import RecordDisplacementTimeseries, CheckHyperCubicContainer, CheckHyperSphericalContainer, RecordStepsTimeseries, RecordDisp2Histogram
from mcpele.monte_carlo import MetropolisTest, RandomCoordsDisplacement
from basinvolume.monte_carlo import SampleUniformSphereGaussian
from mcpele.monte_carlo import SampleGaussian
from basinvolume.monte_carlo import Findk
from basinvolume.utils import write_2d_array_to_hdf5

try:
    from mcpele.monte_carlo import ConfTestOR
    from mcpele.monte_carlo import RecordCoordsTimeseries
except Exception as e:
    print(e)

#for plotting histogram
from itertools import cycle
from scipy.integrate import quad

try:
    import matplotlib.pyplot as plt
    #more stuff for plotting histogram and comparing to prediction
    #######################SET LATEX OPTIONS###################
    plt.rc('text', usetex=False) #True = bugs on the cluster!
    plt.rc('font',**{'family':'serif','serif':['Computer Modern']})
    #rc('text.latex',preamble=r'\usepackage{times}')
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['xtick.major.pad'] = 8
    plt.rcParams['ytick.major.pad'] = 8
    ##########################################################
    ####SET COLOUR MAP######
    cm = plt.get_cmap('Dark2')
    ########################
    #####################LINE STYLE CYCLER####################
    lines = ["-","--","-."]
    linecycler = cycle(lines)
    color_cycle=[cm(1. * i / 6) for i in range(6)]
    ##########################################################
except ImportError as err:
    print(err)


class HypercubeMCrunner(_BaseMCRunner):
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 sidelength=1, k=1.0, acceptance=0.2, adjustf=0.9,
                 hmin=0, hmax=1, hbinsize=0.001,
                 adjustf_niter=1e4, adjustf_navg=100, pt_eq_niter=0,
                 ts_niter=None, ts_freq=1, seeds=None,
                 record_steps_timeseries=False, record_steps_timeseries_every=[1],
                 record_trajectory=False,
                 record_trajectory_npoints=1e4,
                 single=False, record_histogram=False):
        #construct base class
        super(HypercubeMCrunner, self).__init__(potential, full_coords, temperature, niter)

        self.nparticles = 1
        self.bdim = len(full_coords)
        self.ndof = self.bdim
        self.origin = np.array(origin)
        self.red_origin = origin #necessary for pt
        self.rattlers = np.ones(self.bdim)
        self.sidelength = sidelength
        self.set_control(k)
        self.equilibration_steps = adjustf_niter + pt_eq_niter
        self.adjustf_niter = adjustf_niter
        # actions parameters
        if ts_niter is None:
            ts_niter = niter
        self.ts_niter = ts_niter
        self.ts_freq = ts_freq
        self.record_trajectory = record_trajectory
        self.record_trajectory_npoints = record_trajectory_npoints
        self.record_steps_timeseries = record_steps_timeseries
        self.record_steps_timeseries_every = record_steps_timeseries_every
        self.record_histogram = record_histogram
        self.hmin = hmin
        self.hmax = hmax
        self.hbinsize = hbinsize
        # takestep paramters
        self.adjustf_navg = adjustf_navg
        self.adjustf = adjustf
        self.acceptance = acceptance
        self.single = single
        print(self.sidelength)
        #compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max),
                    seed_metropolis=np.random.randint(i32max))
        self.seeds = seeds

        #set up pele:MC
        self._set_takestep(stepsize)
        self._set_accept_tests()
        self._set_conf_tests()
        self._set_actions()
        self._set_report_steps()

    def _set_takestep(self, stepsize):
        self.takestep = RandomCoordsDisplacement(self.seeds['seed_takestep'], stepsize,
                                                report_interval=self.adjustf_navg,
                                                factor=self.adjustf, min_acc_ratio=self.acceptance,
                                                max_acc_ratio=self.acceptance, single=self.single,
                                                nparticles=self.nparticles, bdim=self.bdim)
        self.set_takestep(self.takestep)
        
    def _set_accept_tests(self):
        self.metropolis = MetropolisTest(self.seeds['seed_metropolis'])
        self.add_accept_test(self.metropolis) #metropolis uses the harmonic potential

    def _set_conf_tests(self):
        self.conftest = ConfTestOR()
        conftest = CheckHyperCubicContainer(np.zeros(self.ndof), self.sidelength, self.bdim)
        self.conftest.add_test(conftest)
        self.add_late_conf_test(self.conftest)

    def _set_actions(self):
        self.action_record_displ = RecordDisplacementTimeseries(self.origin, self.bdim, self.ts_niter,
                                                                self.ts_freq, fix_com=False)
        self.add_action(self.action_record_displ)
        if self.record_histogram:
            self.binsize = self.hbinsize
            self.histogram = RecordDisp2Histogram(self.origin, self.rattlers, self.bdim, self.hmin, self.hmax,
                                                  self.binsize, self.equilibration_steps, fix_com=False)
            self.add_action(self.histogram)
        if self.record_trajectory:
            rte = max(int((self.niter-self.equilibration_steps)/self.record_trajectory_npoints),1)
            self.record_trajectory = RecordCoordsTimeseries(self.ndim,
                                                            record_every=rte,
                                                            eqsteps=self.equilibration_steps)
            self.add_action(self.record_trajectory)
        if self.record_steps_timeseries:
            self.steps_timeseries_list = []
            self.record_steps_timeseries_every = self.record_steps_timeseries_every
            for freq in self.record_steps_timeseries_every:
                self.steps_timeseries_list.append(RecordStepsTimeseries(self.origin, self.rattlers, self.bdim, self.ts_niter, freq))
            for action in self.steps_timeseries_list:
                self.add_action(action)
        
    def _set_report_steps(self):
        self.set_report_steps(self.adjustf_niter)


    def set_control(self, c, reset=True):
        """set temperature, canonical control parameter"""
        self.k = c
        self.potential.set_k(c)
        if reset:
            self.reset_energy()

    def get_stepsize(self):
        return self.takestep.get_stepsize()

    def run_kmin(self):
        print("run kmin")
        print("coords initial", self.get_coords())
        self.set_print_progress()
        self.run()
        print("coords final", self.get_coords())

    def get_displ2_kmin(self):
        print("recorded steps for displ2", self.histogram.get_count())
        return self.histogram.get_mean_variance()

    def dump_timeseries(self, fname, clear=True):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.action_record_displ.get_time_series())
        np.savetxt(fname, timeseries)
        if clear:
            self.action_record_displ.clear()
        return timeseries

    def get_timeseries(self, clear=False):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.action_record_displ.get_time_series())
        if clear:
            self.action_record_displ.clear()
        return timeseries

    def check_convergence(self, nr_steps_to_check=10000, rel_std_threshold=0.05):
        return self.action_record_displ.check_convergence(nr_steps_to_check=nr_steps_to_check,
                                                   rel_std_threshold=rel_std_threshold)

    def get_mean_variance_coordinate_vector(self):
        """
        returns the average coordinate vector from the sampling and the elementwise variance
        """
        mean_coord, var_coord = self.record_trajectory.get_mean_variance_time_series()
        return mean_coord, var_coord

    def dump_trajectory(self, fname, clear=True):
        """write time series to fname, returns the timeseries"""
        trajectory = self.get_trajectory()
        write_2d_array_to_hdf5(trajectory, 'trajectory', fname)
        if clear:
            self.clear_trajectory()
        return trajectory

    def get_trajectory(self):
        trajectory = self.record_trajectory.get_time_series()
        return trajectory

    def clear_trajectory(self):
        self.record_trajectory.clear()
        
    def get_complete_state(self):
        return BV_MCRunner_State(
            coords=self.get_coords(), energy=self.get_energy(), k=self.k,
            stepsize=self.takestep.get_stepsize(), counters=self.get_counters(),
            takestep_count=self.takestep.get_count(),
            step_adaptation_counters=self.takestep.get_adaptation_counters())

    def set_complete_state(self, mcrunner_state):
        self.set_config(mcrunner_state.coords, mcrunner_state.energy)
        self.set_control(mcrunner_state.k, reset=False)
        self.set_counters(mcrunner_state.counters)
        self.takestep.set_stepsize(mcrunner_state.stepsize)
        self.takestep.set_count(mcrunner_state.takestep_count)
        self.takestep.set_adaptation_counters(mcrunner_state.step_adaptation_counters)


class HypercubeFindkMCrunner(_BaseMCRunner):
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 sidelength=1, ktarget = 0.9, knavg=500, ktol=0.05, avgcount=1e6,
                 hmin=0, hmax=1, hbinsize=0.001, seeds=None):
        #construct base class
        super(HypercubeFindkMCrunner, self).__init__(potential, full_coords, temperature, niter)

        self.nparticles = 1
        self.bdim = len(full_coords)
        self.ndof = self.bdim
        self.origin = np.array(origin)
        self.rattlers = np.ones(self.bdim)
        self.sidelength = sidelength

        #compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max))
        self.seeds = seeds

        #findk parameters
        self.avgcount = avgcount
        self.ktarget = ktarget
        self.knavg=knavg
        self.ktol=ktol
        self.hmin = hmin
        self.hmax = hmax
        self.hbinsize = hbinsize

        #construct test/action classes
        self.takestep = SampleGaussian(self.seeds['seed_takestep'], stepsize, self.origin)

        self.conftest = ConfTestOR()
        conftest = CheckHyperCubicContainer(np.zeros(self.ndof), self.sidelength, self.bdim)
        self.conftest.add_test(conftest)

        self.findk = Findk(self.origin, self.rattlers, self.bdim, self.avgcount, self.ktarget,
                           self.knavg, self.ktol, self.hmin, self.hmax, self.hbinsize,
                           fix_com=False)

        #set up pele:MC
        self.set_takestep(self.takestep)
        self.add_conf_test(self.conftest)
        self.add_action(self.findk)

    def set_control(self, c):
        """set k"""
        print("WARNING: findk set control is not defined, spring constant is set through stepsize", file=sys.stderr)

    def get_stepsize(self):
        return self.takestep.get_stepsize()

    def get_status(self):
        """
        overloading the base class method to include stepsize
        """
        status = super(HypercubeFindkMCrunner, self).get_status()
        status.stepsize = self.get_stepsize()
        return status

    def get_k(self):
        """in findk, potential is pretty much fictitious, k is adjusted through the stepsize"""
        stepsize = self.get_stepsize()
        k = 1.0 / (stepsize * stepsize)
        #k = self.bdim*len(self.hs_radii)/(stepsize*stepsize)##############
        return k

    def get_entries(self):
        return self.findk.get_entries()

    def show_histogram(self):
        hist = self.findk.get_histogram()
        val = np.array([i * self.hbinsize for i in range(len(hist))]) + 0.5 * self.hbinsize
        plt.hist(val, weights=hist, bins=len(hist), density=True, stacked=True)
        ###analytical
        k = self.get_k()
        and2 = val[:-1]**(self.ndof/2 - 1) * np.exp(-0.5 * k * val[:-1]**1)
        norm = and2.sum() * self.hbinsize
        and2/=norm
        plt.plot(val[:-1], and2, linewidth=2.5, ls='--', color=color_cycle[-1])
        plt.savefig('findk_histogram.eps')
        plt.show()

class BV_MCRunner_State(object):
    """
    This class saves the state of an BV_MCrunner in a NumPy array
    """
    def __init__(self, state=None, coords=None, energy=0., k=0.,
                 stepsize=0., counters=None, takestep_count=0,
                 step_adaptation_counters=None):
        if state is None:
            self.coords = coords
            self.energy = energy
            self.k = k
            self.stepsize = stepsize
            self.counters = counters
            self.takestep_count = takestep_count
            self.step_adaptation_counters = step_adaptation_counters
        else:
            self._set_state(state)

    def _set_state(self, state):
        self.coords = state.coords
        self.energy = state.energy
        self.k = state.k
        self.stepsize = state.stepsize
        self.counters = state.counters
        self.takestep_count = state.takestep_count
        self.step_adaptation_counters = state.step_adaptation_counters

class HypercubeInnerSphereMCrunner(_BaseMCRunner):
    """
    """
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 sidelength=1, hmin=0, hmax=1, hbinsize=0.001, ts_niter=None, ts_freq=1, seeds=None,
                 record_histogram=False, gaussian_step=True):
        #construct base class
        super(HypercubeInnerSphereMCrunner, self).__init__(potential, full_coords, temperature, niter)

        self.nparticles = 1
        self.bdim = len(full_coords)
        self.ndof = self.bdim
        self.origin = np.array(origin)
        self.red_origin = origin #necessary for pt
        self.rattlers = np.ones(self.bdim)
        self.sidelength = sidelength
        self.k = 1.0 / (stepsize * stepsize)
        self.equilibration_steps = 0
        self.gaussian_step = gaussian_step
        if ts_niter is None:
            ts_niter = niter

        #compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max))
        self.seeds = seeds

        #construct test/action classes
        if record_histogram:
            self.binsize = hbinsize
            self.histogram = RecordDisp2Histogram(self.origin, self.rattlers, self.ndof, hmin, hmax,
                                                  self.binsize, self.equilibration_steps, fix_com=False)
            self.add_action(self.histogram)

        self.conftest = ConfTestOR()
        conftest = CheckHyperCubicContainer(np.zeros(self.ndof), self.sidelength, self.bdim)
        self.conftest.add_test(conftest)
        #conftest2 = CheckHyperSphericalContainer(np.array(self.origin), sidelength, self.bdim)
        #self.conftest.add_test(conftest2)

        self.time_series = RecordDisplacementTimeseries(self.origin, self.ndof, ts_niter, ts_freq, fix_com=False)

        self.set_report_steps(0)
        if self.gaussian_step == True:
            self.takestep = SampleUniformSphereGaussian(self.seeds['seed_takestep'], stepsize, self.origin)
        else:
            self.takestep = UniformSphericalSampling(self.seeds['seed_takestep'], stepsize, origin=self.origin)

        #set up pele:MC
        self.set_takestep(self.takestep)
        self.add_conf_test(self.conftest)
        self.add_action(self.time_series)

    def set_control(self, c):
        """set k"""
        print("WARNING: set control is not defined, spring constant is set through stepsize", file=sys.stderr)

    def get_stepsize(self):
        return self.takestep.get_stepsize()

    def get_k(self):
        """potential is pretty much fictitious, k is adjusted through the stepsize"""
        stepsize = self.get_stepsize()
        k = 1.0 / (stepsize * stepsize)
        #k = self.bdim*len(self.hs_radii)/(stepsize*stepsize)##############
        return k

    def get_status(self):
        """
        overloading the base class method to include stepsize
        """
        status = super(HypercubeInnerSphereMCrunner, self).get_status()
        status.stepsize = self.get_stepsize()
        return status

    def dump_histogram(self, fname):
        """write histogram to fname"""
        Emin, Emax = self.histogram.get_bounds_val()
        histl = self.histogram.get_histogram()
        hist = np.array(histl)
        Energies, step = np.linspace(Emin, Emax, num=len(hist), endpoint=False, retstep=True)
        Energies += 0.5 * step
        assert(abs(step - self.binsize) < self.binsize / 100)
        np.savetxt(fname, np.column_stack((Energies,hist)), delimiter='\t')
        mean, variance = self.histogram.get_mean_variance()
        return mean, variance

    def dump_timeseries(self, fname, clear=True):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.time_series.get_time_series())
        np.savetxt(fname, timeseries)
        if clear:
            self.time_series.clear()
        return timeseries

    def get_timeseries(self):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.time_series.get_time_series())
        return timeseries

    def check_convergence(self, nr_steps_to_check=10000, rel_std_threshold=0.05):
        return self.time_series.check_convergence(nr_steps_to_check=nr_steps_to_check,
                                                   rel_std_threshold=rel_std_threshold)

    def show_histogram(self):
        hist = self.histogram.get_histogram()
        val = np.array([i * self.binsize for i in range(len(hist))]) + 0.5 * self.binsize
        plt.hist(val, weights=hist, bins=len(hist))
        plt.show()

    def show_histogram_analytical(self, output_directory=''):
        """
        shows the histogram against the analytical curve when k=kmax
        this function is useful for testing
        """
        plt.clf()
        timeseries = self.get_timeseries()
        n, bins, patch = plt.hist(timeseries, bins=500, range=(np.amin(timeseries), np.amax(timeseries)), density=True, stacked=True,
                           alpha=0.4, edgecolor=color_cycle[0], color=color_cycle[0])
        ###analytical
        k = self.k
        #and2 = np.exp(-0.5 * k * bincenters) * np.sqrt(k) / np.sqrt(2*np.pi*bincenters)
        and2 = n[0] * np.exp(-0.5 * k * bins[:-1]**2)
        plt.plot(bins[:-1], and2, linewidth=2.5, ls='--', color=color_cycle[-1])
        #plt.xlim(0,1)
        plt.xlabel(r'$|{\bf r}-{\bf r}_0|^2$')
        plt.ylabel(r'frequency $\times 10$')
        plt.tight_layout()
        plt.savefig(output_directory+'/innersphere_histogram.eps')
        plt.show()

if __name__ == "__main__":
    #to run harmonic potential go to tests

    import time

    ndim = 100
    origin = np.zeros(ndim)
    potential = NullPotential()
    #build start configuration
    full_coords = np.array(origin)
    k = 2 #0.4261331121440447 # k=25
    stepsize = np.sqrt(1.0 / k)
    if False:
        print("Find k test: \n\n\n")
        test = HypercubeFindkMCrunner(potential, full_coords, 1, stepsize, int(1e8), origin, sidelength=1)
        start = time.time()
        test.run()
        end = time.time()
        print(end - start)
    if False:
        print("MC test: \n\n\n")
        potential = Harmonic(origin, k, bdim=ndim, com=False)
        test = HypercubeMCrunner(potential, full_coords, 1, stepsize, int(1e5), origin, sidelength=1, record_histogram=True)
        start = time.time()
        test.run()
        end = time.time()
        print('displ2_kmin', test.get_displ2_kmin())
        print(end - start)
    if True:
        print("Inner sphere test: \n\n\n")
        test = HypercubeInnerSphereMCrunner(potential, full_coords, 1, stepsize, int(1e5), origin, sidelength=1, record_histogram=True)
        start = time.time()
        test.run()
        end = time.time()
        print(end - start)
        test.show_histogram()
        test.show_histogram_analytical()
