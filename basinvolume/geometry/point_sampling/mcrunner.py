from __future__ import division, print_function
from builtins import range
import numpy as np
import sys

from pele.potentials import Harmonic
from mcpele.monte_carlo import NullPotential, UniformSphericalSampling
from basinvolume.geometry import _BaseGeomMCrunner
from basinvolume.monte_carlo import RecordDisplacementTimeseries, CheckHyperCubicContainer, RecordStepsTimeseries, RecordDisp2Histogram
from basinvolume.monte_carlo import CheckHyperSphericalContainer, CheckExponentiallyDecayingProfile, CheckPowerDecayingProfile
from mcpele.monte_carlo import MetropolisTest, RandomCoordsDisplacement
from basinvolume.monte_carlo import SampleUniformSphereGaussian
from mcpele.monte_carlo import SampleGaussian
from basinvolume.monte_carlo import Findk
from basinvolume.utils import write_2d_array_to_hf5

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
    plt.rc('text', usetex=False)
    # plt.rc('font',**{'family':'serif','serif':['Computer Modern']})
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


class HyperElemMCrunner(_BaseGeomMCrunner):
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 geometry="cube", geom_params=[1], k=1.0, acceptance=0.2, adjustf=0.9,
                 hmin=0, hmax=1, hbinsize=0.001,
                 report_steps=1e4, adjustf_navg=100, pt_eq_niter=0,
                 ts_niter=None, ts_freq=1, seeds=None, 
                 record_steps_timeseries=False, record_steps_timeseries_every=[1],
                 record_trajectory=False,
                 record_trajectory_npoints=1e4, 
                 single=False, record_histogram=False):
        self.nparticles=1
        self.acceptance = acceptance
        self.adjustf = adjustf
        self.adjustf_navg = adjustf_navg
        self.ts_niter = niter if ts_niter is None else ts_niter
        self.ts_freq = ts_freq
        self.single = single
        self.record_steps_timeseries = record_steps_timeseries
        self.record_steps_timeseries_every = record_steps_timeseries_every
        self.record_trajectory = record_trajectory
        self.record_trajectory_npoints = record_trajectory_npoints
        self.record_histogram = record_histogram
        super(HyperElemMCrunner, self).__init__(potential, full_coords, temperature, stepsize, niter, origin,
                                                geometry=geometry, geom_params=geom_params,
                                                report_steps=report_steps, pt_eq_niter=pt_eq_niter,
                                                k=k, hmin=hmin, hmax=hmax, hbinsize=hbinsize, seeds=seeds)

    def _set_takestep(self, stepsize):
        self.takestep = RandomCoordsDisplacement(self.seeds['seed_takestep'], stepsize, report_interval=self.adjustf_navg,
                                                 factor=self.adjustf, min_acc_ratio=self.acceptance, max_acc_ratio=self.acceptance,
                                                 single=self.single, nparticles=self.nparticles, bdim=self.bdim)
        self.set_takestep(self.takestep)

    def _set_accept_tests(self):
        self.metropolis = MetropolisTest(self.seeds['seed_metropolis'])
        self.add_accept_test(self.metropolis)  # metropolis uses the harmonic potential

    def _set_conf_tests(self):
        if self.geometry == "cube":
            sidelength = self.geom_params[0]
            self.conftest = CheckHyperCubicContainer(np.array(self.origin), sidelength, self.bdim)
        elif self.geometry == "sphere":
            radius = self.geom_params[0]
            self.conftest = CheckHyperSphericalContainer(np.array(self.origin), radius, self.bdim)
        elif self.geometry == "cube_exp_decay":
            sidelength = self.geom_params[0]
            decay_length = self.geom_params[1]
            self.conftest = CheckExponentiallyDecayingProfile(np.array(self.origin), sidelength, decay_length, cubic=True)
        elif self.geometry == "sphere_exp_decay":
            radius = self.geom_params[0]
            decay_length = self.geom_params[1]
            self.conftest = CheckExponentiallyDecayingProfile(np.array(self.origin), radius, decay_length, cubic=False)
        elif self.geometry == "sphere_pow_decay":
            radius = self.geom_params[0]
            exponent = self.geom_params[1]
            self.conftest = CheckPowerDecayingProfile(np.array(self.origin), radius, exponent)
        else:
            raise NotImplementedError
        # self.conftest = ConfTestOR()
        # self.conftest.add_test(conftest)
        # self.conftest.add_test(conftest2)
        self.add_late_conf_test(self.conftest)

    def _set_actions(self):
        self.action_record_displ = RecordDisplacementTimeseries(self.origin, self.bdim, self.ts_niter,
                                                                self.ts_freq, fix_com=False)
        self.add_action(self.action_record_displ)
        if self.record_histogram:
            self.histogram = RecordDisp2Histogram(self.origin, self.rattlers, self.bdim, self.hmin, self.hmax,
                                                  self.hbinsize, self.equilibration_steps, fix_com=False)
            self.add_action(self.histogram)
        if self.record_trajectory:
            rte = max(int((self.niter - self.equilibration_steps) / self.record_trajectory_npoints), 1)
            self.record_trajectory = RecordCoordsTimeseries(self.ndim,
                                                            record_every=rte,
                                                            eqsteps=self.equilibration_steps)
            self.add_action(self.record_trajectory)
        if self.record_steps_timeseries:
            self.steps_timeseries_list = []
            self.record_steps_timeseries_every = self.record_steps_timeseries_every
            for freq in self.record_steps_timeseries_every:
                self.steps_timeseries_list.append(
                    RecordStepsTimeseries(self.origin, self.rattlers, self.bdim, self.ts_niter, freq))
            for action in self.steps_timeseries_list:
                self.add_action(action)

    def set_control(self, c):
        """set temperature, canonical control parameter"""
        self.k = c
        self.potential.set_k(c)
        self.reset_energy()

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
    
    def get_timeseries(self):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.action_record_displ.get_time_series())
        return timeseries
    
    def get_mean_variance_coordinate_vector(self):
        """
        returns the average coordinate vector from the sampling and the elementwise variance
        """
        mean_coord, var_coord = self.record_trajectory.get_mean_variance_time_series()
        return mean_coord, var_coord
    
    def dump_trajectory(self, fname, clear=True):
        """write time series to fname, returns the timeseries"""
        trajectory = self.get_trajectory()
        write_2d_array_to_hf5(trajectory, 'trajectory', fname)
        if clear:
            self.clear_trajectory()
        return trajectory
    
    def get_trajectory(self):
        trajectory = self.record_trajectory.get_time_series()
        return trajectory
    
    def clear_trajectory(self):
        self.record_trajectory.clear()

class HyperElemFindkMCrunner(_BaseGeomMCrunner):
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 geometry="cube", geom_params=[1], ktarget = 0.9, knavg=500, ktol=0.05,
                 avgcount=1e6, hmin=0, hmax=1, hbinsize=0.001, seeds=None):
        self.nparticles=1
        self.avgcount = avgcount
        self.ktarget = ktarget
        self.knavg = knavg
        self.ktol = ktol
        super(HyperElemFindkMCrunner, self).__init__(potential, full_coords, temperature, stepsize, niter, origin,
                                                     geometry=geometry, geom_params=geom_params,
                                                     report_steps=0, pt_eq_niter=0, k=1,
                                                     hmin=hmin, hmax=hmax, hbinsize=hbinsize, seeds=seeds)
    def _set_takestep(self, stepsize):
        self.takestep = SampleGaussian(self.seeds['seed_takestep'], stepsize, self.origin)
        self.set_takestep(self.takestep)

    def _set_conf_tests(self):
        if self.geometry == "cube":
            sidelength = self.geom_params[0]
            self.conftest = CheckHyperCubicContainer(np.array(self.origin), sidelength, self.bdim)
        elif self.geometry == "sphere":
            radius = self.geom_params[0]
            self.conftest = CheckHyperSphericalContainer(np.array(self.origin), radius, self.bdim)
        elif self.geometry == "cube_exp_decay":
            sidelength = self.geom_params[0]
            decay_length = self.geom_params[1]
            self.conftest = CheckExponentiallyDecayingProfile(np.array(self.origin), sidelength, decay_length,
                                                              cubic=True, seed=self.seeds['seed_oracle'])
        elif self.geometry == "sphere_exp_decay":
            radius = self.geom_params[0]
            decay_length = self.geom_params[1]
            self.conftest = CheckExponentiallyDecayingProfile(np.array(self.origin), radius, decay_length,
                                                              cubic=False, seed=self.seeds['seed_oracle'])
        elif self.geometry == "sphere_pow_decay":
            radius = self.geom_params[0]
            exponent = self.geom_params[1]
            self.conftest = CheckPowerDecayingProfile(np.array(self.origin), radius, exponent,
                                                      seed=self.seeds['seed_oracle'])
        else:
            raise NotImplementedError
        # self.conftest = ConfTestOR()
        # self.conftest.add_test(conftest)
        # self.conftest.add_test(conftest2)
        self.add_late_conf_test(self.conftest)

    def _set_actions(self):
        self.findk = Findk(self.origin, self.rattlers, self.bdim, self.avgcount, self.ktarget,
                           self.knavg, self.ktol, self.hmin, self.hmax, self.hbinsize,
                           fix_com=False)
        self.add_action(self.findk)

    def _set_accept_tests(self):
        pass

    def set_control(self, c):
        """set k"""
        print("WARNING: findk set control is not defined, spring constant is set through stepsize",
              file=sys.stderr)

    def get_k(self):
        """in findk, potential is pretty much fictitious, k is adjusted through the stepsize"""
        stepsize = self.get_stepsize()
        k = 1.0 / (stepsize * stepsize)
        #k = self.bdim*len(self.hs_radii)/(stepsize*stepsize)##############
        return k

    def get_entries(self):
        return self.findk.get_entries()

class HyperElemInnerSphereMCrunner(_BaseGeomMCrunner):
    """
    """
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 geometry="cube", geom_params=[1], hmin=0, hmax=1, hbinsize=0.001,
                 ts_niter=None, ts_freq=1, seeds=None, record_histogram=False,
                 gaussian_step=True):
        self.nparticles = 1
        self.ts_niter = niter if ts_niter is None else ts_niter
        self.ts_freq = ts_freq
        self.record_histogram = record_histogram
        self.k = 1.0 / (stepsize * stepsize)
        self.gaussian_step = gaussian_step
        super(HyperElemInnerSphereMCrunner, self).__init__(potential, full_coords, temperature, stepsize,
                                                           niter, origin, report_steps=0, pt_eq_niter=0,
                                                           geometry=geometry, geom_params=geom_params,
                                                           k=self.k, hmin=hmin, hmax=hmax, hbinsize=hbinsize, seeds=seeds)

    def _set_takestep(self, stepsize):
        if self.gaussian_step:
            self.takestep = SampleUniformSphereGaussian(self.seeds['seed_takestep'], stepsize, self.origin)
        else:
            self.takestep = UniformSphericalSampling(self.seeds['seed_takestep'], stepsize, origin=self.origin)
        self.set_takestep(self.takestep)

    def _set_accept_tests(self):
        pass

    def _set_conf_tests(self):
        if self.geometry == "cube":
            sidelength = self.geom_params[0]
            self.conftest = CheckHyperCubicContainer(np.array(self.origin), sidelength, self.bdim)
        elif self.geometry == "sphere":
            radius = self.geom_params[0]
            self.conftest = CheckHyperSphericalContainer(np.array(self.origin), radius, self.bdim)
        elif self.geometry == "cube_exp_decay":
            sidelength = self.geom_params[0]
            decay_length = self.geom_params[1]
            self.conftest = CheckExponentiallyDecayingProfile(np.array(self.origin), sidelength, decay_length,
                                                              cubic=True, seed=self.seeds['seed_oracle'])
        elif self.geometry == "sphere_exp_decay":
            radius = self.geom_params[0]
            decay_length = self.geom_params[1]
            self.conftest = CheckExponentiallyDecayingProfile(np.array(self.origin), radius, decay_length,
                                                              cubic=False, seed=self.seeds['seed_oracle'])
        elif self.geometry == "sphere_pow_decay":
            radius = self.geom_params[0]
            exponent = self.geom_params[1]
            self.conftest = CheckPowerDecayingProfile(np.array(self.origin), radius, exponent,
                                                      seed=self.seeds['seed_oracle'])
        else:
            raise NotImplementedError
        # self.conftest = ConfTestOR()
        # self.conftest.add_test(conftest)
        # self.conftest.add_test(conftest2)
        self.add_late_conf_test(self.conftest)

    def _set_actions(self):
        self.time_series = RecordDisplacementTimeseries(self.origin, self.ndof,
                                                        self.ts_niter, self.ts_freq, fix_com=False)
        self.add_action(self.time_series)
        if self.record_histogram:
            self.histogram = RecordDisp2Histogram(self.origin, self.rattlers, self.ndof, self.hmin, self.hmax,
                                                  self.hbinsize, self.equilibration_steps, fix_com=False)
            self.add_action(self.histogram)

    def set_control(self, c):
        """set k"""
        print("WARNING: set control is not defined, spring constant is set through stepsize", file=sys.stderr)

    def get_k(self):
        """potential is pretty much fictitious, k is adjusted through the stepsize"""
        stepsize = self.get_stepsize()
        k = 1.0 / (stepsize * stepsize)
        #k = self.bdim*len(self.hs_radii)/(stepsize*stepsize)##############
        return k

    def dump_histogram(self, fname):
        """write histogram to fname"""
        Emin, Emax = self.histogram.get_bounds_val()
        histl = self.histogram.get_histogram()
        hist = np.array(histl)
        Energies, step = np.linspace(Emin, Emax, num=len(hist), endpoint=False, retstep=True)
        Energies += 0.5 * step
        assert(abs(step - self.hbinsize) < self.hbinsize / 100)
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

    def show_histogram(self):
        hist = self.histogram.get_histogram()
        val = np.array([i * self.hbinsize for i in range(len(hist))]) + 0.5 * self.hbinsize
        plt.hist(val, weights=hist, bins=len(hist))
        plt.show()

    def show_histogram_analytical(self):
        """
        shows the histogram against the analytical curve when k=kmax
        this function is useful for testing
        """
        timeseries = self.get_timeseries()
        n, bins, patch = plt.hist(timeseries, bins=500, range=(np.amin(timeseries), np.amax(timeseries)), normed=True,
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
        plt.savefig('innersphere_histogram.eps')
        plt.show()

if __name__ == "__main__":
    #to run harmonic potential go to tests
    
    import time
    
    ndim = 2
    origin = np.zeros(ndim)
    potential = NullPotential()
    #build start configuration
    full_coords = np.array(origin)
    k=0.
    stepsize = np.sqrt(1.0 / k) if k != 0. else 0.5
    geometry="sphere"
    if False:       
        test = HyperElemFindkMCrunner(potential, full_coords, 1, stepsize, int(1e8), origin,
                                      geometry=geometry, geom_params=[1])
        start = time.time()
        test.run()
        end = time.time()
        print(end - start)
    if True:
        potential = Harmonic(origin, k, bdim=ndim, com=False)
        test = HyperElemMCrunner(potential, full_coords, 1, stepsize, int(1e5), origin,
                                 geometry=geometry, geom_params=[1], record_histogram=True, k=k)
        start = time.time()
        test.run()
        end = time.time()
        print('displ2_kmin', test.get_displ2_kmin())
        print(end - start)
        print(test.get_status())
    if False:
        test = HyperElemInnerSphereMCrunner(potential, full_coords, 1, stepsize, int(1e5), origin,
                                            geometry=geometry, geom_params=[1], record_histogram=True)
        start = time.time()
        test.run()
        end = time.time()
        print(end - start)
        test.show_histogram()
        test.show_histogram_analytical()
        
