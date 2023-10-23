from __future__ import division, print_function
from builtins import range
import numpy as np
from pele.potentials import Harmonic
from mcpele.monte_carlo import NullPotential
from basinvolume.geometry import _BaseGeomMCrunner
from basinvolume.monte_carlo import RecordCloudDisplacementTimeseries, CheckHyperCubicContainer, RecordAcceptanceHistogram
from basinvolume.monte_carlo import CheckHyperSphericalContainer, CheckExponentiallyDecayingProfile, CheckPowerDecayingProfile
from mcpele.monte_carlo import CloudTest, RandomCoordsDisplacement, RecordCloudR2, RecordCloudDropsTimeseries
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

class HyperElemOracleMCrunner(_BaseGeomMCrunner):
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 geometry="sphere", geom_params=[1.],
                 cloud_radius=1., nr_cloud_points=10,
                 hypersphere_radius=1.0,
                 k=1.0, acceptance=0.2, adjustf=0.9,
                 hmin=0, hmax=1, hbinsize=0.001,
                 report_steps=0, adjustf_navg=100, pt_eq_niter=0,
                 ts_niter=None, ts_freq=1, seeds=None, single=False):
        assert report_steps == 0  # there should be no step adaptation for now
        self.nparticles = 1
        self.acceptance = acceptance
        self.adjustf = adjustf
        self.adjustf_navg = adjustf_navg
        self.ts_niter = niter if ts_niter is None else ts_niter
        self.ts_freq = ts_freq
        self.single = single
        self.cloud_radius, self.nr_cloud_points = self._get_cloud_params(cloud_radius, nr_cloud_points)
        super(HyperElemOracleMCrunner, self).__init__(potential, full_coords, temperature, stepsize, niter,
                                                      origin, geometry=geometry, geom_params=geom_params,
                                                      report_steps=report_steps, pt_eq_niter=pt_eq_niter,
                                                      k=k, hmin=hmin, hmax=hmax, hbinsize=hbinsize, seeds=seeds)

    def _get_cloud_params(self, cloud_radius, nr_cloud_points):
        return cloud_radius, nr_cloud_points

    def _get_oracle(self):
        if self.geometry == "cube":
            sidelength = self.geom_params[0]
            oracle = CheckHyperCubicContainer(np.array(self.origin), sidelength, self.bdim)
        elif self.geometry == "sphere":
            radius = self.geom_params[0]
            oracle = CheckHyperSphericalContainer(np.array(self.origin), radius, self.bdim)
        elif self.geometry == "cube_exp_decay":
            sidelength = self.geom_params[0]
            decay_length = self.geom_params[1]
            oracle = CheckExponentiallyDecayingProfile(np.array(self.origin), sidelength, decay_length,
                                                       cubic=True, seed=self.seeds['seed_oracle'])
        elif self.geometry == "sphere_exp_decay":
            radius = self.geom_params[0]
            decay_length = self.geom_params[1]
            oracle = CheckExponentiallyDecayingProfile(np.array(self.origin), radius, decay_length,
                                                       cubic=False, seed=self.seeds['seed_oracle'])
        elif self.geometry == "sphere_pow_decay":
            radius = self.geom_params[0]
            exponent = self.geom_params[1]
            oracle = CheckPowerDecayingProfile(np.array(self.origin), radius, exponent, seed=self.seeds['seed_oracle'])
        else:
            raise NotImplementedError
        return oracle

    def _set_accept_tests(self):
        self.oracle = self._get_oracle()
        self.cloud_test = CloudTest(self.seeds['seed_metropolis'], self.seeds['seed_cloud'],
                                    self.nr_cloud_points, self.cloud_radius, self.potential)
        self.cloud_test.add_conf_test(self.oracle)
        self.add_accept_test(self.cloud_test)

    def _set_actions(self):
        self.action_record_drops = RecordCloudDropsTimeseries(self.origin, self.ts_freq, self.equilibration_steps)
        self.action_record_accept_hist = RecordAcceptanceHistogram(self.origin, self.hmin, self.hmax,
                                                                   (self.hmax - self.hmin) / self.hbinsize,
                                                                   self.equilibration_steps)
        self.cloud_measure_r2 = RecordCloudR2(self.equilibration_steps, self.origin)
        self.add_action(self.cloud_measure_r2)
        self.add_action(self.action_record_drops)
        self.add_action(self.action_record_accept_hist)

    def _set_takestep(self, stepsize):
        self.takestep = RandomCoordsDisplacement(self.seeds['seed_takestep'], stepsize,
                                                 report_interval=self.adjustf_navg,
                                                 factor=self.adjustf, min_acc_ratio=self.acceptance,
                                                 max_acc_ratio=self.acceptance,single=self.single,
                                                 nparticles=self.nparticles, bdim=self.bdim)
        self.set_takestep(self.takestep)

    def _set_conf_tests(self):
        pass

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
        return self.cloud_measure_r2.get_mean(), self.cloud_measure_r2.get_variance()

    def dump_timeseries(self, fname, clear=True):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.action_record_drops.get_time_series())
        write_2d_array_to_hf5(timeseries, 'drops_ts', fname)
        if clear:
            self.action_record_drops.clear()
        return timeseries

    def get_timeseries(self):
        """write time series to fname, returns the timeseries"""
        drops_timeseries = np.array(self.action_record_drops.get_time_series())
        return drops_timeseries

if __name__ == "__main__":
    #to run harmonic potential go to tests
    
    import time
    
    ndim = 2
    origin = np.zeros(ndim)
    potential = NullPotential()
    #build start configuration
    full_coords = np.array(origin)
    k=0.
    geometry="sphere"
    if True:
        stepsize=0.5
        potential = Harmonic(origin, 0, bdim=ndim, com=False)
        test = HyperElemOracleMCrunner(potential, full_coords, 1, stepsize, int(1e5), origin,
                                           geometry=geometry, geom_params=[1], pt_eq_niter=1e4,
                                           cloud_radius=0.5, nr_cloud_points=100, k=k)
        start = time.time()
        test.run()
        end = time.time()
        print('displ2_kmin', test.get_displ2_kmin())
        print(end - start)
        
