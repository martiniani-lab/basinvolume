from __future__ import division, print_function
import numpy as np
import sys

from pele.potentials import Harmonic
from mcpele.monte_carlo import _BaseMCRunner, NullPotential
from basinvolume.monte_carlo import RecordDisplacementTimeseries, CheckHyperCubicContainer, CheckHyperSphericalContainer, RecordStepsTimeseries, RecordDisp2Histogram
from mcpele.monte_carlo import MetropolisTest, RandomCoordsDisplacement, RecordMeanCoordVector, RecordCoordsTimeseries
from basinvolume.monte_carlo import SampleUniformSphereGaussian
from mcpele.monte_carlo import SampleGaussian, ConfTestOR
from basinvolume.monte_carlo import Findk

#for plotting histogram
from itertools import cycle
from scipy.integrate import quad

try:
    import matplotlib.pyplot as plt
    #more stuff for plotting histogram and comparing to prediction
    #######################SET LATEX OPTIONS###################                            
    plt.rc('text', usetex=True)
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
    color_cycle=[cm(1. * i / 6) for i in xrange(6)]
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
        if ts_niter is None:
            ts_niter = niter
        print(self.sidelength)
        #compute seeds
        if not seeds:
            i32max = np.iinfo(np.int32).max
            seeds = dict(seed_takestep=np.random.randint(i32max),
                    seed_metropolis=np.random.randint(i32max))
        self.seeds = seeds
            
        
        self.conftest = ConfTestOR()
        conftest = CheckHyperCubicContainer(self.origin, self.sidelength, self.bdim)
        self.conftest.add_test(conftest) 
        #conftest2 = CheckHyperSphericalContainer(np.array(self.origin), sidelength, self.bdim)
        #self.conftest.add_test(conftest2)
        
        self.action_record_displ = RecordDisplacementTimeseries(self.origin, self.bdim, ts_niter, 
                                                                ts_freq, fix_com=False)
        self.metropolis = MetropolisTest(self.seeds['seed_metropolis'])
        
        self.set_report_steps(adjustf_niter)
        self.takestep = RandomCoordsDisplacement(self.seeds['seed_takestep'], stepsize, report_interval=adjustf_navg,
                                                  factor=adjustf, min_acc_ratio=acceptance, max_acc_ratio=acceptance,
                                                  single=single, nparticles=self.nparticles, bdim=self.bdim)
        
        if record_histogram:
            self.binsize = hbinsize
            self.histogram = RecordDisp2Histogram(self.origin, self.rattlers, self.bdim, hmin, hmax,
                                                  self.binsize, self.equilibration_steps, fix_com=False)
            self.add_action(self.histogram)
        self.record_mcv = RecordMeanCoordVector(self.ndim, self.equilibration_steps)
        self.record_trajectory = RecordCoordsTimeseries(int((self.niter-self.equilibration_steps)/100), self.equilibration_steps)
            
        #set up pele:MC
        self.set_takestep(self.takestep)
        self.add_accept_test(self.metropolis) #metropolis uses the harmonic potential
        self.add_late_conf_test(self.conftest)
        self.add_action(self.action_record_displ)
        self.add_action(self.record_mcv)
        self.add_action(self.record_trajectory)
        if record_steps_timeseries:
            self.steps_timeseries_list = []
            self.record_steps_timeseries_every = record_steps_timeseries_every
            for freq in self.record_steps_timeseries_every:
                self.steps_timeseries_list.append(RecordStepsTimeseries(self.origin, self.rattlers, self.bdim, ts_niter, freq))
            for action in self.steps_timeseries_list:
                self.add_action(action)
        
    def set_control(self, c):
        """set temperature, canonical control parameter"""
        self.k = c
        self.potential.set_k(c)
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
    
    def get_timeseries(self):
        """write time series to fname, returns the timeseries"""
        timeseries = np.array(self.action_record_displ.get_time_series())
        return timeseries
    
    def check_convergence(self, nr_steps_to_check=10000, rel_std_threshold=0.05):
        return self.action_record_displ.check_convergence(nr_steps_to_check=nr_steps_to_check,
                                                   rel_std_threshold=rel_std_threshold)
    
    def get_mean_variance_coordinate_vector(self):
        """
        returns the average coordinate vector from the sampling and the elementwise variance
        """
        mean_coord, var_coord = self.record_mcv.get_mean_variance_coordinate_vector()
        return mean_coord, var_coord
    
    def get_trajectory(self):
        trajectory = self.record_trajectory.get_time_series()
        return trajectory
    
    def clear_trajectory(self):
        self.record_trajectory.clear()
        
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
        conftest = CheckHyperCubicContainer(self.origin, self.sidelength, self.bdim)
        self.conftest.add_test(conftest) 
        #conftest2 = CheckHyperSphericalContainer(np.array(self.origin), sidelength, self.bdim)
        #self.conftest.add_test(conftest2)
        
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

class HypercubeInnerSphereMCrunner(_BaseMCRunner):
    """
    """
    def __init__(self, potential, full_coords, temperature, stepsize, niter, origin,
                 sidelength=1, hmin=0, hmax=1, hbinsize=0.001, ts_niter=None, ts_freq=1, seeds=None, 
                 record_histogram=False):
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
        conftest = CheckHyperCubicContainer(self.origin, self.sidelength, self.bdim)
        self.conftest.add_test(conftest) 
        #conftest2 = CheckHyperSphericalContainer(np.array(self.origin), sidelength, self.bdim)
        #self.conftest.add_test(conftest2)
        
        self.time_series = RecordDisplacementTimeseries(self.origin, self.ndof, ts_niter, ts_freq, fix_com=False)
        
        self.set_report_steps(0)
        self.takestep = SampleUniformSphereGaussian(self.seeds['seed_takestep'], stepsize, self.origin)
        
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
        val = np.array([i * self.binsize for i in xrange(len(hist))]) + 0.5 * self.binsize
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
    k=25
    stepsize = np.sqrt(1.0 / k)
    if False:       
        test = HypercubeFindkMCrunner(potential, full_coords, 1, stepsize, int(1e8), origin, sidelength=1)
        start = time.time()
        test.run()
        end = time.time()
        print(end - start)
    if False:
        potential = Harmonic(origin, k, bdim=ndim, com=False)
        test = HypercubeMCrunner(potential, full_coords, 1, stepsize, int(1e5), origin, sidelength=1, record_histogram=True)
        start = time.time()
        test.run()
        end = time.time()
        print('displ2_kmin', test.get_displ2_kmin())
        print(end - start)
    if True:
        test = HypercubeInnerSphereMCrunner(potential, full_coords, 1, stepsize, int(1e5), origin, sidelength=1, record_histogram=True)
        start = time.time()
        test.run()
        end = time.time()
        print(end - start)
        test.show_histogram()
        test.show_histogram_analytical()
        
