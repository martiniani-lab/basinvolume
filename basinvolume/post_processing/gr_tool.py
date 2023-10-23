from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
from scipy.special import gamma

def cround(r):
    if r > 0.0:
        r = np.floor(r + 0.5)
    else:
        r = np.ceil(r - 0.5)
    return r

class Histogram(object):
    
    def __init__(self, xmin, xmax, bin_size):
        self.xmax = np.floor((xmax / bin_size) + 1) * bin_size
        self.xmin = np.floor(xmin / bin_size) * bin_size
        self.bin_size = bin_size
        self.eps = 2e-16
        self.N = (self.xmax - self.xmin) / self.bin_size
        self.hist = np.zeros(self.N)
        if self.xmin >= self.xmax - self.bin_size:
            raise Exception("illegal input")
    
    def add(self, x):
        if x > self.xmax or x < self.xmin:
            raise Exception("illegal input")
        x_ = x + self.eps
        i = np.floor((x_ - self.xmin) / self.bin_size)
        self.hist[i] += 1
        
    def get_position(self, index):
        return self.xmin + (0.5 + index) * self.bin_size
        
    def get_entry(self, index):
        return self.hist[index]
        
        
class PairDistHistogram(object):
    """
    Compute g(r) from input snapshots.
    """
    def __init__(self, boxvector, nr_bins):
        self.boxvector = boxvector
        self.boxdim = len(self.boxvector)
        self.nr_bins = nr_bins
        self.min_dist = 0
        self.max_dist = 0.5 * np.amax(self.boxvector)
        self.delta_bin = (self.max_dist - self.min_dist) / self.nr_bins
        self.histogram = Histogram(self.min_dist, self.max_dist, self.delta_bin)
        self.nr_configs = 0
        
    def add_configuration(self, coords):
        self.nr_configs += 1
        nr_particles = int(len(coords) / self.boxdim)
        for i in range(nr_particles):
            for j in range(i + 1, nr_particles):
                self.add_distance(i, j, coords)
                
    def add_distance(self, i, j, coords):
        r2 = self.get_r2(i, j, coords)
        if r2 <= self.max_dist**2:
            self.histogram.add(np.sqrt(r2))
        
    def get_r2(self, i, j, coords):
        self.bdim = self.boxdim
        self.boxv = self.boxvector
        deltaij = np.zeros(self.boxdim)
        for k in range(self.boxdim):
            deltaij[k] = ((coords[j*self.bdim+k] - coords[i*self.bdim+k]) - cround((coords[j*self.bdim+k] - coords[i*self.bdim+k]) / self.boxv[k]) * self.boxv[k])
        return np.sum(deltaij**2)
    
    def volume_nball(self, radius, n):
        volume = np.power(np.pi, n / 2) * np.power(radius, n) / gamma(n / 2 + 1)
        return volume
        
    def get_hist_r(self):
        return [self.histogram.get_position(i) for i in range(self.nr_bins)]
        
    def get_hist_gr(self, number_density, nr_particles):
        result = np.zeros(self.nr_bins)
        for i in range(self.nr_bins):
            r = self.histogram.get_position(i)
            delta_r = self.histogram.bin_size
            shell_volume_r = self.volume_nball(r + 0.5 * delta_r, self.boxdim) - self.volume_nball(r - 0.5 * delta_r, self.boxdim)
            nid = shell_volume_r * number_density
            normalization = 2 / (self.nr_configs * nr_particles * nid)
            result[i] = normalization * self.histogram.get_entry(i)
        return result        

import matplotlib.pyplot as plt
from pele.potentials import HS_WCA
from pele.optimize import LBFGS_CPP
from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import RandomCoordsDisplacement
from mcpele.monte_carlo import MetropolisTest

class MC(_BaseMCRunner):
    def set_control(self, temp):
        self.set_temperature(temp)
    def run(self, nr_steps, gr_acc, eq_steps):
        for _ in range(int(nr_steps)):
            print(("_", _))
            self.one_iteration()
            if self.get_iterations_count() > eq_steps:
                gr_acc.add_configuration(self.get_coords())
        

class ComputeGR(object):
    def __init__(self, boxdim=2, nr_particles=100, hard_phi=0.4,
             nr_steps=1e2, epsilon=1, alpha=0.1, verbose=False):
        # Settings.
        np.random.seed(42)
        # Input parameters.
        self.boxdim = boxdim
        self.nr_particles = nr_particles
        self.hard_phi = hard_phi
        self.nr_steps = nr_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.verbose = verbose
        # Derived quantities.
        self.hard_radii = np.ones(self.nr_particles)
        def volume_nball(radius, n):
            return np.power(np.pi, n / 2) * np.power(radius, n) / gamma(n / 2 + 1)
        self.box_length = np.power(np.sum(np.asarray([volume_nball(r, self.boxdim) for r in self.hard_radii])) / self.hard_phi, 1 / self.boxdim)
        self.box_vector = np.ones(self.boxdim) * self.box_length
        # HS-WCA potential.
        self.potential = HS_WCA(use_periodic=True, use_cell_lists=True,
                                ndim=self.boxdim, eps=self.epsilon,
                                sca=self.alpha, radii=self.hard_radii,
                                boxvec=self.box_vector)
        # Initial configuration by minimization.
        self.nr_dof = self.boxdim * self.nr_particles
        self.x = np.random.uniform(-0.5 * self.box_length, 0.5 * self.box_length, self.nr_dof)
        optimizer = LBFGS_CPP(self.x, self.potential)
        optimizer.run()
        if not optimizer.get_result().success:
            print ("warning: minimization has not converged")
        self.x = optimizer.get_result().coords.copy()
        # Potential and MC rules.
        self.temperature = 1
        self.mc = MC(self.potential, self.x, self.temperature, self.nr_steps)
        self.step = RandomCoordsDisplacement(42, 1, single=True, nparticles=self.nr_particles, bdim=self.boxdim)
        if self.verbose:
            print ("initial MC stepsize")
            print(self.step.get_stepsize())
        self.gr = PairDistHistogram(self.box_vector, 50)
        self.mc.set_takestep(self.step)
        self.eq_steps = self.nr_steps / 2
        self.mc.set_report_steps(self.eq_steps)
        self.test = MetropolisTest(44)
        self.mc.add_accept_test(self.test)
    
    def run(self):
        self.mc.set_print_progress()
        if not self.verbose:
            self.mc.disable_input_warnings()
        self.mc.run(self.nr_steps, self.gr, self.eq_steps)
        if self.verbose:
            print ("adapted MC stepsize")
            print(self.step.get_stepsize())
        
    def show_result(self):
        r = self.gr.get_hist_r()
        number_density = self.nr_particles / np.prod(self.box_vector)
        gr = self.gr.get_hist_gr(number_density, self.nr_particles)
        plt.plot(r, gr, "o-", label="Equilibrium")
        plt.xlabel(r"Distance $r$")
        plt.ylabel(r"Radial distr. function $g(r)$")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    box_dimension = 2
    nr_particles = 100
    hard_volume_fraction = 0.4
    nr_steps = 2e2
    alpha = 0.48
    verbose = False
    simulation = ComputeGR(boxdim=box_dimension,
                           nr_particles=nr_particles,
                           hard_phi=hard_volume_fraction,
                           nr_steps=nr_steps,
                           alpha=alpha,
                           verbose=verbose)
    simulation.run()
    simulation.show_result()
