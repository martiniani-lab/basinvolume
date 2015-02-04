from __future__ import division
import numpy as np
from basinvolume.monte_carlo import CheckOverlapPeriodicCellLists
from pele.potentials import Harmonic
from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import RandomCoordsDisplacement

class MC(_BaseMCRunner):
    def set_control(self, temp):
        self.set_temperature(temp)

class EqFluidSnapshots(object):
    """
    Runs the equilibrium fluid corresponding to a given sample of radii
    and initial conditions.
    This should also print snapshots at equal and specified intervals.
    """
    def __init__(self, radii, coordinates, boxvec, nr_steps=1e8, step_seed=4242, nr_images=42):
        self.radii = np.array(radii)
        self.coordinates = np.array(coordinates)
        self.boxvec = np.array(boxvec)
        self.nr_steps = nr_steps
        self.step_seed = step_seed
        self.nr_images = nr_images
        #
        self.eq_steps = self.nr_steps // 2
        self.overlap_check = CheckOverlapPeriodicCellLists(self.radii, self.boxvec)
        self.temperature = 1
        self.mock_potential = Harmonic(self.coordinates, 42, bdim=2) # This is not used.
        self.mc = MC(self.mock_potential, self.coordinates, self.temperature, self.nr_steps)
        self.step = RandomCoordsDisplacement(self.step_seed, 1, single=True, nparticles=1, bdim=2)
        self.mc.set_report_steps(self.eq_steps)
        self.mc.set_takestep(self.step)
        self.mc.add_conf_test(self.overlap_check)
        self.printed_images = 0
    def run(self):
        print("finding number of decorrelation steps")
        
        print("finding number of decorrelation steps -- done")
        print("running reference fluid")
        while self.printed_images < self.nr_images:
            self.print_next_image()
        print("running reference fluid -- done")
    def print_next_image(self):
        print("printing image", self.printed_images, "out of", self.nr_images)
        for _ in xrange(self.nr_decorrelation_steps):
            self.mc.one_iteration()
        x = self.mc.get_coords()
        self.print_Lorenzo_style(x)
        print("printed image", self.printed_images - 1, "out of", self.nr_images)
    def print_Lorenzo_style(self, x)
        
