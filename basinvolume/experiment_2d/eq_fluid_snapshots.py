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
    def __init__(self, radii, coordinates, boxvec, nr_steps=1e8):
        self.radii = np.array(radii)
        self.coordinates = np.array(coordinates)
        self.boxvec = np.array(boxvec)
        self.nr_steps = nr_steps
        #
        self.eq_steps = self.nr_steps // 2
        self.overlap_check = CheckOverlapPeriodicCellLists(self.radii, self.boxvec)
        self.temperature = 1
        self.mock_potential = Harmonic(self.coordinates, 42, bdim=2) # This is not used.
        self.mc = MC(self.mock_potential, self.coordinates, self.temperature, self.nr_steps)
        self.step = RandomCoordsDisplacement(self.step_seed, 1, single=True, nparticles=1, bdim=2)
        self.mc.set_report_steps(self.eq_steps)
        self.mc.set_takestep(self.step)
        self.mc.add_config_test(self.overlap_check)
    def run(self):
        print("running reference fluid")
        
        print("running reference fluid -- done")
        
