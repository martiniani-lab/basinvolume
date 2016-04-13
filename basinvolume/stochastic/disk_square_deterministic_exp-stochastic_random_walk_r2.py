from __future__ import division
import numpy as np

from basinvolume.monte_carlo import CheckExponentiallyDecayingProfile
from basinvolume.utils import *

def get_disk_mean_r2(radius):
    return radius ** 2 / 2
    
def get_square_mean_r2(side_length):
    return side_length ** 2 / 6
    
class DeterministicPlot(BasicPlot):
    

#class StochasticPlot(DeterministicPlot):

if __name__ == "__main__":
    disc_radii = range(1, 10, 0.5)
    square_sides = range(1, 10, 0.5)
    dp = DeterministicPlot(disc_radii, square_sides)
    dp.run()
    #sp = StochasticPlot(disc_radii, square_sides)
    #sp.run()
