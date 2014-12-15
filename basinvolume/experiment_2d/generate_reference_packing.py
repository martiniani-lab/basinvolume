from __future__ import division
import numpy as np
import argparse as ap
from radii_sampler import RadiiSampler
from throw_and_quench import ThrowAndQuench
from eq_fluid_snapshots import EqFluidSnapshots

class HSExpReferenceGeneratePacking(object):
    """
    Runs equilibrium HS fluid with same parameters as an experimental
    image.
    
    Intended use is to import the experimental radii distribution and
    volume fraction, then sample a legal configuration of spheres from
    this and finally run an equlibrium fluid with this.
    This should then be stored in the same format as for the
    experimental input so that it can be fed in the same script for
    comparison.
    The idea is to run the same script that extracts the small packings
    from the experimental images on the output of this script.
    Therefore the output format of this script has to match the one of
    the experimental data files.
    
    Parameters
    ----------
    nr_particles : integer
        The number of particles in the "experimental image" that is
        being generated.
    nr_images : integer
        The number of "experimental images" that is generated.
        These are the fluid snapshots that should have the same format
        as the experimental data sets and which will be split into
        smaller, circular packings later.
    exp_data_set_index: integer
        Selects the experimental image / data set from which we are
        importing the radii and the volume fraction.
    exp_data_set_name_begin: string
        Beginning of the experimental data set name.
        This is something like "PackingsData_".
    """
    def __init__(self, nr_particles=1000, nr_images=1, exp_data_set_index=1, exp_data_set_name_begin="PackingsData_", data_dir=None, show_radii_distribution=False, hard_phi=0.67):
        # begin: store input
        self.nr_particles = nr_particles
        self.nr_images = nr_images
        self.exp_data_set_index = exp_data_set_index
        self.exp_data_set_name_begin = exp_data_set_name_begin
        self.data_dir = data_dir
        self.show_radii_distribution = show_radii_distribution
        self.hard_phi = hard_phi
        # end: store input
        self.radii_sampler = RadiiSampler(self.exp_data_set_index, self.exp_data_set_name_begin, self.data_dir, self.nr_particles, show_distribution=self.show_radii_distribution)
        self.radii = self.radii_sampler.radii
        self.initial_condition = ThrowAndQuench(self.nr_particles, self.hard_phi, self.radii)
        #self.initial_coordinates = self.initial_condition.coordinates
        #self.boxvec = self.initial_condition.boxvec
        #self.fluid = EqFluidSnapshots(self.radii, self.coordinates, self.boxvec)
        #self.fluid.run()
        
if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Generate reference equilibrium fluid snapshot from experimental radii distribution")
    parser.add_argument("nr_particles", type=int, help="number of particles")
    parser.add_argument("--nr_images", type=int, default=1, help="number of printed snapshots")
    parser.add_argument("--exp_data_set_index", type=int, default=1, help="index of underlying experimental image")
    parser.add_argument("--exp_data_set_name_begin", type=str, default="PackingsData_")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--show_radii_distribution", action="store_true", help="display experimental radii distribution, learned distribution, sampled radii", default=False)
    parser.add_argument("--hard_phi", type=float, default=0.67, help="hard disc volume fraction")
    pars = parser.parse_args()
    print("input parameters:")
    print(pars)
    HSExpReferenceGeneratePacking(nr_particles=pars.nr_particles, nr_images=pars.nr_images, exp_data_set_index=pars.exp_data_set_index, exp_data_set_name_begin=pars.exp_data_set_name_begin, data_dir=pars.data_dir, show_radii_distribution=pars.show_radii_distribution, hard_phi=pars.hard_phi)
