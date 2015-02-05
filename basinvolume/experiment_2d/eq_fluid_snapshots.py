from __future__ import division
import numpy as np
from pele.potentials import Harmonic
from mcpele.monte_carlo import _BaseMCRunner
from mcpele.monte_carlo import RandomCoordsDisplacement
from basinvolume.monte_carlo import CheckOverlapPeriodicCellLists
from basinvolume.spheres import HS_MCrunnerOptDiffusion

class MC(_BaseMCRunner):
    def set_control(self, temp):
        self.set_temperature(temp)

class EqFluidSnapshots(object):
    """
    Runs the equilibrium fluid corresponding to a given sample of radii
    and initial conditions.
    This should also print snapshots at equal and specified intervals.
    """
    def __init__(self, radii, coordinates, boxvec, nr_steps=1e8, step_seed=4242, nr_images=42, base_out_file_name="exp_reference_packing"):
        self.radii = np.array(radii)
        self.coordinates = np.array(coordinates)
        self.boxvec = np.array(boxvec)
        self.nr_steps = nr_steps
        self.step_seed = step_seed
        self.nr_images = nr_images
        self.base_out_file_name = base_out_file_name
        #
        self.nr_particles = self.radii.size
        self.eq_steps = 0 # Adapting stepsize and finding nr of decorrelation steps should be done by the diffusion test MC. Therefore, we do not need eq_steps (report steps) in the 'second' MC (which prints the fluid snapshots).
        self.overlap_check = CheckOverlapPeriodicCellLists(self.radii, self.boxvec)
        self.temperature = 1
        self.mock_potential = Harmonic(self.coordinates, 42, bdim=2) # This is not used.
        self.stepsize = 1
        self.find_nr_decorrelation_steps()
        self.mc = MC(self.mock_potential, self.coordinates, self.temperature, self.nr_steps)
        self.step = RandomCoordsDisplacement(self.step_seed, self.stepsize, single=True, nparticles=1, bdim=2)
        self.mc.set_report_steps(self.eq_steps)
        self.mc.set_takestep(self.step)
        self.mc.add_conf_test(self.overlap_check)
        self.printed_images = 0
    def run(self):
        print("running reference fluid")
        while self.printed_images < self.nr_images:
            self.print_next_image()
        print("running reference fluid -- done")
    def find_nr_decorrelation_steps(self):
        print("finding number of decorrelation steps")
        diffusion_test_mc = HS_MCrunnerOptDiffusion(self.mock_potential, self.coordinates, self.temperature, self.stepsize, 1e9, self.radii, self.boxvec, adjustf=0.9, acceptance=0.15, adjustf_niter=1e6, single=True)
        diffusion_test_mc.run()
        self.stepsize = diffusion_test_mc.get_stepsize()
        self.nr_decorrelation_steps = diffusion_test_mc.get_nr_decorrelation_steps()
        self.coordinates, energy = diffusion_test_mc.get_config()
        self.nr_steps = max(self.nr_steps, self.nr_images * self.nr_decorrelation_steps)
        print("finding number of decorrelation steps -- done -- results:")
        print("stepsize", self.stepsize)
        print("nr decorrelation steps", self.nr_decorrelation_steps)
        print("maximum total nr steps", self.nr_steps)
    def print_next_image(self):
        print("printing image", self.printed_images + 1, "out of", self.nr_images)
        for _ in xrange(self.nr_decorrelation_steps):
            self.mc.one_iteration()
        x = self.mc.get_coords()
        self.print_Lorenzo_style(x)
        print("printed image", self.printed_images, "out of", self.nr_images)
    def print_Lorenzo_style(self, x):
        out_file = open(self.base_out_file_name + "_" + str(self.printed_images), "w")
        for particle_index in xrange(self.nr_particles):
            out_file.write(self.get_Lorenzo_style_string(particle_index))
        out_file.close()
        self.printed_images += 1
    def get_Lorenzo_style_string(self, particle_index):
        """
        Lorenzo's file format, for his images of jammed experimental
        packings, is as follows: 'The files *.dat contain the x-y
        coordinates, the radius and a binary and a binary variable which
        is 1 if the particle belongs to the green component (larger
        particles).  All distances are in pixel.'
        (x, y, radius, large particle flag)
        Examples:
        42.827,1493.5,16.434,1
        42.529,1809.8,19.726,0
        44.158,345.29,15.846,0
        46.431,715.81,14.904,0
        48.376,206.32,16.691,0
        Procedure:
        This is replicated in the following, such that the fluid
        snapshots printed here can be used directly with the same
        scripts that analyze the experimental images.
        Warning:
        For now, I did not bother to replicate the large particle flag.
        This could be done, e.g., using the median of the sampled radii.
        Since we are not using the large particle flag in the
        experimental basinvolume scripts, I did not implement that and
        it will just be set to 1 for all particles, irrespective of
        their radius.
        """
        x = self.get_x(particle_index)
        y = self.get_y(particle_index)
        r = self.get_r(particle_index)
        large_particle_flag = self.get_large_particle_flag(particle_index)
        return ",".join([str(x), str(y), str(r), str(large_particle_flag)])
    def get_x(self, particle):
        return self.coordinates[particle * 2]
    def get_y(self, particle):
        return self.coordinates[particle * 2 + 1]
    def get_r(self, particle):
        return self.radii[particle]
    def get_large_particle_flag(self, particle):
        return 1
