from __future__ import division
from math import sqrt, floor, pi
import numpy as np
from numpy import linalg as la
import argparse
import os

class Splitting_Grid(object):
    """
    Holds the information on how the experimental packing is split up.
    This is a simple square grid and the spacing is estimated from the nr_of_mobile_particles
    (desired number of non-frozen particles in resulting split packings).
    distance_from_boundary_x: Gives the width of the discarded boundary stripes on the left and right sides
    of the experimental packing in per-cent of the horizontal length.
    distance_from_boundary_y: Gives the width of the discarded boundary stripes on the bottom and top sides
    of the experimental packing in per-cent of the vertical length.
    """
    def __init__(self, nr_of_mobile_particles, distance_from_boundary_x, distance_from_boundary_y, min_x, max_x, min_y, max_y, total_nr_of_particles):
        self.nr_of_mobile_particles = nr_of_mobile_particles
        self.distance_from_boundary_x = distance_from_boundary_x
        self.distance_from_boundary_y = distance_from_boundary_y
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        self.number_density = total_nr_of_particles/(delta_x*delta_y)
        #dimensions of box excluding the discarded stripes at the boundary
        min_xg = min_x + delta_x*self.distance_from_boundary_x
        max_xg = max_x - delta_x*self.distance_from_boundary_x
        min_yg = min_y + delta_y*self.distance_from_boundary_y
        max_yg = max_y - delta_y*self.distance_from_boundary_y
        delta_xg = max_xg - min_xg
        delta_yg = max_yg - min_yg
        #determination of grid parameters (for simple square grid)
        nr_of_cells_one_direction = int(floor(sqrt(self.number_density*delta_xg*delta_yg/self.nr_of_mobile_particles)))
        self.nr_of_cells = nr_of_cells_one_direction*nr_of_cells_one_direction
        #recoding positions of grid cell centres
        self.center_x = np.zeros(self.nr_of_cells)
        self.center_y = np.zeros(self.nr_of_cells)
        spacing_x = (delta_xg/nr_of_cells_one_direction)
        spacing_y = (delta_yg/nr_of_cells_one_direction)
        self.center_x[0] = min_xg + 0.5*spacing_x
        self.center_y[0] = min_yg + 0.5*spacing_y
        for i in xrange(1,self.nr_of_cells):
            self.center_x[i] = self.center_x[0] + (i%nr_of_cells_one_direction)*spacing_x
            self.center_y[i] = self.center_y[0] + (i//nr_of_cells_one_direction)*spacing_y

class Small_Packing_Information(object):
    def __init__(self, x, y, z, d, f):
        self.x = x
        self.y = y
        self.z = z
        self.d = d
        self.f = f

class Experimental_Packing(object):
    """
    This reads in and holds the experimental packing data as required for the
    cutting of (smaller) packings below.
    """
    def __init__(self, input_file_name, nr_of_mobile_particles, distance_from_boundary_x, distance_from_boundary_y, frozen_shell_thickness):
        self._read_in_data(input_file_name)
        self.grid = Splitting_Grid(nr_of_mobile_particles, distance_from_boundary_x, distance_from_boundary_y, min(self.x), max(self.x), min(self.y), max(self.y), self.total_nr_of_particles)
        self.frozen_shell_thickness = frozen_shell_thickness
    
    def _read_in_data(self, input_file_name):
        self.input_file_name = input_file_name
        self.input_file = open(self.input_file_name,"r")
        self.x = [] 
        self.y = [] 
        self.r = [] 
        self.large = [] 
        while True:
            this_line = self.input_file.readline()
            if not this_line:
                break
            #split this_line (which is not empty) at comma and store data for processing
            line_contents = this_line.split(',')
            self.x.append(line_contents[0])
            self.y.append(line_contents[1])
            self.r.append(line_contents[2])
            self.large.append(line_contents[3].strip())
        self.input_file.close()
        self.x = map(float, self.x)
        self.y = map(float, self.y)
        self.r = map(float, self.r)
        self.large = map(bool, self.large)
        self.total_nr_of_particles = len(self.x)
        self.average_particle_radius = np.mean(self.r)
        
    def print_particles(self, output_name):
        out_file = open(output_name,"w")
        for i in xrange(self.total_nr_of_particles):
            out_file.write(str(self.x[i])+"\t"+str(self.y[i])+"\n")
        
    def print_grid(self, output_name):
        out_file = open(output_name,"w")
        for i in xrange(self.grid.nr_of_cells):
            out_file.write(str(self.grid.center_x[i])+"\t"+str(self.grid.center_y[i])+"\n")
    
    def _extract_neighborhood(self, packing_index, particle_indices, particle_frozen):
        center_x = self.grid.center_x[packing_index]
        center_y = self.grid.center_y[packing_index]
        nr_of_mobile_particles = self.grid.nr_of_mobile_particles
        mobile_particle_radius = sqrt(nr_of_mobile_particles/self.grid.number_density/pi) #initial guess, based on particle density
        nr_mobile_found = self._get_nr_particles_in_circle(center_x, center_y, mobile_particle_radius)
        nr_iterations = 0
        while nr_mobile_found != nr_of_mobile_particles:
            nr_iterations += 1
            mobile_particle_radius = self._adapt_radius(mobile_particle_radius, nr_mobile_found, nr_of_mobile_particles, nr_iterations)
            nr_mobile_found = self._get_nr_particles_in_circle(center_x, center_y, mobile_particle_radius)
            #print("mobile_particle_radius: %f" % mobile_particle_radius)
            #print("nr_mobile_found: %d" % nr_mobile_found)
        frozen_particle_radius = mobile_particle_radius + self.frozen_shell_thickness*(2*self.average_particle_radius)
        for i in xrange(self.total_nr_of_particles):
            dd = la.norm([self.x[i] - center_x, self.y[i] - center_y])
            if dd <= frozen_particle_radius:
                particle_indices.append(i)
                if dd <= mobile_particle_radius:
                    particle_frozen.append(False)
                else:
                    particle_frozen.append(True)
        
    def _get_nr_particles_in_circle(self, center_x, center_y, radius):
        result = 0
        for i in xrange(self.total_nr_of_particles):
            if la.norm([self.x[i] - center_x, self.y[i] - center_y]) <= radius:
                result += 1
        return result
    
    def _adapt_radius(self, old_radius, found_particles, desired_particles, nr_iterations):
        coupling = 1.0/nr_iterations #can be adapted to damp oscillations
        return old_radius*( (1-coupling) + coupling*sqrt(desired_particles/found_particles) )
    
    def _get_small_packing_information(self, indices, frozen):
        x = []
        y = []
        z = []
        d = []
        f = frozen
        for i in xrange(len(indices)):
            idx = indices[i]
            x.append(self.x[idx])
            y.append(self.y[idx])
            z.append(0)
            d.append(2*self.r[idx])
        #TODO: shift and rescale coords as it is convenient for the simulations; check output format
        #here one could shift, rescale the coordinates
        ########################################
        return Small_Packing_Information(x, y, z, d, f)
    
    def extract_small_packing(self, packing_index):
        small_packing_particles_indices = []
        small_packing_particles_frozen = []
        self._extract_neighborhood(packing_index, small_packing_particles_indices, small_packing_particles_frozen)
        return self._get_small_packing_information(small_packing_particles_indices, small_packing_particles_frozen)

class Cut_Out_Packings(object):
    """
    Out of the data set with the given index, this cuts out a given number of
    packings with the specified number of (non-frozen) particles.
    """
    def __init__(self):
        self._read_parameters()
        self._read_experimental_data()
    
    def _read_parameters(self):
        self.parser = argparse.ArgumentParser(description='Split experimental data.')
        self.parser.add_argument('data_set_index', type=int, nargs='?', default=0, help='selects experimental dataset')
        self.parser.add_argument('nr_of_particles', type=int, nargs='?', default=8, help='number of non-frozen particles')
        self.parser.add_argument('nr_of_packings', type=int, nargs='?', default=10, help='number of extracted packings')
        self.parser.add_argument('path_to_data',type=str, nargs='?', default='data', help='path to data files')
        self.parser.add_argument('distance_from_boundary_x',type=float, nargs='?', default=0.01, help='discarded margins left and right, per-cent')
        self.parser.add_argument('distance_from_boundary_y',type=float, nargs='?', default=0.01, help='discarded margins bottom and top, per-cent')
        self.parser.add_argument('frozen_shell_thickness',type=float, nargs='?', default=2, help='number of average particle diameters in frozen shell')
        self.args = self.parser.parse_args()
        if self.args.nr_of_particles <= 0:
            raise Exception('expect finite number of particles')
        if self.args.nr_of_packings <= 0:
            raise Exception('expect finite number of packings')
        if self.args.data_set_index < 0 or self.args.data_set_index >= 4:
            raise Exception('expect data set index to be between 0 and 3')
        self.data_file_idx = [1,3,4,5]
        self.data_set_index = self.data_file_idx[self.args.data_set_index]
        self.nr_of_particles = self.args.nr_of_particles
        self.nr_of_packings = self.args.nr_of_packings
        self.data_file_name = "PackingsData_%d.dat" % self.data_set_index
        self.path_to_data = self.args.path_to_data
        self.path_to_file = "/".join([self.path_to_data,self.data_file_name])
        self.distance_from_boundary_x = self.args.distance_from_boundary_x
        self.distance_from_boundary_y = self.args.distance_from_boundary_y
        self.frozen_shell_thickness = self.args.frozen_shell_thickness
        
    def _read_experimental_data(self):
        self.all_particles = Experimental_Packing(self.path_to_file, self.nr_of_particles, self.distance_from_boundary_x, self.distance_from_boundary_y, self.frozen_shell_thickness)
        
    def _test_grid(self):
        self.all_particles.print_particles("test_particles.xy")
        self.all_particles.print_grid("test_grid.xy")
    
    def _split_packings(self):
        self.small_packings = []
        for i in xrange(self.nr_of_packings):
            self._find_one_small_packing(i)
            print("found packing %d of %d" % (i, self.nr_of_packings))
            
    def _find_one_small_packing(self, index):
        self.small_packings.append(self.all_particles.extract_small_packing(index))
    
    def _dump_packings(self):
        self.path_to_output_small_packings = "/".join([self.path_to_data,"output"])
        for i in xrange(len(self.small_packings)):
            self._print_small_packing(i, self.small_packings[i])
            
    def _print_small_packing(self, packing_index, packing_information):
        #TODO: check that this prints the split packings as needed
        #TODO: print also parameters of generated packings
        #TODO: if more packings requested than possible, reset to maximum value and print warning
        #TODO: if exploration of neighborhood goes to far into boundary region, trow, and suggest to increase safety margins
        if not os.path.exists(self.path_to_output_small_packings): 
            os.makedirs(self.path_to_output_small_packings)
        output_file = open("/".join([self.path_to_output_small_packings,"split_packing_"+str(packing_index)+".xyzdf"]), "w")
        output_file_f = open("/".join([self.path_to_output_small_packings,"frozen_only_split_packing_"+str(packing_index)+".xyzdf"]), "w")
        for i in xrange(len(packing_information.x)):
            output_file.write('{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\n'.format(packing_information.x[i], packing_information.y[i], packing_information.z[i], packing_information.d[i], packing_information.f[i]))
            if packing_information.f[i]==True:
                output_file_f.write('{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\n'.format(packing_information.x[i], packing_information.y[i], packing_information.z[i], packing_information.d[i], packing_information.f[i]))
        output_file.close()
        output_file_f.close()
        ########################################
        ########################################
    
    def run(self):
        self._test_grid()
        self._split_packings()
        self._dump_packings()
        
if __name__ == "__main__":
    sys = Cut_Out_Packings()
    sys.run()
