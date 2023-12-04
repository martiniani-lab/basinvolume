from __future__ import division
from __future__ import print_function
from builtins import map
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
from math import sqrt, floor, pi
import numpy as np
from numpy import linalg as la
import argparse
import os
import sys


class Splitting_Grid_0(object):
    """
    Holds the information on how the experimental packing is split up.
    This is a simple square grid and the spacing is estimated from the nr_of_mobile_particles
    (desired number of non-frozen particles in resulting split packings).
    distance_from_boundary_x: Gives the width of the discarded boundary stripes on the left and right sides
    of the experimental packing in per-cent of the horizontal length.
    distance_from_boundary_y: Gives the width of the discarded boundary stripes on the bottom and top sides
    of the experimental packing in per-cent of the vertical length.
    """

    def __init__(
        self,
        nr_of_mobile_particles,
        distance_from_boundary_x,
        distance_from_boundary_y,
        min_x,
        max_x,
        min_y,
        max_y,
        total_nr_of_particles,
    ):
        self.nr_of_mobile_particles = nr_of_mobile_particles
        self.distance_from_boundary_x = distance_from_boundary_x
        self.distance_from_boundary_y = distance_from_boundary_y
        self.delta_x = max_x - min_x
        self.delta_y = max_y - min_y
        self.number_density = total_nr_of_particles / (self.delta_x * self.delta_y)
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        # dimensions of box excluding the discarded stripes at the boundary
        self.min_xg = min_x + self.delta_x * self.distance_from_boundary_x
        self.max_xg = max_x - self.delta_x * self.distance_from_boundary_x
        self.min_yg = min_y + self.delta_y * self.distance_from_boundary_y
        self.max_yg = max_y - self.delta_y * self.distance_from_boundary_y
        self.delta_xg = self.max_xg - self.min_xg
        self.delta_yg = self.max_yg - self.min_yg
        # determination of grid parameters (for simple square grid)
        self.nr_of_cells_one_direction = int(
            floor(
                sqrt(
                    self.number_density
                    * self.delta_xg
                    * self.delta_yg
                    / self.nr_of_mobile_particles
                )
            )
        )
        self.nr_of_cells = (
            self.nr_of_cells_one_direction * self.nr_of_cells_one_direction
        )
        # recoding positions of grid cell centres
        self.center_x = np.zeros(self.nr_of_cells)
        self.center_y = np.zeros(self.nr_of_cells)
        self.spacing_x = self.delta_xg / self.nr_of_cells_one_direction
        self.spacing_y = self.delta_yg / self.nr_of_cells_one_direction
        self.center_x[0] = self.min_xg + 0.5 * self.spacing_x
        self.center_y[0] = self.min_yg + 0.5 * self.spacing_y
        for i in range(1, self.nr_of_cells):
            self.center_x[i] = (
                self.center_x[0] + (i % self.nr_of_cells_one_direction) * self.spacing_x
            )
            self.center_y[i] = (
                self.center_y[0]
                + (i // self.nr_of_cells_one_direction) * self.spacing_y
            )


class Splitting_Grid_1(object):
    """
    Regular grid shifted right-up.
    """

    def __init__(
        self,
        nr_of_mobile_particles,
        distance_from_boundary_x,
        distance_from_boundary_y,
        min_x,
        max_x,
        min_y,
        max_y,
        total_nr_of_particles,
    ):
        base = Splitting_Grid_0(
            nr_of_mobile_particles,
            distance_from_boundary_x,
            distance_from_boundary_y,
            min_x,
            max_x,
            min_y,
            max_y,
            total_nr_of_particles,
        )
        self.nr_of_cells_one_direction = base.nr_of_cells_one_direction - 1
        self.nr_of_cells = (
            self.nr_of_cells_one_direction * self.nr_of_cells_one_direction
        )
        self.center_x = np.zeros(self.nr_of_cells)
        self.center_y = np.zeros(self.nr_of_cells)
        # shift right and up
        self.center_x[0] = base.min_xg + base.spacing_x
        self.center_y[0] = base.min_yg + base.spacing_y
        for i in range(1, self.nr_of_cells):
            self.center_x[i] = (
                self.center_x[0] + (i % self.nr_of_cells_one_direction) * base.spacing_x
            )
            self.center_y[i] = (
                self.center_y[0]
                + (i // self.nr_of_cells_one_direction) * base.spacing_y
            )


class Splitting_Grid_2(object):
    """
    Regular grid shifted right.
    """

    def __init__(
        self,
        nr_of_mobile_particles,
        distance_from_boundary_x,
        distance_from_boundary_y,
        min_x,
        max_x,
        min_y,
        max_y,
        total_nr_of_particles,
    ):
        base = Splitting_Grid_0(
            nr_of_mobile_particles,
            distance_from_boundary_x,
            distance_from_boundary_y,
            min_x,
            max_x,
            min_y,
            max_y,
            total_nr_of_particles,
        )
        nr_of_cells_x = base.nr_of_cells_one_direction - 1
        nr_of_cells_y = base.nr_of_cells_one_direction
        self.nr_of_cells = nr_of_cells_x * nr_of_cells_y
        self.center_x = np.zeros(self.nr_of_cells)
        self.center_y = np.zeros(self.nr_of_cells)
        # shift right
        self.center_x[0] = base.min_xg + base.spacing_x
        self.center_y[0] = base.min_yg + 0.5 * base.spacing_y
        for i in range(1, self.nr_of_cells):
            self.center_x[i] = self.center_x[0] + (i % nr_of_cells_x) * base.spacing_x
            self.center_y[i] = self.center_y[0] + (i // nr_of_cells_x) * base.spacing_y


class Splitting_Grid_3(object):
    """
    Regular grid shifted up.
    """

    def __init__(
        self,
        nr_of_mobile_particles,
        distance_from_boundary_x,
        distance_from_boundary_y,
        min_x,
        max_x,
        min_y,
        max_y,
        total_nr_of_particles,
    ):
        base = Splitting_Grid_0(
            nr_of_mobile_particles,
            distance_from_boundary_x,
            distance_from_boundary_y,
            min_x,
            max_x,
            min_y,
            max_y,
            total_nr_of_particles,
        )
        nr_of_cells_x = base.nr_of_cells_one_direction
        nr_of_cells_y = base.nr_of_cells_one_direction - 1
        self.nr_of_cells = nr_of_cells_x * nr_of_cells_y
        self.center_x = np.zeros(self.nr_of_cells)
        self.center_y = np.zeros(self.nr_of_cells)
        # shift up
        self.center_x[0] = base.min_xg + 0.5 * base.spacing_x
        self.center_y[0] = base.min_yg + base.spacing_y
        for i in range(1, self.nr_of_cells):
            self.center_x[i] = self.center_x[0] + (i % nr_of_cells_x) * base.spacing_x
            self.center_y[i] = self.center_y[0] + (i // nr_of_cells_x) * base.spacing_y


class Splitting_Grid(object):
    """
    Handles the different grid types.
    0) usual regular grid
    1) shifted right-up
    2) shifted right
    3) shifted up
    4) all of the above
    """

    def __init__(
        self,
        nr_of_mobile_particles,
        distance_from_boundary_x,
        distance_from_boundary_y,
        min_x,
        max_x,
        min_y,
        max_y,
        total_nr_of_particles,
        grid_version,
        get_all_packings,
    ):
        self.info = Splitting_Grid_0(
            nr_of_mobile_particles,
            distance_from_boundary_x,
            distance_from_boundary_y,
            min_x,
            max_x,
            min_y,
            max_y,
            total_nr_of_particles,
        )
        self.delta_x = self.info.delta_x
        self.delta_y = self.info.delta_y
        self.nr_of_mobile_particles = self.info.nr_of_mobile_particles
        self.number_density = self.info.number_density
        self.min_x = self.info.min_x
        self.max_x = self.info.max_x
        self.min_y = self.info.min_y
        self.max_y = self.info.max_y
        if grid_version > 3:
            raise Exception(
                "Splitting_Grid: illegal grid type selected; possible are: 0, 1, 2, 3, --all"
            )
        grids = []
        if grid_version == 0 or get_all_packings:
            grids.append(
                Splitting_Grid_0(
                    nr_of_mobile_particles,
                    distance_from_boundary_x,
                    distance_from_boundary_y,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    total_nr_of_particles,
                )
            )
        if grid_version == 1 or get_all_packings:
            grids.append(
                Splitting_Grid_1(
                    nr_of_mobile_particles,
                    distance_from_boundary_x,
                    distance_from_boundary_y,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    total_nr_of_particles,
                )
            )
        if grid_version == 2 or get_all_packings:
            grids.append(
                Splitting_Grid_2(
                    nr_of_mobile_particles,
                    distance_from_boundary_x,
                    distance_from_boundary_y,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    total_nr_of_particles,
                )
            )
        if grid_version == 3 or get_all_packings:
            grids.append(
                Splitting_Grid_3(
                    nr_of_mobile_particles,
                    distance_from_boundary_x,
                    distance_from_boundary_y,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    total_nr_of_particles,
                )
            )
        self.nr_of_cells = 0
        self.center_x = []
        self.center_y = []
        for i in range(len(grids)):
            self.nr_of_cells += grids[i].nr_of_cells
            self.center_x.extend(grids[i].center_x)
            self.center_y.extend(grids[i].center_y)


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

    def __init__(
        self,
        input_file_name,
        nr_of_mobile_particles,
        distance_from_boundary_x,
        distance_from_boundary_y,
        frozen_shell_thickness,
        grid_version,
        get_all_packings,
    ):
        self._read_in_data(input_file_name)
        self.grid_version = grid_version
        self.get_all_packings = get_all_packings
        self.grid = Splitting_Grid(
            nr_of_mobile_particles,
            distance_from_boundary_x,
            distance_from_boundary_y,
            min(self.x),
            max(self.x),
            min(self.y),
            max(self.y),
            self.total_nr_of_particles,
            self.grid_version,
            self.get_all_packings,
        )
        self.frozen_shell_thickness = frozen_shell_thickness
        min_distance_from_boundary_x = (
            self.frozen_shell_thickness
            * (2 * self.average_particle_radius)
            / self.grid.delta_x
        )
        min_distance_from_boundary_y = (
            self.frozen_shell_thickness
            * (2 * self.average_particle_radius)
            / self.grid.delta_y
        )
        x_small = distance_from_boundary_x < min_distance_from_boundary_x
        y_small = distance_from_boundary_y < min_distance_from_boundary_y
        if x_small or y_small:
            sys.stderr.write(
                "WARNING: distance_from_boundary chosen too small; was reset to approx. minimum\n"
            )
            if x_small:
                distance_from_boundary_x = min_distance_from_boundary_x
            if y_small:
                distance_from_boundary_y = min_distance_from_boundary_y
            self.grid = Splitting_Grid(
                nr_of_mobile_particles,
                distance_from_boundary_x,
                distance_from_boundary_y,
                min(self.x),
                max(self.x),
                min(self.y),
                max(self.y),
                self.total_nr_of_particles,
            )

    def _read_in_data(self, input_file_name):
        self.input_file_name = input_file_name
        self.input_file = open(self.input_file_name, "r")
        self.x = []
        self.y = []
        self.r = []
        self.large = []
        while True:
            this_line = self.input_file.readline()
            if not this_line:
                break
            # split this_line (which is not empty) at comma and store data for processing
            line_contents = this_line.split(",")
            self.x.append(line_contents[0])
            self.y.append(line_contents[1])
            self.r.append(line_contents[2])
            self.large.append(line_contents[3].strip())
        self.input_file.close()
        self.x = list(map(float, self.x))
        self.y = list(map(float, self.y))
        self.r = list(map(float, self.r))
        self.large = list(map(bool, self.large))
        self.total_nr_of_particles = len(self.x)
        self.average_particle_radius = np.mean(self.r)

    def print_particles(self, output_name):
        out_file = open(output_name, "w")
        for i in range(self.total_nr_of_particles):
            out_file.write(str(self.x[i]) + "\t" + str(self.y[i]) + "\n")

    def print_grid(self, output_name):
        out_file = open(output_name, "w")
        for i in range(self.grid.nr_of_cells):
            out_file.write(
                str(self.grid.center_x[i]) + "\t" + str(self.grid.center_y[i]) + "\n"
            )

    def _extract_neighborhood(self, packing_index, particle_indices, particle_frozen):
        center_x = self.grid.center_x[packing_index]
        center_y = self.grid.center_y[packing_index]
        nr_of_mobile_particles = self.grid.nr_of_mobile_particles

        # adaptively compute mobile particle radius starting from a guess
        mobile_particle_radius = sqrt(
            nr_of_mobile_particles / self.grid.number_density / pi
        )  # initial guess, based on particle density
        nr_mobile_found = self._get_nr_particles_in_circle(
            center_x, center_y, mobile_particle_radius
        )
        nr_iterations = 0
        while nr_mobile_found != nr_of_mobile_particles:
            nr_iterations += 1
            if nr_iterations > 100:
                nr_iterations = 1
                (
                    center_x,
                    center_y,
                ) = self._change_center_pathological_configuration(center_x, center_y)
            mobile_particle_radius = self._adapt_radius(
                mobile_particle_radius,
                nr_mobile_found,
                nr_of_mobile_particles,
                nr_iterations,
            )
            nr_mobile_found = self._get_nr_particles_in_circle(
                center_x, center_y, mobile_particle_radius
            )

        # note that self.mobile_particle_radius and self.frozen_particle_radius are instantiated as members
        self.mobile_particle_radius = mobile_particle_radius
        self.frozen_particle_radius = (
            self.mobile_particle_radius
            + self.frozen_shell_thickness * (2 * self.average_particle_radius)
        )
        for i in range(self.total_nr_of_particles):
            dd = la.norm([self.x[i] - center_x, self.y[i] - center_y])
            if dd <= self.frozen_particle_radius:
                particle_indices.append(i)
                if dd <= self.mobile_particle_radius:
                    particle_frozen.append(False)
                    self._check_distance_to_boundary(i)
                else:
                    particle_frozen.append(True)

    def _get_nr_particles_in_circle(self, center_x, center_y, radius):
        return np.count_nonzero(
            [
                la.norm([x - center_x, y - center_y]) <= radius
                for x, y in zip(self.x, self.y)
            ]
        )

    def _adapt_radius(
        self, old_radius, found_particles, desired_particles, nr_iterations
    ):
        coupling = 1.0 / nr_iterations  # can be adapted to damp oscillations
        return old_radius * (
            (1 - coupling) + coupling * sqrt(desired_particles / found_particles)
        )

    def _check_distance_to_boundary(self, idx):
        safe_nr_diameters = 2  # depends on boundary shape of experimental packing
        safe_distance = safe_nr_diameters * (2 * self.average_particle_radius)
        if (
            min(
                [
                    abs(self.x[idx] - self.grid.max_x),
                    abs(self.x[idx] - self.grid.min_x),
                    abs(self.y[idx] - self.grid.max_y),
                    abs(self.y[idx] - self.grid.min_y),
                ]
            )
            < safe_distance
        ):
            raise Exception(
                "Experimental_Packing: distance to boundary smaller than %f average diameters"
                % safe_nr_diameters
            )

    def _change_center_pathological_configuration(self, center_x, center_y):
        displacement = 0.2 * self.average_particle_radius
        center_x += displacement
        center_y += displacement
        return center_x, center_y

    def _get_small_packing_information(self, indices, frozen):
        x = [self.x[idx] for idx in indices]
        y = [self.y[idx] for idx in indices]
        z = np.zeros(len(indices))
        d = [2 * self.r[idx] for idx in indices]
        f = frozen
        # here one could shift, rescale the coordinates, as convenient for simulations
        return Small_Packing_Information(x, y, z, d, f)

    def extract_small_packing(self, packing_index):
        small_packing_particles_indices = []
        small_packing_particles_frozen = []
        self._extract_neighborhood(
            packing_index,
            small_packing_particles_indices,
            small_packing_particles_frozen,
        )
        return self._get_small_packing_information(
            small_packing_particles_indices, small_packing_particles_frozen
        )


class Cut_Out_Packings(object):
    """
    Out of the data set with the given index, this cuts out a given number of
    packings with the specified number of (non-frozen) particles.
    """

    def __init__(self):
        self._read_parameters()
        self._read_experimental_data()
        if self.all:
            self.nr_of_packings = self.all_particles.grid.nr_of_cells
        if self.all_particles.grid.nr_of_cells < self.nr_of_packings:
            sys.stderr.write(
                "WARNING: number of requested packings (%d) was larger than possible\n for chosen grid/parameters (max. %d); reset to max.\n"
                % (self.nr_of_packings, self.all_particles.grid.nr_of_cells)
            )
            self.nr_of_packings = self.all_particles.grid.nr_of_cells

    def _read_parameters(self):
        self.parser = argparse.ArgumentParser(description="Split experimental data.")
        self.parser.add_argument(
            "--data_set_index",
            type=int,
            nargs="?",
            default=0,
            help="selects experimental dataset",
        )
        self.parser.add_argument(
            "--nr_of_particles",
            type=int,
            nargs="?",
            default=8,
            help="number of non-frozen particles",
        )
        self.parser.add_argument(
            "--nr_of_packings",
            type=int,
            nargs="?",
            default=10,
            help="number of extracted packings",
        )
        self.parser.add_argument(
            "--path_to_data",
            type=str,
            nargs="?",
            default="data",
            help="path to data files",
        )
        self.parser.add_argument(
            "--distance_from_boundary_x",
            type=float,
            nargs="?",
            default=0.04,
            help="discarded margins left and right, per-cent",
        )
        self.parser.add_argument(
            "--distance_from_boundary_y",
            type=float,
            nargs="?",
            default=0.04,
            help="discarded margins bottom and top, per-cent",
        )
        self.parser.add_argument(
            "--frozen_shell_thickness",
            type=float,
            nargs="?",
            default=2,
            help="number of average particle diameters in frozen shell",
        )
        self.parser.add_argument(
            "--grid_version",
            type=int,
            nargs="?",
            default=0,
            help="selects type of grid for splitting",
        )
        self.parser.add_argument(
            "--all",
            action="store_true",
            help="extract maximum number of packings",
        )
        self.args = self.parser.parse_args()
        if self.args.nr_of_particles <= 0:
            raise Exception("expect finite number of particles")
        if self.args.nr_of_packings <= 0:
            raise Exception("expect finite number of packings")
        if self.args.data_set_index < 0 or self.args.data_set_index >= 4:
            raise Exception("expect data set index to be between 0 and 3")
        self.data_file_idx = [1, 3, 4, 5]
        self.data_set_index = self.data_file_idx[self.args.data_set_index]
        self.nr_of_particles = self.args.nr_of_particles
        self.nr_of_packings = self.args.nr_of_packings
        self.data_file_name = "PackingsData_%d.dat" % self.data_set_index
        self.path_to_data = self.args.path_to_data
        self.path_to_file = "/".join([self.path_to_data, self.data_file_name])
        self.distance_from_boundary_x = self.args.distance_from_boundary_x
        self.distance_from_boundary_y = self.args.distance_from_boundary_y
        self.frozen_shell_thickness = self.args.frozen_shell_thickness
        self.grid_version = self.args.grid_version
        self.all = self.args.all

    def _read_experimental_data(self):
        self.all_particles = Experimental_Packing(
            self.path_to_file,
            self.nr_of_particles,
            self.distance_from_boundary_x,
            self.distance_from_boundary_y,
            self.frozen_shell_thickness,
            self.grid_version,
            self.all,
        )

    def _test_grid(self):
        self.all_particles.print_particles("test_particles.xy")
        self.all_particles.print_grid("test_grid.xy")

    def _split_packings(self):
        self.small_packings = []
        for i in range(self.nr_of_packings):
            self._find_one_small_packing(i)
            print("found packing %d of %d" % (i + 1, self.nr_of_packings))

    def split_packings(self):
        """
        returns a list of Small_Packing_Information objects
        """
        self._split_packings()
        return self.small_packings

    def _find_one_small_packing(self, index):
        self.small_packings.append(self.all_particles.extract_small_packing(index))

    def _dump_packings(self):
        self.path_to_output_small_packings = "/".join([self.path_to_data, "output"])
        for i in range(len(self.small_packings)):
            self._print_small_packing(i, self.small_packings[i])

    def _print_small_packing(self, packing_index, packing_information):
        if not os.path.exists(self.path_to_output_small_packings):
            os.makedirs(self.path_to_output_small_packings)
        self.grid_descriptor = str(self.grid_version)
        if self.all:
            self.grid_descriptor = "all"
        output_file = open(
            "/".join(
                [
                    self.path_to_output_small_packings,
                    "packing_"
                    + str(packing_index)
                    + "_nr_particles_"
                    + str(self.nr_of_particles)
                    + "_exp_file_"
                    + str(self.data_set_index)
                    + "_grid_option_"
                    + self.grid_descriptor
                    + ".xydf",
                ]
            ),
            "w",
        )
        # here one could print an extra file with only the frozen particles of the packing
        # output_file_f = open("/".join([self.path_to_output_small_packings,"frozen_only_split_packing_"+str(packing_index)+".xyzdf"]), "w")
        for i in range(len(packing_information.x)):
            output_file.write(
                "{:<12}\t{:<12}\t{:<12}\t{:<12}\n".format(
                    packing_information.x[i],
                    packing_information.y[i],
                    packing_information.d[i],
                    packing_information.f[i],
                )
            )
            # if packing_information.f[i]==True:
            #    output_file_f.write('{:<12}\t{:<12}\t{:<12}\t{:<12}\t{:<12}\n'.format(packing_information.x[i], packing_information.y[i], packing_information.z[i], packing_information.d[i], packing_information.f[i]))
        output_file.close()
        # output_file_f.close()

    def run(self):
        # self._test_grid()
        self._split_packings()
        self._dump_packings()


if __name__ == "__main__":
    sys = Cut_Out_Packings()
    sys.run()
