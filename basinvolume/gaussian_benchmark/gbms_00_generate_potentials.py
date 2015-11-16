from __future__ import division

from multi_gauss_wrap import MultiGaussWrap

def generate_potentials(potential_dir, nr_gaussians, nr_dimensions, nr_samples):
    for pot_index in xrange(nr_samples):
        pot = MultiGaussWrap(potential_dir)
        if not pot.exists(nr_gaussians, nr_dimensions, pot_index):
            pot.generate(nr_gaussians, nr_dimensions, pot_index)
            pot.select_benchmark_basins(nr_gaussians, nr_dimensions, pot_index)

if __name__ == "__main__":
    nr_samples = 20
    potential_dir = os.path.join(os.getcwd(), "potentials")
    for nr_gaussians in [5]:
        for nr_dimensions in [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]:
            generate_potentials(potential_dir, nr_gaussians, nr_dimensions, nr_samples)
