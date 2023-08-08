"""Module for a full run with the inverse power soft sphere potential.


The module should do the following 


1. Generate a list of jammed packings at a given packing fraction.
   These packings are the minima for the basins of attraction whose
   volumes we aim to calculate.
2. For each packing we find the kmax value
3. For each packing we find the kmin value
4. We run a bunch of Parallel Tempering simulations
5. Calculating the free energy difference (volume) between the kmax and kmin
   along with an estimate for volume for kmax should give us a volume estimate
   for kmin



Note that this module will not be able to capture the volume. We will need to
run longer jobs on the cluster. However it serves as a template for job scripts
"""

import numpy as np
import os
import yaml

from basinvolume.spheres import Findk_MCrunner
from soft_sphere_ensemble import setup_bidisperse


class BaseBasinVolumeCalculator:
    def __init__():
        """Abstract class for Basin Volume Calculations"""
        self.attractor = None

    def load_attractor(self):
        """
        Load the attractor for the runyaml
        """
        return None

    def find_kmax(self):
        """Finds the kmax corresponding to a sphere corresponding to 90 percent of the volume"""
        return None

    def find_kmin(self):
        """Finds the kmin that roughly overlaps 90 percent with the k=0 case"""
        return None

    def run_parallel_tempering(self):
        """Run parallel tempering simulations to generate `energy` series"""
        return None

    def calculate_volumes(self):
        """Calculates volumes based off the series"""
        return None


class SoftSphereRunner(BaseBasinVolumeCalculator):
    def __init__(
        self,
        base_folder,
        attractor_path,
    ):
        self.attractor_path = attractor_path
        self.attractor = np.load(attractor_path)

    def load_system_config(self):
        return 0

    def run(self):
        self.find_kmax()
        self.find_kmin()
        self.run_parallel_tempering()
        self.calculate_volumes()


def load_data(fpath, **kwargs):
    """Loading factory function forms a common interface for all files

    currently supports .csv and .npy files

    Parameters
    ----------
    fname : str
        Path to the file
    delimiter : str, optional
        any delimiter, by default None

    Returns
    -------
    np.ndarray
        The data in the file
    """
    extension = os.path.splitext(fpath)[1]

    if extension == ".csv":
        return np.loadtxt(fpath, delimiter=",", **kwargs)
    elif extension == ".npy":
        return np.load(fpath, **kwargs)
    else:
        raise ValueError(f"Extension {extension} not supported")


class SoftSphereBasinVolumeCalculator(BaseBasinVolumeCalculator):
    def __init__(
        self,
        attractor_path,
    ):
        self.attractor_path = attractor_path
        self.attractor_coords = load_data(attractor_path).flatten()
        self.attractor_directory = os.path.dirname(attractor_path)
        self.simulation_dir = os.path.basename(self.attractor_directory)
        with open(
            os.path.join(self.attractor_directory, "parameters.yaml")
        ) as param_f:
            self.parameters = yaml.load(param_f, Loader=yaml.UnsafeLoader)
        self.potential = setup_bidisperse(
            self.parameters, self.parameters["seed"]
        )["potential"]
        os.chdir(self.simulation_dir)

    def run(self):
        self.find_kmax()
        self.find_kmin()
        self.run_parallel_tempering()
        self.calculate_volumes()

    def find_kmax(self, k_guess=2000, niter=int(1e3), **kwargs):
        """Finds the kmax value for the attractor
        kmax is the spring constant for where 90 percent of the points in a
        random walk are in the basin of attraction

        Parameters
        ----------
        k_guess : int, optional
            _description_, by default 150
        niter : int, optional
            _description_, by default int(1e8)
        """
        stepsize = 1 / k_guess
        mcrunner = Findk_MCrunner(
            self.potential,
            self.attractor_coords,
            temperature=1.0,  # Temperature is redundant with the potential
            stepsize=stepsize,
            niter=niter,
            origin=self.attractor_coords,
            hs_radii=self.parameters["radii"],
            boxv=np.array(
                [self.parameters["box_length"]] * int(self.parameters["ndim"])
            ),
            sca=0,  # This means there is no hard shell in the potential
            rattlers=None,
            **kwargs,
        )
        mcrunner.run()
        self.kmax = mcrunner.kmax
        self.prob = mcrunner.prob

    def _print_results_kmax(self):
        fname = self.configfile
        f = open(fname, "a")
        f.write("[FINDK_MCRUNNER_STATUS]\n")
        status = self.mcrunner.get_status()
        for key, value in list(status.items()):
            f.write("{}: {}\n".format(key, value))
        f.write("[FINDK]\n")
        f.write("kmax: {:.16f}\n".format(self.kmax))
        f.write("prob: {:.16f}\n".format(self.prob))
        f.close()


if __name__ == "__main__":
    print("Running inverse soft sphere potential")
    # end to end test
    attractor_path = "/home/praharsh/simulation/packings/minimum_0.csv"

    bv = SoftSphereBasinVolumeCalculator(attractor_path)
    bv.find_kmax()
