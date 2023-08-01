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


class BaseBasinVolumeCalculator:
    def __init__():
        """Abstract class for Basin Volume Calculations"""
        self.attractor = None

    def load_attractor(self):
        """
        Load the attractor for the run
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
    def __init__(foldpath):
        self.foldpath = foldpath

    def run(self):
        self.jammed_run()
        self.find_kmax()
        self.find_kmin()
        self.run_parallel_tempering()
        self.calculate_volumes()
