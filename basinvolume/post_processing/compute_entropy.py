"""
Usage:
To compute entropies for packings in a folder, say, ./n32_phi88_2D
run
python ~/PathToBasinvolume/basinvolume/post_processing/compute_entropy.py -d n32_phi88_2D
Entropy results are written to file
./n32_phi88_2D/entropy_AFP
and
./n32_phi88_2D/entropy_LogOmega
The fit used for the histogram un-biasing is written to
./n32_phi88_2D/unbiasing_fit.pdf
Other free energy histograms and data is written to
./n32_phi88_2D/volume_histogram_*

The entropy is computed in different ways according to different definitions
and techniques:
1.) APF Entropy gives the p log p entropy according to Asenjo14:
10.1103/PhysRevLett.112.098002
2.) JackLogOmega gives the LogOmega entropy (log of number of basins), after
unbiasing the distribution with fit to generalised gaussian CDF and numerical
integration as in Asenjo14.
3.) MLLogOmega gives LogOmega from ML estimate for LogOmega, after maximum
likelihood fit of generalised Gaussian.
TODO: 4.) BayesianLogOmega uses Bayesian inference to get the parameters of the
generalised Gaussian.
TODO: 5.) NonParametricLogOmega constructs a non-parametric description of the
biased distribution and integrates that with the un-biasing factor to get
LogOmega without the assumption of the genealised Gaussian.
This needs some kernel density estimation or smoothing or similar to get the
PDF description.
"""

from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import re
    import matplotlib.pyplot as plt
    import traceback
    import copy
    import multiprocessing as mp
    import pele.utils.fix_multiprocessing
    from scipy import integrate
    from scipy.optimize import curve_fit
    from scipy.special import gamma
    from basinvolume.utils import to_string, save_pdf, log_factorial
    from basinvolume.utils import ResultsFile, OutlierDetection
    from basinvolume.utils import MomentsAcc, CDFAccumulator, trymakedir
    from basinvolume.post_processing import APFEntropy
    from basinvolume.post_processing import VolumeSanityCheck, PackingFailureStatistics
    from basinvolume.post_processing import OutlierRemovalUnbiasingEntropyLogOmega
    from basinvolume.post_processing import GeneralisedGauss
    from basinvolume.post_processing import MLLogOmega, KernelDensityLogOmegaJackKnife
    from basinvolume.post_processing import PTFailures, assert_pt_success
    from basinvolume.post_processing import BasinAnalysis
except ImportError as err:
    print err
    
"""
def compute_entropy_etc(self):
    print("---numerical packing---")
    packing_configpath = self.get_packing_configpath(42)
    volume_sanity_check = VolumeSanityCheck(packing_configpath, numerical_moments=self.numerical_moments)
    # -p log g entropy
    self.APF_entropy = APFEntropy(self.F0_wo_outliers, volume_sanity_check)
    self.APF_entropy.compute_and_write_entropy(os.path.join(self.output_path, "entropy_AFP"))
    # non-parametric: kernel density estimate of pdf plus numerical integration like for cdf fits
    self.kernel_density_log_omega = KernelDensityLogOmegaJackKnife(self.F0_wo_outliers, volume_sanity_check)
    self.kernel_density_log_omega.compute_and_write_entropy(os.path.join(self.output_path, "entropy_kernel_density"))
    # fit to cdf, numerical integration for un-biasing
    self.outlier_removal_unbiasing_entropy_log_omega = OutlierRemovalUnbiasingEntropyLogOmega(self.F0_wo_outliers, self.output_path)
    try:
        self.outlier_removal_unbiasing_entropy_log_omega.compute_log_omega_entropy(volume_sanity_check)
    except Exception, e:
        print e
    # fit to pdf with ML method
    self.ML_log_omega = MLLogOmega(self.F0_wo_outliers, volume_sanity_check)
    try:
        self.ML_log_omega.compute_and_write_entropy(os.path.join(self.output_path, "entropy_ML_LogOmega"))
    except Exception, e:
        print e
"""

class ComputeEntropy(object):
    """
    Use either ComputeEntropyNumerical or ComputeEntropyExperimental,
    based on the name of the folder containing the MC data
    ("packings_dir").
    """
    def __init__(self, workspace):
        #
        self.workspace = os.path.abspath(workspace)
        #
        self.analysis = BasinAnalysis(workspace=self.workspace)
        self.analysis.collect_data_all_set()
        print ("self.analysis.packing_datasets", self.analysis.packing_datasets)
        for (nr_particles, data_set) in self.analysis.packing_datasets:
            self._compute_write_entropies(data_set)
    def _compute_write_entropies(self, data_set):
        print(nr_particles, data_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute entropy from F0 data obtained via independent compute_volumes script")
    parser.add_argument("-w", "--workspace", type=str, help="top-level dir containing the packings folders of format n32_phi88_2D", default=os.getcwd())
    args = parser.parse_args()
    ComputeEntropy(args.workspace)
