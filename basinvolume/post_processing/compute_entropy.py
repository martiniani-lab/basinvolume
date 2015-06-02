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
    from joblib import Parallel, delayed
except ImportError as err:
    print err

def _entropy(name, out_path, sanity_check, free_energies):
    computer = name(free_energies, sanity_check)
    try:
        computer.compute_and_write_entropy(out_path)
    except Exception as ex:
        print("Exception occured in ", name)
        print(ex)

def _compute_write_entropies(data_set):
    print("compute and write entropies for dataset with name", data_set.set_name)
    entropy_base_output_path = os.path.join(data_set.set_path, 'entropy_analysis_all')
    print("entropy_base_output_path", entropy_base_output_path)
    trymakedir(entropy_base_output_path)
    volume_sanity_check = VolumeSanityCheck(data_set.packing_data[0].configpath_packing)
    outliers = OutlierDetection(data_set.free_energies, p=0.5, D=3*np.std(data_set.free_energies))
    free_energies = np.array(data_set.free_energies)[np.array(outliers.non_outliers_indexes, dtype="i")]
    unbias_log_omega = OutlierRemovalUnbiasingEntropyLogOmega(free_energies, entropy_base_output_path)
    try:
        unbias_log_omega.compute_log_omega_entropy(volume_sanity_check)
    except Exception as ex:
        print("Exception occured in unbiasing for log omega:", ex)
    _entropy(APFEntropy, os.path.join(entropy_base_output_path, "entropy_APF"), volume_sanity_check, free_energies)
    _entropy(KernelDensityLogOmegaJackKnife, os.path.join(entropy_base_output_path, "entropy_kernel_density"), volume_sanity_check, free_energies)
    _entropy(MLLogOmega, os.path.join(entropy_base_output_path, "entropy_ML_LogOmega"), volume_sanity_check, free_energies)

def _compute_write_entropies_msf(data_set, ik=0.18878315, a=0.9807471):
    """
    compute free energy from pressure and use these to compute entropy
    """
    print("compute and write entropies for dataset with name", data_set.set_name)
    entropy_base_output_path = os.path.join(data_set.set_path, 'msf_entropy_analysis_all')
    print("entropy_base_output_path", entropy_base_output_path)
    trymakedir(entropy_base_output_path)
    volume_sanity_check = VolumeSanityCheck(data_set.packing_data[0].configpath_packing)
    outliers = OutlierDetection(data_set.free_energies, p=0.5, D=3*np.std(data_set.free_energies))
    gamma = np.array(data_set.pressures)[np.array(outliers.non_outliers_indexes, dtype="i")]
    #msf relation
    free_energies = (data_set.nparticles*ik)*np.log(gamma) + a*data_set.nparticles
    unbias_log_omega = OutlierRemovalUnbiasingEntropyLogOmega(free_energies, entropy_base_output_path)
    try:
        unbias_log_omega.compute_log_omega_entropy(volume_sanity_check)
    except Exception as ex:
        print("Exception occured in unbiasing for log omega:", ex)
    _entropy(APFEntropy, os.path.join(entropy_base_output_path, "entropy_APF"), volume_sanity_check, free_energies)
    _entropy(KernelDensityLogOmegaJackKnife, os.path.join(entropy_base_output_path, "entropy_kernel_density"), volume_sanity_check, free_energies)
    _entropy(MLLogOmega, os.path.join(entropy_base_output_path, "entropy_ML_LogOmega"), volume_sanity_check, free_energies)

class ComputeEntropy(object):
    """
    Read free energies with data set tools.
    Compute different entropies from them.
    """
    def __init__(self, workspace, ncores=6):
        #
        self.workspace = os.path.abspath(workspace)
        #
        self.analysis = BasinAnalysis(workspace=self.workspace)
        self.analysis.collect_data_all_set(no_pickle=True)
        results = Parallel(n_jobs=ncores)(delayed(_compute_write_entropies)(data_set) for data_set in self.analysis.packing_datasets)
        #results = Parallel(n_jobs=ncores)(delayed(_compute_write_entropies_msf)(data_set) for data_set in self.analysis.packing_datasets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute entropy from F0 data obtained via independent compute_volumes script")
    parser.add_argument("-w", "--workspace", type=str, help="top-level dir containing the packings folders of format n32_phi88_2D", default=os.getcwd())
    parser.add_argument("-j", "--ncores", type=int, help="number of parallel jobs to run, default 6", default=6)
    args = parser.parse_args()
    ComputeEntropy(args.workspace, ncores=args.ncores)
