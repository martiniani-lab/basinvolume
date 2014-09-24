"""
Usage:
To compute volumes and entropies for packings in a folder, say, ./n32_phi88_2D
run
python ~/PathToBasinvolume/basinvolume/post_processing/compute_entropy.py -d n32_phi88_2D
To skip the volume computation, run
python ~/PathToBasinvolume/basinvolume/post_processing/compute_entropy.py -d n32_phi88_2D -plot_only
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
1.) APFEntropy gives the p log p entropy according to Asenjo14:
10.1103/PhysRevLett.112.098002
2.) JackLogOmega gives the LogOmega entropy (log of number of basins), after
unbiasing the distribution with fit to generalised gaussian CDF and numerical
integration as in Asenjo14.
3.) MLLogOmega gives LogOmega from ML estimate for LogOmega, after maximum
likelihood fit of generalised gaussian.
TODO: 4.) BayesianLogOmega uses Bayesian inference to get the parameters of the
generalised gaussian.
TODO: 5.) NonParametricLogOmega constructs a non-parametric description of the
biased distribution and integrates that with the un-biasing factor to get
LogOmega without the assumption of the genealised gaussian.
This needs some kernel density estimation or smoothing or similar to get the
PDF description.
"""

from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.special import gamma
    from basinvolume.utils import to_string, save_pdf, log_factorial, ResultsFile, OutlierDetection, MomentsAcc, CDFAccumulator
    from basinvolume.post_processing import F_acc_Gaussian_Poly_HS_Fluid, APFEntropy, BestIntegrationSelection, VolumeSanityCheck, PackingFailureStatistics, OutlierRemovalUnbiasingEntropyLogOmega, GeneralisedGauss
    from basinvolume.post_processing import MLLogOmega, KernelDensityLogOmega
    from scipy import integrate
except ImportError as err:
    print err
                    
class ComputeEntropy(object):
    def __init__(self, packings_dir, plot_ts_integrand_data = False, skip_volume_computation = False, max_relative_GL_error = 0.2, kmax_threshold = 1000):
        self.packings_dir = packings_dir
        self.plot_ts_integrand_data = plot_ts_integrand_data
        self.skip_volume_computation = skip_volume_computation
        self.max_relative_GL_error = max_relative_GL_error
        self.kmax_threshold = kmax_threshold
        self.best_integration_selection = BestIntegrationSelection(max_relative_GL_error = self.max_relative_GL_error, kmax_threshold = self.kmax_threshold)
        self.volume_sanity_check = VolumeSanityCheck(self.packings_dir + "/packings/packings.config")
        self.explore_dirs = [self.packings_dir + "/" + f for f in os.listdir(self.packings_dir) if f.startswith("explore_bv_jammed_packing")]
        # compute F0 for all basins
        if not self.skip_volume_computation:
            self._compute_F0()
        # collect computed F0 data
        self._gather_data()
        # check that basins fit in box (w. HS constraints)
        self.best_integration_selection.perform_sanity_check_on_final_F0(self.volume_sanity_check)
        self.best_integration_selection.print_fail_information(self.packings_dir)
        # perform outlier removal
        self.F0_final_integration_selection = self.best_integration_selection.F0_final
        self.outlier_detection = OutlierDetection(self.F0_final_integration_selection, p=0.5, D=2*np.sqrt(np.var(self.F0_final_integration_selection)), verbose = True)
        self.F0_wo_outliers = np.asarray(self.outlier_detection.non_outliers)
        # plot various datasets
        self._generate_plots()
        # compute different entropies
        # fit to cdf, numerical integration for un-biasing
        self.outlier_removal_unbiasing_entropy_log_omega = OutlierRemovalUnbiasingEntropyLogOmega(self.F0_wo_outliers, self.packings_dir)
        self.outlier_removal_unbiasing_entropy_log_omega.compute_log_omega_entropy(self.volume_sanity_check)
        # -p log g entropy
        self.APF_entropy = APFEntropy(self.F0_wo_outliers, self.volume_sanity_check)
        self.APF_entropy.compute_and_write_entropy(self.packings_dir + "/entropy_AFP")
        # fit to pdf with ML method
        self.ML_log_omega = MLLogOmega(self.F0_wo_outliers, self.volume_sanity_check)
        self.ML_log_omega.compute_and_write_entropy(self.packings_dir + "/entropy_ML_LogOmega")
        # non-parametric: kernel density estimate of pdf plus numerical integration like for cdf fits
        self.kernel_density_log_omega = KernelDensityLogOmega(self.F0_wo_outliers, self.volume_sanity_check)
        self.kernel_density_log_omega.compute_and_write_entropy(self.packings_dir + "/entropy_kernel_density")
    def _compute_F0(self):
        self.packing_strings = ["jammed_" + (s.split("/")[-1]).split("_")[3] for s in self.explore_dirs]
        from basinvolume.spheres import _collect_u2_vs_k
        sim = _collect_u2_vs_k()
        self.packing_stat = PackingFailureStatistics(len(self.explore_dirs))
        for (path, fname) in zip(self.explore_dirs, self.packing_strings):
            try:
                sim(fname = fname, explore_dir = path, packings_dir = os.path.abspath(self.packings_dir + "/jammed_packings"), plot_ts_integrand_data = self.plot_ts_integrand_data)
                volf = ConfigParser.ConfigParser()
                volf.read(str(path + "/analysis/volume_data"))
                F0 = volf.getfloat('VOLUME_FULL_PT', 'F0')
                #self.volume_sanity_check.check(F0, "F0", path)
                self.packing_stat.add_success()
            except:
                print "failed packing!"
                print "name: ", fname
                print "path:", path
                self.packing_stat.add_failure()
            self.packing_stat.print_progress_info(fname)
        self.packing_stat.print_failure_info()
    def _gather_data(self):
        self.volume_files = [f + "/analysis/volume_data" for f in self.explore_dirs]
        self.F0 = []
        self.unit_box_F0 = []
        self.sigF0 = []
        self.F0_approx = []
        self.F0_approx_error = []
        self.unit_box_F0_approx = []
        self.F0_approx_PTu2k0 = []
        self.F0_approx_PTu2k0_error = []
        self.unit_box_F0_approx_PTu2k0 = []
        for vf in self.volume_files:
            self._read_from_volume_file(vf)
    def _read_from_volume_file(self, vf):
        volf = ConfigParser.ConfigParser()
        volf.read(str(vf))
        try:
            self.F0_approx.append(volf.getfloat('VOLUME_APPROXIMATED', 'F0_approx'))
            self.F0_approx_error.append(volf.getfloat("VOLUME_APPROXIMATED", "F0_approx_error"))
            self.unit_box_F0_approx.append(volf.getfloat('VOLUME_APPROXIMATED', 'unit_box_F0_approx'))
            self.F0_approx_PTu2k0.append(volf.getfloat('VOLUME_PTU2_APPROXIMATED', 'F0_approx_PTu2k0'))
            self.F0_approx_PTu2k0_error.append(volf.getfloat("VOLUME_PTU2_APPROXIMATED", "F0_approx_PTu2k0_error"))
            self.unit_box_F0_approx_PTu2k0.append(volf.getfloat('VOLUME_PTU2_APPROXIMATED', 'unit_box_F0_approx_PTu2k0'))
            self.F0.append(volf.getfloat('VOLUME_FULL_PT', 'F0'))
            self.unit_box_F0.append(volf.getfloat('VOLUME_FULL_PT', 'unit_box_F0'))
            self.sigF0.append(volf.getfloat('VOLUME_FULL_PT', 'sigF0'))
            try:
                self.best_integration_selection.check_next_F0(self.volume_sanity_check, self, vf)
            except Exception, e:
                print "Exception: ", e
                print "integration selection failed"
                print "location:", vf
        except Exception, e:
            print "Exception: ", e
            print "insufficient data available"
            print "location:", vf
    def _generate_plots(self):
        """
        try:
            self._print_histogram_and_data(self.F0, "/volume_histogram_F0")
            self._print_histogram_and_data(self.unit_box_F0, "/volume_histogram_unit_box_F0")
            self._print_histogram_and_data(self.F0_approx, "/volume_histogram_F0_approx")
            self._print_histogram_and_data(self.unit_box_F0_approx, "/volume_histogram_unit_box_F0_approx")
        except RuntimeError as err:
            print err
        """
        try:
            self._print_histogram_and_data(self.best_integration_selection.F0_final, "/volume_histogram_F0_final")
            self._print_histogram_and_data(self.F0_wo_outliers, "/volume_histogram_F0_final_removed_outliers")
        except RuntimeError as err:
            print err
    def _print_histogram_and_data(self, data, name):
        np.savetxt(self.packings_dir + name + ".data", data)
        desired_binsize = 1.5
        bins = np.abs(np.amax(data) - np.amin(data)) / desired_binsize
        hist, bin_edges = np.histogram(data, density = True, bins = bins)
        plt.hist(data, bins = bins, normed = True)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        def _gauss(x, sig, mu):
            return 1 / np.sqrt(2 * np.pi * sig ** 2) * np.exp( -(x - mu) ** 2 / (2 * sig ** 2))
        opt, error = curve_fit(_gauss, bin_centres, hist, [np.sqrt(np.var(data)), np.mean(data)])
        gauss_fit_opt = opt
        gauss_fit_opt[0] = np.abs(gauss_fit_opt[0]) #make printed sigma positive
        gauss_fit_error = error
        gauss_fit_names = ["sigma", "mean"]
        gauss_fit = [gauss_fit_opt, gauss_fit_names]
        generalised_gauss = GeneralisedGauss(alpha_min = 0.01, zeta_min = 0.01)
        print "name:", name
        generalised_gauss.fit(bin_centres, hist)
        gen_gauss_fit_opt = [generalised_gauss.mu_fit, generalised_gauss.alpha_fit, generalised_gauss.zeta_fit]
        gen_gauss_fit_error = generalised_gauss.fit_error
        gen_gauss_fit_names = ["mean", "alpha", "zeta"]
        gen_gauss_fit = [gen_gauss_fit_opt, gen_gauss_fit_names]
        def _set_hist_basics(plt):
            xp = np.linspace(bin_centres[0], bin_centres[-1], num = 500)
            plt.plot(xp, [_gauss(xpi, opt[0], opt[1]) for xpi in xp], "g--", label = "Gaussian")
            plt.plot(xp, [generalised_gauss.get_fitted(xpi) for xpi in xp], "r", label = "Generalised Gaussian")
            plt.legend()
            plt.xlabel(r"Free energy $F$")
            plt.ylabel(r"Probability density")
        #plot in lin-lin scale
        _set_hist_basics(plt)
        save_pdf(plt, self.packings_dir + name + ".pdf")
        plt.hist(data, bins = bins, normed = True)
        #plot in ylog scale
        _set_hist_basics(plt)
        plt.yscale('log', nonposy='clip')
        plt.axis(ymin = 0.25 / len(data))
        save_pdf(plt, self.packings_dir + name + "_ylog" + ".pdf")
        self._print_fitting_results(name, gauss_fit, gen_gauss_fit)
    def _print_fitting_results(self, name, gauss_fit, gen_gauss_fit):
        self.fit_results_dir = self.packings_dir + name + ".fit_results"
        f = ResultsFile(self.fit_results_dir)
        def _print_function_parameters(fit_info):
            for parameter in xrange(len(fit_info[0])):
                f.to_file(fit_info[1][parameter], fit_info[0][parameter])
        f.set_heading("GAUSS_FIT_PARAMETERS")
        _print_function_parameters(gauss_fit)
        f.set_heading("GENERALISED_GAUSS_FIT_PARAMETERS")
        _print_function_parameters(gen_gauss_fit)
        f.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare F0 form full PT data with F0 from integral approximation")
    parser.add_argument("-d", "--packings_dir", type = str, help = "top-level dir containing the packings, e.g. n32_phi88_2D")
    parser.add_argument("-plot_only", "--plot_only", action='store_true', help = "flag to switch off the actual volume computing and to only do the plotting part")
    # if the relative error of the GL integral over the PT data is estimatedto be larger than max_relative_GL_error, the approximated integral is used instead to compute F0
    parser.add_argument("-max_relative_GL_error", "--max_relative_GL_error", default = 0.1, type = float, help = "parameter that selects between GL integral from PT data and approx integral")
    parser.add_argument("-kmax_threshold", "--kmax_threshold", default = 1000, type = float, help = "largest kmax value that is not considered to be huge")
    args = parser.parse_args()
    packings_dir = os.path.abspath(args.packings_dir)
    ComputeEntropy(packings_dir, plot_ts_integrand_data = False, skip_volume_computation = args.plot_only, max_relative_GL_error = args.max_relative_GL_error, kmax_threshold = args.kmax_threshold)
    
    
