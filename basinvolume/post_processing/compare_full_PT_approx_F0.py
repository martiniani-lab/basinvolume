from __future__ import division
try:
    import numpy as np
    import argparse
    import ConfigParser
    import os
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.optimize import curve_fit
    from scipy.special import gamma
    from basinvolume.utils import to_string, log_factorial, ResultsFile, OutlierDetection
    from basinvolume.post_processing import F_acc_Gaussian_Poly_HS_Fluid
except ImportError as err:
    print err

def save_pdf(plt, file_name):
    pdf = PdfPages(file_name)
    plt.savefig(pdf, format="pdf")
    pdf.close()
    plt.close()
            
class PackingFailureStatistics(object):
    def __init__(self, total_nr):
        self.total_nr = total_nr
        self.total_count = 0
        self.success_count = 0
    def add_success(self):
        self.add_any()
        self.success_count += 1
    def add_failure(self):
        self.add_any()
    def add_any(self):
        self.total_count += 1
    def get_nr_failures(self):
        return self.total_count - self.success_count
    def print_failure_info(self):
        print self.get_nr_failures(), "out of", self.total_count, "failed"
        print "corresponding failure ratio", self.get_nr_failures() / self.total_count
        print 100 * self.get_nr_failures() / self.total_count, "per-cent"
    def print_progress_info(self, packing_string):
        print "done", self.total_count, "out of", self.total_nr 
        print to_string(self.total_count / self.total_nr * 100, 2), "per-cent"
        print "packing was", packing_string
        
class VolumeSanityCheck(object):
    def __init__(self, v_acc_parameter_file):
        self.v_acc_parameter_file = v_acc_parameter_file
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.v_acc_parameter_file))
        self.nr_particles = configf.getint("PACKING", "nparticles")
        self.box_dimension = configf.getint("PACKING", "boxdim")
        self.phiHD = configf.getfloat("PACKING", "packing_fraction")
        boxv = configf.get("PACKING", "boxv")
        boxv = np.array([float(x) for x in boxv.split()])
        self.V_box = np.prod(boxv)
        self.diameter_mean = 2 * configf.getfloat("PACKING", "radii_mean")
        radii_stdev = configf.getfloat("PACKING", "radii_stdev")
        self.diameter_variance = (2 * radii_stdev) ** 2
        self.ideal_gas_V_acc = self.V_box ** self.nr_particles
        self.F0_acc = F_acc_Gaussian_Poly_HS_Fluid(self.phiHD, self.V_box, self.nr_particles, self.box_dimension, self.diameter_mean, self.diameter_variance) 
        self.V_acc = np.exp(- self.F0_acc)
        if np.log(self.V_acc) > np.log(self.ideal_gas_V_acc):
            raise Exception("VolumeSanityCheck: polyHS fluid failure")
        print "VolumeSanityCheck: "
        print "F0_acc, HS fluid", self.F0_acc
        print "F0_acc, ideal gas", - np.log(self.ideal_gas_V_acc)
    def is_insane(self, F0):
        if F0 < self.F0_acc:
            return True
        else:
            return False
    def check(self, F0, F0_name, vf_path):
        if F0 < self.F0_acc:
            print "failed F0 value", F0
            print "-log(V_acc)", self.F0_acc
            print "failed F0 name", F0_name
            print "failed packing", ([f for f in vf_path.split("/") if "jammed_packing" in f][0])[11:]
            raise Exception("VolumeSanityCheck: illegal free energy")
        if F0 < - np.log(self.ideal_gas_V_acc):
            print "failed F0 value -- failed ideal gas box test"
            print "-log(V_acc, ideal)", - np.log(self.ideal_gas_V_acc)
            print "failed F0 name", F0_name
            print "failed packing", ([f for f in vf_path.split("/") if "jammed_packing" in f][0])[11:]
            raise Exception("VolumeSanityCheck: illegal free energy")

class GeneralisedGauss(object):
    """
    Implements the generalised gaussian distribution, see e.g.: http://en.wikipedia.org/wiki/Generalized_normal_distribution
    Parameters are as follows:
    PDF(mu, alpha, zeta; x) = zeta / (2 * alpha * gamma(1 / zeta)) * exp[-|x - mu|**zeta / alpha**zeta]
    """
    def __init__(self, mu_initial = 1, alpha_initial = 1, zeta_initial = 1, alpha_min = 1e-10, zeta_min = 1e-10):
        if alpha_min < 0:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: alpha_min")
        if zeta_min < 0:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: zeta_min")
        self.alpha_min = alpha_min
        self.zeta_min = zeta_min
        self.mu = mu_initial
        self.alpha_offset = alpha_initial - self.alpha_min
        self.zeta_offset = zeta_initial - self.zeta_min
    def set_mu(self, mu):
        self.mu = mu
    def set_alpha(self, alpha):
        if alpha < self.alpha_min:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: alpha")
        self.alpha_offset = alpha - self.alpha_min
    def set_zeta(self, zeta):
        if zeta < self.zeta_min:
            raise Exception("GeneralisedGauss: attempt to set illegal parameter value: zeta")
        self.zeta_offset = zeta - self.zeta_min
    def get_alpha(self, alpha_offset):
        return np.abs(alpha_offset) + self.alpha_min
    def get_zeta(self, zeta_offset):
        return np.abs(zeta_offset) + self.zeta_min
    def get(self, x, mu, alpha_offset, zeta_offset):
        return self.get_zeta(zeta_offset) / (2 * self.get_alpha(alpha_offset) * gamma(1 / self.get_zeta(zeta_offset))) * np.exp(- np.power((np.abs(x - mu) / self.get_alpha(alpha_offset)), self.get_zeta(zeta_offset)))
    def get_times_expx(self, x, mu, alpha_offset, zeta_offset):
        return self.get_zeta(zeta_offset) / (2 * self.get_alpha(alpha_offset) * gamma(1 / self.get_zeta(zeta_offset))) * np.exp(- np.power((np.abs(x - mu) / self.get_alpha(alpha_offset)), self.get_zeta(zeta_offset)) + x)
    def get_fitted(self, x):
        return self.get(x, self.mu, self.alpha_offset, self.zeta_offset)
    def get_fitted_times_expx(self, x):
        return self.get_times_expx(x, self.mu, self.alpha_offset, self.zeta_offset)
    def fit(self, data_x, data_y):
        opt_gen, error_gen = curve_fit(self.get, data_x, data_y, [np.mean(data_x), 2 * np.var(data_x), 2])
        self.mu = opt_gen[0]
        self.alpha_offset = opt_gen[1]
        self.zeta_offset = opt_gen[2]
        self.mu_fit = self.mu
        self.alpha_fit = self.get_alpha(self.alpha_offset)
        self.zeta_fit = self.get_zeta(self.zeta_offset)
        self.fit_error = error_gen
        print "self.mu", self.mu
        print "self.alpha_offset", self.alpha_offset
        print "self.zeta_offset", self.zeta_offset

class GLPTNotUsedStatistics(object):
    """
    Collect statistics on how many times the GL (PT) result was not used due to
    large integration error (for huge kmax).
    In these cases the analytical intrgral approximation is used.
    The number of these cases should be realtively low.
    """
    def __init__(self):
        self.total_nr = 0
        self.used_approx = 0
    def add_PT_GL(self):
        self.add_any()
    def add_approx(self):
        self.add_any()
        self.used_approx += 1
    def add_any(self):
        self.total_nr += 1
    def get_approx_use_fraction(self):
        return self.used_approx / self.total_nr
    def print_statistics(self):
        print "GLPTNotUsedStatistics:"
        print "total number of F0 values:", self.total_nr
        print "number of times GL failed:", self.used_approx
        print "GL failure (approx usage) fraction:", self.get_approx_use_fraction()

class BestIntegrationSelection(object):
    """
    Handles part of the analysis of F0 values.
    This version only considers packings where both, the kindk runs, and PT
    completed successfully.
    In case there is a huge kmax, we allow for the GL integration to fail and
    use an analytic approximation instead.
    In case there is no huge kmax and the GL integration still fails, something
    went wrong and we throw a warning and exception.
    In case the final F0, be it from GL integration or from the approximated
    integral, fails the constraint given by the box size, we throw a warning
    and exception.
    """
    def __init__(self, max_relative_GL_error = 0.2, kmax_threshold = 1000):
        if max_relative_GL_error < 0:
            raise Exception("BestIntegrationSelection: illegal input: max_relative_GL_error")
        self.max_relative_GL_error = max_relative_GL_error
        if kmax_threshold < 0:
            raise Exception("BestIntegrationSelection: illegal input: kmax_threshold")
        self.kmax_threshold = kmax_threshold
        self.F0_final = []
        self.F0_error_final = []
        self.bad_volumes_larger_than_Vacc = []
        self.bad_volumes_failed_GL_integration = []
        self.bad_volumes_huge_kmax = []
        self.GLPT_not_used_statistics = GLPTNotUsedStatistics()
    def check_next_F0(self, volume_sanity_check, volume_data, volume_file_path):
        F0 = volume_data.F0[-1]
        F0_error = volume_data.sigF0[-1]
        F0_approx_PTu2k0 = volume_data.F0_approx_PTu2k0[-1]
        F0_approx_PTu2k0_error = volume_data.F0_approx_PTu2k0_error[-1]
        kmax = self.get_kmax(volume_file_path)
        fail_information = "F0 from PT:", to_string(F0, 3), "kmax:", to_string(kmax, 3), "packing_label:", ([f for f in volume_file_path.split("/") if "jammed_packing" in f][0])[11:]
        if volume_sanity_check.is_insane(F0):
            self.bad_volumes_larger_than_Vacc.append(fail_information)
        if self.kmax_is_huge(kmax):
            self.bad_volumes_huge_kmax.append(fail_information)
        if (np.abs(F0_error) / np.abs(F0)) > self.max_relative_GL_error:
            if self.kmax_is_huge(kmax) is False:
                print "---WARNING: GL integration failed, even though kmax is not huge!---"
                raise Exception("GL integration failed, even though kmax is not huge!")
            if volume_sanity_check.is_insane(F0_approx_PTu2k0):
                print "---WARNING: GL integration failed and approximation is also wrong!---"
                raise Exception("GL integration failed and approximation is also wrong!")
            else:
                self.F0_final.append(F0_approx_PTu2k0)
                self.F0_error_final.append(F0_approx_PTu2k0_error)
                self.bad_volumes_failed_GL_integration.append(fail_information)
                self.GLPT_not_used_statistics.add_approx()
        else:
            self.F0_final.append(F0)
            self.F0_error_final.append(F0_error)
            self.GLPT_not_used_statistics.add_PT_GL()
    def get_kmax(self, volume_file):
        path_with_kmax_info_file = os.path.split(os.path.split(volume_file)[0])[0]
        kmax_file = [path_with_kmax_info_file + "/" + f for f in os.listdir(path_with_kmax_info_file) if f.endswith(".config") and f.startswith("findk_jammed_packing")][0]
        configf = ConfigParser.ConfigParser()
        configf.read(str(kmax_file))
        return configf.getfloat("FINDK", "kmax")
    def kmax_is_huge(self, kmax):
        if kmax > self.kmax_threshold:
            return True
        else:
            return False
    def perform_sanity_check_on_final_F0(self, volume_sanity_check):
        self.GLPT_not_used_statistics.print_statistics()
        if len(self.F0_final) is not len(self.F0_error_final):
            raise Exception("BestIntegrationSelection: perform_sanity_check_on_final_F0: error handling failed")
        for F0_final in self.F0_final:
            if volume_sanity_check.is_insane(F0_final):
                raise Exception("BestIntegrationSelection: perform_sanity_check_on_final_F0: F0 is insane")
    def print_fail_information(self, packings_dir):
        def failed_to_file(path, info):
            if len(info) is 0:
                return
            f = open(path, "w")
            for line in info:
                f.write("failed packing:\n")
                for word in line:
                    f.write(str(word) + " ")
                f.write("\n")
            f.close()
        failed_to_file(packings_dir + "/bad_volumes_larger_than_Vacc", self.bad_volumes_larger_than_Vacc)
        failed_to_file(packings_dir + "/bad_volumes_failed_GL_integration", self.bad_volumes_failed_GL_integration)
        failed_to_file(packings_dir + "/bad_volumes_huge_kmax", self.bad_volumes_huge_kmax)

class F0MeanError(object):
    """
    The mean is not weithed by the error because we want to have the mean as
    sampled with bias, see APFEntropy below.
    """
    def __init__(self, F0, error_F0):
        if len(F0) is not len(error_F0):
            raise Exception("F0MeanError: illegal shape of F0, F0_error arrays")
        F0 = np.array(F0)
        error_F0 = np.array(error_F0)
        self.mean = np.mean(F0)
        squared_errors = error_F0 ** 2
        self.error = np.sqrt(np.sum(squared_errors)) / len(squared_errors)
        self.sample_variance_error = np.sqrt(np.var(F0) / len(F0))

class APFEntropy(object):
    def __init__(self, F0, error_F0, volume_sanity_check):
        self.F0 = F0
        self.error_F0 = error_F0
        self.F0_acc = volume_sanity_check.F0_acc
        self.V_acc = volume_sanity_check.V_acc
        self.nr_particles = volume_sanity_check.nr_particles
    def compute_and_write_entropy(self, entropy_file_path):
        """
        Granular entropy according to Asenjo14: 10.1103/PhysRevLett.112.098002
        S_\text{APF}^* = \langle F \rangle_\text{biased} + \log(V_\text{acc})
        S_\text{APF} = S_\text{APF}^* - \log(N!)
        """
        F0_stat = F0MeanError(self.F0, self.error_F0)
        self.S_star = F0_stat.mean - self.F0_acc
        if self.S_star < 0:
            raise Exception("APFEntropy: compute_and_write_entropy: entropy computation failed")
        self.S = self.S_star - log_factorial(self.nr_particles)
        #self.error_S_star = F0_stat.error
        self.error_S_star = F0_stat.sample_variance_error
        self.error_S = self.error_S_star
        print "Granular entropy according to APF:"
        print "S_star:", self.S_star, "+/-", self.error_S_star 
        print "S:", self.S, "+/-", self.error_S
        self.write_to_file(entropy_file_path)
    def write_to_file(self, entropy_file_path):
        f = ResultsFile(entropy_file_path)
        f.set_heading("ENTROPY_APF")
        f.to_file("S_star", self.S_star)
        f.to_file("error_S_star", self.error_S_star)
        f.to_file("S", self.S)
        f.to_file("error_S", self.error_S)
        f.close()

class OutlierRemovalUnbiasingEntropyLogOmega(object):
    """
    Fits generalised gaussian to histogram of F0 data.
    Integrates that to get the unbiased mean basin volume.
    Uses that to compute
    S^\star = \log \Omega = \log(V_\text{acc}) - \log(unbiased_mean_volume)
    and
    S = S^\star - \log(N!)
    """
    def __init__(self, F0):
        self.F0_full = F0
        self.outlier_detection = OutlierDetection(F0, p = 0.5, D = 10, verbose = True)
        self.F0 = self.outlier_detection.non_outliers
    def compute_log_omega_entropy(self, volume_sanity_check):
        self.generalised_gauss = GeneralisedGauss(alpha_min = 0.01, zeta_min = 0.01)
        desired_binsize = 0.8
        bins = np.abs(np.amax(self.F0) - np.amin(self.F0)) / desired_binsize
        hist, bin_edges = np.histogram(self.F0, density = True, bins = bins)
        #plt.hist(self.F0, bins = bins, normed = True)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.generalised_gauss.fit(bin_centres, hist)
        self.compute_integral(volume_sanity_check)
        self.S_star = - volume_sanity_check.F0_acc + np.log(self.integral)
        self.S = self.S_star - log_factorial(volume_sanity_check.nr_particles)
        self.error_S_star = None
        self.error_S = None
        self.write_to_file()
        assert(self.S_star > 0)
    def compute_integral(self, volume_sanity_check):
        from scipy import integrate
        self.integral = integrate.quad(self.generalised_gauss.get_fitted_times_expx, volume_sanity_check.F0_acc, np.inf)[0]
        assert(self.integral > 0)
    def write_to_file(self):
        print "Log of Omega entropy:"
        print "S_star:", self.S_star, "+/-", self.error_S_star 
        print "S:", self.S, "+/-", self.error_S

class GenerateComparisonPlotPTApprox(object):
    def __init__(self, packings_dir, plot_ts_integrand_data = False, skip_volume_computation = False, max_relative_GL_error = 0.2, kmax_threshold = 1000):
        self.packings_dir = packings_dir
        self.plot_ts_integrand_data = plot_ts_integrand_data
        self.skip_volume_computation = skip_volume_computation
        self.max_relative_GL_error = max_relative_GL_error
        self.kmax_threshold = kmax_threshold
        self.best_integration_selection = BestIntegrationSelection(max_relative_GL_error = self.max_relative_GL_error, kmax_threshold = self.kmax_threshold)
        self.volume_sanity_check = VolumeSanityCheck(self.packings_dir + "/packings/packings.config")
        self.explore_dirs = [self.packings_dir + "/" + f for f in os.listdir(self.packings_dir) if f.startswith("explore_bv_jammed_packing")]
        if not self.skip_volume_computation:
            self._compute_F0()
        self._gather_data()
        self.best_integration_selection.perform_sanity_check_on_final_F0(self.volume_sanity_check)
        self.outlier_removal_unbiasing_entropy_log_omega = OutlierRemovalUnbiasingEntropyLogOmega(self.best_integration_selection.F0_final)
        self.outlier_removal_unbiasing_entropy_log_omega.compute_log_omega_entropy(self.volume_sanity_check)
        self._generate_plots()
        self.best_integration_selection.print_fail_information(self.packings_dir)
        self.APF_entropy = APFEntropy(self.best_integration_selection.F0_final, self.best_integration_selection.F0_error_final, self.volume_sanity_check)
        self.APF_entropy.compute_and_write_entropy(self.packings_dir + "/entropy_AFP")
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
        self._print_histogram_and_data(self.F0, "/volume_histogram_F0")
        self._print_histogram_and_data(self.unit_box_F0, "/volume_histogram_unit_box_F0")
        self._print_histogram_and_data(self.F0_approx, "/volume_histogram_F0_approx")
        self._print_histogram_and_data(self.unit_box_F0_approx, "/volume_histogram_unit_box_F0_approx")
        self._print_histogram_and_data(self.best_integration_selection.F0_final, "/volume_histogram_F0_final")
        self._print_histogram_and_data(self.outlier_removal_unbiasing_entropy_log_omega.F0, "/volume_histogram_F0_final_removed_outliers")
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
    GenerateComparisonPlotPTApprox(packings_dir, plot_ts_integrand_data = False, skip_volume_computation = args.plot_only, max_relative_GL_error = args.max_relative_GL_error, kmax_threshold = args.kmax_threshold)
    
    
    