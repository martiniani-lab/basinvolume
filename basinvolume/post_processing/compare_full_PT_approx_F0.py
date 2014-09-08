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
    from basinvolume.utils import to_string
    from basinvolume.post_processing import F_acc_Gaussian_Poly_HS_Fluid
except ImportError as err:
    print err

def save_pdf(plt, file_name):
    pdf = PdfPages(file_name)
    plt.savefig(pdf, format="pdf")
    pdf.close()
    plt.close()
    
class FitResultsFile(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.f = open(self.file_name, "w")
        self.f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
    def set_heading(self, title):
        self.f.write("[" + title + "]\n")
    def to_file(self, name, value):
        self.f.write((name + ": {}\n").format(to_string(value)))
    def close(self):
        self.f.close()
        
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
    def check(self, F0, F0_name, vf_path):
        if F0 < self.F0_acc :
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
    
class GenerateComparisonPlotPTApprox(object):
    def __init__(self, packings_dir, plot_ts_integrand_data = False):
        self.packings_dir = packings_dir
        self.plot_ts_integrand_data = plot_ts_integrand_data
        self._set_up_volume_check()
        self._compute_F0()
        self._gather_data()
        self._generate_plots()
    def _compute_F0(self):
        self.explore_dirs = [self.packings_dir + "/" + f for f in os.listdir(self.packings_dir) if f.startswith("explore_bv_jammed_packing")]
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
                self.volume_sanity_check.check(F0, "F0", path)
                self.packing_stat.add_success()
            except:
                print "failed packing!"
                print "name: ", fname
                print "path:", path
                self.packing_stat.add_failure()
            self.packing_stat.print_progress_info(fname)
        self.packing_stat.print_failure_info()
    def _set_up_volume_check(self):
        self.v_acc_parameter_file = self.packings_dir + "/packings/packings.config"
        self.volume_sanity_check = VolumeSanityCheck(self.v_acc_parameter_file)
    def _gather_data(self):
        self.volume_files = [f + "/analysis/volume_data" for f in self.explore_dirs]
        self.F0 = []
        self.unit_box_F0 = []
        self.sigF0 = []
        self.F0_approx = []
        self.unit_box_F0_approx = []
        for vf in self.volume_files:
            self._read_from_volume_file(vf)
    def _read_from_volume_file(self, vf):
        volf = ConfigParser.ConfigParser()
        volf.read(str(vf))
        try:
            self.F0_approx.append(volf.getfloat('VOLUME_APPROXIMATED', 'F0_approx'))
            self.unit_box_F0_approx.append(volf.getfloat('VOLUME_APPROXIMATED', 'unit_box_F0_approx'))
        except:
            print "no approx integral data available"
            print "location:", vf
        try:
            self.F0.append(volf.getfloat('VOLUME_FULL_PT', 'F0'))
            self.unit_box_F0.append(volf.getfloat('VOLUME_FULL_PT', 'unit_box_F0'))
            self.sigF0.append(volf.getfloat('VOLUME_FULL_PT', 'sigF0'))
        except:
            print "no PT data availible"
            print "location:", vf
    def _generate_plots(self):
        self._print_histogram_and_data(self.F0, "/volume_histogram_F0")
        self._print_histogram_and_data(self.unit_box_F0, "/volume_histogram_unit_box_F0")
        self._print_histogram_and_data(self.F0_approx, "/volume_histogram_F0_approx")
        self._print_histogram_and_data(self.unit_box_F0_approx, "/volume_histogram_unit_box_F0_approx") 
    def _print_histogram_and_data(self, data, name):
        np.savetxt(self.packings_dir + name + ".data", data)
        bins = 100
        hist, bin_edges = np.histogram(data, density = True, bins = bins)
        plt.hist(data, bins = 100, normed = True)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        def _gauss(x, sig, mu):
            return 1 / np.sqrt(2 * np.pi * sig ** 2) * np.exp( -(x - mu) ** 2 / (2 * sig ** 2))
        def _generalized_gauss(x, mu, alpha, zeta):
            return np.sqrt(zeta ** 2) / (2 * np.sqrt(alpha ** 2) * gamma(1 / np.sqrt(zeta ** 2))) * np.exp(-np.abs(x - mu) ** np.sqrt(zeta ** 2) / np.sqrt(alpha ** 2) ** np.sqrt(zeta ** 2))
        opt, error = curve_fit(_gauss, bin_centres, hist, [np.sqrt(np.var(data)), np.mean(data)])
        gauss_fit_opt = opt
        gauss_fit_opt[0] = np.abs(gauss_fit_opt[0]) #make printed sigma positive
        gauss_fit_error = error
        gauss_fit_names = ["sigma", "mean"]
        gauss_fit = [gauss_fit_opt, gauss_fit_names]
        opt_gen, error_gen = curve_fit(_generalized_gauss, bin_centres, hist, [np.mean(data), 2 * np.var(data), 1])
        gen_gauss_fit_opt = opt_gen
        gen_gauss_fit_opt[1] = np.abs(gen_gauss_fit_opt[1]) #make printed parameters positive
        gen_gauss_fit_opt[2] = np.abs(gen_gauss_fit_opt[2])
        gen_gauss_fit_error = error_gen
        gen_gauss_fit_names = ["mean", "alpha", "zeta"]
        gen_gauss_fit = [gen_gauss_fit_opt, gen_gauss_fit_names]
        def _set_hist_basics(plt):
            xp = np.linspace(bin_centres[0], bin_centres[-1], num = 500)
            plt.plot(xp, [_gauss(xpi, opt[0], opt[1]) for xpi in xp], "g--", label = "Gaussian")
            plt.plot(xp, [_generalized_gauss(xpi, opt_gen[0], opt_gen[1], opt_gen[2]) for xpi in xp], "r", label = "Generalised Gaussian")
            plt.legend()
            plt.xlabel(r"Free energy $F$")
            plt.ylabel(r"Probability density")
        #plot in lin-lin scale
        _set_hist_basics(plt)
        save_pdf(plt, self.packings_dir + name + ".pdf")
        plt.hist(data, bins = 100, normed = True)
        #plot in ylog scale
        _set_hist_basics(plt)
        plt.yscale('log', nonposy='clip')
        plt.axis(ymin = 0.25 / len(data))
        save_pdf(plt, self.packings_dir + name + "_ylog" + ".pdf")
        self._print_fitting_results(name, gauss_fit, gen_gauss_fit)
    def _print_fitting_results(self, name, gauss_fit, gen_gauss_fit):
        self.fit_results_dir = self.packings_dir + name + ".fit_results"
        f = FitResultsFile(self.fit_results_dir)
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
    parser.add_argument("-d", "--packings_dir", help = "top-level dir containing the packings, e.g. n32_phi88_2D")
    args = parser.parse_args()
    print args
    packings_dir = os.path.abspath(args.packings_dir)
    GenerateComparisonPlotPTApprox(packings_dir, plot_ts_integrand_data = False)