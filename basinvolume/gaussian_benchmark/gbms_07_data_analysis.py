from __future__ import division

import collections
import copy
import numpy as np
import os

from basinvolume.utils import BasicPlot
from basinvolume.utils import trymakedir

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError as err:
    print err
    
class TimeSeriesComparison(object):
    """
    Utils to compare scalar time series.
    
    Parameters
    ----------
    keys : array
        List of keys to identify the time series.
    """
    def __init__(self, keys, series, analysis_parameters):
        self.data = dict([(k, s) for k, s in zip(keys, series)])
        self.analysis_parameters = analysis_parameters
        self.long_time_mean = np.mean(np.asarray([s[-1] for s in self.data.values()]))
        self.max_deviation_from_long_mean = np.amax([np.absolute(s[-1] - self.long_time_mean) / self.long_time_mean for s in self.data.values()])
        if self.max_deviation_from_long_mean > self.analysis_parameters["target_relative_error"]:
            print("self.max_deviation_from_long_mean", self.max_deviation_from_long_mean)
            print("self.analysis_parameters['target_relative_error']", self.analysis_parameters["target_relative_error"])
            raise Exception("Target relative error too low.")
        self.latest_converged_iteration = dict([(k, self.get_latest_conv_iteration(k)) for k in keys])
        
    def get_latest_conv_iteration(self, k):
        it = len(self.data[k])
        while it > 1:
            if np.absolute(self.data[k][it - 1] - self.long_time_mean) / self.long_time_mean > self.analysis_parameters["target_relative_error"]:
                return it
            it -= 1
        return 0
        
class SeriesComparison(object):
    def __init__(self, three_series_dir, analysis_parameters):
        self.three_series_dir = three_series_dir
        self.analysis_parameters = analysis_parameters
        self.methods = self.analysis_parameters["methods"]
        self.data_files = os.listdir(three_series_dir)
        """
        brute_evaluations.txt  brute_iterations.txt  ti_evaluations.txt  ti_iterations.txt  traj_evaluations.txt  traj_iterations.txt
        brute_ini_evals.txt    brute_volume.txt      ti_ini_evals.txt    ti_volume.txt      traj_ini_evals.txt    traj_volume.txt
        """
    @property
    def incomplete(self):
        for m in self.methods:
            if not (m + "_evaluations.txt" in self.data_files and m + "_ini_evals.txt" in self.data_files and m + "_iterations.txt" in self.data_files and m + "_volume.txt" in self.data_files):
                return True
        return False
        
    def analyse(self):
        if self.incomplete:
            raise Exception("This assumes that the three-series-set is complete.")
        comp = TimeSeriesComparison(self.methods, [self.get_volume_series(m) for m in self.methods], self.analysis_parameters)
        for m in self.methods:
            converged_iteration = comp.latest_converged_iteration[m]
            self.write_converged_evaluation(m, converged_iteration)
        
    def get_volume_series(self, m):
        return np.loadtxt(os.path.join(self.three_series_dir, m + "_volume.txt"))
        
    def write_converged_evaluation(self, method, converged_iteration):
        np.savetxt(os.path.join(self.three_series_dir, method + "_res_evals.txt"), np.asarray([self.get_converged_evaluation(method, converged_iteration)]))
        
    def get_converged_evaluation(self, method, converged_iteration):
        print("method", method)
        print("converged_iteration", converged_iteration)
        result = self.get_converged_evaluation_basic(method, converged_iteration)
        if self.analysis_parameters["subtract_ini_evals"]:
            result -= self.get_ini_evals(method)
        print("converged_evaluation", result)
        return result
        
    def get_converged_evaluation_basic(self, method, converged_iteration):
        return np.loadtxt(os.path.join(self.three_series_dir, method + "_evaluations.txt"))[converged_iteration]
        
    def get_ini_evals(self, method):
        ini_path = os.path.join(self.three_series_dir, method + "_ini_evals.txt")
        if not os.path.exists(ini_path):
            raise Exception("Ini evals file not found", self.three_series_path, method)
        return np.loadtxt(ini_path)

class BenchmarkPlot(BasicPlot):
    """
    Makes benchmark plot.
    """
    def __init__(self, gauss_parameters, analysis_parameters, dirs):
        self.gauss_parameters = gauss_parameters
        self.analysis_parameters = analysis_parameters
        self.dirs = dirs
        self.evaluations = dict([(m, []) for m in analysis_parameters["methods"]])
        self.evaluations_error = copy.deepcopy(self.evaluations)
        self.dimensions = []
        self.get_data()
        self.make_plot()
        
    def get_data(self):
        # large_basin_results/5/
        base_dir = os.path.join(self.dirs["ls_basin_results_dir"],
            str(self.gauss_parameters["nr_gaussians"]))
        for dim in os.listdir(base_dir):
            # large_basin_results/5/2/
            add_dim = False
            for index in os.listdir(os.path.join(base_dir, dim)):
                # large_basin_results/5/2/0
                three_series_dir = os.path.join(base_dir, dim, index)
                self.run_three_series_analysis(three_series_dir)
                if "brute_res_evals.txt" in os.listdir(three_series_dir) and dim not in self.dimensions:
                    add_dim = True
            if add_dim:
                self.dimensions.append(int(dim))
        self.dimensions = sorted(self.dimensions)
        print("self.dimenisons", self.dimensions)
        #large_basin_results/5/2/0/brute_res_evals.txt
        for d in self.dimensions:
            for m in self.evaluations.keys():
                evals_list = [np.loadtxt(os.path.join(base_dir, str(dim), i, m + "_res_evals.txt")) for i in os.listdir(os.path.join(base_dir, str(dim)))]
                print("evals_list", evals_list)
                self.evaluations[m].append(np.mean(evals_list))
                self.evaluations_error[m].append(np.std(evals_list) / np.sqrt(len(evals_list) - 1))
    
    def run_three_series_analysis(self, three_series_dir):
        print(three_series_dir)
        sc = SeriesComparison(three_series_dir, self.analysis_parameters)
        if sc.incomplete:
            print("incomplete")
            return
        print("complete")
        sc.analyse()
    
    def make_plot(self):
        self.out_name = "gbms_data_analysis_" + self.gauss_parameters["ls_basin_label"] + ".pdf"
        symbols = ["o", "s", "^"]
        for i, m in enumerate(self.evaluations.keys()):
            plt.errorbar(self.dimensions, self.evaluations[m],
                yerr=self.evaluations_error[m], fmt=symbols[i], label=m)
        self.save_and_close()
    
def run_analysis(ls_basin_label):
    """
    Analyse benchmark nr. of function calls data.
    
    Parameter
    ---------
    
    ls_basin_label : string
        Needs to be "large" or "small" and indicates size label of basin to be computed.
    """
    if ls_basin_label is not "large" and ls_basin_label is not "small":
        raise Exception("ls_basin_label: illegal input, can be large or small only")
    gauss_parameters = dict([("nr_samples", 10),
                             ("nr_gaussians", 5),
                             ("dimensions", [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]),
                             ("ls_basin_label", ls_basin_label)])
    analysis_parameters = dict([("target_relative_error", 0.30),
                                ("subtract_ini_evals", True),
                                ("methods", ["traj", "ti", "brute"])])
    dirs = dict([("potential_dir", os.path.join(os.getcwd(), "potentials")),
                 ("ls_basin_results_dir", os.path.join(os.getcwd(), ls_basin_label + "_basin_results"))])
    BenchmarkPlot(gauss_parameters, analysis_parameters, dirs)

if __name__ == "__main__":
    run_analysis("large")
    run_analysis("small")
