from __future__ import division
from __future__ import print_function

from builtins import zip
from builtins import str
from builtins import object
import argparse as ap
import collections
import copy
import numpy as np
import os

from basinvolume.utils import BasicPlot
from basinvolume.utils import trymakedir
from basinvolume.utils import MomentsAcc

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError as err:
    print(err)


class TimeSeriesComparison(object):
    """
    Utils to compare scalar time series.

    Parameters
    ----------
    keys : array
        List of keys to identify the time series.
    series : array of arrays
        List of time series.
    analysis_parameters: dict
        Parameters required for analysis.
    """

    def __init__(self, keys, series, analysis_parameters):
        if len(keys) == 2:
            min_length = np.amin([len(s) for s in series])
            series = [s[0:min_length] for s in series]
        self.keys = keys
        self.series = series
        self.data = dict([(k, s) for k, s in zip(keys, series)])
        self.analysis_parameters = analysis_parameters
        self.long_time_mean = np.mean(np.asarray([s[-1] for s in list(self.data.values())]))
        self.max_deviation_from_long_mean = np.amax(
            [
                np.absolute(s[-1] - self.long_time_mean) / self.long_time_mean
                for s in list(self.data.values())
            ]
        )
        print(
            (
                "self.max_deviation_from_long_mean",
                self.max_deviation_from_long_mean,
            )
        )
        if (
            len(keys) > 2
            and self.max_deviation_from_long_mean
            > self.analysis_parameters["target_relative_error"]
        ):
            print(
                (
                    "self.analysis_parameters['target_relative_error']",
                    self.analysis_parameters["target_relative_error"],
                )
            )
            raise Exception("Target relative error too low.")

    def compute_conv_it(self):
        self.latest_converged_iteration = dict(
            [(k, self.get_latest_conv_iteration(k)) for k in self.keys]
        )
        print(
            (
                "self.latest_converged_iteration",
                self.latest_converged_iteration,
            )
        )

    def get_latest_conv_iteration(self, k):
        it = len(self.data[k])
        while it > 1:
            if (
                np.absolute(self.data[k][it - 1] - self.long_time_mean) / self.long_time_mean
                > self.analysis_parameters["target_relative_error"]
            ):
                return it
            it -= 1
        return 0


class TimeSeriesComparison2(TimeSeriesComparison):
    """
    Utils to compare convergence of 2 scalar time series.

    Parameters
    ----------
    keys : array
        List of keys to identify the time series.
    series : array of arrays
        List of time series.
    analysis_parameters: dict
        Parameters required for analysis.
    """

    def __init__(self, keys, series, analysis_parameters):
        super(TimeSeriesComparison2, self).__init__(keys, series, analysis_parameters)
        self.final_delta = np.absolute(self.series[0][-1] - self.series[1][-1])

    def get_latest_conv_iteration(self, k):
        it = len(self.data[k])
        while it > 1:
            if np.absolute(self.data[k][it - 1] - self.long_time_mean) > self.final_delta:
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
            if not (
                m + "_evaluations.txt" in self.data_files
                and m + "_ini_evals.txt" in self.data_files
                and m + "_iterations.txt" in self.data_files
                and m + "_volume.txt" in self.data_files
            ):
                return True
        return False

    def analyse(self):
        if self.incomplete:
            raise Exception("This assumes that the three-series-set is complete.")
        comp = None
        print(("self.three_series_dir", self.three_series_dir))
        try:
            comp = None
            if len(self.methods) == 2:
                comp = TimeSeriesComparison2(
                    self.methods,
                    [self.get_volume_series(m) for m in self.methods],
                    self.analysis_parameters,
                )
            else:
                comp = TimeSeriesComparison(
                    self.methods,
                    [self.get_volume_series(m) for m in self.methods],
                    self.analysis_parameters,
                )
            comp.compute_conv_it()
            for m in self.methods:
                converged_iteration = comp.latest_converged_iteration[m]
                self.write_converged_evaluation(m, converged_iteration)
        except Exception as e:
            print(e)
            print("warning")

    def get_volume_series(self, m):
        return np.loadtxt(os.path.join(self.three_series_dir, m + "_volume.txt"))

    def write_converged_evaluation(self, method, converged_iteration):
        np.savetxt(
            os.path.join(self.three_series_dir, method + "_res_evals.txt"),
            np.asarray([self.get_converged_evaluation(method, converged_iteration)]),
        )

    def get_converged_evaluation(self, method, converged_iteration):
        # print("method", method)
        # print("converged_iteration", converged_iteration)
        result = self.get_converged_evaluation_basic(method, converged_iteration)
        if self.analysis_parameters["subtract_ini_evals"]:
            result -= self.get_ini_evals(method)
        # print("converged_evaluation", result)
        return result

    def get_converged_evaluation_basic(self, method, converged_iteration):
        return np.loadtxt(os.path.join(self.three_series_dir, method + "_evaluations.txt"))[
            converged_iteration
        ]

    def get_ini_evals(self, method):
        ini_path = os.path.join(self.three_series_dir, method + "_ini_evals.txt")
        if not os.path.exists(ini_path):
            raise Exception("Ini evals file not found", self.three_series_path, method)
        return np.loadtxt(ini_path)


class TrajDataFile(object):
    def __init__(self, name_ending):
        self.name_ending = name_ending
        self.path = dict()

    def check_append(self, name, index):
        if name.endswith(self.name_ending):
            self.path[index] = name


class SingleSeriesConvergence(object):
    def __init__(self, volume_path, evals_path, target_relative_error):
        self.volume_path = volume_path
        self.evals_path = evals_path
        self.target_relative_error = target_relative_error
        self.volume = np.loadtxt(self.volume_path)
        self.evals = np.loadtxt(self.evals_path)
        self.final_volume = self.volume[-1]
        self.converged_iteration = self.find_converged_iteration()
        self.converged_evaluation = self.find_converged_evaluation()

    def find_converged_iteration(self):
        it = len(self.volume)
        while it > 1:
            it -= 1
            if (
                np.abs(self.final_volume - self.volume[it]) / self.final_volume
                > self.target_relative_error
            ):
                return it
        return it

    def find_converged_evaluation(self):
        return np.loadtxt(self.evals_path)[self.converged_iteration]


class TrajOnlyAnalysis(object):
    def __init__(self, dim_dir, analysis_parameters):
        # large_basin_results/5/2/ not large_basin_results/5/2/0, 0 is index
        self.dim_dir = dim_dir
        self.analysis_parameters = analysis_parameters
        self.indices = os.listdir(self.dim_dir)
        self.volume_files = TrajDataFile("traj_volume.txt")
        self.evaluations_files = TrajDataFile("traj_evaluations.txt")
        self.ini_evals_files = TrajDataFile("traj_ini_evals.txt")
        for index in self.indices:
            for f in os.listdir(os.path.join(self.dim_dir, index)):
                path_to_f = os.path.join(self.dim_dir, index, f)
                self.volume_files.check_append(path_to_f, index)
                self.evaluations_files.check_append(path_to_f, index)
                self.ini_evals_files.check_append(path_to_f, index)
        self.run()

    def run(self):
        acc = MomentsAcc()
        for i in list(self.volume_files.path.keys()):
            evaluations = self.get_evals(self.volume_files.path[i], self.evaluations_files.path[i])
            if self.analysis_parameters["subtract_ini_evals"]:
                evaluations -= self.get_ini_evals(self.ini_evals_files.path[i])
            acc.update(evaluations)
        self.nr_samples = acc.count
        self.evals = acc.get_mean()
        self.error_evals = acc.get_error()

    def get_evals(self, volume_path, evals_path):
        ssc = SingleSeriesConvergence(
            volume_path,
            evals_path,
            self.analysis_parameters["target_relative_error"],
        )
        return ssc.converged_evaluation

    def get_ini_evals(self, ini_evals_file):
        return np.loadtxt(ini_evals_file)


class BenchmarkPlot(BasicPlot):
    """
    Makes benchmark plot.
    """

    def __init__(self, gauss_parameters, analysis_parameters, dirs):
        self.gauss_parameters = gauss_parameters
        self.analysis_parameters = analysis_parameters
        self.dirs = dirs
        self.evaluations = dict([(m, []) for m in analysis_parameters["methods"]])
        self.traj_only_evaluations = []
        self.evaluations_error = copy.deepcopy(self.evaluations)
        self.traj_only_evaluations_error = []
        self.nr_samples = copy.deepcopy(self.evaluations)
        self.traj_only_nr_samples = []
        self.converged_sets = dict()
        self.get_data()
        self.make_plot()

    def get_data(self):
        # large_basin_results/5/
        base_dir = os.path.join(
            self.dirs["ls_basin_results_dir"],
            str(self.gauss_parameters["nr_gaussians"]),
        )
        for dim in os.listdir(base_dir):
            # large_basin_results/5/2/
            for index in os.listdir(os.path.join(base_dir, dim)):
                # large_basin_results/5/2/0
                three_series_dir = os.path.join(base_dir, dim, index)
                self.run_three_series_analysis(three_series_dir)
                if "traj_res_evals.txt" in os.listdir(three_series_dir):
                    if not int(dim) in self.converged_sets:
                        self.converged_sets[int(dim)] = []
                    self.converged_sets[int(dim)].append(index)
        self.dimensions = sorted(self.converged_sets.keys())
        print(("self.converged_sets", self.converged_sets))
        # large_basin_results/5/2/0/brute_res_evals.txt
        for dim in self.dimensions:
            for m in list(self.evaluations.keys()):
                evals_list = [
                    np.loadtxt(os.path.join(base_dir, str(dim), i, m + "_res_evals.txt"))
                    for i in self.converged_sets[dim]
                ]
                # print("evals_list", evals_list)
                assert len(evals_list) == len(self.converged_sets[dim])
                self.nr_samples[m].append(len(evals_list))
                self.evaluations[m].append(np.mean(evals_list))
                self.evaluations_error[m].append(np.std(evals_list) / np.sqrt(len(evals_list) - 1))
        # Traj only analysis, not conflated with TI data
        for dim in self.dimensions:
            nr_samples, evals, error_evals = self.traj_only_analysis(
                os.path.join(base_dir, str(dim))
            )
            self.traj_only_nr_samples.append(nr_samples)
            self.traj_only_evaluations.append(evals)
            self.traj_only_evaluations_error.append(error_evals)

    def traj_only_analysis(self, dim_dir):
        to = TrajOnlyAnalysis(dim_dir, self.analysis_parameters)
        return to.nr_samples, to.evals, to.error_evals

    def run_three_series_analysis(self, three_series_dir):
        # print(three_series_dir)
        if self.analysis_parameters["plot_only"]:
            return
        sc = SeriesComparison(three_series_dir, self.analysis_parameters)
        if sc.incomplete:
            # print("incomplete")
            return
        # print("complete")
        sc.analyse()

    def make_plot(self):
        # self.methods_for_plot = [m for m in self.evaluations.keys() if m is not "brute"]
        self.methods_for_plot = ["traj", "ti"]
        self.methods_label_names = dict([("traj", "Trajectories"), ("ti", "TI")])

        if self.analysis_parameters["logy"]:
            plt.yscale("log")
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.xlabel(r"Potential dimensionality, $D$", fontsize=22)
        plt.ylabel(r"Number of function calls, $N_{EFE}/10^8$", fontsize=22)
        symbols = ["s", "^", "o"]
        if not self.analysis_parameters["traj_plot_only"]:
            self.out_name = (
                "gbms_data_analysis_" + self.gauss_parameters["ls_basin_label"] + ".pdf"
            )
            for i, m in enumerate(self.methods_for_plot):
                print(("self.dimensions", self.dimensions))
                print(("self.evaluations[m]", self.evaluations[m]))
                print(("self.nr_samples[m]", self.nr_samples[m]))
                eval_plot = np.asarray(self.evaluations[m]) / 10**8
                yerr_plot = np.asarray(self.evaluations_error[m]) / 10**8
                plt.errorbar(
                    self.dimensions,
                    eval_plot,
                    yerr=yerr_plot,
                    fmt=symbols[i],
                    label=self.methods_label_names[m],
                )
            plt.legend(loc=2, prop={"size": 18})
        else:
            self.out_name = (
                "gbms_data_analysis_traj_only_" + self.gauss_parameters["ls_basin_label"] + ".pdf"
            )
            eval_plot = np.asarray(self.traj_only_evaluations) / 10**8
            yerr_plot = np.asarray(self.traj_only_evaluations_error) / 10**8
            plt.errorbar(self.dimensions, eval_plot, yerr=yerr_plot, fmt=symbols[0])
        plt.tick_params(labelsize=22)
        pdf = PdfPages(self.out_name)
        # http://stackoverflow.com/questions/18572234/matplotlib-axes-set-aspectequal-doesnt-behave-like-expected
        plt.axes().set_xlim([0, 35])
        plt.axes().set_aspect(1 / plt.axes().get_data_ratio())
        plt.savefig(pdf, format="pdf", bbox_inches="tight")
        pdf.close()
        plt.close()


def run_analysis(ls_basin_label):
    """
    Analyse benchmark nr. of function calls data.

    Parameter
    ---------

    ls_basin_label : string
        Needs to be "large" or "small" and indicates size label of basin to be computed.
    """
    p = ap.ArgumentParser()
    p.add_argument("--plot_only", action="store_true", default=False)
    p.add_argument("--traj_plot_only", action="store_true", default=False)
    args = p.parse_args()
    if ls_basin_label is not "large" and ls_basin_label is not "small":
        raise Exception("ls_basin_label: illegal input, can be large or small only")
    gauss_parameters = dict(
        [
            ("nr_samples", 10),
            ("nr_gaussians", 5),
            ("dimensions", [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]),
            ("ls_basin_label", ls_basin_label),
        ]
    )
    analysis_parameters = dict(
        [
            ("target_relative_error", 0.01),
            ("subtract_ini_evals", True),
            ("logy", False),
            ("methods", ["traj", "ti"]),
            # ("methods", ["traj", "ti", "brute"]),
            ("plot_only", args.plot_only),
            ("traj_plot_only", args.traj_plot_only),
        ]
    )
    dirs = dict(
        [
            ("potential_dir", os.path.join(os.getcwd(), "potentials")),
            (
                "ls_basin_results_dir",
                os.path.join(
                    "/scratch/kjs73/basin_traj_data/",
                    ls_basin_label + "_basin_results",
                ),
            ),
        ]
    )
    BenchmarkPlot(gauss_parameters, analysis_parameters, dirs)


if __name__ == "__main__":
    run_analysis("large")
    run_analysis("small")
