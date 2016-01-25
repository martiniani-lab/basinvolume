from __future__ import division

import copy
import numpy as np
import os

from basinvolume.utils import BasicPlot
from basinvolume.utils import trymakedir

class TimeSeriesComparison(object):
    """
    Utils to compare scalar time series.
    
    Parameters
    ----------
    keys : array
        List of keys to identify the time series.
    """
    def __init__(self, keys, series):
        self.data = dict([(k, s) for k, s in zip(keys, series)])
        self.long_time_mean = np.mean(np.asarray(s[-1] for s in self.data.values()))
        print("self.long_time_mean", self.long_time_mean)
        assert(False)
        
class FunctionCallData(object):
    """
    Data is read from files with paths analogous to the following:
    large_basin_results/5/2/0/brute_evaluations.txt
    large_basin_results/5/2/0/brute_ini_evals.txt
    large_basin_results/5/2/0/brute_iterations.txt
    large_basin_results/5/2/0/brute_volume.txt
    """
    def __init__(self, m, d, gauss_parameters, analysis_parameters, dirs):
        self.m = m
        self.d = d
        self.gauss_parameters = gauss_parameters
        self.analysis_parameters = analysis_parameters
        self.dirs = dirs
        for measurement_name in ..............:
            self.measurement_name = np.loadtxt(.....)
        

class NrFunctionCallsStatistics(object):
    """
    Compute and write number of function call information for fixed
    (d, gauss_parameters).
    """
    def __init__(self, d, gauss_parameters, analysis_parameters, dirs):
        self.d = d
        self.gauss_parameters = gauss_parameters
        self.analysis_parameters = analysis_parameters
        self.dirs = dirs
        self.data = dict([(m, FunctionCallData(m, self.d,
                            self.gauss_parameters, self.analysis_parameters,
                            self.dirs)) for m in self.analysis_parameters["methods"]])
        
    def run_analysis(self):
        """
        Collect data files for (d, gauss_parameters), compute number
        of function calls and error.
        """
        # List of methods for which there is data.
        self.converged_methods = [m for m in self.analysis_parameters.methods if self.data[m].converged]
        if len(self.converged_methods) < 2:
            raise Exception("NrFunctionCallsStatistics: too few data for analysis")
        # Mean of last data points in volume time series.
        self.long_time_mean = np.mean(np.asarray([self.data[m].volume[-1] for m in self.converged_methods]))
        # Deviations of the final volume data points from the mean, in percent.
        
    def print_results(self):
        """
        Write nr function calls results to disk.
        """
        assert(False)
        

def run_preprocessing(gauss_parameters, analysis_parameters, dirs):
    """
    Subtract equilibration steps.
    Find number needed eval etc.
    
    Parameters
    ----------
    
    gauss_parameters : dict
        Parameters of gaussian potential landscape.
        
    analysis_parameters : dict
        Parameters of analysis method.
        
    dirs : dict
        Path parameters for simulation results and analysis output.
    """
    trymakedir(dirs["analysis_dir"])
    for d in gauss_parameters["dimensions"]:
        stat = NrFunctionCallsStatistics(d, gauss_parameters, analysis_parameters, dirs)
        stat.run_analysis()
        stat.print_results()

class BenchmarkPlot(BasicPlot):
    """
    Makes benchmark plot.
    """
    def __init__(self, gauss_parameters, analysis_parameters):
        self.gauss_parameters = gauss_parameters
        self.analysis_parameters = analysis_parameters
        self.evaluations = dict([(m, []) for m in analysis_parameters["methods"]])
        self.evaluations_error = copy.deepcopy(self.evaluations)
        self.dimensions = self.gauss_parameters["dimensions"]
        self.get_data()
        
    #def get_data(self):
    #    for m in self.evaluations.keys():
    #        for d in self.dimensions:
                #e, ee = self.get_evaluations_error(m, d)
                #self.evaluations[m]append(e)
                #self.evaluations_error[m].append(ee)
                
    #def get_evaluations_error(self, method, dim):
    #    """
    #    Read evaluations needed of that method
    #    """
    
    def make_plot(self):
        self.out_name = "gbms_07_data_analysis.pdf"
        symbols = ["o", "s", "^"]
        for i, m in enumerate(self.evaluations.keys()):
            plt.errorbar(self.dimensions, self.self.evaluations[m], yerr=self.evaluations_error[m], fmt=symbols[i], label=m)
        self.save_and_close()
    
def make_plot(gauss_parameters, analysis_parameters, dirs):
    """
    Given preprocessing results, turn these into plots.
    
    Parameters
    ----------
    
    gauss_parameters : dict
        Parameters of gaussian potential landscape.
        
    analysis_parameters : dict
        Parameters of analysis method.
    """
    plot = BenchmarkPlot(gauss_parameters, analysis_parameters)
    plot.make_plot()

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
    analysis_parameters = dict([("target_relative_error", 0.10),
                                ("subtract_ini_evals", True),
                                ("methods", ["traj", "ti", "brute"])])
    dirs = dict([("potential_dir", os.path.join(os.getcwd(), "potentials")),
                 ("ls_basin_results_dir", os.path.join(os.getcwd(), ls_basin_label + "_basin_results")),
                 ("analysis_dir", os.path.join(os.getcwd(), ls_basin_label + "_basin_analysis"))])
    run_preprocessing(gauss_parameters, analysis_parameters, dirs)
    make_plot(gauss_parameters, analysis_parameters, dirs)

if __name__ == "__main__":
    run_analysis("large")
    run_analysis("small")
