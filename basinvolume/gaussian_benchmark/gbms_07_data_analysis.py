from __future__ import division

import copy

from basinvolume.utils import BasicPlot

def run_preprocessing(gauss_parameters, analysis_parameters, dimension):
    """
    Subtract equilibration steps.
    Find number needed eval etc.
    
    Parameters
    ----------
    
    gauss_parameters : dict
        Parameters of gaussian potential landscape.
        
    analysis_parameters : dict
        Parameters of analysis method.
        
    dimension : int
        Number of Euclidean dimensions.
    """

class BenchmarkPlot(BasicPlot):
    """
    Makes benchmark plot.
    """
    def __init__(self, gauss_parameters, analysis_parameters):
        self.gauss_parameters = gauss_parameters
        self.analysis_parameters = analysis_parameters
        self.evaluations = dict([(m, []) for m in analysis_parameters["methods"]])
        self.evaluations_error = copy.deepcopy(self.evaluations)
        self.dimensions = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]
        self.get_data()
        
    def get_data(self):
        for m in self.evaluations.keys():
            for d in self.dimensions:
                e, ee = self.get_evaluations_error(m, d)
                self.evaluations[m]append(e)
                self.evaluations_error[m].append(ee)
                
    def get_evaluations_error(self, method, dim):
        """
        Read evaluations needed of that method
        """
    
    def make_plot(self):
        self.out_name = "gbms_07_data_analysis.pdf"
        symbols = ["o", "s", "^"]
        for i, m in enumerate(self.evaluations.keys()):
            plt.errorbar(self.dimensions, self.self.evaluations[m], yerr=self.evaluations_error[m], fmt=symbols[i], label=m)
        self.save_and_close()
    
def make_plot(gauss_parameters, analysis_parameters):
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
    gauss_parameters = dict([("nr_samples", 20),
                             ("nr_gaussians", 5),
                             ("dimensions", [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]),
                             ("ls_basin_label", ls_basin_label)])
    analysis_parameters = dict([("target_relative_error", 0.10),
                                ("methods", ["traj", "ti", "brute"])])
    for dimension in gauss_parameters["dimensions"]:
        run_preprocessing(gauss_parameters, analysis_parameters, dimension)
    make_plot(gauss_parameters, analysis_parameters)

if __name__ == "__main__":
    run_analysis("large")
    run_analysis("small")
