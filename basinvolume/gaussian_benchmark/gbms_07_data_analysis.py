from __future__ import division

def run_preprocessing(gauss_parameters, analysis_parameters, dimension):
    """
    Determine number of evaluations for 
    
    Parameters
    ----------
    
    gauss_parameters : dict
        Parameters of gaussian potential landscape.
        
    analysis_parameters : dict
        Parameters of analysis method.
        
    dimension : int
        Number of Euclidean dimensions.
    """
    
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
    analysis_parameters = dict([("target_relative_error", 0.05)])
    for dimension in gauss_parameters["dimensions"]:
        run_preprocessing(gauss_parameters, analysis_parameters, dimension)
    make_plot(gauss_parameters, analysis_parameters)

if __name__ == "__main__":
    run_analysis("large")
    run_analysis("small")