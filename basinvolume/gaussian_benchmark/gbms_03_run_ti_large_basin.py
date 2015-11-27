from __future__ import division

from computer_common import ComputerCommon
from computer_common import run_computer



class TIComputer(ComputerCommon):
    """
    Computer volume of basin in the gaussian landscape by thermodynamic integration
    as function of the number of energy function calls and print that to disk.
    """
    def __init__(self, results_dir, opt_parameters, pes_parameters,
        vol_parameters, method_parameters, pot):
        super(TIComputer, self).__init__(results_dir, opt_parameters,
            pes_parameters, vol_parameters, method_parameters, pot)
        self.ti_engine = TIEngine(self.method_parameters,
            self.pes_parameters, self.pot, self.opt_parameters)
            
    def get_method_labels(self):
        return "ti"
        
    def get_evaluations_volume_one_iteration(self):
        self.ti_engine.one_iteration()
        evaluations = self.ti_engine.evaluations
        volume = self.ti_engine.volume
        return evaluations, volume

def run_ti(ls_basin_label):
    """
    Execute ti basin volume computation for large or small basin.
    
    Parameter
    ---------
    
    ls_basin_label : string
        Needs to be "large" or "small" and indicates size label of basin to be computed.
    """
    if ls_basin_label is not "large" and ls_basin_label is not "small":
        raise Exception("ls_basin_label: illegal input, can be large or small only")
    nr_samples = 20
    opt_parameters = dict([("opt_dtmax", 1), ("opt_tol", 1e-8),
        ("opt_nsteps", 1e8), ("opt_maxstep", 0.1), ("verbosity", 0)])
    vol_parameters = dict([("max_iterations", 100)])
    ti_parameters = dict([("nr_samples_increment", 1000)])
    potential_dir = os.path.join(os.getcwd(), "potentials")
    ls_basin_results_dir = os.path.join(os.getcwd(), ls_basin_label + "_basin_results")
    for nr_gaussians in [5]:
        for nr_dimensions in [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 80]:
            pes_parameters = dict([("csm_dtol", 1),
                ("nr_dimensions", nr_dimensions),
                ("radius_container", 10)])
            run_computer(potential_dir, ls_basin_results_dir, ls_basin_label,
                nr_gaussians, nr_dimensions, nr_samples, TIComputer,
                opt_parameters, pes_parameters, vol_parameters,
                ti_parameters)

if __name__ == "__main__":
    run_ti("large")