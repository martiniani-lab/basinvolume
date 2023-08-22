"""
Automatically figures out which runs have finished/haven't finished in a run folder
and submits the appropriate jobs to the cluster.
"""
from enum import Enum, unique
import os

@unique
class SimStage(Enum):
    JAMMED_PACKING = 1
    KMAX = 2
    KMIN = 3
    PT = 4
    INNER_SPHERE = 5
    ANALYSIS_STAGE = 6
    ANALYSIS_DONE = 7
    



def get_calculation_stage(simulation_dir, jammed_packing_fname):
    """ Gets the calculation stage of the simulation in the given directory
    Note 

    Parameters
    ----------
    simulation_dir : str
        The directory where the simulation is located
        
    Returns
    -------
    SimStage : Enum
        The stage of the simulation that needs to be run
    """
    jammed_packing_dir = os.path.join(simulation_dir, "jammed_packings")
    first_jammed_packing_config = os.path.join(jammed_packing_dir, jammed_packing_fname)
    fname_wo_ext = os.path.splitext(jammed_packing_fname)[0]
    explore_dir = os.path.join(simulation_dir, f"explore_bv_{fname_wo_ext}")
    if not os.path.exists(first_jammed_packing_config):
        return SimStage.JAMMED_PACKING
    elif not os.path.exists(os.path.join(explore_dir, f"findk_{fname_wo_ext}.config")):
        return SimStage.KMAX
    elif not os.path.exists(os.path.join(explore_dir, f"kmin_{fname_wo_ext}.config")):
        return SimStage.KMIN
    elif not os.path.exists(os.path.join(explore_dir, f"explore_{fname_wo_ext}.config")):
        return SimStage.PT
    elif not os.path.exists(os.path.join(explore_dir, f"inner_sphere_{fname_wo_ext}.config")):
        return SimStage.INNER_SPHERE
    elif not os.path.isdir(os.path.join(explore_dir, "analysis")):
        return SimStage.ANALYSIS_STAGE
    else:
        return SimStage.ANALYSIS_DONE
    

    