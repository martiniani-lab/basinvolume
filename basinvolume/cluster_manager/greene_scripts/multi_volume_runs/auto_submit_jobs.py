"""
Automatically figures out which runs have finished/haven't finished in a run folder
and submits the appropriate jobs to the cluster.
"""
from enum import Enum, unique
from greene_submission import SimStage, calculate_volume
import os


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
        return SimStage.ANALYSIS
    else:
        return SimStage.COMPLETE

def submit_jobs(simulation_dir):
    simulation_dir_name = os.path.basename(simulation_dir)
    jammed_packings_dir = os.path.join(simulation_dir, "jammed_packings")
    jammed_packing_fnames = os.listdir(jammed_packings_dir)
    jammed_packing_fnames = [fname for fname in jammed_packing_fnames if fname.endswith(".xydr") or fname.endswith(".xyzdr")]
    for jammed_packing_fname in jammed_packing_fnames:
        simstage = get_calculation_stage(simulation_dir, jammed_packing_fname)
        print(f"{simulation_dir_name} in stage {simstage.name}")
        if simstage == SimStage.COMPLETE:
            print(f"{simulation_dir_name} is complete")
            continue
        calculate_volume(simulation_dir, jammed_packing_fname, simstage, submit=True)
    return