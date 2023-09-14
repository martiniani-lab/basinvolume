"""
Automatically figures out which runs have finished/haven't finished in a run folder
and submits the appropriate jobs to the cluster.
"""
from enum import Enum, unique
from greene_submission import SimStage, calculate_volume
import os
from basinvolume.enums import Minimizer


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
    elif not os.path.exists(os.path.join(explore_dir, f"innersphere_{fname_wo_ext}.config")):
        return SimStage.INNER_SPHERE
    elif not os.path.isfile(os.path.join(explore_dir, "analysis", "mbar_volume_data")):
        return SimStage.ANALYSIS
    else:
        return SimStage.COMPLETE

def submit_jobs(simulation_dir):
    simulation_dir_name = os.path.basename(simulation_dir)
    jammed_packings_dir = os.path.join(simulation_dir, "jammed_packings")
    jammed_packing_fnames = os.listdir(jammed_packings_dir)
    jammed_packing_fnames = [fname for fname in jammed_packing_fnames if fname.endswith(".xydr") or fname.endswith(".xyzdr")]
    n_analysis = 0
    # simulation dir is assumed to be of the form {minimizer}_{n_particles}_{packing_fraction}
    minimizer_name = simulation_dir_name.split("_")[0]
    minimizer = Minimizer[minimizer_name]
    global_kwargs=dict(
        opt_tol=1e-10,
        opt_nsteps=1e5,
        dtol=1e-2,
        opt_dtmax=1,
        minimizer=minimizer,
    )
    
    n_prev_stages = 0
    for jammed_packing_fname in jammed_packing_fnames:
        simstage = get_calculation_stage(simulation_dir, jammed_packing_fname)
        print(f"{simulation_dir_name} in stage {simstage.name}")
        if simstage == SimStage.COMPLETE:
            print(f"{simulation_dir_name} is complete")
            continue
        if simstage != SimStage.ANALYSIS and SimStage != SimStage.COMPLETE:
            n_prev_stages += 1
        
        if simstage == SimStage.ANALYSIS:
            n_analysis += 1
            print(f"analysis waiting for {simulation_dir_name}")
            continue
        calculate_volume(simulation_dir, jammed_packing_fname, simstage, submit=True)
    if n_prev_stages == 0 and n_analysis != 0:
        print(f"Submitting analysis for {simulation_dir_name}")
        calculate_volume(simulation_dir, jammed_packing_fnames[0], SimStage.ANALYSIS, submit=True)
    return


def main():
    folder = "/scratch/ps4586/multi_number/"
    for simfolder in os.listdir(folder):
        submit_jobs(os.path.join(folder, simfolder))
if __name__ == "__main__":
    main()