"""
Automatically figures out which runs have finished/haven't finished in a run folder
and submits the appropriate jobs to the cluster.
"""
from greene_submission import SimStage, calculate_volume
import os
import argparse
import toml
import shutil
import configparser


def get_calculation_stage(simulation_dir, jammed_packing_fname):
    """Gets the calculation stage of the simulation in the given directory
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
    
    kmax_config = os.path.join(explore_dir, f"findk_{fname_wo_ext}.config")
    kmin_config =  os.path.join(explore_dir, f"kmin_{fname_wo_ext}.config")
    pt_config =  os.path.join(explore_dir, f"explore_{fname_wo_ext}.config")
    innersphere_config =  os.path.join(explore_dir, f"innersphere_{fname_wo_ext}.config")

    if not os.path.exists(first_jammed_packing_config):
        return SimStage.JAMMED_PACKING
    elif not os.path.exists(kmax_config) or not check_success(kmax_config):
        return SimStage.KMAX
    elif not os.path.exists(kmin_config) or not check_success(kmin_config):
        return SimStage.KMIN
    elif not os.path.exists(  # PT not started
        pt_config
    ) or (
        os.path.exists(os.path.join(explore_dir, "checkpoint.dmp"))  # PT not finished
    ) or not check_success(pt_config, pt=True):
        return SimStage.PT
    elif not os.path.exists(
       innersphere_config
    ) or not check_success(innersphere_config) :
        return SimStage.INNER_SPHERE
    elif not os.path.isfile(os.path.join(explore_dir, "analysis", "mbar_volume_data")):
        return SimStage.ANALYSIS
    else:
        return SimStage.COMPLETE


def submit_jobs(simulation_dir, generate_packings=False):
    if generate_packings:
        simstage = SimStage.JAMMED_PACKING
        RUN_PARAMS_CONFIG_FILE = os.path.join(simulation_dir, "run_params.toml")
        if os.path.exists(RUN_PARAMS_CONFIG_FILE):
            run_params = toml.load(RUN_PARAMS_CONFIG_FILE)
        else:
            print(
                "No param file in destination folder, copying template from basinvolume source"
            )
            EMPTY_PARAMS_CONFIG_FILE = os.path.join(os.getcwd(), "run_params.toml")
            shutil.copy(EMPTY_PARAMS_CONFIG_FILE, RUN_PARAMS_CONFIG_FILE)
            run_params = toml.load(RUN_PARAMS_CONFIG_FILE)
        calculate_volume(
            simulation_dir,
            "",
            simstage,
            run_params,
            submit=True,
        )
        return

    simulation_dir_name = os.path.basename(simulation_dir)
    jammed_packings_dir = os.path.join(simulation_dir, "jammed_packings")
    jammed_packing_fnames = os.listdir(jammed_packings_dir)
    jammed_packing_fnames = [
        fname
        for fname in jammed_packing_fnames
        if fname.endswith(".xydr") or fname.endswith(".xyzdr")
    ]
    n_analysis = 0
    # simulation dir is assumed to be of the form {minimizer}_{n_particles}_{packing_fraction}
    minimizer_name = simulation_dir_name.split("_")[0]
    # minimizer = Minimizer[minimizer_name]
    minimizer = minimizer_name

    RUN_PARAMS_CONFIG_FILE = os.path.join(simulation_dir, "../run_params.toml")
    run_params = toml.load(RUN_PARAMS_CONFIG_FILE)
    run_params["kmax"]["minimizer"] = minimizer

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

        if simstage == SimStage.PT:
            fname_wo_ext = os.path.splitext(jammed_packing_fname)[0]
            explore_dir = os.path.join(simulation_dir, f"explore_bv_{fname_wo_ext}")
            if os.path.exists(os.path.join(explore_dir, "checkpoint.dmp")):
                checkpoint_file = os.path.join(explore_dir, "checkpoint.dmp")
            else:
                checkpoint_file = None
        else:
            checkpoint_file = None

        calculate_volume(
            simulation_dir,
            jammed_packing_fname,
            simstage,
            run_params,
            submit=True,
            checkpoint_file=checkpoint_file,
        )
    if n_prev_stages == 0 and n_analysis != 0:
        print(f"Submitting analysis for {simulation_dir_name}")
        calculate_volume(
            simulation_dir,
            jammed_packing_fnames[0],
            SimStage.ANALYSIS,
            run_params,
            submit=True,
        )
    return

def check_success(config_file, pt = False):
    ''' Check if a step was successful or not '''
    
    configf = configparser.ConfigParser()
    configf.read(config_file)
    
    if pt:
        success = conf_getboolean_default(configf, "STATUS", "success_rank0", False)
    else:
        success = conf_getboolean_default(configf, "STATUS", "success", False)
    
    return success

def conf_getboolean_default(configf, region, option, default):
    if configf.has_option(region, option):
        return configf.getboolean(region, option)
    else:
        return default

def main():
    parser = argparse.ArgumentParser(
        description="Automatically submits the next step of the basin volume calculation to a slurm interface. Assumes that generate_packin has already been run."
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Head directory containing the OPTIMIZER_N_PHI directories",
    )

    args = parser.parse_args()
    folder = args.folder

    simlist = os.listdir(folder)
    if simlist != []:
        simlist.remove("run_params.toml")

    if simlist == []:
        print("Empty directory: starting packing generation")
        submit_jobs(folder, generate_packings=True)
    else:
        for simfolder in simlist:
            submit_jobs(os.path.join(folder, simfolder))

if __name__ == "__main__":
    main()
