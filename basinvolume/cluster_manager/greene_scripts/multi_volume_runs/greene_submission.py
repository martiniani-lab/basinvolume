"""Script to submit multi volume calculations on the cluster.
"""


import configparser
from enum import unique, Enum
import os
import toml
import numpy as np
import subprocess

# hardcoded because we don't want to import this, and the relative paths are set
# importing basinvolume can only be done in singularity
# and this script just manages job submission
current_directory = os.getcwd()
BASINVOLUME_PATH = os.path.join(current_directory, "../../..")

# Load possibly changing paths and environment variables from a toml config file
CONFIG_FILE = os.path.join(current_directory, "local_config.toml")
config = toml.load(CONFIG_FILE)
# User-dependent values
USER_EMAIL = config["user"]["email"]
EMAIL_TYPE = config["user"]["email_type"]
EXT3_FILE = config["user"]["ext3_file"]
CONDA_ENV = config["user"]["conda_env"]
# Cluster-dependent values, may change over time and/or between clusters even with similar architectures
GREENE_SINGULARITY_OVERLAY = config["cluster"]["singularity_overlay"]

RESOURCE_FILE = os.path.join(current_directory, "resource_requirements.toml")
RESOURCE_CONFIG = toml.load(RESOURCE_FILE)

MAX_PROC_NUMBER = RESOURCE_CONFIG["cpu"]["max_proc_number"]

# Load defaults from the relevant config file
DEFAULTS_CONFIG_FILE = os.path.join(current_directory, "default_params.toml")
DEFAULT_CONFIG = toml.load(DEFAULTS_CONFIG_FILE)


@unique
class SimStage(Enum):
    """Enum for the different types of runs that can be submitted."""

    JAMMED_PACKING = 0
    KMAX = 1
    KMIN = 2
    PT = 3  # parallel tempering
    INNER_SPHERE = 4
    ANALYSIS = 5
    COMPLETE = 6


GREENE_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --time={time_str}
#SBATCH --ntasks={ntasks}
#SBATCH --partition=cs
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem_str}
#SBATCH --mail-type={email_type}
#SBATCH --mail-user={email}
#SBATCH --job-name={out_file}
#SBATCH --output={out_file}.out

export OMP_NUM_THREADS=1;

cd {simulation_folder};
singularity exec --overlay {ext3_file}:ro \
    {singularity_overlay} \
    /bin/bash -c "source ~/.bashrc;
    export OMP_NUM_THREADS=1;
    conda activate {conda_env};
    {run_command};"
"""


def format_args_from_dict(arg_dict):
    """
    Convert a dictionary of arguments into a formatted string suitable for command-line usage.

    Args:
    - arg_dict (dict): Dictionary of arguments.

    Returns:
    - str: Formatted string of command-line arguments.
    """
    arg_str = ""

    for key, value in arg_dict.items():
        if value is None:
            continue
        # Convert underscore in keys to hyphen
        formatted_key = "--" + key

        # For boolean flags
        if isinstance(value, bool):
            if value:  # Only include True flags in the string
                arg_str += f" {formatted_key}"
        else:
            # For other arguments
            arg_str += f" {formatted_key} {value}"

    return arg_str


def calculate_volume(
    simulation_folder,
    packing_file,
    simulation_type,
    run_params,
    submit=True,
    checkpoint_file=None,
    checkpoint_fraction = 0.8
):
    # global args that should be the same across scripts
    # only kmax sees the optimizer kwargs, the following steps just read them off from the kmax config file

    # Pass time_str to everyone using opt_kwargs
    # TODO: set these times based on problem dimension
    kmax_dict = DEFAULT_CONFIG["kmax_defaults"]
    if run_params["kmax"] != {}:
        kmax_dict.update(run_params["kmax"])
    minimizer = kmax_dict["minimizer"]
    time_str, time = make_time_str(
        minimizer, simulation_folder, simulation_type, RESOURCE_CONFIG["time"], DEFAULT_CONFIG["hard_sphere_packing_defaults"]["boxdim"]
    )

    # Always checkpoint after a fraction of required wall time to avoid bad surprises
    # This one time is in minutes, not hours, so it needs a factor of 60
    checkpoint_time = int(checkpoint_fraction * 60 * time)
    if simulation_type == SimStage.JAMMED_PACKING:
        mem_str = RESOURCE_CONFIG["memory"]["generate"]
        setup_generate_jammed_data(simulation_folder, run_params, time_str, mem_str, submit=submit)
    elif simulation_type == SimStage.KMAX:
        mem_str = RESOURCE_CONFIG["memory"]["kmax"]
        setup_kmax(
            simulation_folder,
            run_params,
            packing_file,
            time_str,
            mem_str,
            submit=submit,
        )
    elif simulation_type == SimStage.KMIN:
        mem_str = RESOURCE_CONFIG["memory"]["kmin"]
        setup_kmin(
            simulation_folder,
            run_params,
            packing_file,
            time_str,
            mem_str,
            submit=submit,
        )
    elif simulation_type == SimStage.PT:
        mem_str = RESOURCE_CONFIG["memory"]["pt"]
        setup_parallel_tempering(
            simulation_folder,
            run_params,
            packing_file,
            time_str,
            mem_str,
            checkpoint_time=checkpoint_time,
            checkpoint_file=checkpoint_file,
            submit=submit,
        )
    elif simulation_type == SimStage.INNER_SPHERE:
        mem_str = RESOURCE_CONFIG["memory"]["innersphere"]
        setup_inner_sphere(
            simulation_folder,
            run_params,
            packing_file,
            time_str,
            mem_str,
            submit=submit,
        )
    elif simulation_type == SimStage.ANALYSIS:
        mem_str = RESOURCE_CONFIG["memory"]["analysis"]
        setup_compute_volume(simulation_folder, packing_file, run_params, time_str, mem_str, submit=submit)
    else:
        raise NotImplementedError("simulation type not implemented")


def setup_generate_jammed_data(simulation_folder, run_params, time_str, mem_str, submit=True):
    # single core args
    ntasks = 1
    cpus_per_task = 1

    # Most of these should be defaults but you can change them at the script level
    jammed_data_kwargs = DEFAULT_CONFIG["jammed_data_defaults"]
    jammed_data_kwargs.update(run_params["jammed_data"])
    hard_sphere_packing_kwargs = DEFAULT_CONFIG["hard_sphere_packing_defaults"]
    hard_sphere_packing_kwargs.update(run_params["hard_sphere_packing"])
    hard_sphere_packing_kwargs["hsf-niter-dif"] = int(hard_sphere_packing_kwargs["hsf-niter-dif"])
    jammed_packing_kwargs = DEFAULT_CONFIG["jammed_packing_defaults"]
    jammed_packing_kwargs.update(run_params["jammed_packing"])

    # update defaults with global kwargs
    packing_script_subpath = "spheres/generate_packing.py"
    jammed_packing_script_subpath = "spheres/generate_jammed_packing.py"
    job_name_prefix = "generate_jammed_packing"

    # This one is a bit special in the sense that the kwargs are lists of arguments
    submit_initial_jobs(
        simulation_folder,
        ntasks,
        cpus_per_task,
        jammed_data_kwargs,
        hard_sphere_packing_kwargs,
        jammed_packing_kwargs,
        packing_script_subpath,
        jammed_packing_script_subpath,
        time_str,
        mem_str,
        job_name_prefix,
        submit=submit,
    )

    return 0


def setup_kmax(simulation_folder, run_params, packing_file, time_str, mem_str, submit=True):
    # single core args
    ntasks = 1
    cpus_per_task = 1

    # Most of these should be defaults but you can change them at the script level
    kmax_default_kwargs = DEFAULT_CONFIG["kmax_defaults"]

    # update defaults with global kwargs
    script_subpath = "spheres/bv_find_kmax.py"
    job_name_prefix = "bv_kmax"

    submit_job(
        simulation_folder,
        run_params["kmax"],
        packing_file,
        ntasks,
        cpus_per_task,
        kmax_default_kwargs,
        script_subpath,
        time_str,
        mem_str,
        job_name_prefix,
        submit=submit,
        kmax=True,
    )
    return 0


def setup_kmin(simulation_folder, run_params, packing_file, time_str, mem_str, submit=True):
    ntasks = 1
    cpus_per_task = 1
    # defaults but you can change them at the script level
    kmin_default_kwargs = DEFAULT_CONFIG["kmin_defaults"]
    script_subpath = "spheres/bv_find_kmin.py"
    job_name_prefix = "bv_kmin"

    submit_job(
        simulation_folder,
        run_params["kmin"],
        packing_file,
        ntasks,
        cpus_per_task,
        kmin_default_kwargs,
        script_subpath,
        time_str,
        mem_str,
        job_name_prefix,
        submit=submit,
    )
    return 0


def setup_parallel_tempering(
    simulation_folder,
    run_params,
    packing_file,
    time_str,
    mem_str,
    checkpoint_time=None,
    checkpoint_file=None,
    submit=True,
):
    ntasks = 1
    pt_default_kwargs = DEFAULT_CONFIG["pt_defaults"]
    run_params["pt"]["checkpoint-time"] = checkpoint_time
    run_params["pt"]["load-checkpoint"] = checkpoint_file
    pt_default_kwargs.update(run_params["pt"])
    
    if pt_default_kwargs["nreplicas"] == "auto":
        # load dim from jammed_packing config file
        jammed_packing_folder = os.path.join(simulation_folder, "jammed_packings")
        # get the first file ending with an integer followed by [.config]
        fnames = os.listdir(jammed_packing_folder)
        jammed_fname = next(
            fname for fname in fnames if fname.endswith(".config") # and fname.split("_")[-1].isdigit() # XXX this broke the code and I don't get why it's here
        )
        config_file = os.path.join(jammed_packing_folder, jammed_fname)
        configf = configparser.ConfigParser()
        configf.read(config_file)
        dim = int(configf["JAMMED_PACKING"]["ndim"])
        max_replicas = max(64, int(dim/4))
        max_replicas = max_replicas if max_replicas % 4 == 0 else max_replicas - max_replicas % 4 + 4
        run_params["pt"]["nreplicas"] = max_replicas
    else:
        run_params["pt"]["nreplicas"] = run_params["pt"].get("nreplicas", pt_default_kwargs["nreplicas"])
    mpi_procs = int(run_params["pt"]["nreplicas"] / 4)  # Best performance according to Johannes
    if (
        mpi_procs > MAX_PROC_NUMBER
    ):  # Bound by a config-file specified max value that depends on the cluster
        mpi_procs = MAX_PROC_NUMBER
    cpus_per_task = mpi_procs
    script_subpath = "spheres/bv_parallel_tempering.py"
    job_name_prefix = "bv_pt"
    
    if checkpoint_file is not None:
        job_name_prefix += "_fromcheckpoint"

    # give the explore directory as the argument
    packing_fname = os.path.splitext(packing_file)[0]
    explore_dir = f" explore_bv_{packing_fname}"

    submit_job(
        simulation_folder,
        run_params["pt"],
        packing_file,
        ntasks,
        cpus_per_task,
        pt_default_kwargs,
        script_subpath,
        time_str,
        mem_str,
        job_name_prefix,
        script_run_prefix=f"mpiexec -n {mpi_procs} python",
        extra_args=explore_dir,
        submit=submit,
    )
    return 0


def setup_inner_sphere(
    simulation_folder, run_params, packing_file, time_str, mem_str, submit=True
):
    ntasks = 1
    cpus_per_task = 1
    inner_sphere_default_kwargs = DEFAULT_CONFIG["innersphere_defaults"]
    script_subpath = "mbar_spheres/bv_innersphere_dos.py"
    job_name_prefix = "bv_inner_sphere"
    # give the explore directory as an argument
    submit_job(
        simulation_folder,
        run_params["innersphere"],
        packing_file,
        ntasks,
        cpus_per_task,
        inner_sphere_default_kwargs,
        script_subpath,
        time_str,
        mem_str,
        job_name_prefix,
        submit=submit,
    )
    return 0


def setup_compute_volume(simulation_folder, packing_file, run_params, time_str, mem_str, submit=True):
    # TODO make this more like the others with fewer hardcoded values
    ntasks = 1
    cpus_per_task = 1

    script_run_prefix = "python"
    script_location = os.path.join(BASINVOLUME_PATH, "mbar_spheres/mbar_compute_volume.py")

    job_name_prefix = "bv_computevolume"
    
    # give the explore directory as the argument
    packing_fname = os.path.splitext(packing_file)[0]

    out_folder = os.path.join(simulation_folder, "job_out")
    out_file = f"{out_folder}/{job_name_prefix}_{packing_fname}"
    explore_dir_prefix = "explore_bv_jammed_packing"

    # Use the same bias as in PT here
    pt_default_kwargs = DEFAULT_CONFIG["pt_defaults"]
    pt_default_kwargs.update(run_params["pt"])
    bias = pt_default_kwargs["bias"]
    

    run_command = f"{script_run_prefix} {script_location} -w {simulation_folder} --bias {bias} -f {packing_fname}"

    script = GREENE_SCRIPT_TEMPLATE.format(
        time_str=time_str,
        ntasks=ntasks,
        cpus_per_task=cpus_per_task,
        mem_str=mem_str,
        out_file=out_file,
        run_command=run_command,
        simulation_folder=simulation_folder,
        email=USER_EMAIL,
        email_type=EMAIL_TYPE,
        ext3_file=EXT3_FILE,
        conda_env=CONDA_ENV,
        singularity_overlay=GREENE_SINGULARITY_OVERLAY,
    )
    
    script_save_folder = os.path.join(simulation_folder, "job_scripts")
    script_path = os.path.join(script_save_folder,"compute_volume_"+packing_fname+".sh")
    
    # check if job with same script name is still running
    user = USER_EMAIL.split("@")[0]  # XXX This might be a bit too us-dependent, could adapt this
    jobs_list = subprocess.check_output(f'squeue -u {user} -o "%o"', shell=True)
    conflict = script_path in jobs_list.decode()
    if conflict:
        print(f"Job already running for script {script_path}! Skipping.")
    else:
        # write the script
        with open(script_path, "w") as script_file:
            script_file.write(script)
        if submit:
            os.system(f"sbatch {script_path}")
            
    return 0


def make_time_str(minimizer, simulation_folder, simstage, time_dict, box_dim=2):
    if simstage == SimStage.JAMMED_PACKING:
        time = 2
        return hours_to_slurm_time(time), time

    sim_folder = os.path.basename(simulation_folder)
    sim_folder_parts = sim_folder.split("_")
    n_particles = int(sim_folder_parts[1])
    nearest_power_of_two = 2 ** int(np.log2(n_particles))
    if nearest_power_of_two < 8:
        nearest_power_of_two == 8
    elif nearest_power_of_two > 256:
        nearest_power_of_two == 256
    time = time_dict[str(nearest_power_of_two)]

    if minimizer == "CVODE" or minimizer == "MXD":
        time *= 4
        
    if simstage == SimStage.PT:
        time *= 4
    if simstage == SimStage.ANALYSIS:
        time *= 4
    if box_dim == 3:
        time *=2
        
    # max job time
    if time > 168:
        time = 168
    return hours_to_slurm_time(time), time


def hours_to_slurm_time(hours):
    """
    Convert hours to a SLURM time string in the format "days-hours:minutes:seconds".

    Parameters:
    hours (float): The number of hours.

    Returns:
    str: A SLURM time string.
    """

    if hours <= 0:
        raise ValueError("Hours must be a positive number.")

    # Convert hours to seconds
    total_seconds = int(hours * 3600)

    # Calculate days, hours, minutes, and seconds
    days = total_seconds // 86400
    total_seconds %= 86400
    hours = total_seconds // 3600
    total_seconds %= 3600
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    # Create a SLURM time string
    slurm_time_str = f"{days}-{hours:02}:{minutes:02}:{seconds:02}"

    return slurm_time_str


def submit_initial_jobs(
    simulation_folder,
    ntasks,
    cpus_per_task,
    jammed_data_kwargs,
    hard_sphere_packing_kwargs,
    jammed_packing_kwargs,
    generate_packing_script_subpath,
    generate_jammed_packing_script_subpath,
    time_str,
    mem_str,
    job_name_prefix,
    script_run_prefix="python",
    extra_args="",
    submit=True,
):
    # Paths to the script files
    generate_packing_script_location = os.path.join(
        BASINVOLUME_PATH, generate_packing_script_subpath
    )
    generate_jammed_packing_script_location = os.path.join(
        BASINVOLUME_PATH, generate_jammed_packing_script_subpath
    )

    # Isolate loop arguments
    minimizer_list = jammed_data_kwargs["minimizer_list"]
    ss_packing_fraction_list = jammed_data_kwargs["ss_packing_fraction_list"]
    n_particles_list = jammed_data_kwargs["n_particles_list"]

    # Loop of jobs to submit
    for minimizer in minimizer_list:
        for packing_fraction in ss_packing_fraction_list:
            for n_particles in n_particles_list:
                # generate a directory for the experiment
                experiment_dir = os.path.join(
                    simulation_folder,
                    f"{minimizer}_{n_particles}_{packing_fraction}",
                )
                os.makedirs(experiment_dir, exist_ok=True)
                os.chdir(experiment_dir)
                job_script_dir = os.path.join(experiment_dir, "job_scripts")
                job_out_dir = os.path.join(experiment_dir, "job_out")
                packings_dir = os.path.join(experiment_dir, "packings")
                os.makedirs(job_script_dir, exist_ok=True)
                os.makedirs(job_out_dir, exist_ok=True)
                os.makedirs(packings_dir, exist_ok=True)

                # Add loop arguments to the dictionaries
                local_packing_kwargs = {"npackings": jammed_data_kwargs["n_ensemble"]}
                local_jammed_packing_kwargs = {
                    "minimizer": minimizer,
                    "density": packing_fraction,
                }
                loop_packing_kwargs = {
                    **hard_sphere_packing_kwargs,
                    **local_packing_kwargs,
                }
                loop_jammed_packing_kwargs = {
                    **jammed_packing_kwargs,
                    **local_jammed_packing_kwargs,
                }

                # Translate to argument string then to the run command
                packing_args_str = format_args_from_dict(loop_packing_kwargs)
                jammed_packing_args_str = format_args_from_dict(loop_jammed_packing_kwargs)
                packing_run_command = f"{script_run_prefix} {generate_packing_script_location} {n_particles} {packing_args_str}"
                jammed_packing_run_command = f"{script_run_prefix} {generate_jammed_packing_script_location} {extra_args} {jammed_packing_args_str}"
                # Run both in sequence
                run_command = f"{packing_run_command};\n{jammed_packing_run_command}"

                # Interface with slurm
                scripts_folder = os.path.join(experiment_dir, "job_scripts")
                out_folder = os.path.join(experiment_dir, "job_out")
                os.makedirs(scripts_folder, exist_ok=True)
                os.makedirs(out_folder, exist_ok=True)
                out_file = f"{out_folder}/{job_name_prefix}"
                script = GREENE_SCRIPT_TEMPLATE.format(
                    time_str=time_str,
                    ntasks=ntasks,
                    cpus_per_task=cpus_per_task,
                    mem_str=mem_str,
                    out_file=out_file,
                    run_command=run_command,
                    simulation_folder=experiment_dir,
                    email=USER_EMAIL,
                    email_type=EMAIL_TYPE,
                    ext3_file=EXT3_FILE,
                    conda_env=CONDA_ENV,
                    singularity_overlay=GREENE_SINGULARITY_OVERLAY,
                )
                script_path = os.path.join(scripts_folder, f"{job_name_prefix}.sh")

                # check if job with same script name is still running
                user = USER_EMAIL.split("@")[0]  # XXX This might be a bit too us-dependent, could adapt this
                jobs_list = subprocess.check_output(f'squeue -u {user} -o "%o"', shell=True)
                conflict = script_path in jobs_list.decode()
                if conflict:
                    print(f"Job already running for script {script_path}! Skipping.")
                else:
                    # write the script
                    print("script_path", script_path)
                    with open(script_path, "w") as script_file:
                        script_file.write(script)
                    if submit:
                        os.system(f"sbatch {script_path}")


def submit_job(
    simulation_folder,
    run_specific_kwargs,
    packing_file,
    ntasks,
    cpus_per_task,
    default_kwargs,
    script_subpath,
    time_str,
    mem_str,
    job_name_prefix,
    script_run_prefix="python",
    extra_args="",
    submit=True,
    kmax=False,
):
    script_kwargs = default_kwargs.copy()
    script_kwargs.update(run_specific_kwargs)

    if kmax:
        minimizer = script_kwargs["minimizer"]
        actual_opt_kwargs = {"opt_kwargs": script_kwargs["opt_kwargs"][minimizer][0]}
        opt_kwargs_file = os.path.join(simulation_folder, "opt_kwargs.toml")
        with open(opt_kwargs_file, "w") as f:
            toml.dump(actual_opt_kwargs, f)
        script_kwargs["opt_kwargs_file"] = opt_kwargs_file
        del script_kwargs["opt_kwargs"]

    args_str = format_args_from_dict(script_kwargs)
    script_location = os.path.join(BASINVOLUME_PATH, script_subpath)

    packing_file_name = os.path.splitext(packing_file)[0]
    run_command = f"{script_run_prefix} {script_location} {packing_file} {extra_args} {args_str}"
    # replace spaces with underscores
    args_str = args_str.replace(" ", "_")
    scripts_folder = os.path.join(simulation_folder, "job_scripts")
    out_folder = os.path.join(simulation_folder, "job_out")
    os.makedirs(scripts_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)
    out_file = f"{out_folder}/{job_name_prefix}_{packing_file_name}"

    script = GREENE_SCRIPT_TEMPLATE.format(
        time_str=time_str,
        ntasks=ntasks,
        cpus_per_task=cpus_per_task,
        mem_str=mem_str,
        out_file=out_file,
        run_command=run_command,
        simulation_folder=simulation_folder,
        email=USER_EMAIL,
        email_type=EMAIL_TYPE,
        ext3_file=EXT3_FILE,
        conda_env=CONDA_ENV,
        singularity_overlay=GREENE_SINGULARITY_OVERLAY,
    )

    script_path = os.path.join(scripts_folder, f"{job_name_prefix}_{packing_file_name}.sh")

    # check if job with same script name is still running
    user = USER_EMAIL.split("@")[0]  # XXX This might be a bit too us-dependent, could adapt this
    jobs_list = subprocess.check_output(f'squeue -u {user} -o "%o"', shell=True)
    conflict = script_path in jobs_list.decode()
    if conflict:
        print(f"Job already running for script {script_path}! Skipping.")
    else:
        # write the script
        print("script_path", script_path)
        with open(script_path, "w") as script_file:
            script_file.write(script)
        print("script_file written")
        if submit:
            os.system(f"sbatch {script_path}")
