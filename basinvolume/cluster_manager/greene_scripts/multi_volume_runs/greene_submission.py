"""Script to submit multi volume calculations on the cluster.
"""


from enum import unique, Enum
import os
import toml

# hardcoded because we don't want to import this, and the relative paths are set
# importing basinvolume can only be done in singularity
# and this script just manages job submission
current_directory = os.getcwd()
BASINVOLUME_PATH = os.path.join(current_directory, "../../..")

# Load possibly changing paths and environment variables from a toml config file
CONFIG_FILE = os.path.join(current_directory, "local_config.toml")
config = toml.load(CONFIG_FILE)
# User-dependent values
email = config["user"]["email"]
email_type = config["user"]["email_type"]
ext3_file = config["user"]["ext3_file"]
conda_env = config["user"]["conda_env"]
# Cluster-dependent values, may change over time and/or between clusters even with similar architectures
singularity_overlay = config["cluster"]["singularity_overlay"]

# Load defaults from the relevant config file
DEFAULTS_CONFIG_FILE = os.path.join(current_directory, "default_params.toml")
default_config = toml.load(DEFAULTS_CONFIG_FILE)

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
    checkpoint_file = None,
):
    # global args that should be the same across scripts
    # only kmax sees the optimizer kwargs, the following steps just read them off from the kmax config file
    
    # Pass time_str to everyone using opt_kwargs
    # TODO: set these times based on problem dimension
    kmax_dict = default_config["kmax_defaults"]
    if run_params["kmax"] != {}:
        kmax_dict = kmax_dict.update(run_params["kmax"])
    minimizer = kmax_dict["minimizer"]
    time_str = make_time_str(minimizer, simulation_folder, simulation_type)
    
    # Always checkpoint after 6 days if not over yet, always start from checkpoint if it exists
    checkpoint_time = 8640 
    if simulation_type == SimStage.JAMMED_PACKING:
        setup_generate_jammed_data(simulation_folder, run_params, time_str, submit=submit)
    elif simulation_type == SimStage.KMAX:
        setup_kmax(
            simulation_folder, run_params, packing_file, time_str, submit=submit
        )
    elif simulation_type == SimStage.KMIN:
        setup_kmin(
            simulation_folder, run_params, packing_file, time_str, submit=submit
        )
    elif simulation_type == SimStage.PT:
        setup_parallel_tempering(
            simulation_folder, run_params, packing_file, time_str, checkpoint_time=checkpoint_time, checkpoint_file=checkpoint_file, submit=submit
        )
    elif simulation_type == SimStage.INNER_SPHERE:
        setup_inner_sphere(
            simulation_folder, run_params, packing_file, time_str, submit=submit
        )
    elif simulation_type == SimStage.ANALYSIS:
        setup_compute_volume(simulation_folder, submit=submit)
    else:
        raise NotImplementedError("simulation type not implemented")

def setup_generate_jammed_data(simulation_folder, run_params, time_str, submit=True):
    # single core args
    ntasks = 1
    cpus_per_task = 1
    
    # Most of these should be defaults but you can change them at the script level
    jammed_data_kwargs = default_config["jammed_data_defaults"]
    jammed_data_kwargs.update(run_params["jammed_data"])
    hard_sphere_packing_kwargs = default_config["hard_sphere_packing_defaults"]
    hard_sphere_packing_kwargs.update(run_params["hard_sphere_packing"])
    jammed_packing_kwargs = default_config["jammed_packing_defaults"]
    jammed_packing_kwargs.update(run_params["jammed_packing"])
    mem_str = "4GB"
    
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
        submit=submit
    )
    
    return 0

def setup_kmax(simulation_folder, run_params, packing_file, time_str, submit=True):
    # single core args
    ntasks = 1
    cpus_per_task = 1

    # Most of these should be defaults but you can change them at the script level
    kmax_default_kwargs = default_config["kmax_defaults"]
    mem_str = "4GB"

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
    )
    return 0


def setup_kmin(simulation_folder, run_params, packing_file, time_str, submit=True):
    ntasks = 1
    cpus_per_task = 1
    # defaults but you can change them at the script level
    kmin_default_kwargs = default_config["kmin_defaults"]
    script_subpath = "spheres/bv_find_kmin.py"
    job_name_prefix = "bv_kmin"
    mem_str = "4GB"

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
    simulation_folder, run_params, packing_file, time_str, checkpoint_time = None, checkpoint_file = None, submit=True
):
    # TODO Adapt this to replicas
    mpi_procs = 16
    ntasks = 1
    cpus_per_task = mpi_procs
    pt_default_kwargs = default_config["pt_defaults"]
    pt_default_kwargs["checkpoint-time"] = checkpoint_time
    pt_default_kwargs["load-checkpoint"] = checkpoint_file
    script_subpath = "spheres/bv_parallel_tempering.py"
    job_name_prefix = "bv_pt"

    mem_str = "20GB"
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
        submit=submit
    )
    return 0


def setup_inner_sphere(
    simulation_folder, run_params, packing_file, time_str, submit=True
):
    ntasks = 1
    cpus_per_task = 1
    inner_sphere_default_kwargs = default_config["innersphere_defaults"]
    script_subpath = "mbar_spheres/bv_innersphere_dos.py"
    job_name_prefix = "bv_inner_sphere"
    mem_str = "8GB"
    # give the explore directory as an argument
    packing_fname = os.path.splitext(packing_file)[0]
    explore_dir = f" explore_bv_{packing_fname}"
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


def setup_compute_volume(simulation_folder, submit=True):
    ntasks = 1
    cpus_per_task = 1
    script_location = os.path.join(
        BASINVOLUME_PATH, "mbar_spheres/mbar_compute_volume.py"
    )

    out_folder = os.path.join(simulation_folder, "job_out")
    script = GREENE_SCRIPT_TEMPLATE.format(
        time_str="04:00:00",
        mem_str="16GB",
        ntasks=ntasks,
        cpus_per_task=cpus_per_task,
        job_name="bv_compute_volume",
        run_command=f"python {script_location}",
        out_file=os.path.join(out_folder, "compute_volume"),
        simulation_folder=simulation_folder,
    )
    script_save_folder = os.path.join(simulation_folder, "job_scripts")
    with open(
        os.path.join(script_save_folder, "compute_volume.sh"), "w"
    ) as script_file:
        script_file.write(script)
    if submit:
        os.system(
            f"sbatch {os.path.join(script_save_folder, 'compute_volume.sh')}"
        )
    return 0


def make_time_str(minimizer, simulation_folder, simstage):
    
    if simstage == SimStage.JAMMED_PACKING:
        time = 1
        return hours_to_slurm_time(time)
    
    slurm_time_dict = {
        8: 1,
        32: 1,
        64: 4,
        128: 168,
        256: 32,
    }
    sim_folder = os.path.basename(simulation_folder)
    sim_folder_parts = sim_folder.split("_")
    n_particles = int(sim_folder_parts[1])
    time = slurm_time_dict[n_particles]

    if minimizer == "CVODE":
        time *= 4

    if simstage == SimStage.PT:
        time *= 4

    # max job time
    if time > 168:
        time = 168
    return hours_to_slurm_time(time)


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
    generate_packing_script_location = os.path.join(BASINVOLUME_PATH, generate_packing_script_subpath)
    generate_jammed_packing_script_location = os.path.join(BASINVOLUME_PATH, generate_jammed_packing_script_subpath)
    
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
                    simulation_folder, f"{minimizer}_{n_particles}_{packing_fraction}"
                )
                os.makedirs(experiment_dir, exist_ok=True)
                os.chdir(experiment_dir)
                job_script_dir = os.path.join(experiment_dir, "job_scripts")
                job_out_dir = os.path.join(experiment_dir, "job_out")
                os.makedirs(job_script_dir, exist_ok=True)
                os.makedirs(job_out_dir, exist_ok=True)
                
                # Add loop arguments to the dictionaries
                local_packing_kwargs = {"n_particles": n_particles, "npackings": jammed_data_kwargs["n_ensemble"]}
                local_jammed_packing_kwargs = {"minimizer": minimizer, "density": packing_fraction, "n_particles": n_particles}
                loop_packing_kwargs = {**hard_sphere_packing_kwargs, **local_packing_kwargs}
                loop_jammed_packing_kwargs = {**jammed_packing_kwargs, **local_jammed_packing_kwargs}
                
                # Translate to argument string then to the run command
                packing_args_str = format_args_from_dict(loop_packing_kwargs)
                jammed_packing_args_str = format_args_from_dict(loop_jammed_packing_kwargs)
                packing_run_command = f"{script_run_prefix} {generate_packing_script_location} {extra_args} {packing_args_str}"
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
                    email = email,
                    email_type = email_type,
                    ext3_file=ext3_file,
                    conda_env=conda_env,
                    singularity_overlay=singularity_overlay
                )
                script_path = os.path.join(
                scripts_folder, f"{job_name_prefix}.sh"
                )
                # write the script
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
    submit=True
):
    script_kwargs = default_kwargs.update(run_specific_kwargs)
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
        email = email,
        email_type = email_type,
        ext3_file=ext3_file,
        conda_env=conda_env,
        singularity_overlay=singularity_overlay
    )

    script_path = os.path.join(
        scripts_folder, f"{job_name_prefix}_{packing_file_name}.sh"
    )
    # write the script
    with open(script_path, "w") as script_file:
        script_file.write(script)

    if submit:
        os.system(f"sbatch {script_path}")
