"""Script to submit multi volume calculations on the cluster.
"""


from enum import unique, Enum
import os


# hardcoded because we don't want to import this
# importing basinvolume can only be done in singularity
# and this script just manages job submission
BASINVOLUME_PATH = "/home/ps4586/bv_lib/basinvolume/basinvolume"


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
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ps4586@nyu.edu
#SBATCH --job-name={out_file}
#SBATCH --output={out_file}.out

export OMP_NUM_THREADS=1;

cd {simulation_folder};
singularity exec --overlay /scratch/ps4586/conda/overlay-10GB-400K.ext3:ro \
    /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
    /bin/bash -c "source ~/.bashrc;
    export OMP_NUM_THREADS=1;
    conda activate cb3-3.9;
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
    submit=True,
    global_kwargs=dict(
        opt_tol=1e-10,
        opt_nsteps=1e5,
        dtol=1e-2,
        opt_dtmax=1,
        minimizer="LBFGS",
    ),
):
    # global args that should be the same across scripts
    if simulation_type == SimStage.KMAX:
        setup_kmax(
            simulation_folder, global_kwargs, packing_file, submit=submit
        )
    elif simulation_type == SimStage.KMIN:
        setup_kmin(
            simulation_folder, global_kwargs, packing_file, submit=submit
        )
    elif simulation_type == SimStage.PT:
        setup_parallel_tempering(
            simulation_folder, global_kwargs, packing_file, submit=submit
        )
    elif simulation_type == SimStage.INNER_SPHERE:
        setup_inner_sphere(
            simulation_folder, global_kwargs, packing_file, submit=submit
        )
    elif simulation_type == SimStage.ANALYSIS:
        setup_compute_volume(simulation_folder, submit=submit)
    else:
        raise NotImplementedError("simulation type not implemented")


def setup_kmax(simulation_folder, global_kwargs, packing_file, submit=True):
    # single core args
    ntasks = 1
    cpus_per_task = 1

    # Most of these should be defaults but you can change them at the script level
    kmax_kwargs = {
        "kstart": 500,
        "packings-dir": "jammed_packings",
        "explore-dir": "explore_bv_jammed_packing",
        "nocell": False,
        "minimizer": "FIRE",
        "verbose": False,
        "seed-takestep": None,
        "niter": 1e8,
        "dtol": 1e-2,
        "eps": 1.0,
        "ktarget": 0.9,
        "knavg": 1e4,
        "ktol": 0.025,
        "opt_dtmax": 1,
        "opt_tol": 1e-5,
        "opt_nsteps": 1e5,
    }
    mem_str = "4GB"
    time_str = make_time_str(kmax_kwargs["minimizer"], simulation_folder)

    # update defaults with global kwargs
    script_subpath = "spheres/bv_find_kmax.py"
    job_name_prefix = "bv_kmax"

    submit_job(
        simulation_folder,
        global_kwargs,
        packing_file,
        ntasks,
        cpus_per_task,
        kmax_kwargs,
        script_subpath,
        time_str,
        mem_str,
        job_name_prefix,
        submit=submit,
    )
    return 0


def setup_kmin(simulation_folder, global_kwargs, packing_file, submit=True):
    ntasks = 1
    cpus_per_task = 1
    # defaults but you can change them at the script level
    kmin_kwargs = {
        "packings-dir": "jammed_packings",
        "explore-dir": "explore_bv_jammed_packing",
        "niter": 1e5,
        "adjustf-niter": 1e4,
        "nocell": False,
        "moveall": False,
        "minimizer": "FIRE",
        "rsts": False,
        "rsts-only": False,
        "verbose": False,
        "seed-takestep": None,
        "seed-metropolis": None,
        "k": 0,
        "stepsize": 1e-1,
        "dtol": 1e-2,
        "eps": 1.0,
        "hmin": 0,
        "hmax": 1000,
        "hbinsize": 1,
        "acceptance": 0.2,
        "adjustf": 0.9,
        "opt_dtmax": 1,
        "opt_tol": 1e-10,
        "opt_nsteps": 1e5,
        "record_trajectory_npoints": int(1e4),
    }
    script_subpath = "spheres/bv_find_kmin.py"
    job_name_prefix = "bv_kmin"
    # TODO: set these times based on problem dimension
    time_str = make_time_str(kmin_kwargs["minimizer"], simulation_folder)
    mem_str = "4GB"

    submit_job(
        simulation_folder,
        global_kwargs,
        packing_file,
        ntasks,
        cpus_per_task,
        kmin_kwargs,
        script_subpath,
        time_str,
        mem_str,
        job_name_prefix,
        submit=submit,
    )
    return 0


def make_time_str(minimizer, simulation_folder, parallel_tempering=False):
    slurm_time_dict = {
        32: 1,
        64: 4,
        128: 16,
        256: 32,
    }
    sim_folder = os.path.basename(simulation_folder)
    sim_folder_parts = sim_folder.split("_")
    n_particles = int(sim_folder_parts[1])
    time = slurm_time_dict[n_particles]

    if minimizer == "CVODE":
        time *= 4

    if parallel_tempering:
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


def submit_job(
    simulation_folder,
    simulation_global_kwargs,
    packing_file,
    ntasks,
    cpus_per_task,
    script_kwargs,
    script_subpath,
    time_str,
    mem_str,
    job_name_prefix,
    script_run_prefix="python",
    extra_args="",
    submit=True,
):
    script_kwargs.update(simulation_global_kwargs)
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
    )

    script_path = os.path.join(
        scripts_folder, f"{job_name_prefix}_{packing_file_name}.sh"
    )
    # write the script
    with open(script_path, "w") as script_file:
        script_file.write(script)

    if submit:
        os.system(f"sbatch {script_path}")


def setup_parallel_tempering(
    simulation_folder, global_kwargs, packing_file, submit=True
):
    replicas = 64
    mpi_procs = 16
    ntasks = 1
    cpus_per_task = mpi_procs
    pt_kwargs = {
        "mintotniter": 5e5,
        "maxtotniter": 2e6,
        "adjustf-niter": None,
        "numnegk": 23,
        "lownegk": -0.5,
        "relstderr": 0.05,
        "nocell": False,
        "moveall": False,
        "adjustf-navg": 100,
        "minimizer": "FIRE",
        "verbose": False,
        "collect-minima": False,
        "packings-dir": "jammed_packings",
        "delraw": False,
        "nreplicas": None,
        "sleep-seconds": 0.0001,
        "exchange-scheme": "NEIGHBOR_EXCHANGE",
        "checkpoint-time": None,
        "load-checkpoint": None,
        "stepsize": 1e-1,
        "dtol": 1e-2,
        "opt_tol": 1e-10,
        "opt_nsteps": 1e5,
        "hmin": 0,
        "hmax": 1000,
        "hbinsize": 1e-1,
        "acceptance": 0.2,
        "adjustf": 0.9,
        "k_spreading": "positionlinspace",
    }
    pt_kwargs["nreplicas"] = replicas
    script_subpath = "spheres/bv_parallel_tempering.py"
    job_name_prefix = "bv_pt"
    time_str = make_time_str(
        pt_kwargs["minimizer"], simulation_folder, parallel_tempering=True
    )

    mem_str = "20GB"
    # give the explore directory as the argument
    packing_fname = os.path.splitext(packing_file)[0]
    explore_dir = f" explore_bv_{packing_fname}"

    submit_job(
        simulation_folder,
        global_kwargs,
        packing_file,
        ntasks,
        cpus_per_task,
        pt_kwargs,
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
    simulation_folder, global_kwargs, packing_file, submit=True
):
    ntasks = 1
    cpus_per_task = 1
    inner_sphere_kwargs = {
        "packings-dir": "jammed_packings",
        "explore-dir": "explore_bv_jammed_packing",
        "nocell": False,
        "minimizer": "FIRE",
        "verbose": False,
        "niter": 1e5,
        "dtol": 1e-2,
        "eps": 1.0,
        "opt_dtmax": 1,
        "opt_tol": 1e-10,
        "opt_nsteps": 1e5,
    }
    script_subpath = "mbar_spheres/bv_innersphere_dos.py"
    job_name_prefix = "bv_inner_sphere"
    # TODO: set these times based on problem dimension
    time_str = make_time_str(
        inner_sphere_kwargs["minimizer"], simulation_folder
    )
    mem_str = "8GB"
    # give the explore directory as an argument
    packing_fname = os.path.splitext(packing_file)[0]
    explore_dir = f" explore_bv_{packing_fname}"
    submit_job(
        simulation_folder,
        global_kwargs,
        packing_file,
        ntasks,
        cpus_per_task,
        inner_sphere_kwargs,
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


if __name__ == "__main__":
    test_folder = "/scratch/ps4586/test_volume"
    calculate_volume(
        test_folder, "jammed_packing0.xydr", SimStage.KMAX, submit=False
    )
    calculate_volume(
        test_folder, "jammed_packing0.xydr", SimStage.KMIN, submit=False
    )
    calculate_volume(
        test_folder,
        "jammed_packing0.xydr",
        SimStage.PT,
        submit=False,
    )
    calculate_volume(
        test_folder,
        "jammed_packing0.xydr",
        SimStage.INNER_SPHERE,
        submit=False,
    )
    calculate_volume(
        test_folder,
        "jammed_packing0.xydr",
        SimStage.ANALYSIS,
        submit=False,
    )
