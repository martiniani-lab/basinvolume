# Generates jammed packings across a range of volume fractions and saves them to a directory


from greene_submission import GREENE_SCRIPT_TEMPLATE
import os


def main():
    BASE_DIR = "/scratch/ps4586/volume_runs_multi_packing2"
    minimizer_list = ["FIRE", "CG", "CVODE", "LBFGS", "MXD"]
    minimizer_list = ["CVODE"]
    packing_fraction_list = [0.85, 0.86, 0.87, 0.88, 0.90]
    packing_fraction_list = [0.85]
    n_particles_list = [32]
    n_ensemble = 4

    for minimizer in minimizer_list:
        for packing_fraction in packing_fraction_list:
            for n_particles in n_particles_list:
                generate_packings(
                    BASE_DIR,
                    minimizer,
                    n_particles,
                    packing_fraction,
                    n_ensemble,
                )


def generate_packings(
    base_dir, minimizer, n_particles, packing_fraction, n_ensemble
):

    # generate a directory for the experiment
    experiment_dir = os.path.join(
        base_dir, f"{minimizer}_{n_particles}_{packing_fraction}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    n_ensemble = 4
    os.chdir(experiment_dir)
    job_script_dir = os.path.join(experiment_dir, "job_scripts")
    job_out_dir = os.path.join(experiment_dir, "job_out")
    os.makedirs(job_script_dir, exist_ok=True)
    os.makedirs(job_out_dir, exist_ok=True)
    # note that these arguments are placeholders since the code was written for HS_WCA
    pack_comm = f"python /home/ps4586/bv_lib/basinvolume/basinvolume/spheres/generate_packing.py"
    pack_comm += f" {n_particles} -n {n_ensemble} -d 2 -p 0.7 -u 1 -s 0.1"

    jpack_com = f"python /home/ps4586/bv_lib/basinvolume/basinvolume/spheres/generate_jammed_packing.py"
    jpack_com += f" -p {packing_fraction} --minimizer CVODE --interaction INVERSE_POWER"  # keep the minimizer same for all
    full_command = f"{pack_comm};\n{jpack_com}"
    time_str = "00:15:00"
    script = GREENE_SCRIPT_TEMPLATE.format(
        time_str=time_str,
        mem_str="2GB",
        ntasks=1,
        cpus_per_task=1,
        out_file=os.path.join(job_out_dir, "generate_packing.out"),
        simulation_folder=experiment_dir,
        run_command=full_command,
    )
    with open(os.path.join(job_script_dir, "generate_packing.sh"), "w") as f:
        f.write(script)
    os.system(f"sbatch {os.path.join(job_script_dir, 'generate_packing.sh')}")
    os.chdir("..")
    return


if __name__ == "__main__":
    main()
