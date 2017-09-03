#!/bin/bash
##Example SLURM script to submit job chains using submit_bv
#SBATCH --job-name=submit_chain_jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00-01:00:00
#SBATCH --output=submit_%j.out

batch_system=SLURM
mpi_implementation=Intel
ndim=2
workdir=${SLURM_SUBMIT_DIR}
path_to_script="/home/jk695/basinvol/basinvolume/basinvolume/spheres/"
job_label="100_16"
k_walltime=12
pt_walltime=12
cores_per_node=16
threads=1
pt_threads=1
nojmin=0
nojmax=500
minimizer=LBFGS
kmax_start=500
pt_workers=4
pt_replicas=16
pt_adjustf_navg=100
pt_sleep_seconds=0.0001
pt_exchange=INDEPENDENCE_SAMPLING
pt_relstderr=0.1
pt_checkpoint_time=700

cd ${SLURM_SUBMIT_DIR}

echo Starting job $SLURM_JOBID
echo
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2
echo
echo "Running ${SLURM_JOB_NAME}"
echo
python ~/basinvol/basinvolume/basinvolume/cluster_manager/submit_bv.py chain \
$ndim $workdir $path_to_script $job_label $k_walltime $pt_walltime \
--batch-system $batch_system --mpi-implementation $mpi_implementation \
--cores-per-node $cores_per_node --threads $threads --pt-threads $pt_threads \
--nojmin $nojmin --nojmax $nojmax --minimizer $minimizer \
--kmax-start $kmax_start --pt-workers $pt_workers --pt-replicas $pt_replicas \
--pt-adjustf-navg $pt_adjustf_navg --pt-sleep-seconds $pt_sleep_seconds \
--pt-exchange-scheme $pt_exchange --relstderr $pt_relstderr \
--pt-checkpoint-time $pt_checkpoint_time --sort --delraw
echo
echo "Job finished. SLURM details are:"
echo
scontrol show job ${SLURM_JOBID}
echo
echo Finished at `date`
