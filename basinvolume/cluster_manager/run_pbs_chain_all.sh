##Example Torque script to submit job chains using submit_bv
#PBS -N submit_chain_jobs
#PBS -q test
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:01:00:00
#PBS -j oe

ndim=2
workdir=${PBS_O_WORKDIR}
path_to_script="/home/sm958/Work/basinvolume/basinvolume/spheres/"
job_label="32_70_88_2D"
k_queue_type="s16"
k_walltime=12
pt_queue_type="s16"
pt_walltime=24
cores_per_node=16
threads=1
nojmin=0
nojmax=500
minimizer=LBFGS
pt_workers=4
pt_runners=16
pt_sleep_seconds=0.0001
pt_exchange=INDEPENDENCE_SAMPLING
pt_relstderr=0.05
pt_checkpoint_time=1410

cd ${PBS_O_WORKDIR}

echo Starting job $PBS_JOBID
echo
echo PBS assigned me this node:
cat $PBS_NODEFILE
echo
echo "Running ${PBS_JOBNAME}"
echo
python ~/Work/basinvolume/basinvolume/cluster_manager/submit_bv.py chain \
$ndim $workdir $path_to_script $job_label $k_queue_type $k_walltime \
$pt_queue_type $pt_walltime --cores-per-node $cores_per_node \
--threads $threads --nojmin $nojmin --nojmax $nojmax --minimizer $minimizer \
--pt-workers $pt_workers --pt-runners $pt_runners \
--pt-sleep-seconds $pt_sleep_seconds --pt-exchange-scheme $pt_exchange \
--relstderr $pt_relstderr --pt-checkpoint-time $pt_checkpoint_time --sort --delraw
echo
echo "Job finished. PBS details are:"
echo
qstat -f ${PBS_JOBID}
echo
echo Finished at `date`
