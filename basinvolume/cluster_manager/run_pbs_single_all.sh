##Example Torque script to submit jobs individually using submit_bv
#PBS -N submit_single_jobs
#PBS -q test
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:01:00:00
#PBS -j oe

ndim=2
workdir=${PBS_O_WORKDIR}
path_to_script="/home/sm958/Work/basinvolume/basinvolume/spheres/"
job_label="32_70_88_2D"
queue_type="s16"
walltime=12
cores_per_node=16
threads=1
nojmin=0
nojmax=500
minimizer=LBFGS
pt_workers=4
pt_runners=16
pt_sleep_seconds=0.0001
pt_exchange=INDEPENDENCE_SAMPLING
#options are --kmin and --kmax or --pt
option="--pt"

cd ${PBS_O_WORKDIR}

echo Starting job $PBS_JOBID
echo
echo PBS assigned me this node:
cat $PBS_NODEFILE
echo
echo "Running ${PBS_JOBNAME}"
echo
python ~/Work/basinvolume/basinvolume/cluster_manager/submit_bv.py single \
$ndim $workdir $path_to_script $job_label $queue_type \
$walltime $option --cores-per-node $cores_per_node \
--threads $threads --nojmin $nojmin --nojmax $nojmax --minimizer $minimizer \
--pt-workers $pt_workers --pt-runners $pt_runners \
--pt-sleep-seconds $pt_sleep_seconds --pt-exchange-scheme $pt_exchange --sort --delraw
echo
echo "Job finished. PBS details are:"
echo
qstat -f ${PBS_JOBID}
echo
echo Finished at `date`
