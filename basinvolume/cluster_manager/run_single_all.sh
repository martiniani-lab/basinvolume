##Example Torque script to submit jobs individually using BVSubmitPBS
#PBS -N submit_single_jobs 
#PBS -q test 
#PBS -l nodes=1:ppn=1 
#PBS -l walltime=00:01:00:00 

ndim=2
workdir=${PBS_O_WORKDIR}
job_label="32_70_88_2D"
queue_type="s16"
nodes=1
cores=7
walltime=12
nojmin=0
nojmax=500
path_to_script="/home/sm958/Work/basinvolume/basinvolume/spheres/"
#options are --kmin and --kmax or --pt
option="--pt"

cd ${PBS_O_WORKDIR} 

echo Starting job $PBS_JOBID 
echo
echo PBS assigned me this node: 
cat $PBS_NODEFILE 
echo 
echo "Running ${job_name}" 
echo 
python ~/Work/basinvolume/basinvolume/cluster_manager/BVSubmitPBS.py single $ndim $workdir $path_to_script $job_label $queue_type \
$nodes $cores $walltime $option --nojmin $nojmin --nojmax $nojmax 
echo 
echo "Job finished. PBS details are:" 
echo 
qstat -f ${PBS_JOBID} 
echo 
echo Finished at \`date\`
