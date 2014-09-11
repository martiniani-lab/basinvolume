##Example Torque script to submit job chains using BVSubmitPB
#PBS -N submit_chain_jobs 
#PBS -q test 
#PBS -l nodes=1:ppn=1 
#PBS -l walltime=00:01:00:00 

ndim=2
workdir=${PBS_O_WORKDIR}
job_label="32_70_88_2D"
k_queue_type="s16"
k_nodes=1
k_cores=1
k_walltime=12
pt_queue_type="s16"
pt_nodes=1
pt_cores=7
pt_walltime=12
nojmin=0
nojmax=500
path_to_script="/home/sm958/Work/basinvolume/basinvolume/spheres/"

cd ${PBS_O_WORKDIR} 

echo Starting job $PBS_JOBID 
echo
echo PBS assigned me this node: 
cat $PBS_NODEFILE 
echo 
echo "Running ${job_name}" 
echo 
python ~/Work/basinvolume/basinvolume/cluster_manager/BVSubmitPBS.py chain $ndim $workdir $path_to_script $job_label $k_queue_type \
$k_nodes $k_cores $k_walltime $pt_queue_type $pt_nodes $pt_cores $pt_walltime --nojmin $nojmin --nojmax $nojmax  
echo 
echo "Job finished. PBS details are:" 
echo 
qstat -f ${PBS_JOBID} 
echo 
echo Finished at \`date\`
