##Example Torque script to submit jobs individually using BVSubmitPBS
#PBS -N submit_single_jobs 
#PBS -q s1
#PBS -l nodes=1:ppn=1 
#PBS -l walltime=00:03:00:00

ndim=2
geometry="sphere"
geom_params="1."
stepsize=0.5
cloud_radius=0.25
nr_cloud_points=10
qq
path_to_script="/home/sm958/Work/basinvolume/basinvolume/geometry/bv_run_geom.py"

cd ${PBS_O_WORKDIR} 

echo Starting job $PBS_JOBID 
echo
echo PBS assigned me this node: 
cat $PBS_NODEFILE 
echo 
echo "Running ${job_name}" 
echo 
python $path_to_script -d ${ndim} -g ${geometry} -r ${cloud_radius} -n ${nr_cloud_points} -x ${stepsize} -p ${geom_params}
echo 
echo "Job finished. PBS details are:" 
echo 
qstat -f ${PBS_JOBID} 
echo 
echo Finished at \`date\`
