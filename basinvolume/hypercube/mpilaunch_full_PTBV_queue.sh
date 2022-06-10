#!/bin/bash

BASEFOLDER="/home/mc9287/basinvolumelibs/basinvolume/basinvolume/hypercube"
CODEFILENAMEBV="${BASEFOLDER}/mkrunner_queue.py"
LOGFOLDER="/vast/mc9287/slurm_logs"

NEGATIVESPRINGNUMBER=16
POSITIVESPRINGNUMBER=48 #Best to have TOTAL spring number ~ 4 * cpu number if using Johannes's code..... Which is not the case here!
CPUNUMBER=16

FLOATTIME=5000000
printf -v MINTOTNITER '%d' ${FLOATTIME}


# Create a script.sub with the right arguments within 

rm -rf script_slurm.sh

echo "#!/bin/bash">script_slurm.sh
echo "#SBATCH --job-name=QHypercubeBVPT">>script_slurm.sh
echo "#SBATCH --output=$LOGFOLDER/QHypercubeBVPT_%j.out">>script_slurm.sh
echo "#SBATCH --error=$LOGFOLDER/QHypercubeBVPT_%j.err">>script_slurm.sh

echo "#SBATCH --nodes=1">>script_slurm.sh
echo "#SBATCH --ntasks-per-node=$CPUNUMBER">>script_slurm.sh
echo "#SBATCH --cpus-per-task=1">>script_slurm.sh
echo "#SBATCH --mem=128GB">>script_slurm.sh
echo "#SBATCH --time=3-0:00:00">>script_slurm.sh

echo "#SBATCH --mail-type=ALL">>script_slurm.sh
echo "#SBATCH --mail-user=mc9287@nyu.edu">>script_slurm.sh
# echo "srun nice -n 19 /home/mc9287/singularity_rust_startup.sh cargo run --release \$@">>script_slurm.sh
echo "/home/mc9287/singularity_condawithopenmpi_startup.sh \$@">>script_slurm.sh

for n in 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 5000 10000
do

printf -v NDIM '%d' $n
ACTUALOUTPUTFOLDER="/scratch/mc9287/remote_no_copy/hypercube/explore_bv_hypercube_n${NDIM}_l1/"
LINKTOFOLDER="/home/mc9287/basinvolumelibs/basinvolume/basinvolume/hypercube/"
mkdir -p $ACTUALOUTPUTFOLDER
ln -s $ACTUALOUTPUTFOLDER $LINKTOFOLDER

# Send back list of arguments to queue
# sbatch script_slurm.sh "--jobs $CPUNUMBER $N $Phi $CPUNUMBER $FULLFOLDER";
sbatch script_slurm.sh "mpirun -n $CPUNUMBER python $CODEFILENAMEBV ${NDIM} --positivespringnumber=$POSITIVESPRINGNUMBER --negativespringnumber=$NEGATIVESPRINGNUMBER --min_tot_niter=$MINTOTNITER -n_spheres=1 -force_k=True -k_sprd=positionlinspace --auto_replica_number";

done


