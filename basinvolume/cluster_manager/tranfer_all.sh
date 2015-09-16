#example script to transfer output to \frenkelscratch type crontab -e
#and add a call to a copy of this bash script, example crontab line
#that will run at 23:30 every day: 
#30 23 * * bash /sharedscratch/sm958/Jobs/basin_volume/n32_phi52705938645472294_phi74148048969306102_3D/transfer_all.sh

wdir="/sharedscratch/sm958/Jobs/basin_volume/n32_phi52705938645472294_phi74148048969306102_3D/"
attempt_move_all="/home/sm958/Work/basinvolume/basinvolume/cluster_manager/attempt_move_all_complete.sh"
dest="sm958@dexter:/frenkelscratch/sm958/crystal_dos/n32_phi52705938645472294_phi74148048969306102_3D/"

echo "transferring data"
for dname in $(ls $wdir)
do
    if [[ -d $wdir$dname ]] && [[ $dname == "n32_phi52705938645472294_phi74148048969306102_3D_"* ]]; then
    echo "destination $dest$dname"
    (cd $wdir$dname && yes | nohup sh $attempt_move_all  $dest$dname > $wdir$dname"/transfer_output.txt" 2>&1 &)
    fi
done
