echo "getting data from dexter"
for f in large_basin_results small_basin_results
    do
        rsync -Pva dexter:/sharedscratch/sm958/Jobs/trajectories/benchmark/$f/ /scratch/kjs73/basin_traj_data/$f/
    done
echo "done"
