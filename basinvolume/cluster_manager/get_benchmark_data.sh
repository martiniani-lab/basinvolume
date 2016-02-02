echo "getting data from dexter"
for f in large_basin_results small_basin_results
    do
        rsync -Pvua dexter:/sharedscratch/sm958/Jobs/trajectories/benchmark/$f/ ./$f/
    done
echo "done"
