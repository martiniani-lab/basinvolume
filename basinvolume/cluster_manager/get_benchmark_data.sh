echo "getting benchmark data from zero and deathstar"
for c in deathstar zero
    do
        for f in large_basin_results small_basin_results
            do
                rsync -Pvua kjs73@$c:/sharedscratch/kjs73/Jobs/trajectories/benchmark/$f/ ./$f/
            done
    done
echo "getting data from dexter"
for f in large_basin_results small_basin_results
    do
        rsync -Pvua dexter:/sharedscratch/sm958/Jobs/trajectories/benchmark/$f/ ./$f/
    done
echo "done"
