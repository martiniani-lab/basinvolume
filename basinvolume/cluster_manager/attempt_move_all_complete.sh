# Attempt to move explore jammed packing output to destination in $1
explore_files=$(find explore_bv_jammed_packing* -maxdepth  0)
for f in $explore_files
do
    echo "moving "$f
    sh ~/projects/basinvolume/basinvolume/cluster_manager/move_data.sh $f $1
done

exit 0

