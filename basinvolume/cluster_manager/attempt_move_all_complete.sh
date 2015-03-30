# Attempt to move explore jammed packing output to destination in $1
# http://stackoverflow.com/questions/1744595/extract-the-last-directory-of-a-pwd-output
top_dir=$(basename `pwd`)
echo "top directory is "$top_dir
remote_top_dir=$(echo $1 | sed 's#.*/##')
echo "remote top directory "$remote_top_dir
if [ "$top_dir" != "$remote_top_dir" ];
then
    echo "top directories are not matching -- terminating"
    exit 42
fi
explore_files=$(find explore_bv_jammed_packing* -maxdepth  0)
for f in $explore_files
do
    echo "moving "$f
    sh ~/Work/basinvolume/basinvolume/cluster_manager/move_data.sh $f $1
done

exit 0

