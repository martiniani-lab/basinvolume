#!/bin/bash
# Collects runtime data to predict runtimes for future runs.
# Runtime prediction is done by e.g. runtime_dexter.py.
#

# Set parameters.
remote_computer=nemesis.ch.private.cam.ac.uk
remote_folder=/media/exhhd/new_data
runtime_folder=runtime_data
#
echo "This is collect_runtime_data.sh."

# Print parameters.
echo "List of input parameters:"
echo "remote_computer: "$remote_computer
echo "remote_folder: "$remote_folder

# Check that external disk is accessible.
echo "Finding data folders."
if (ssh $remote_computer '[ -d $remote_folder ]')
then
    echo "Origin exists."
else 
    echo "Origin does not exist."
    exit 42
fi

# Prepare runtime data storage.
mkdir -p $runtime_folder

# Find different system sizes and get files.
nN_data_folders=$( ( ssh $remote_computer ls $remote_folder ) )
for f in $nN_data_folders
do
    echo "add data for "$f
    mkdir -p $runtime_folder/$f
    explore_folders=$( ( ssh $remote_computer ls $remote_folder/$f ) )
    for ef in $explore_folders
    do
        if [[ $ef == "explore_bv_jammed_packing"* ]]
        then
            echo "add "$ef
            rt_file_name=$runtime_folder/$f/$ef.runtime
            if [ ! -f $rt_file_name ]
            then
                time_line=$( ( ssh $remote_computer cat $remote_folder/$f/$ef/bv_*pt*.o* | grep "resources_used.walltime" ) )
                echo $time_line > $rt_file_name
            fi
        fi
    done
done
