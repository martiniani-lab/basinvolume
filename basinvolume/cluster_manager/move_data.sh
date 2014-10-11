#!/bin/bash
# Moves output of BV computation from cluster to a defined remote location.
#
# Intended procedure:
# 0. Check: Folder to move exists.
# 1. Check: Computation terminated properly.
# 2. Check: Destination can be rached / is a valid path / dir exists.
# 3. Check: Enough disc space available at destination.
# 4. Do: At the end of PT, or manually, scp the data in batch mode, roughly with
#    scp -BCvr explore_bv_jammed_packing500 "bazinga:/media/My\ Passport/LinuxPartition/n32_phi50_phi88_3D"
# 5. Check that the transfer was successful, i.e. that data at origin and destination is identical.
# 6. Do: Erase folder at origin.
#
# Output: data moving success?
# Input parameters:
# $1: folder to move, this should be something like explore_bv_jammed_packing500
# $2: destination for folder, this should be something like bazinga:/media/My\ Passport/LinuxPartition/n32_phi50_phi88_3D

# Step 0.
echo "attempting to move folder "$1
if [ -d "$1" ];
then
    echo "origin data folder exists"
else
    echo "origin data folder does not exist -- terminating"
    exit 42
fi

# Step 1.
# Assumption: job has finished properly if there is a file like bv_32_50_70_3D_pt4.o372227 in folder $1.
if [ -f $1/bv_*_pt*.o* ];
then
    echo "job terminated correctly"
else
    echo "job failed"
    exit 42
fi

# Step 2.
echo "destination is "$2
remote_computer=$(echo $2 | awk '{split($0,a,":"); print a[1]}')
echo "remote computer "$remote_computer
remote_folder=$(echo $2 | awk '{split($0,a,":"); print a[2]}')
echo "remote folder "$remote_folder
if (ssh $remote_computer '[ -d $remote_folder ]')
then
    echo "destination exists"
else 
    echo "destination does not exist"
    exit 42
fi

# Step 3.
data_size_=$(du -s $1)
data_size=$(echo $data_size_ | awk '{split($0,a," "); print a[1]}')
echo "data size "$data_size
free_space_=$(df $2)
free_space=$(echo $free_space_ | awk '{split($0,a," "); print a[11]}')
echo "free space "$free_space
# Assumption: multiply actual folder size by some safety factor larger 1
double_data_size=$((2 * data_size))
if [ "$free_space" -lt "$double_data_size" ];
then
    echo "not enough disk space at destination"
    exit 42
fi

# Step 4.
scp -BCvr "$1" "$2"

# Step 5.
differences=$(diff -rq "$1" "$2"/"$1")
if [ -z "$differences" ];
then
    echo "data transfer successful"
else
    echo "differences "$differences
    echo "data transfer failed"
    exit 42
fi

# Step 6.
# For now, this makes a tar.gz of the folder, leaves that in place, and erases the folder.
# To save more space we should leave ot the tar generation step, but then we have no backup.
tar -zcvf $1".tar.gz" $1
#rm -rf $1

exit 0
