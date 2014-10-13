#!/bin/bash
# Moves output of BV computation from cluster to a defined remote location.
#
# Intended procedure:
# -1.To set up ssh with public key, use something like ssh-copy-id -i ~/.ssh/id_rsa.pub bazinga
# http://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/
# 0. Check: Folder to move exists.
# 1. Check: Computation terminated properly.
# 2. Check: Destination can be rached / is a valid path / dir exists.
# 3. Check: Enough disc space available at destination.
# 4. Do: At the end of PT, or manually, scp the data in batch mode, roughly with
#    scp -BCvr explore_bv_jammed_packing500 bazinga:/scratch/kjs73/link_to_disk/n32_phi50_phi88_3D
# 5. Check that the transfer was successful, i.e. that data at origin and destination is identical.
# 6. Do: Erase folder at origin.
#
# Output: data moving success?
# Input parameters:
# $1: folder to move, this should be something like explore_bv_jammed_packing500
# $2: destination for folder, this should be something like bazinga:/scratch/kjs73/link_to_disk/n32_phi50_phi88_3D

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
remote_folder="$remote_folder"
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

free_space_=$(ssh $remote_computer 'df $remote_folder')
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
# Reference for error collection: http://stackoverflow.com/questions/12738460/how-to-get-output-of-a-bash-command-in-a-variable
# "running diff via ssh --> use rsync": http://serverfault.com/questions/16661/how-can-i-diff-two-redhat-linux-servers/16665#16665
# http://serverfault.com/questions/63894/rsync-c-i-flags-identical-files-as-different
differences=$(rsync -rni --delete --checksum "$1""/" "$2"/"$1""/" 2>&1)
diff_exit_status=$?
echo "diff_exit_status "$diff_exit_status
if [ "$diff_exit_status" -eq 0 ];
then
    echo "differences exit status detected"
else
    echo "error in difference detection"
    exit 42
fi
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
tar -zcvf $1".tar.gz" $1 2>&1
tar_exit_status=$?
if [ "$tar_exit_status" -eq 0 ];
then
    rm -rf $1
else
    echo "compression failed"
    exit 42
fi

exit 0
