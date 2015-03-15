#!/bin/bash
# Moves output of BV computation from cluster to a defined remote location.
#
# Intended procedure:
# -1.To set up ssh with public key, use something like ssh-copy-id -i ~/.ssh/id_rsa.pub bazinga
# http://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/
# 0. Check: Folder to move exists.
# 1. Check: Computation terminated properly.
# 1.1. Check: Output files indicate that the time series have converged.
# 2. Check: Destination can be rached / is a valid path / dir exists.
# 3. Check: Enough disc space available at destination.
# 3.1. Check: If there exists remote folder of the same name and that folder has the same contents as the local folder, then compress the local folder and terminate with status 0.
#             If that remote folder has different content than the local folder (Assumption: local folder contains successfully terminated run output),
#             then the remote folder is erased and then the local folder copied to the remote location.
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
    echo ".o file of PT run exists"
else
    echo ".o file of PT run does not exist -- terminating"
    exit 42
fi

# Step 1.1
# Checks, in addition to above step 1, that the output files have the success flags inicating that kmax, kmin, and PT have finished properly.
# Assumption: there is only one file of the type $1/explore_jammed_packing*.config 
pt_success_file=$(find $1/explore_jammed_packing*.config)
if [ "$pt_success_file" ];
then
    echo "pt_success_file exists"
else
    echo "pt_success_file does not exist -- terminating"
    exit 42
fi
rank_success_lines=$(grep "success_rank" $pt_success_file)
for l in $rank_success_lines
do
    if [ "$l" = "False" ];
    then
        echo "PT time series convergence failed -- terminating"
        exit 42
    fi
done

# Step 2.
echo "destination is "$2
remote_computer=$(echo $2 | awk '{split($0,a,":"); print a[1]}')
echo "remote computer "$remote_computer
remote_folder=$(echo $2 | awk '{split($0,a,":"); print a[2]}')
remote_folder="$remote_folder"
echo "remote folder "$remote_folder
ssh  -f $remote_computer mkdir -p $remote_folder

# Step 3.
data_size_=$(du -s $1)
data_size=$(echo $data_size_ | awk '{split($0,a," "); print a[1]}')
echo "data size "$data_size

#free_space_=$(ssh $remote_computer 'df $remote_folder')
#free_space=$(echo $free_space_ | awk '{split($0,a," "); print a[11]}')
#echo "free space "$free_space
# Assumption: multiply actual folder size by some safety factor larger 1
#double_data_size=$((2 * data_size))
#if [ "$free_space" -lt "$double_data_size" ];
#then
#    echo "not enough disk space at destination"
#    exit 42
#fi

# Step 3.1
# Check if remote folder with (probably corruped data) exists and leave it, or erase it.
# Check whether remote folder exists, otherwise goto next step.
remote_folder_expl="$remote_folder"/$1
echo "remote_folder_expl "$remote_folder_expl
erase_remote=0
x=`ssh  -f $remote_computer ls -l $remote_folder_expl` 2>&1
folder_find_status=$?
echo $folder_find_status
echo $x
#if (ssh $remote_computer '[ -d $remote_folder_expl ]') # This gave the wrong answer, but I don't see why.
if [ "$x" ];
then
    echo "remote expl folder of same name exists for "$1
    # If contents is identical, terminate whole script without data transfer.
    # If there is an error in comparing the contents or the contents is different,
    # erase the remote folder and proceed with data transfer from local to remote folder.
    differences_=$(rsync -rni --delete --checksum "$1""/" "$2"/"$1""/" 2>&1)
    diff_exit_status_=$?
    if [ "$diff_exit_status_" -eq 0 ];
    then
        if [ -z "$differences_" ];
        then
            echo "remote folder is identical to local folder for "$1
            echo "compressing local folder"
            tar -zcvf $1".tar.gz" $1 2>&1
            tar_exit_status=$?
            if [ "$tar_exit_status" -eq 0 ];
            then
                rm -rf $1
                exit 0
            else
                echo "compression failed"
                exit 42
            fi
        else
            echo "remote folder is not identical to local folder for "$1
            echo "differences "$differences_
            erase_remote=1
        fi
    else
        erase_remote=1
    fi
else
    echo "no remote expl folder of same name exists for "$1
fi
echo "erase_remote "$erase_remote
if [ "$erase_remote" -eq 1 ];
then
    echo "erasing remote folder "$1
    # http://unix.stackexchange.com/questions/17466/how-to-delete-a-file-on-remote-machine-via-ssh-by-using-a-shell-script
    echo "about to erase "$remote_folder_expl" on "$remote_computer
    # http://stackoverflow.com/questions/1885525/how-do-i-prompt-a-user-for-confirmation-in-bash-script
    # http://stackoverflow.com/questions/3231804/in-bash-how-to-add-are-you-sure-y-n-to-any-command-or-alias
    read -r -p "Are you sure? [y/N] " response
    echo    # (optional) move to a new line
    if [ "$response" = "y" ];
    then
        echo "erasing remote folder "$remote_folder_expl
        ssh $remote_computer rm -rf $remote_folder_expl
    else
        echo "erasing remote folder aborted -- check folder by hand "$1
        echo $remote_folder_expl
        echo "terminating"
        exit 42
    fi
else
    echo "do not erase remote folder for "$1
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
