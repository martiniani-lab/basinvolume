"""
script to copy initial conditions across minimizers
"""

import os
import shutil

# Assuming the directories are in the current working directory
base_directory = "/scratch/ps4586/volume_runs_multi_packing_new"

# Given directory list
directory_list = os.listdir(base_directory)

# minimizer list
name_list = ["MXD", "LBFGS", "FIRE", "CG"]
n_copies = 14

def create_copies(src_name, n, cwd):
    for i in range(n):
        dest_name = f"{src_name}_{i}"
        src = os.path.join(cwd, src_name)
        dest = os.path.join(cwd, dest_name)
        shutil.copytree(src, dest)


for directory in directory_list:
    # Split directory name by underscores to extract parts
    parts = directory.split("_")
    # Extract the numbers
    num1 = parts[1]
    num2 = parts[2]
    
    create_copies(directory, n_copies, base_directory)

    for name in name_list:
        # Create new directory name based on the format
        new_directory = f"{name}_{num1}_{num2}"

        # Full paths
        src = os.path.join(base_directory, directory)
        dest = os.path.join(base_directory, new_directory)

        # Copy the directory and its contents
        shutil.copytree(src, dest)
        
        create_copies(new_directory, n_copies, base_directory)
        
    
    
    