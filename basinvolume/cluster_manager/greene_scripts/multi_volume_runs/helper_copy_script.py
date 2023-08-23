"""
script to copy initial conditions across minimizers
"""

import os
import shutil

# Assuming the directories are in the current working directory
base_directory = "/scratch/ps4586/volume_runs_multi_packing"

# Given directory list
directory_list = os.listdir(base_directory)

# minimizer list
name_list = ["MXD", "LBFGS", "FIRE", "CG"]

for directory in directory_list:
    # Split directory name by underscores to extract parts
    parts = directory.split("_")
    # Extract the numbers
    num1 = parts[1]
    num2 = parts[2]

    for name in name_list:
        # Create new directory name based on the format
        new_directory = f"{name}_{num1}_{num2}"

        # Full paths
        src = os.path.join(base_directory, directory)
        dest = os.path.join(base_directory, new_directory)

        # Copy the directory and its contents
        shutil.copytree(src, dest)