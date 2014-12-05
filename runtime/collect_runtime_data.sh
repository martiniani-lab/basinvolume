#!/bin/bash
# Collects runtime data to predict runtimes for future runs.
# Runtime prediction is done by e.g. runtime_dexter.py.
#
# Parameters.
remote_computer=nemesis.ch.private.cam.ac.uk
remote_folder=/media/exhhd/new_data/
#
#nemesis.ch.private.cam.ac.uk:/media/exhhd/new_data/n24_phi50_phi70_3D
echo "This is collect_runtime_data.sh."
echo "List of input parameters:"
echo "remote_computer: "$remote_computer
echo "remote_folder: "$remote_folder
echo "Finding data folders."
if (ssh $remote_computer '[ -d $remote_folder ]')
then
    echo "Origin exists."
else 
    echo "Origin does not exist."
    exit 42
fi
