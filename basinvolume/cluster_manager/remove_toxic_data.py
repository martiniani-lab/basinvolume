from __future__ import division
import subprocess
import shlex
import shutil
import os
from pipes import quote
import re
import ConfigParser
import numpy as np
import argparse
from basinvolume.cluster_manager import BuildPBSScript
from basinvolume.utils import trymakedir

def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

class BVRemoveToxicData(object):
    """
    This class is specific to the basin volume repository and is responsible of 
    looping through a particular directory containing the explore_bv_jammed_packingsubdirectories
    and removing all the data that have been produced with a specific git version of the code
    *workdir is the directory containing all the explore_bv_* subdirectories
    *job_label should help distinguish between different densities and packing numbers
    """
    def __init__(self, git_toxic_version, workdir=None, explore_dir='explore_bv_jammed_packing', kmax_config='findk_jammed_packing', 
                 kmin_config='kmin_jammed_packing', pt_config='explore_jammed_packing', packing_naming='jammed_packing', 
                 structures_dir='jammed_packings'):
        if not workdir:
            workdir = os.getcwd()
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
        self.workdir = workdir
        self.structures_dir = structures_dir
        self.explore_dir = explore_dir
        self.kmax_config = kmax_config 
        self.kmin_config = kmin_config
        self.pt_config = pt_config
        self.packing_naming = packing_naming
        self.git_toxic_version = git_toxic_version
        self.kmin_toxic_list = []
        self.kmax_toxic_list = []
        self.pt_toxic_list = []
        self.pt_output_files = ["exchanges","rem_permutations","temperatures"]
    
    def remove_kmax_toxic_data(self):
        self._find_kmax_toxic_data()
        self._remove_toxic_data(self.kmax_toxic_list, self.kmax_config, output_signature="bv*kmax*.o*", pt=False)
    
    def remove_kmin_toxic_data(self):
        self._find_kmin_toxic_data()
        self._remove_toxic_data(self.kmin_toxic_list, self.kmin_config, output_signature="bv*kmin*.o*", pt=False)
    
    def remove_pt_toxic_data(self):
        self._find_pt_toxic_data()
        self._remove_toxic_data(self.pt_toxic_list, self.pt_config, output_signature="bv*pt*.o*", pt=True)
    
    def _find_kmax_toxic_data(self):
        """
        """
        command = shlex.split("grep -r -i -l --include {}\*.config \"{}\" .".format(self.kmax_config, self.git_toxic_version))
        try:
            list = subprocess.check_output(command).rstrip('\n').split('\n')
            for i,path in enumerate(list):
                list[i] = os.path.basename(os.path.dirname(path))
                self.kmax_toxic_list = list
        except Exception,e:
            print "\n no kmax data are toxic, \n",e
            
    def _find_kmin_toxic_data(self):
        """
        """
        command = shlex.split("grep -r -i -l --include {}\*.config \"{}\" .".format(self.kmin_config, self.git_toxic_version))
        try:
            list = subprocess.check_output(command).rstrip('\n').split('\n')
            for i,path in enumerate(list):
                list[i] = os.path.basename(os.path.dirname(path))
            self.kmin_toxic_list = list
        except Exception,e:
            print "\n no kmin data are toxic, \n",e
    
    def _find_pt_toxic_data(self):
        """
        """
        command = shlex.split("grep -r -i -l --include {}\*.config \"{}\" .".format(self.pt_config, self.git_toxic_version))
        try:
            list = subprocess.check_output(command).rstrip('\n').split('\n')
            for i,path in enumerate(list):
                list[i] = os.path.basename(os.path.dirname(path))
            self.pt_toxic_list = list
        except Exception,e:
            print "\n no pt data are toxic, \n",e
    
    def _remove_toxic_data(self, toxic_list, config_fname, output_signature="bv\*kmax\*.o\*", pt=False):
        subdirs = get_immediate_subdirectories(self.workdir)
        assert(self.structures_dir in subdirs)
        for toxic_folder in toxic_list:
            if self.explore_dir in toxic_folder:
                toxic_dir_path = os.path.join(self.workdir, toxic_folder)
                if pt:
                    for root, dirs, files in os.walk(toxic_dir_path):
                        for dir in dirs:
                            if dir.isdigit():
                                shutil.rmtree(os.path.join(root, dir))
                        for file in files:
                            if file in self.pt_output_files:
                                os.remove(os.path.join(root, file))
                #remove config file and pbs output
                p = subprocess.call(shlex.split("find {} -maxdepth 1 -type f -name \"{}\" -exec rm -f '{{}}' \;".format(toxic_dir_path, output_signature)))
                if p != 0:
                    raise Exception("removing pbs output file failed")
                p = subprocess.call(shlex.split("find {} -maxdepth 1 -type f -name \"{}\" -exec rm -f '{{}}' \;".format(toxic_dir_path, config_fname+"*.config")))
                if p != 0:
                    raise Exception("removing pbs output file failed")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="perform parallel tempering for basin volume method")
        
    parser.add_argument("workdir", type=str, help="working directory (folder containing the packings and jammed_packings subfolders)")
    parser.add_argument("toxic_git_version", type=str, help="toxic git version")
    parser.add_argument("--kmin", action='store_true', help="compute kmin",default=False)
    parser.add_argument("--kmax", action='store_true', help="compute kmax",default=False)
    parser.add_argument("--pt", action='store_true', help="perform parallel tempering",default=False)
    parser.add_argument("--all", action='store_true', help="perform parallel tempering",default=False)        
    
    args = parser.parse_args()
    print args
        
    bvrm = BVRemoveToxicData(args.toxic_git_version, workdir=args.workdir)
       
    if args.kmin or args.all:
        bvrm.remove_kmin_toxic_data()
    if args.kmax or args.all:
        bvrm.remove_kmax_toxic_data()
    if args.pt or args.all:
        bvrm.remove_pt_toxic_data()
        