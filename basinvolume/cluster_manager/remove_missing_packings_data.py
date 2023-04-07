from __future__ import division
from __future__ import print_function
from builtins import object
import shutil
import os
import re
import argparse
from basinvolume.utils import query_yes_no

def get_immediate_subdirectories_and_tar(dir, explore_dir):
    return [name for name in os.listdir(dir) if (os.path.isdir(os.path.join(dir, name)) or (".tar.gz" in name and explore_dir in name))]

class BVRemoveMissingPackingsData(object):
    """
    This class is specific to the basin volume repository and is responsible of 
    looping through a particular directory containing the explore_bv_jammed_packingsubdirectories
    and removing all the data that don't have a corresponding structure in the jammed_packings
    repository
    *workdir is the directory containing all the explore_bv_* subdirectories
    """
    def __init__(self, workdir=None, explore_dir='explore_bv_jammed_packing', packing_naming='jammed_packing',
                 ext="xyzdr", packing_folder='jammed_packings'):
        if not workdir:
            workdir = os.getcwd()
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
        assert os.path.isdir(workdir)
        self.workdir = workdir
        if not os.path.isabs(packing_folder):
            packing_folder = os.path.join(self.workdir, packing_folder)
        assert os.path.isdir(packing_folder)
        self.packing_folder = packing_folder
        self.explore_dir = explore_dir
        self.packing_naming = packing_naming
        self.ext = ext
    
    def remove_toxic_data(self):
        """
        remove folders and compressed folders
        """
        subdirs = get_immediate_subdirectories_and_tar(self.workdir, self.explore_dir)
        for folder in subdirs:
            if self.explore_dir in folder:
                npack = re.findall('\d+', folder)[0]
                if not os.path.isfile(os.path.join(self.packing_folder, self.packing_naming+npack+"."+self.ext)):
                    print("{} is toxic".format(folder))
                    if ".tar.gz" in folder:
                        os.remove(os.path.join(self.workdir, folder))
                    else:
                        toxic_dir_path = os.path.join(self.workdir, folder)
                        shutil.rmtree(toxic_dir_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="perform parallel tempering for basin volume method")
        
    parser.add_argument("workdir", type=str, help="working directory (folder containing the packings and jammed_packings subfolders)")
    parser.add_argument("ext", type=str, help="jammed packing extension, default xyzdr")        
    
    args = parser.parse_args()
    print(args)
    assert args.ext == "xyzdr" or args.ext == "xydr" or args.ext == "xyzdfr" or args.ext == "xydfr", "{} not a valid extension".format(args.ext)
    
    check = query_yes_no("Confirm that the right file extension is \"{}\" ".format(args.ext), default="no")
    if check:
        bvrm = BVRemoveMissingPackingsData(workdir=args.workdir, ext=args.ext)
        bvrm.remove_toxic_data()
    else:
        print("Check failed, change extension! and be careful!")
