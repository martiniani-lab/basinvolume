from __future__ import division
import shutil
import os
import re
import argparse

def get_immediate_files(dir):
    return [str(name) for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]

class BVPackingsConfigNewFormat(object):
    """
    This class is specific to the basin volume repository and is responsible of 
    copying the old format packings/jammed_packings.config to the new format
    packings/jammed_packing{number}.config
    
    *workdir is the directory containing all the explore_bv_* subdirectories
    """
    def __init__(self, workdir=None, 
                 old_packing_naming='packings', old_jammed_packing_naming='jammed_packings',
                 new_packing_naming='packing', new_jammed_packing_naming='jammed_packing',
                 packing_folder='packings', jammed_packing_folder='jammed_packings'):
        if not workdir:
            workdir = os.getcwd()
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
        assert os.path.isdir(workdir)
        self.workdir = workdir
        if not os.path.isabs(packing_folder):
            packing_folder = os.path.join(self.workdir, packing_folder)
        assert os.path.isdir(packing_folder)
        if not os.path.isabs(jammed_packing_folder):
            jammed_packing_folder = os.path.join(self.workdir, jammed_packing_folder)
        assert os.path.isdir(jammed_packing_folder)
        
        self.packing_folder = packing_folder
        self.jammed_packing_folder = jammed_packing_folder
        self.old_packing_naming = old_packing_naming
        self.old_jammed_packing_naming = old_jammed_packing_naming
        self.new_packing_naming = new_packing_naming
        self.new_jammed_packing_naming = new_jammed_packing_naming
    
    def copy_packing_config_files(self):
        files = get_immediate_files(os.path.join(self.workdir, self.packing_folder))
        src = os.path.join(self.workdir, self.packing_folder, self.old_packing_naming+".config")
        
        if os.path.isfile(src):
            for fn in files:
                if ".xy" in fn:
                    npack = re.findall('\d+', fn)[0]
                    dst = os.path.join(self.workdir, self.packing_folder, self.new_packing_naming + npack + ".config")
                    shutil.copyfile(src, dst)
            shutil.copyfile(src, os.path.join(self.workdir, self.packing_folder, self.old_packing_naming + ".bak"))
            os.remove(src)
            
    def copy_jammed_packing_config_files(self):
        files = get_immediate_files(os.path.join(self.workdir, self.jammed_packing_folder))
        src = os.path.join(self.workdir, self.jammed_packing_folder, self.old_jammed_packing_naming+".config")
        
        if os.path.isfile(src):
            for fn in files:
                if ".xy" in fn:
                    npack = re.findall('\d+', fn)[0]
                    dst = os.path.join(self.workdir, self.jammed_packing_folder, self.new_jammed_packing_naming + npack + ".config")
                    shutil.copyfile(src, dst)
            shutil.copyfile(src, os.path.join(self.workdir, self.jammed_packing_folder, self.old_jammed_packing_naming + ".bak"))
            os.remove(src)
            
    def copy_all(self):
        self.copy_packing_config_files()
        self.copy_jammed_packing_config_files()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="perform parallel tempering for basin volume method")
        
    parser.add_argument("workdir", type=str, help="working directory (folder containing the packings and jammed_packings subfolders)")        
    
    args = parser.parse_args()
    print args
    
    bvcp = BVPackingsConfigNewFormat(workdir=args.workdir)
    bvcp.copy_all()
