from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object
import subprocess
import shlex
import shutil
import os
import argparse

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
                 kmin_config='kmin_jammed_packing', pt_config='explore_jammed_packing', packing_naming='jammed_packing'):
        if not workdir:
            workdir = os.getcwd()
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
        self.workdir = workdir
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
    
    def remove_kmax_toxic_data(self, where='at', gitrepo_path=None):
        self._find_kmax_toxic_data(where=where, gitrepo_path=gitrepo_path)
        print("kmax toxic list", self.kmax_toxic_list)
        self._remove_toxic_data(self.kmax_toxic_list, self.kmax_config, output_signature="bv*kmax*.o*", pt=False)
    
    def remove_kmin_toxic_data(self, where='at', gitrepo_path=None):
        self._find_kmin_toxic_data(where=where, gitrepo_path=gitrepo_path)
        print("kmin toxic list", self.kmin_toxic_list)
        self._remove_toxic_data(self.kmin_toxic_list, self.kmin_config, output_signature="bv*kmin*.o*", pt=False)
    
    def remove_pt_toxic_data(self, where='at', gitrepo_path=None):
        self._find_pt_toxic_data(where=where, gitrepo_path=gitrepo_path)
        print("pt toxic list", self.pt_toxic_list)
        self._remove_toxic_data(self.pt_toxic_list, self.pt_config, output_signature="bv*pt*.o*", pt=True)
    
    def _find_kmax_toxic_data(self, where='at', gitrepo_path=None):
        """
        where: {'at', 'older', later'}
            older and later are strictly less and greater than a particular hash, respectively
            in other words later is "strictly more recent than"
        """
        if where == 'at':
            self.kmax_toxic_list = self._find_toxic_data(self.kmax_config)
        else:
            assert(gitrepo_path is not None)
            hash_table = self._get_hash_history(gitrepo_path)
            assert self.git_toxic_version in hash_table, "toxic_hash not in table, try to update log"
            if where == 'later':
                hash_list = [git_hash for git_hash,index in list(hash_table.items()) if index < hash_table[self.git_toxic_version]]
            elif where == 'older':
                hash_list = [git_hash for git_hash,index in list(hash_table.items()) if index > hash_table[self.git_toxic_version]]
            else:
                raise NotImplementedError('option \"{}\" not known')
            self.kmax_toxic_list = []
            for item in hash_list:
                self.kmax_toxic_list.extend(self._find_toxic_data(self.kmax_config, git_hash=item))
            
    def _find_kmin_toxic_data(self, where='at', gitrepo_path=None):
        """
        where: {'at', 'older', later'}
            older and later are strictly less and greater than a particular hash, respectively
            in other words later is "strictly more recent than"
        """
        if where == 'at':
            self.kmin_toxic_list = self._find_toxic_data(self.kmin_config)
        else:
            assert(gitrepo_path is not None)
            hash_table = self._get_hash_history(gitrepo_path)
            assert self.git_toxic_version in hash_table, "toxic_hash not in table, try to update log"
            if where == 'later':
                hash_list = [git_hash for git_hash,index in list(hash_table.items()) if index < hash_table[self.git_toxic_version]]
            elif where == 'older':
                hash_list = [git_hash for git_hash,index in list(hash_table.items()) if index > hash_table[self.git_toxic_version]]
            else:
                raise NotImplementedError('option \"{}\" not known')
            self.kmin_toxic_list = []
            for item in hash_list:
                self.kmin_toxic_list.extend(self._find_toxic_data(self.kmin_config, git_hash=item))
    
    def _find_pt_toxic_data(self, where='at', gitrepo_path=None):
        """
        where: {'at', 'older', later'}
            older and later are strictly less and greater than a particular hash, respectively
            in other words later is "strictly more recent than"
        """
        if where == 'at':
            self.pt_toxic_list = self._find_toxic_data(self.pt_config)
        else:
            assert(gitrepo_path is not None)
            hash_table = self._get_hash_history(gitrepo_path)
            print("building toxic hash list...", end=' ')
            assert self.git_toxic_version in hash_table, "toxic_hash not in table, try to update log"
            if where == 'later':
                hash_list = [git_hash for git_hash,index in list(hash_table.items()) if index < hash_table[self.git_toxic_version]]
            elif where == 'older':
                hash_list = [git_hash for git_hash,index in list(hash_table.items()) if index > hash_table[self.git_toxic_version]]
            else:
                raise NotImplementedError('option \"{}\" not known')
            print("DONE")
            self.pt_toxic_list = []
            for item in hash_list:
                self.pt_toxic_list.extend(self._find_toxic_data(self.pt_config, git_hash=item))
    
    def _find_toxic_data(self, config_file, git_hash=None):
        if git_hash is None:
            git_hash = self.git_toxic_version
        print("finding toxic data for git-hash {}...".format(git_hash), end=' ')
        command = shlex.split("grep -r -i -l --include {}\*.config \"{}\" .".format(config_file, git_hash))
        try:
            list = subprocess.check_output(command).rstrip('\n').split('\n')
            print("DONE")
            for i,path in enumerate(list):
                list[i] = os.path.basename(os.path.dirname(path))
        except Exception:
            list = []
        return list
    
    def _get_hash_history(self, gitrepo_path):
        print("retrieving hash history...", end=' ')
        command = shlex.split("git --git-dir {}/.git log --pretty=oneline".format(gitrepo_path))
        try:
            raw_list = subprocess.check_output(command).split()
            raw_list = [word for word in raw_list if len(word) == 40]
            hash_table = dict(list(zip(raw_list, list(range(len(raw_list))))))
        except Exception:
            raise RuntimeError("\n could not build hash history \n")
        print("DONE")
        return hash_table
    
    def _remove_toxic_data(self, toxic_list, config_fname, output_signature="bv\*kmax\*.o\*", pt=False):
        subdirs = get_immediate_subdirectories(self.workdir)
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
    parser.add_argument("--repopath", type=str, help="path to git repository", default=None)
    parser.add_argument("--where", type=str, help="range (all strictly): at (default), older, later", default='at')
    parser.add_argument("--kmin", action='store_true', help="compute kmin",default=False)
    parser.add_argument("--kmax", action='store_true', help="compute kmax",default=False)
    parser.add_argument("--pt", action='store_true', help="perform parallel tempering",default=False)
    parser.add_argument("--all", action='store_true', help="perform parallel tempering",default=False)        
    
    args = parser.parse_args()
    print(args)
        
    bvrm = BVRemoveToxicData(args.toxic_git_version, workdir=args.workdir)
       
    if args.kmin or args.all:
        bvrm.remove_kmin_toxic_data(where=args.where, gitrepo_path=args.repopath)
    if args.kmax or args.all:
        bvrm.remove_kmax_toxic_data(where=args.where, gitrepo_path=args.repopath)
    if args.pt or args.all:
        bvrm.remove_pt_toxic_data(where=args.where, gitrepo_path=args.repopath)
        