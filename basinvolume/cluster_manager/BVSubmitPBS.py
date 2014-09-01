from __future__ import division
from subprocess import Popen, PIPE
from pipes import quote
import os
import re
import ConfigParser
import numpy as np
import argparse
from basinvolume.cluster_manager import BuildPBSScript

class BVSubmitPBS(object):
    """
    This class is specific to the basin volume repository and is responsible of 
    looping through a particular directory containing the jammed_packings 
    and explore_bv_jammed_packingsubdirectories
    *workdir is the directory containing all the explore_bv_* subdirectories
    *job_label should help distinguish between different densities and packing numbers
    """
    def __init__(self, ndim, workdir=None, job_label='32_70_88_2D', explore_dir='explore_bv_jammed_packing', kmax_config='findk_jammed_packing', 
                 kmin_config='kmin_jammed_packing', pt_config='explore_jammed_packing', packing_naming='jammed_packing'):
        if not workdir:
            workdir = os.getcwd()
        self.workdir = workdir
        self.explore_dir = explore_dir
        self.kmax_config = kmax_config 
        self.kmin_config = kmin_config
        self.pt_config = pt_config
        self.packing_naming = packing_naming
        self.label = job_label
        if ndim == 2:
            self.ext = '.xydr'
        else:
            self.ext = '.xyzdr'
    
    def _check_kmax_config_file_ready(self, kmax_configpath):
        """
        checks whether config files are ready (hence the necessary calculations have already been launched or have terminated), 
        returns false if they are not
        """
        if not self._check_config_file_exist(kmax_configpath):
            return False
        configf = ConfigParser.ConfigParser()
        try:
            configf.read(str(kmax_configpath))
            kmax = configf.getfloat('FINDK','kmax')
            prob_kmax = configf.getfloat('FINDK','prob')
            displ_k_max = configf.getfloat('FINDK','displ_k_max')
            var_displ_k_max = configf.getfloat('FINDK','var_displ_k_max')
        except:
            return False
        return True
    
    def _check_kmin_config_file_ready(self, kmin_configpath):
        """
        checks whether config files are ready (hence the necessary calculations have already been launched or have terminated), 
        returns false if they are not
        """
        if not self._check_config_file_exist(kmin_configpath):
            return False
        configf = ConfigParser.ConfigParser()
        try:
            configf.read(str(kmin_configpath))
            displ_k_min = configf.getfloat('KMIN','displ_k_min')
            var_displ_k_min = configf.getfloat('KMIN','var_displ_k_min')
        except:
            return False
        return True
    
    def _check_config_file_exist(self, configpath):
        return os.path.isfile(configpath)
    
    def _get_findk_command(self, noj, path_to_script, script='bv_find_kmin.py'):
        """
        this function returns the correct command line
        """
        packing = self.packing_naming + noj + self.ext
        findk_script = os.path.join(path_to_script, script)
        command = 'python {0} {1} -p \${{PBS_O_WORKDIR}}/jammed_packings'.format(findk_script, packing)
        return command
    
    def submit_kmin_calculations(self, queue_type, nodes, cores, walltime, path_to_script):
        """
        launch kmin calculations manually if they have not been launched yet
        (this method only checks that the config file is not ready or present, 
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *pbs object of type BuildPBSScript
        *path_to_script, excluding the filename with .py extension
        """
        for root, dirs, files in os.walk(self.workdir):
            for dir in dirs:
                if self.explore_dir in dir:
                    noj = re.findall(r'\d+', dir)[0]   #extract packing number from explor_dir string
                    path = os.path.join(root,dir)      #build a full path
                    kmin_path = os.path.join(path, self.kmin_config + noj + '.config')
                    if not self._check_kmin_config_file_ready(kmin_path):
                        if not os.path.isabs(path_to_script):
                            path_to_script = os.path.abspath(path_to_script)
                        command = self._get_findk_command(noj, path_to_script, script='bv_find_kmin.py')
                        pbs = BuildPBSScript(queue_type, nodes, cores, walltime, command)
                        pbs.submit_PBS('bv_kmin'+noj+'.sh', 'bv_'+self.label+'_kmin'+noj)
                    else:
                        pass
                    
    def submit_kmax_calculations(self, queue_type, nodes, cores, walltime, path_to_script):
        """
        launch kmax calculations manually if they have not been launched yet
        (this method only checks that the config file is not ready or present, 
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *pbs object of type BuildPBSScript
        *path_to_script, excluding the filename with .py extension
        """
        for root, dirs, files in os.walk(self.workdir):
            for dir in dirs:
                if self.explore_dir in dir:
                    noj = re.findall(r'\d+', dir)[0]   #extract packing number from explor_dir string
                    path = os.path.join(root,dir)      #build a full path
                    kmax_path = os.path.join(path, self.kmax_config + noj + '.config')
                    if not self._check_kmax_config_file_ready(kmax_path):
                        if not os.path.isabs(path_to_script):
                            path_to_script = os.path.abspath(path_to_script)
                        command = self._get_findk_command(noj, path_to_script, script='bv_find_kmax.py')
                        pbs = BuildPBSScript(queue_type, nodes, cores, walltime, command)
                        pbs.submit_PBS('bv_kmax'+noj+'.sh', 'bv_'+self.label+'_kmax'+noj)
                    else:
                        pass
    
    def _get_pt_command(self, noj, path_to_script, script='bv_parallel_tempering.py'):
        """
        this function returns the correct command line for the parallel tempering calculation
        """
        packing = self.packing_naming + noj + self.ext
        explore_dir = self.explore_dir + noj
        pt_script = os.path.join(path_to_script, script)
        command = 'python {0} {1} \${{PBS_O_WORKDIR}}/{2}'.format(pt_script, packing, explore_dir)
        return command
    
    def submit_pt_calculations(self, queue_type, nodes, cores, walltime, path_to_script):
        """
        launch pt calculations manually if they have not been launched yet
        (this method only checks that the config files are not ready or present, 
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *pbs object of type BuildPBSScript
        *path_to_script, excluding the filename with .py extension
        """
        for root, dirs, files in os.walk(self.workdir):
            for dir in dirs:
                if self.explore_dir in dir:
                    noj = re.findall(r'\d+', dir)[0]   #extract packing number from explor_dir string
                    path = os.path.join(root,dir)      #build a full path
                    kmax_path = os.path.join(path, self.kmax_config + noj + '.config')
                    kmin_path = os.path.join(path, self.kmin_config + noj + '.config')
                    pt_path = os.path.join(path, self.pt_config + noj + '.config')
                    if self._check_kmax_config_file_ready(kmax_path) \
                    and self._check_kmin_config_file_ready(kmin_path) \
                    and not self._check_config_file_exist(pt_path):
                        if not os.path.isabs(path_to_script):
                            path_to_script = os.path.abspath(path_to_script)
                        command = self._get_pt_command(noj, path_to_script)
                        pbs = BuildPBSScript(queue_type, nodes, cores, walltime, command) 
                        pbs.submit_PBS('bv_pt'+noj+'.sh', 'bv_'+self.label+'_pt'+noj)
                    else:
                        pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="perform parallel tempering for basin volume method")
    parser.add_argument("ndim", type=int, help="dimensionality")
    parser.add_argument("workdir", type=str, help="working directory (folder containing the packings and jammed_packings subfolders)")
    parser.add_argument("job_label", type=str, help="suggested: Nn_Pp_Pp_nD: 32_70_80_2D")
    parser.add_argument("queue_type", type=str, help="queue type")
    parser.add_argument("nodes", type=int, help="number of nodes to use")
    parser.add_argument("cores", type=int, help="number of processors per node to use")
    parser.add_argument("walltime_hours", type=int, help="wall-time in hours")
    parser.add_argument("path_to_script", type=str, help="path to the file to execute")
    parser.add_argument("--kmin", action='store_true', help="compute kmin",default=False)
    parser.add_argument("--kmax", action='store_true', help="compute kmax",default=False)
    parser.add_argument("--pt", action='store_true', help="perform parallel tempering",default=False)        
    args = parser.parse_args()
    
    assert(not ((args.kmin is True or args.kmax is True) and args.pt is True))
    
    bvpbs = BVSubmitPBS(args.ndim, workdir=args.workdir, job_label=args.job_label)
    if args.kmin:
        bvpbs.submit_kmin_calculations(args.queue_type, args.nodes, args.cores, args.walltime_hours, args.path_to_script)
    if args.kmax:
        bvpbs.submit_kmax_calculations(args.queue_type, args.nodes, args.cores, args.walltime_hours, args.path_to_script)
    if args.pt:
        bvpbs.submit_pt_calculations(args.queue_type, args.nodes, args.cores, args.walltime_hours, args.path_to_script)
        