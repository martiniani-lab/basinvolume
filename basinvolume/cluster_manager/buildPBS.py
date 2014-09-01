from __future__ import division
from subprocess import Popen, PIPE
from pipes import quote
import os
import re
import ConfigParser
import numpy as np

def sec_to_pbs_time(seconds):
    """
    clean solution from
    https://stackoverflow.com/questions/21323692/convert-seconds-to-weeks-days-hours-minutes-seconds-in-python
    "The idea behind this is that the number of seconds in your answer is the remainder after dividing 
    them in minutes; minutes are the remainder of dividing all minutes into hours etc... This version 
    is better because you can easily adjust it to months, years, etc..."
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    weeks, days = divmod(days, 7)
    return "{:02d}:{:02d}:{:02d}:{:02d}".format(int(days),int(hours),int(minutes),int(seconds))
    

class BuildPBSScript(object):
    """
    *queue_type [string]
    *nodes [int]
    *core [int] processors per node
    *walltime [hours]
    *command [string]: command line to execute e.g. python parallel_tempering.py args
    """
    def __init__(self, queue_type, nodes, cores, walltime, command):
        self.qtype = queue_type
        self.nodes = nodes
        self.cores = cores
        self.s_wtime = walltime*60*60 #convert hours to seconds
        self.dhms_wtime = sec_to_pbs_time(self.s_wtime) #DD:HH:MM:SS time
        self.command = command
        self.pbs_ready = False
            
    def writePBSscript(self, fname, job_name):
        """
        *fname [string]: name of the pbs bash script where to write
        *job_name [string]: name of the pbs job
        """
        print "writing PBS file"
        if ".sh" not in fname:
            fname += ".sh"
        f = open(fname,'w')
        f.write('#PBS -N {0} \n'.format(job_name))
        f.write('#PBS -q {0} \n'.format(self.qtype))
        f.write('#PBS -l nodes={0}:ppn={1} \n'.format(self.nodes,self.cores))
        f.write('#PBS -l walltime={0} \n'.format(self.dhms_wtime))
        f.write('\n')
        f.write('cd \${PBS_O_WORKDIR} \n')
        f.write('\n')
        f.write('echo Starting job $PBS_JOBID \n')
        f.write('echo\n')
        f.write('echo PBS assigned me this node: \n')
        f.write('cat \$PBS_NODEFILE \n')
        f.write('echo \n')
        f.write('echo \"Running ${job_name}\" \n')
        f.write('echo \n')
        f.write('mpirun {0}\n'.format(self.command))
        f.write('echo \n')
        f.write('echo \"Job finished. PBS details are:\" \n')
        f.write('echo \n')
        f.write('qstat -f \${PBS_JOBID} \n')
        f.write('echo \n')
        f.write('echo Finished at \`date\` \n')
        f.close()
        self.pbs_ready = True
    
    def get_PBS_O_WORKDIR(self):
        (stdout, stderr) = Popen(["echo $PBS_O_WORKDIR"], shell=True, stdout=PIPE).communicate()
        stdout=stdout.rstrip()
        return stdout
    
    def get_PBS_JOBID(self):
        (stdout, stderr) = Popen(["echo $PBS_JOBID"], shell=True, stdout=PIPE).communicate()
        stdout=stdout.rstrip()
        return stdout
    
    def goto_PBS_O_WORKDIR(self):
        print "going to PBS_O_WORKDIR"
        pbs_wdir = self.get_PBS_O_WORKDIR()
        if not os.path.isabs(pbs_wdir):
            #this probably unnecessary, more of a safety check
            print "PBS_O_WORKDIR is not absolute, making absolute"
            pbs_wdir = os.path.abspath(pbs_wdir)
        os.chdir(pbs_wdir)
    
    def checkin_PBS_O_WORKDIR(self):
        pbs_wdir = self.get_PBS_O_WORKDIR()
        cwd = os.getcwd()
        return (pbs_wdir == cwd)
    
    def submit_PBS(self, fname, job_name):
        """
        submit pbs file, returns the std output
        *fname [string]: name of the pbs bash script where to write
        *job_name [string]: name of the pbs job
        """
        if ".sh" not in fname:
            fname += ".sh"
        if not self.checkin_PBS_O_WORKDIR():
            self.goto_PBS_O_WORKDIR()
        if not self.pbs_ready:
            self.writePBSscript(fname, job_name)
        (stdout, stderr) = Popen(["qsub {}".format(fname)], shell=True, stdout=PIPE).communicate()
        return stdout
    
    
class BVSubmitPBSscripts(object):
    """
    this class is responsible of looping through a particular directory 
    containing the jammed_packings subdir
    *workdir is the directory containing all the explore_bv_* subdirectories
    *job_label should help distinguish between different densities and packing numbers
    """
    def __init__(self, ndim, job_label='packing', workdir=None, explore_dir='explore_bv_jammed_packing', kmax_config='findk_jammed_packing', 
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
        self._check_config_file_exist(kmax_configpath)
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
        self._check_config_file_exist(kmin_configpath)
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
    
    def _get_findk_command(self, noj, path_to_script):
        """
        this function returns the correct command line
        """
        packing = self.packing_naming + noj + self.ext
        command = 'mpirun python {0}.py {1} -p \${{PBS_O_WORKDIR}}/jammed_packings'.format(path_to_script, packing)
        return command
    
    def launch_kmin_calculations(self, queue_type, nodes, cores, walltime, path_to_script):
        """
        launch kmin calculations manually if they have not been launched yet
        (this method only checks that the config file is not ready or present, 
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *pbs object of type BuildPBSScript
        """
        for root, dirs, files in os.walk(self.workdir):
            for dir in dirs:
                if self.explore_dir in dir:
                    noj = re.findall(r'\d+', dir)[0]   #extract packing number from explor_dir string
                    path = os.path.join(root,dir)      #build a full path
                    kmin_path = os.path.join(path, self.kmin_config + noj)
                    if not self._check_kmin_config_file_ready(kmin_path):
                        if not os.path.isabs(path_to_script):
                            path_to_script = os.path.abspath(path_to_script)
                        command = self._get_findk_command(noj, path_to_script)
                        pbs = BuildPBSScript(queue_type, nodes, cores, walltime, command)
                        pbs.submit_PBS('bv_kmin'+noj+'.sh', 'bv_'+self.label+'_kmin'+noj)
    
    def launch_kmax_calculations(self, queue_type, nodes, cores, walltime, path_to_script):
        """
        launch kmax calculations manually if they have not been launched yet
        (this method only checks that the config file is not ready or present, 
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *pbs object of type BuildPBSScript
        """
        for root, dirs, files in os.walk(self.workdir):
            for dir in dirs:
                if self.explore_dir in dir:
                    noj = re.findall(r'\d+', dir)[0]   #extract packing number from explor_dir string
                    path = os.path.join(root,dir)      #build a full path
                    kmax_path = os.path.join(path, self.kmax_config + noj)
                    if not self._check_kmax_config_file_ready(kmax_path):
                        if not os.path.isabs(path_to_script):
                            path_to_script = os.path.abspath(path_to_script)
                        command = self._get_findk_command(noj, path_to_script)
                        pbs = BuildPBSScript(queue_type, nodes, cores, walltime, command)
                        pbs.submit_PBS('bv_kmax'+noj+'.sh', 'bv_'+self.label+'_kmax'+noj)
    
    def _get_pt_command(self, noj, path_to_script):
        """
        this function returns the correct command line
        """
        packing = self.packing_naming + noj + self.ext
        explore_dir = self.explore_dir + noj
        command = 'mpirun python {0}.py {1} \${{PBS_O_WORKDIR}}/{2}'.format(path_to_script, packing, explore_dir)
    
    def launch_pt_calculations(self, queue_type, nodes, cores, walltime, path_to_script):
        """
        launch pt calculations manually if they have not been launched yet
        (this method only checks that the config files are not ready or present, 
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *pbs object of type BuildPBSScript
        """
        for root, dirs, files in os.walk(self.workdir):
            for dir in dirs:
                if self.explore_dir in dir:
                    noj = re.findall(r'\d+', dir)[0]   #extract packing number from explor_dir string
                    path = os.path.join(root,dir)      #build a full path
                    kmax_path = os.path.join(path, self.kmax_config + noj)
                    kmin_path = os.path.join(path, self.kmin_config + noj)
                    pt_path = os.path.join(path, self.pt_config + noj)
                    if not self._check_kmax_config_file_ready(kmax_path) \
                    and not self._check_kmin_config_file_ready(kmin_path) \
                    and not self._check_config_file_exist(pt_path):
                        if not os.path.isabs(path_to_script):
                            path_to_script = os.path.abspath(path_to_script)
                        command = self._get_pt_command(noj, path_to_script)
                        pbs = BuildPBSScript(queue_type, nodes, cores, walltime, command) 
                        pbs.submit_PBS('bv_pt'+noj+'.sh', 'bv_'+self.label+'_pt'+noj)
                    
    
                    
    
if __name__ == "__main__":
    pbs = BuildPBSScript('test', 1, 1, 0.5, 'python run_test.py args')
    #pbs.writePBSscript('test_job', 'test_job')
    print pbs.get_PBS_O_WORKDIR()
    print pbs.get_PBS_JOBID()
    print pbs.checkin_PBS_O_WORKDIR()
    print pbs.s_wtime
    print pbs.dhms_wtime
    stdout = pbs.submit_PBS('test_job','test_job')
    print stdout
    