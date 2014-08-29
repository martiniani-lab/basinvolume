from __future__ import division
from subprocess import Popen, PIPE
from pipes import quote
import os
import time

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
        self.s_wtime = walltime*86400.0 #convert hours to seconds
        self.dhms_wtime = time.strftime("%D:%H:%M:%S", time.gmtime(self.s_wtime))
        self.command = command
            
    def writePBSscript(self, fname, job_name):
        """
        *fname [string]: name of the pbs bash script where to write
        *job_name [string]: name of the pbs job
        """
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
    
    def get_PBS_O_WORKDIR(self):
        (stdout, stderr) = Popen(["echo $PBS_O_WORKDIR"], shell=True, stdout=PIPE).communicate()
        stdout.replace(" ","")
        stdout.replace("\n","")
        return stdout
    
    def get_PBS_JOBID(self):
        (stdout, stderr) = Popen(["echo $PBS_JOBID"], shell=True, stdout=PIPE).communicate()
        stdout.replace(" ","")
        stdout.replace("\n","")
        return stdout
    
    def checkin_PBS_O_WORKDIR(self):
        pbs_wdir = self.get_PBS_O_WORKDIR()
        return os.path.normpath(pbs_wdir) == os.path.normpath(os.getcwd())
    
class SubmitPBSscripts(object):
    """
    this class is responsible of looping through a particular directory 
    containing the jammed_packings subdir
    """
    
if __name__ == "__main__":
    pbs = BuildPBSScript('test', 1, 7, 6, 'python run_test.py args')
    
    pbs.writePBSscript('test_job', 'test_job')
    print pbs.get_PBS_O_WORKDIR()
    print pbs.get_PBS_JOBID()
    print pbs.checkin_PBS_O_WORKDIR()
        