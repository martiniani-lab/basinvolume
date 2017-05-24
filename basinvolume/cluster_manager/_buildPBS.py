from __future__ import division
from subprocess import Popen, PIPE
import os
import logging

def sec_to_pbs_time(seconds, nodays=False):
    """
    clean solution from
    https://stackoverflow.com/questions/21323692/convert-seconds-to-weeks-days-hours-minutes-seconds-in-python
    "The idea behind this is that the number of seconds in your answer is the remainder after dividing
    them in minutes; minutes are the remainder of dividing all minutes into hours etc... This version
    is better because you can easily adjust it to months, years, etc..."
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if nodays:
        return "{:02d}:{:02d}:{:02d}".format(int(hours),int(minutes),int(seconds))
    else:
        days, hours = divmod(hours, 24)
        return "{:02d}:{:02d}:{:02d}:{:02d}".format(int(days),int(hours),int(minutes),int(seconds))

class BuildPBSScript(object):
    """
    This is a completely general class, it does not and should not know anything about the naming conventions
    and structures of any particular library. If any such knowledge is necessary please make a derived class
    and overload the member functions
    *queue_type [string]
    *nodes [int]
    *core [int] processors per node
    *walltime [hours]
    *command [string]: command line to execute e.g. python parallel_tempering.py args
    *outdir is the directory where to redirect the standard output
    """
    def __init__(self, queue_type, nodes, cores, walltime, command, mpi_procs=1,
                 omp_threads=1, nodays=False, outdir=None):
        self.qtype = queue_type
        self.nodes = nodes
        self.cores = cores
        self.s_wtime = walltime*60*60 #convert hours to seconds
        self.dhms_wtime = sec_to_pbs_time(self.s_wtime, nodays=nodays) #DD:HH:MM:SS time
        self.command = command
        self.pbs_ready = False
        if outdir and not os.path.isabs(outdir):
            outdir = os.path.abspath(outdir)
        self.mpi_procs = mpi_procs
        self.omp_threads = omp_threads
        self.outdir = outdir

    def writePBSscript(self, fname, job_name):
        """
        *fname [string]: name of the pbs bash script where to write
        *job_name [string]: name of the pbs job
        """
        logging.info("Writing PBS file")
        if ".sh" not in fname:
            fname += ".sh"
        f = open(fname,'w')
        f.write('#PBS -N {0} \n'.format(job_name))
        f.write('#PBS -q {0} \n'.format(self.qtype))
        f.write('#PBS -l nodes={0}:ppn={1} \n'.format(self.nodes, self.cores))
        f.write('#PBS -l walltime={0} \n'.format(self.dhms_wtime))
        f.write('#PBS -j oe \n') # this directive merges output and error in the same file
        if self.outdir:
            f.write('#PBS -o {0} \n'.format(self.outdir))
        f.write('\n')
        f.write('cd ${PBS_O_WORKDIR} \n')
        f.write('\n')
        f.write('export OMP_NUM_THREADS={}\n'.format(self.omp_threads))
        f.write('\n')
        f.write('echo Starting job $PBS_JOBID \n')
        f.write('echo\n')
        f.write('echo PBS assigned me this node: \n')
        f.write('cat $PBS_NODEFILE \n')
        f.write('echo \n')
        f.write('echo \"Running ${job_name}\" \n')
        f.write('echo \n')
        f.write('mpirun -n {0} --bynode {1}\n'.format(self.mpi_procs, self.command))
        f.write('echo \n')
        f.write('echo \"Job finished. PBS details are:\" \n')
        f.write('echo \n')
        f.write('qstat -f ${PBS_JOBID} \n')
        f.write('echo \n')
        f.write('echo Finished at `date` \n')
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
        logging.info("Going to PBS_O_WORKDIR")
        pbs_wdir = self.get_PBS_O_WORKDIR()
        if not os.path.isabs(pbs_wdir):
            # This makes the BVSubmitPBS still work when there is no PBS_O_WORKDIR,
            # i.e. when calling the script locally
            pbs_wdir = os.path.abspath(pbs_wdir)
            logging.info("PBS_O_WORKDIR is not absolute, making absolute: {}"
                         .format(pbs_wdir))
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
