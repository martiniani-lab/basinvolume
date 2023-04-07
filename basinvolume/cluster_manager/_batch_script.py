from __future__ import division
from __future__ import print_function
from builtins import object
from subprocess import Popen, PIPE
import os
import logging
from enum import Enum, unique  # Package enum34


@unique
class BatchSystem(Enum):
    PBS = 1
    SLURM = 2


@unique
class MPI_Implementation(Enum):
    INTEL = 1
    OPENMPI = 2


def sec_to_time(seconds, nodays=False, batch_system=BatchSystem.PBS):
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
        if batch_system == BatchSystem.PBS:
            return "{:02d}:{:02d}:{:02d}:{:02d}".format(int(days),int(hours),int(minutes),int(seconds))
        elif batch_system == BatchSystem.SLURM:
            return "{:02d}-{:02d}:{:02d}:{:02d}".format(int(days),int(hours),int(minutes),int(seconds))
        else:
            raise ValueError("Batch system not implemented: {}".format(batch_system))


class BatchScript(object):
    """
    This is a completely general class, it does not and should not know anything about the naming conventions
    and structures of any particular library. If any such knowledge is necessary please make a derived class
    and overload the member functions
    *batch_system [BatchSystem]: PBS or SLURM
    *queue_or_partition [string]
    *walltime [hours]
    *command [string]: command line to execute e.g. python parallel_tempering.py args
    *mpi_procs [int]: Number of MPI processes
    *mpi_oversubscribe [int]: Number of MPI processes than don't require a core
    *omp_threads [int]: OpenMP threads per MPI process
    *cores_per_node [int]: Cores each cluster node has
    *noday [bool]: Don't use a separate day counter, use hours instead
    *outdir [string]: Directory for redirecting the standard output
    """
    def __init__(self, batch_system, queue_or_partition, walltime, command, mpi_procs=1,
                 mpi_oversubscribe=0, omp_threads=1, cores_per_node=16, memory_per_cpu=2,
                 nodays=False, outdir=None, mpi_impl=MPI_Implementation.OPENMPI):
        self.batch_system = batch_system
        if self.batch_system is BatchSystem.PBS:
            if queue_or_partition is None:
                self.queue = self._get_queue(walltime)
            else:
                self.queue = queue_or_partition
        elif self.batch_system is BatchSystem.SLURM:
            self.partition = queue_or_partition
        self.s_wtime = walltime*60*60  # convert hours to seconds
        self.dhms_wtime = sec_to_time(self.s_wtime, nodays=nodays,
                                      batch_system=self.batch_system)  # DD-/:HH:MM:SS time
        self.command = command
        self.ready = False
        if outdir and not os.path.isabs(outdir):
            outdir = os.path.abspath(outdir)
        self.mpi_procs = mpi_procs
        self.mpi_oversubscribe = mpi_oversubscribe
        self.omp_threads = omp_threads
        self.outdir = outdir
        self.nodes = 1
        while self.omp_threads*(self.mpi_procs-self.mpi_oversubscribe) - self.nodes*cores_per_node > 0:
           self.nodes += 1
        self.ncores = min(cores_per_node, self.omp_threads*(self.mpi_procs-self.mpi_oversubscribe))
        self.mpi_impl = mpi_impl
        self.memory_per_cpu = memory_per_cpu

    def _get_queue(self, walltime):
        if walltime <= 1:
            return 'test'
        elif walltime <= 24:
            return 'short'
        elif walltime <= 168:
            return 'long'
        elif walltime <= 672:
            return 'huge'
        else:
            raise ValueError("Walltime is too long for any queue.")

    def _get_mpi_flags(self):
        if self.mpi_impl is MPI_Implementation.OPENMPI:
            return '-n {} --map-by node'.format(self.mpi_procs)
        elif self.mpi_impl is MPI_Implementation.INTEL:
            return '-n {} -rr'.format(self.mpi_procs)
        else:
            raise ValueError("MPI implementation not implemented: {}".format(self.mpi_impl))

    def write(self, fname, job_name):
        """
        *fname [string]: name of the batch script to write
        *job_name [string]: name of the job
        """
        if self.batch_system == BatchSystem.PBS:
            self._write_pbs(fname, job_name)
        elif self.batch_system == BatchSystem.SLURM:
            self._write_slurm(fname, job_name)
        else:
            raise ValueError("Batch system not implemented: {}".format(self.batch_system))

    def _write_pbs(self, fname, job_name):
        logging.info("Writing PBS batch script")
        if ".sh" not in fname:
            fname += ".sh"
        f = open(fname,'w')
        f.write('#PBS -N {}\n'.format(job_name))
        f.write('#PBS -q {}\n'.format(self.queue))
        f.write('#PBS -l nodes={0}:ppn={1}\n'.format(self.nodes, self.ncores))
        f.write('#PBS -l walltime={}\n'.format(self.dhms_wtime))
        f.write('#PBS -l mem={}GB\n'.format(self.ncores*self.memory_per_cpu))
        f.write('#PBS -j oe\n') # this directive merges output and error in the same file
        if self.outdir:
            f.write('#PBS -o {}\n'.format(self.outdir))
        f.write('\n')
        f.write('cd ${PBS_O_WORKDIR}\n')
        f.write('\n')
        f.write('export OMP_NUM_THREADS={}\n'.format(self.omp_threads))
        f.write('\n')
        f.write('echo Starting job ${PBS_JOBID}\n')
        f.write('echo\n')
        f.write('echo PBS assigned me these nodes:\n')
        f.write('cat ${PBS_NODEFILE}\n')
        f.write('echo\n')
        f.write('echo \"Running ${PBS_JOBNAME}\"\n')
        f.write('echo\n')
        f.write('mpirun {} {}\n'.format(self._get_mpi_flags(), self.command))
        f.write('echo\n')
        f.write('echo \"Job finished. PBS details are:\"\n')
        f.write('echo\n')
        f.write('qstat -f ${PBS_JOBID}\n')
        f.write('echo\n')
        f.write('echo Finished at `date`\n')
        f.close()
        self.ready = True

    def _write_slurm(self, fname, job_name):
        logging.info("Writing SLURM batch script")
        if ".sh" not in fname:
            fname += ".sh"
        f = open(fname,'w')
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --job-name={}\n'.format(job_name))
        f.write('#SBATCH --nodes={}\n'.format(self.nodes))
        f.write('#SBATCH --cpus-per-task={}\n'.format(self.ncores))
        f.write('#SBATCH --partition={}\n'.format(self.partition))
        f.write('#SBATCH --mem={}GB\n'.format(self.memory_per_cpu*self.ncores))
        f.write('#SBATCH --time={}\n'.format(self.dhms_wtime))
        if self.outdir:
            f.write('#SBATCH --output={}_%j.out\n'.format(os.path.join(self.outdir, job_name)))
        else:
            f.write('#SBATCH --output={}_%j.out\n'.format(job_name))
        f.write('\n')
        f.write('cd ${SLURM_SUBMIT_DIR}\n')
        f.write('\n')
        f.write('export OMP_NUM_THREADS={}\n'.format(self.omp_threads))
        f.write('\n')
        f.write('echo Starting job ${SLURM_JOBID}\n')
        f.write('echo\n')
        f.write('echo SLURM assigned me these nodes:\n')
        f.write('squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2\n')
        f.write('echo\n')
        f.write('echo \"Running ${SLURM_JOB_NAME}\"\n')
        f.write('echo\n')
        f.write('mpirun {} {}\n'.format(self._get_mpi_flags(), self.command)) # add singularity stuff there?
        f.write('echo\n')
        f.write('echo \"Job finished. SLURM details are:\"\n')
        f.write('echo\n')
        f.write('scontrol show job ${SLURM_JOBID}\n')
        f.write('echo\n')
        f.write('echo Finished at `date`\n')
        f.close()
        self.ready = True

    def get_workdir(self):
        if self.batch_system == BatchSystem.PBS:
            workdir = os.getenv('PBS_O_WORKDIR')
        elif self.batch_system == BatchSystem.SLURM:
            workdir = os.getenv('SLURM_SUBMIT_DIR')
        else:
            raise ValueError("Batch system not implemented: {}".format(self.batch_system))
        if workdir is None:
            logging.info("Environment variable for working directory not found, "
                         "getting current directory.")
            workdir = os.getcwd()
        return workdir

    def get_jobid(self):
        if self.batch_system == BatchSystem.PBS:
            return os.environ['PBS_JOBID']
        elif self.batch_system == BatchSystem.SLURM:
            return os.environ['SLURM_JOBID']
        else:
            raise ValueError("Batch system not implemented: {}".format(self.batch_system))

    def goto_workdir(self):
        logging.info("Going to workdir")
        workdir = self.get_workdir()
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
            logging.info("workdir is not absolute, making absolute: {}"
                         .format(workdir))
        os.chdir(workdir)

    def check_in_workdir(self):
        workdir = self.get_workdir()
        cwd = os.getcwd()
        return (workdir == cwd)

    def submit(self, fname, job_name):
        """
        submit batch script, returns the std output
        *fname [string]: name of the batch script where to write
        *job_name [string]: name of the job
        """
        if ".sh" not in fname:
            fname += ".sh"
        if not self.check_in_workdir():
            self.goto_workdir()
        if not self.ready:
            self.write(fname, job_name)
        if self.batch_system == BatchSystem.PBS:
            (stdout, stderr) = Popen(["qsub {}".format(fname)], shell=True, stdout=PIPE).communicate()
        elif self.batch_system == BatchSystem.SLURM:
            (stdout, stderr) = Popen(["sbatch {}".format(fname)], shell=True, stdout=PIPE).communicate()
        else:
            raise ValueError("Batch system not implemented: {}".format(self.batch_system))
        return stdout

if __name__ == "__main__":
    batch_script = BatchScript('test', 1, 1, 0.5, 'python run_test.py args')
    #batch_script.write('test_job', 'test_job')
    print(batch_script.get_workdir())
    print(batch_script.get_jobid())
    print(batch_script.check_in_workdir())
    print(batch_script.s_wtime)
    print(batch_script.dhms_wtime)
    stdout = batch_script.submit('test_job','test_job')
    print(stdout)
