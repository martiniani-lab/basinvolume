from __future__ import division
import os
import re
import ConfigParser
import numpy as np
import argparse
import shutil
import shlex
import subprocess
import logging
from basinvolume.enums import Interaction
from basinvolume.cluster_manager import BatchScript, BatchSystem
from basinvolume.utils import trymakedir, check_kmax_reasonable
from basinvolume.enums import Minimizer
from basinvolume.spheres import (HS_Generate_Jammed_Packing,
                                 read_jammed_packing_config, ExchangeScheme)


def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


class SubmitBV(object):
    """
    This class is specific to the basin volume repository and is responsible of
    looping through a particular directory containing the jammed_packings
    and explore_bv_jammed_packingsubdirectories
    *workdir is the directory containing all the explore_bv_* subdirectories
    *job_label should help distinguish between different densities and packing numbers
    *nojmin number of minimum job ID to submit (to selectively submit a range of jobs)
    *nojmax number of maximum job ID to submit (to selectively submit a range of jobs)
    *nodays if true use walltime HH:MM:SS format (necessary for some clusters)
    *numnegk is the number of negative ks to use during PT
    """
    def __init__(self, ndim, batch_system=BatchSystem.PBS, workdir=None,
                 job_label='32_70_88_2D',
                 explore_dir='explore_bv_jammed_packing',
                 kmax_config='findk_jammed_packing',
                 kmin_config='kmin_jammed_packing',
                 innersphere_dos_config='innersphere_jammed_packing',
                 pt_config='explore_jammed_packing',
                 packing_naming='jammed_packing',
                 structures_dir='jammed_packings', nojmin=0, nojmax=1e6,
                 nodays=False, experimental=False, minimizer=Minimizer.FIRE,
                 record_steps_timeseries=False, kmax_start=500, mintotniter=5e5,
                 maxtotniter=2e6, relstderr=0.05, numnegk=0, lownegk=-2.5,
                 nocell=False, delraw=False,
                 cores_per_node=16, pt_workers=4, pt_runners=16,
                 pt_exchange_scheme=ExchangeScheme.NEIGHBOR_EXCHANGE,
                 pt_sleep_seconds=0.0001, pt_nocollectminima=False, nthreads=1):
        if not workdir:
            workdir = os.getcwd()
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
        self.batch_system = batch_system
        self.workdir = workdir
        self.explore_dir = explore_dir
        self.kmax_config = kmax_config
        self.kmin_config = kmin_config
        self.innersphere_dos_config = innersphere_dos_config
        self.pt_config = pt_config
        self.packing_naming = packing_naming
        self.label = job_label
        self.structures_dir = structures_dir
        self.nojmin = nojmin
        self.nojmax = nojmax
        self.nodays = nodays
        self.experimental = experimental
        self.minimizer = minimizer
        self.record_steps_timeseries = record_steps_timeseries
        self.mintotniter = int(mintotniter)
        self.maxtotniter = int(maxtotniter)
        self.relstderr = relstderr
        self.lownegk = lownegk
        self.numnegk = numnegk
        self.kmax_start = kmax_start
        self.nocell = nocell
        self.delraw = delraw
        self.cores_per_node = cores_per_node
        self.pt_output_files = ["exchanges", "rem_permutations", "temperatures"]
        self.pt_workers = pt_workers
        self.pt_runners = pt_runners
        self.pt_exchange_scheme = pt_exchange_scheme
        self.pt_sleep_seconds = pt_sleep_seconds
        self.pt_nocollectminima = pt_nocollectminima
        self.nthreads = nthreads
        if ndim == 2:
            if not self.experimental:
                self.ext = '.xydr'
            else:
                self.ext = '.xydfr'
        else:
            if not self.experimental:
                self.ext = '.xyzdr'
            else:
                self.ext = '.xyzdfr'
        if self.batch_system == BatchSystem.PBS:
            self.submit_cmd = 'qsub'
            self.workdir_var = 'PBS_O_WORKDIR'
        elif self.batch_system == BatchSystem.SLURM:
            self.submit_cmd = 'sbatch'
            self.workdir_var = 'SLURM_SUBMIT_DIR'
        else:
            raise ValueError("Batch system not implemented: {}".format(self.batch_system))

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
            kmax = configf.getfloat('FINDK', 'kmax')
            prob_kmax = configf.getfloat('FINDK', 'prob')
            displ_k_max = configf.getfloat('FINDK', 'displ_k_max')
            var_displ_k_max = configf.getfloat('FINDK', 'var_displ_k_max')
            success = configf.getboolean('STATUS', 'success')
        except:
            return False
        return success

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
            displ_k_min = configf.getfloat('KMIN', 'displ_k_min')
            var_displ_k_min = configf.getfloat('KMIN', 'var_displ_k_min')
            success = configf.getboolean('STATUS', 'success')
        except:
            return False
        return success

    def _check_pt_config_file_ready(self, pt_configpath):
        """
        checks whether config files are ready (hence the necessary calculations have already been launched or have terminated),
        returns false if they are not
        """
        if not self._check_config_file_exist(pt_configpath):
            return False
        configf = ConfigParser.ConfigParser()
        try:
            configf.read(str(pt_configpath))
            success_dict = dict(configf.items('STATUS'))
        except:
            return False
        for key, value in success_dict.iteritems():
            if not (value == "True"):
                return False
        return True

    def _check_innersphere_dos_config_file_ready(self, innersphere_dos_configpath):
        """
        checks whether config files are ready (hence the necessary calculations have already been launched or have terminated),
        returns false if they are not
        """
        if not self._check_config_file_exist(innersphere_dos_configpath):
            return False
        configf = ConfigParser.ConfigParser()
        try:
            configf.read(str(innersphere_dos_configpath))
            success = configf.getboolean('STATUS', 'success')
        except:
            return False
        return success

    def _check_config_file_exist(self, configpath):
        return os.path.isfile(configpath)

    def _remove_bv_output(self, explore_dir_path, output_signature):
        p = subprocess.call(
            shlex.split("find {} -maxdepth 1 -type f -name \"{}\" -exec rm -vf '{{}}' \;"
                        .format(explore_dir_path, output_signature)))
        if p != 0:
            raise Exception("removing BV output file failed")

    def _remove_innersphere_dos_old_data(self, explore_dir_path, config_fname,
                                         output_signature="bv*innersphere_dos*.o*",
                                         rmdata="inner_sphere.timeseries"):
        try:
            os.remove(os.path.join(explore_dir_path, rmdata))
            logging.info("Removed {}".format(os.path.join(explore_dir_path, rmdata)))
        except OSError:
            pass  # nothing to remove
        # remove config file and bv output
        self._remove_bv_output(explore_dir_path, output_signature)
        self._remove_bv_output(explore_dir_path, config_fname + "*.config")

    def _remove_pt_old_data(self, explore_dir_path, config_fname, output_signature="bv*pt*.o*", pt=False):
        for root, dirs, files in os.walk(explore_dir_path):
            for dir in dirs:
                if dir.isdigit():
                    logging.info("Removing %s" % os.path.join(root, dir))
                    shutil.rmtree(os.path.join(root, dir))
            for file in files:
                if file in self.pt_output_files:
                    logging.info("Removing %s" % os.path.join(root, file))
                    os.remove(os.path.join(root, file))
        # remove config file and bv output
        self._remove_bv_output(explore_dir_path, output_signature)
        self._remove_bv_output(explore_dir_path, config_fname + "*.config")

    def _get_findk_command(self, noj, path_to_script, script='bv_find_kmin.py', record_steps_timeseries=False):
        """
        this function returns the correct command line
        """
        packing = self.packing_naming + noj + self.ext
        findk_script = os.path.join(path_to_script, script)
        command = 'python {0} {1}'.format(findk_script, packing)
        command += (" -p ${{{0}}}/{1}".format(self.workdir_var, self.structures_dir))

        command += " --explore-dir {}".format(self.explore_dir)
        command += " --minimizer {}".format(self.minimizer.name)
        if record_steps_timeseries:
            command += " --rsts"
        if script == 'bv_find_kmax.py':
            command += " --kstart {}".format(self.kmax_start)
        if self.nocell:
            command += " --nocell"
        return command

    def _get_innersphere_dos_command(self, noj, path_to_script, script='bv_innersphere_dos.py'):
        """
        this function returns the correct command line.

        this methods makes the assumption that path_to_script points to the spheres folder
        so it goes up one folder from path_to_script and replaces 'spheres' with  'mbar_spheres'
        """
        packing = self.packing_naming + noj + self.ext
        innersphere_dos_script = os.path.join(os.path.dirname(os.path.dirname(path_to_script)), 'mbar_spheres', script)
        command = 'python {0} {1}'.format(innersphere_dos_script, packing)
        command += (" -p ${{{0}}}/{1}".format(self.workdir_var, self.structures_dir))
        command += " --explore-dir {}".format(self.explore_dir)
        command += " --minimizer {}".format(self.minimizer.name)
        if self.nocell:
            command += " --nocell"
        return command

    def submit_kmin_calculations(self, queue_type, walltime, path_to_script, force):
        """
        launch kmin calculations manually if they have not been launched yet
        (this method only checks that the config file is not ready or present,
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *batch_script object of type BatchScript
        *path_to_script, excluding the filename with .py extension
        """
        subdirs = get_immediate_subdirectories(self.workdir)
        assert (self.structures_dir in subdirs)
        structures_dir_path = os.path.join(self.workdir, self.structures_dir)
        for root, dirs, files in os.walk(structures_dir_path):
            for file in files:
                if self.ext in file:
                    noj = re.findall(r'\d+', file)[0]  # extract packing number
                    if self.nojmin <= int(noj) <= self.nojmax:
                        explore_dir = self.explore_dir + noj  # build explore_dir name
                        if not os.path.isfile(
                                os.path.join(self.workdir, explore_dir + ".tar.gz")):  # check if there's a tar version
                            path = os.path.join(self.workdir, explore_dir)  # build a full path for explore dir
                            if (explore_dir) not in subdirs:  # check is explore_dir is a subfolder of self.workdir
                                trymakedir(path)
                            kmin_path = os.path.join(path, self.kmin_config + noj + '.config')
                            if not self._check_kmin_config_file_ready(kmin_path) or force:
                                #########remove old BV output#######
                                self._remove_bv_output(explore_dir, "bv_{}_kmin{}.o*".format(self.label, noj))
                                #####################################
                                if not os.path.isabs(path_to_script):
                                    path_to_script = os.path.abspath(path_to_script)
                                command = self._get_findk_command(noj, path_to_script, script='bv_find_kmin.py',
                                                                  record_steps_timeseries=self.record_steps_timeseries)
                                batch_script = BatchScript(
                                    self.batch_system, queue_type, walltime, command,
                                    mpi_procs=1, mpi_oversubscribe=0,
                                    omp_threads=self.nthreads,
                                    cores_per_node=self.cores_per_node,
                                    outdir=path, nodays=self.nodays)
                                batch_script.submit('bv_kmin' + noj + '.sh',
                                                    'bv_' + self.label + '_kmin' + noj)
                            else:
                                pass

    def submit_kmax_calculations(self, queue_type, walltime, path_to_script, force):
        """
        launch kmax calculations manually if they have not been launched yet
        (this method only checks that the config file is not ready or present,
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *batch_script object of type BatchScript
        *path_to_script, excluding the filename with .py extension
        """
        subdirs = get_immediate_subdirectories(self.workdir)
        assert (self.structures_dir in subdirs)
        structures_dir_path = os.path.join(self.workdir, self.structures_dir)
        for root, dirs, files in os.walk(structures_dir_path):
            for file in files:
                if self.ext in file:
                    noj = re.findall(r'\d+', file)[0]  # extract packing number
                    if self.nojmin <= int(noj) <= self.nojmax:
                        explore_dir = self.explore_dir + noj  # build explore_dir name
                        if not os.path.isfile(
                                os.path.join(self.workdir, explore_dir + ".tar.gz")):  # check if there's a tar version
                            path = os.path.join(self.workdir, explore_dir)  # build a full path for explore dir
                            if (explore_dir) not in subdirs:  # check is explore_dir is a subfolder of self.workdir
                                trymakedir(path)
                            kmax_path = os.path.join(path, self.kmax_config + noj + '.config')
                            if not self._check_kmax_config_file_ready(kmax_path) or force:
                                #########remove old BV output#######
                                self._remove_bv_output(explore_dir, "bv_{}_kmax{}.o*".format(self.label, noj))
                                #####################################
                                if not os.path.isabs(path_to_script):
                                    path_to_script = os.path.abspath(path_to_script)
                                command = self._get_findk_command(noj, path_to_script, script='bv_find_kmax.py')
                                batch_script = BatchScript(
                                    self.batch_system, queue_type, walltime, command,
                                    mpi_procs=1, mpi_oversubscribe=0,
                                    omp_threads=self.nthreads,
                                    cores_per_node=self.cores_per_node,
                                    outdir=path, nodays=self.nodays)
                                batch_script.submit('bv_kmax' + noj + '.sh',
                                                    'bv_' + self.label + '_kmax' + noj)
                            else:
                                pass

    def submit_innersphere_dos_calculations(self, queue_type, walltime, path_to_script, force):
        """
        launch innersphere_dos bv_innersphere_dos calculations manually if they have not been launched yet
        (this method only checks that the config file is not ready or present,
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *batch_script object of type BatchScript
        *path_to_script, excluding the filename with .py extension
        """
        subdirs = get_immediate_subdirectories(self.workdir)
        assert (self.structures_dir in subdirs)
        structures_dir_path = os.path.join(self.workdir, self.structures_dir)
        for root, dirs, files in os.walk(structures_dir_path):
            for file in files:
                if self.ext in file:
                    noj = re.findall(r'\d+', file)[0]  # extract packing number
                    if self.nojmin <= int(noj) <= self.nojmax:
                        explore_dir = self.explore_dir + noj  # build explore_dir name
                        if not os.path.isfile(
                                os.path.join(self.workdir, explore_dir + ".tar.gz")):  # check if there's a tar version
                            path = os.path.join(self.workdir, explore_dir)  # build a full path for explore dir
                            if (explore_dir) not in subdirs:  # check is explore_dir is a subfolder of self.workdir
                                trymakedir(path)
                            innersphere_dos_path = os.path.join(path, self.innersphere_dos_config + noj + '.config')
                            kmax_path = os.path.join(path, self.kmax_config + noj + '.config')
                            pt_path = os.path.join(path, self.pt_config + noj + '.config')
                            if (self._check_pt_config_file_ready(pt_path) \
                                        and self._check_kmax_config_file_ready(kmax_path)) \
                                    and (not self._check_innersphere_dos_config_file_ready(
                                        innersphere_dos_path) or force):
                                #########remove old innersphere data#######
                                self._remove_innersphere_dos_old_data(explore_dir, self.innersphere_dos_config + noj,
                                                                      output_signature="bv_{}_innersphere_dos{}.o*".format(
                                                                          self.label, noj))
                                #####################################
                                if not os.path.isabs(path_to_script):
                                    path_to_script = os.path.abspath(path_to_script)
                                command = self._get_innersphere_dos_command(noj, path_to_script,
                                                                            script='bv_innersphere_dos.py')
                                batch_script = BatchScript(
                                    self.batch_system, queue_type, walltime, command,
                                    mpi_procs=1, mpi_oversubscribe=0,
                                    omp_threads=self.nthreads,
                                    cores_per_node=self.cores_per_node,
                                    outdir=path, nodays=self.nodays)
                                batch_script.submit('bv_innersphere_dos' + noj + '.sh',
                                                    'bv_' + self.label + '_innersphere_dos' + noj)
                            else:
                                pass

    def _get_pt_command(self, noj, path_to_script, script='bv_parallel_tempering.py'):
        """
        this function returns the correct command line for the parallel tempering calculation
        """
        packing = self.packing_naming + noj + self.ext
        explore_dir = self.explore_dir + noj
        pt_script = os.path.join(path_to_script, script)
        command = ("python {0} {1} ${{{2}}}/{3} "
                   "--mintotniter {4} --maxtotniter {5} --relstderr {6} "
                   "--nrunners {7}"
                   .format(pt_script, packing, self.workdir_var, explore_dir,
                           self.mintotniter, self.maxtotniter,
                           self.relstderr, self.pt_runners))
        command += (" -p ${{{0}}}/{1}".format(self.workdir_var, self.structures_dir))
        if self.nocell:
            command += " --nocell"
        command += " --minimizer {}".format(self.minimizer.name)
        command += " --exchange-scheme {}".format(self.pt_exchange_scheme.name)
        command += " --sleep-seconds {}".format(self.pt_sleep_seconds)
        if self.pt_nocollectminima:
            command += " --nocollectminima"
        if self.numnegk > 0:
            command += " --numnegk {0} --lownegk {1}".format(self.numnegk, self.lownegk)
        if self.delraw > 0:
            command += " --delraw"
        return command

    def submit_pt_calculations(self, queue_type, walltime, path_to_script, force):
        """
        launch pt calculations manually if they have not been launched yet
        (this method only checks that the config files are not ready or present,
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *batch_script object of type BatchScript
        *path_to_script, excluding the filename with .py extension
        """
        for root, dirs, files in os.walk(self.workdir):
            for dir in dirs:
                if self.explore_dir in dir:  # PT requires that the explore_dir has already been created
                    if not os.path.isfile(
                            os.path.join(self.workdir, dir + ".tar.gz")):  # check if there's a tar version
                        noj = re.findall(r'\d+', dir)[0]  # extract packing number from explor_dir string
                        if self.nojmin <= int(noj) <= self.nojmax:
                            path = os.path.join(root, dir)  # build a full path
                            kmax_path = os.path.join(path, self.kmax_config + noj + '.config')
                            kmin_path = os.path.join(path, self.kmin_config + noj + '.config')
                            pt_path = os.path.join(path, self.pt_config + noj + '.config')
                            if (self._check_kmax_config_file_ready(kmax_path) \
                                        and self._check_kmin_config_file_ready(kmin_path)) \
                                    and (not self._check_pt_config_file_ready(pt_path) or force):
                                #############remove old pt data##############
                                self._remove_pt_old_data(dir, self.pt_config + noj,
                                                         output_signature="bv_{}_pt{}.o*".format(self.label, noj))
                                ##############################################
                                ##now check that kmax has a reasonable value##
                                if check_kmax_reasonable(kmax_path):
                                    if not os.path.isabs(path_to_script):
                                        path_to_script = os.path.abspath(path_to_script)
                                    mpi_procs = min(self.pt_workers+1, self.pt_runners)
                                    if mpi_procs == self.pt_runners:
                                        mpi_oversubscribe = 0
                                    else:
                                        mpi_oversubscribe = 1
                                    command = self._get_pt_command(noj, path_to_script)
                                    batch_script = BatchScript(
                                        self.batch_system, queue_type, walltime, command,
                                        mpi_procs=mpi_procs,
                                        mpi_oversubscribe=mpi_oversubscribe,
                                        omp_threads=self.nthreads,
                                        cores_per_node=self.cores_per_node,
                                        outdir=path, nodays=self.nodays)
                                    batch_script.submit('bv_pt' + noj + '.sh',
                                                        'bv_' + self.label + '_pt' + noj)
                            else:
                                pass

    def submit_chain_calculations(self, k_queue_type, k_walltime, pt_queue_type,
                                  pt_walltime, path_to_script):
        """
        launch a chain of calculations. The strategy is to create all 3 bash files at the start and
        then submit them in the following order: kmin -> kmax -> pt.
        This function assumes that the path_to_script is the same for all 3 executables
        *batch_script object of type BatchScript
        *path_to_script, excluding the filename with .py extension
        """
        subdirs = get_immediate_subdirectories(self.workdir)
        assert (self.structures_dir in subdirs)
        structures_dir_path = os.path.join(self.workdir, self.structures_dir)
        for _, _, files in os.walk(structures_dir_path):
            for file in files:
                noj = re.findall(r'\d+', file)[0]  # extract packing number
                explore_dir = self.explore_dir + noj  # build explore_dir name
                if (self.ext in file
                    and self.nojmin <= int(noj) <= self.nojmax
                    and not os.path.isfile(
                            os.path.join(self.workdir, explore_dir + ".tar.gz"))):

                    path = os.path.join(self.workdir, explore_dir)  # build a full path for explore dir
                    if (explore_dir) not in subdirs:  # check is explore_dir is a subfolder of self.workdir
                        trymakedir(path)
                    kmax_path = os.path.join(path, self.kmax_config + noj + '.config')
                    kmin_path = os.path.join(path, self.kmin_config + noj + '.config')
                    pt_path = os.path.join(path, self.pt_config + noj + '.config')
                    innersphere_dos_path = os.path.join(
                        path, self.innersphere_dos_config + noj + '.config')

                    if not self._check_innersphere_dos_config_file_ready(innersphere_dos_path):
                        #########remove old innersphere data#######
                        self._remove_innersphere_dos_old_data(
                            explore_dir, self.innersphere_dos_config + noj,
                            output_signature="bv_{}_innersphere_dos{}.o*".format(self.label, noj))
                        #####################################
                        if not os.path.isabs(path_to_script):
                            path_to_script = os.path.abspath(path_to_script)
                        # kmax_fname = 'bv_kmax'+noj+'.sh' #unused
                        kmin_fname = 'bv_kmin' + noj + '.sh'
                        pt_fname = 'bv_pt' + noj + '.sh'
                        innersphere_dos_fname = 'bv_innersphere_dos' + noj + '.sh'
                        innersphere_dos_command = self._get_innersphere_dos_command(
                            noj, path_to_script, script='bv_innersphere_dos.py')
                        batch_script = BatchScript(
                            self.batch_system, k_queue_type, k_walltime,
                            innersphere_dos_command, mpi_procs=1, mpi_oversubscribe=0,
                            omp_threads=self.nthreads, cores_per_node=self.cores_per_node,
                            outdir=path, nodays=self.nodays)
                        if self._check_pt_config_file_ready(pt_path):
                            batch_script.submit('bv_innersphere_dos' + noj + '.sh',
                                                'bv_' + self.label + '_innersphere_dos' + noj)
                        else:
                            batch_script.write(innersphere_dos_fname,
                                               'bv_' + self.label + '_innersphere_dos' + noj)
                            #########remove old pt data#######
                            self._remove_pt_old_data(explore_dir, self.pt_config + noj,
                                                     output_signature="bv_{}_pt{}.o*".format(self.label,
                                                                                             noj))
                            ##################################
                            kmax_ready = self._check_kmax_config_file_ready(kmax_path)
                            kmin_ready = self._check_kmin_config_file_ready(kmin_path)
                            # prepare PT command
                            pt_mpi_procs = min(self.pt_workers+1, self.pt_runners)
                            if pt_mpi_procs == self.pt_runners:
                                pt_mpi_oversubscribe = 0
                            else:
                                pt_mpi_oversubscribe = 1
                            pt_command = self._get_pt_command(noj, path_to_script)
                            pt_command += ' && {} ${{{}}}/{}'.format(self.submit_cmd,
                                                                     self.workdir_var,
                                                                     innersphere_dos_fname)
                            batch_script = BatchScript(
                                self.batch_system, pt_queue_type, pt_walltime, pt_command,
                                mpi_procs=pt_mpi_procs,
                                mpi_oversubscribe=pt_mpi_oversubscribe,
                                omp_threads=self.nthreads,
                                cores_per_node=self.cores_per_node,
                                outdir=path, nodays=self.nodays)
                            # if kmax is either not terminated or is reasonable then continue
                            if check_kmax_reasonable(kmax_path):
                                # if kmin and kmax terminated
                                if kmax_ready and kmin_ready:
                                    batch_script.submit('bv_pt' + noj + '.sh',
                                                        'bv_' + self.label + '_pt' + noj)
                                else:
                                    batch_script.write(pt_fname, 'bv_' + self.label + '_pt' + noj)
                                    if not kmin_ready:
                                        #########remove old BV output#######
                                        self._remove_bv_output(explore_dir,
                                                               "bv_{}_kmin{}.o*".format(self.label, noj))
                                        #####################################
                                        kmin_command = self._get_findk_command(
                                            noj, path_to_script, script='bv_find_kmin.py',
                                            record_steps_timeseries=self.record_steps_timeseries)
                                        kmin_command += ' && {} ${{{}}}/{}'.format(self.submit_cmd,
                                                                                   self.workdir_var,
                                                                                   pt_fname)
                                        batch_script = BatchScript(
                                            self.batch_system, k_queue_type, k_walltime,
                                            kmin_command, mpi_procs=1,
                                            mpi_oversubscribe=0,
                                            omp_threads=self.nthreads,
                                            cores_per_node=self.cores_per_node,
                                            outdir=path, nodays=self.nodays)
                                        if kmax_ready:
                                            batch_script.submit('bv_kmin' + noj + '.sh',
                                                                'bv_' + self.label + '_kmin' + noj)
                                        else:
                                            #########remove old BV output#######
                                            self._remove_bv_output(
                                                explore_dir, "bv_{}_kmax{}.o*".format(self.label, noj))
                                            #####################################
                                            batch_script.write(kmin_fname,
                                                               'bv_' + self.label + '_kmin' + noj)
                                            kmax_command = self._get_findk_command(
                                                noj, path_to_script, script='bv_find_kmax.py')
                                            kmax_command += ' && {} ${{{}}}/{}'.format(self.submit_cmd,
                                                                                       self.workdir_var,
                                                                                       kmin_fname)
                                            batch_script = BatchScript(
                                                self.batch_system, k_queue_type, k_walltime,
                                                kmax_command, mpi_procs=1,
                                                mpi_oversubscribe=0,
                                                omp_threads=self.nthreads,
                                                cores_per_node=self.cores_per_node,
                                                outdir=path, nodays=self.nodays)
                                            batch_script.submit('bv_kmax' + noj + '.sh',
                                                                'bv_' + self.label + '_kmax' + noj)
                                    else:
                                        kmax_command = self._get_findk_command(
                                            noj, path_to_script, script='bv_find_kmax.py')
                                        kmax_command += ' && {} ${{{}}}/{}'.format(self.submit_cmd,
                                                                                   self.workdir_var,
                                                                                   pt_fname)
                                        batch_script = BatchScript(
                                            self.batch_system, k_queue_type, k_walltime,
                                            kmax_command, mpi_procs=1,
                                            mpi_oversubscribe=0,
                                            omp_threads=self.nthreads,
                                            cores_per_node=self.cores_per_node,
                                            outdir=path, nodays=self.nodays)
                                        batch_script.submit('bv_kmax' + noj + '.sh',
                                                            'bv_' + self.label + '_kmax' + noj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="perform parallel tempering for basin volume method")
    subparsers = parser.add_subparsers(help="Choose a subparser: single to "
                                            "submit jobs individually, "
                                            "chain to submit a bv jobs chain", dest='mode')
    single_parser = subparsers.add_parser('single', help='submit individual jobs')
    chain_parser = subparsers.add_parser('chain', help='submit a bv chain')

    single_parser.add_argument("ndim", type=int, help="dimensionality")
    single_parser.add_argument("workdir", type=str,
                               help="working directory (folder containing the "
                                    "packings and jammed_packings subfolders)")
    single_parser.add_argument("path_to_script", type=str, help="path to the file to execute")
    single_parser.add_argument("job_label", type=str, help="suggested: Nn_Pp_Pp_nD: 32_70_80_2D")
    single_parser.add_argument("queue_type", type=str, help="queue type")
    single_parser.add_argument("walltime_hours", type=float, help="wall-time in hours")
    single_parser.add_argument("--batch-system", type=str,
                               help="Batch system. Supported: PBS, SLURM. "
                                    "Default: 'PBS'", default='PBS')
    single_parser.add_argument("--kmin", action='store_true', help="compute kmin", default=False)
    single_parser.add_argument("--kmax", action='store_true', help="compute kmax", default=False)
    single_parser.add_argument("--mbar", action='store_true',
                               help="compute innersphere dos", default=False)
    single_parser.add_argument("--pt", action='store_true',
                               help="perform parallel tempering", default=False)
    single_parser.add_argument("--cores-per-node", type=int,
                               help="Number of cores per node. Default: 16",
                               default=16)
    single_parser.add_argument("--nojmin", type=int,
                               help="number of minimum job ID to submit (to "
                                    "selectively submit a range of jobs)",
                               default=0)
    single_parser.add_argument("--nojmax", type=int,
                               help="number of maximum job ID to submit (to "
                                    "selectively submit a range of jobs)",
                               default=1e6)
    single_parser.add_argument("--nodays", action='store_true',
                               help="don't use days in walltime format", default=False)
    single_parser.add_argument("--experimental", action='store_true',
                               help="read experimental data format",
                               default=False)
    single_parser.add_argument("--minimizer", type=str, help="Energy minimization algorithm "
                               "used for quenching. Options: 'CG', 'FIRE', 'LBFGS'. "
                               "Default: 'FIRE'", default='FIRE')
    single_parser.add_argument("--rsts", action='store_true',
                               help="record steps timeseries for diffusion studies, default: False",
                               default=False)
    single_parser.add_argument("--kmax-start", type=float,
                               help="starting value for kmax calculation", default=500)
    single_parser.add_argument("--mintotniter", type=float,
                               help="minimum number of energy evaluation per replica, "
                               "before checking for convergence default: 5e5. "
                               "This sets a lower bound", default=5e5)
    single_parser.add_argument("--maxtotniter", type=float,
                               help="maximum number of energy evaluation per replica, "
                               "This sets an upper bound default: 2e6", default=2e6)
    single_parser.add_argument("--relstderr", type=float,
                               help="relative standard error to test convergence, "
                               "default 0.05", default=0.05)
    single_parser.add_argument("--numnegk", type=int,
                               help="number of negative ks to use", default=0)
    single_parser.add_argument("--lownegk", type=float,
                               help="lowest value of negative k's to use, default -2.5",
                               default=-2.5)
    single_parser.add_argument("--force", action='store_true', help="force run", default=False)
    single_parser.add_argument("--nocell", action='store_true',
                               help="don't use cell lists, default: False",
                               default=False)
    single_parser.add_argument("--delraw", action='store_true',
                               help="Delete raw timeseries textfiles after parallel "
                               "tempering and only use the HDF5 format.", default=False)
    single_parser.add_argument("--explore-dir", type=str,
                               help="Start of the directory name for the basinvolume data. "
                                    "Default: 'explore_bv_jammed_packing'",
                               default='explore_bv_jammed_packing')
    single_parser.add_argument('-p', "--packings-dir", type=str,
                               help="Directory containing the jammed packings. "
                                    "Default: 'jammed_packings'",
                               default='jammed_packings')
    single_parser.add_argument("--threads", type=int,
                               help="Number of OpenMP threads to use. Default: 1",
                               default=1)
    single_parser.add_argument("--pt-workers", type=int,
                               help="Number of workers to use for parallel tempering. "
                                    "Default: 4",
                               default=4)
    single_parser.add_argument("--pt-runners", type=int,
                               help="Number of runners (replica) to use for "
                                    "parallel tempering. Default: 16",
                               default=16)
    single_parser.add_argument("--pt-sleep-seconds", type=float,
                               help="Waiting time between MPI probes for the "
                                    "parallel tempering job queue master. "
                                    "Default: 0.0001 (100us)",
                               default=0.0001)
    single_parser.add_argument("--pt-exchange-scheme", type=str,
                               help="Exchange scheme used in parallel tempering. "
                                    "Options: 'NEIGHBOR_EXCHANGE', 'INDEPENDENCE_SAMPLING'. "
                                    "Default: 'NEIGHBOR_EXCHANGE'",
                               default='NEIGHBOR_EXCHANGE')
    single_parser.add_argument("--pt-nocollectminima", action='store_true',
                               help="Don't collect a database of minima.",
                               default=False)
    single_parser.add_argument("--sort", action='store_true',
                               help="Sort the atoms before running PT. This "
                                    "improves performance, especially in combination "
                                    "with multithreading. Default: False",
                               default=False)

    chain_parser.add_argument("ndim", type=int, help="dimensionality")
    chain_parser.add_argument("workdir", type=str,
                              help="working directory (folder containing the "
                                   "packings and jammed_packings subfolders)")
    chain_parser.add_argument("path_to_script", type=str, help="path to the file to execute")
    chain_parser.add_argument("job_label", type=str, help="suggested: Nn_Pp_Pp_nD: 32_70_80_2D")
    chain_parser.add_argument("k_queue_type", type=str, help="queue type")
    chain_parser.add_argument("k_walltime_hours", type=float, help="wall-time in hours")
    chain_parser.add_argument("pt_queue_type", type=str, help="queue type")
    chain_parser.add_argument("pt_walltime_hours", type=float, help="wall-time in hours")
    chain_parser.add_argument("--batch-system", type=str,
                              help="Batch system. Supported: PBS, SLURM. "
                                   "Default: 'PBS'", default='PBS')
    chain_parser.add_argument("--cores-per-node", type=int,
                              help="Number of cores per node. Default: 16",
                              default=16)
    chain_parser.add_argument("--nojmin", type=int,
                              help="number of minimum job ID to submit (to "
                                   "selectively submit a range of jobs)",
                              default=0)
    chain_parser.add_argument("--nojmax", type=int,
                              help="number of maximum job ID to submit (to "
                                   "selectively submit a range of jobs)",
                              default=1e6)
    chain_parser.add_argument("--nodays", action='store_true',
                              help="don't use days in walltime format", default=False)
    chain_parser.add_argument("--minimizer", type=str, help="Energy minimization algorithm "
                              "used for quenching. Options: 'CG', 'FIRE', 'LBFGS'. "
                              "Default: 'FIRE'", default='FIRE')
    chain_parser.add_argument("--rsts", action='store_true',
                              help="record steps timeseries for diffusion studies, default: False",
                              default=False)
    chain_parser.add_argument("--kmax-start", type=float,
                              help="starting value for kmax calculation", default=500)
    chain_parser.add_argument("--mintotniter", type=float,
                              help="minimum number of energy evaluation per replica, "
                              "before checking for convergence default: 5e5. "
                              "This sets a lower bound", default=5e5)
    chain_parser.add_argument("--maxtotniter", type=float,
                              help="maximum number of energy evaluation per replica, "
                              "This sets an upper bound default: 2e6", default=2e6)
    chain_parser.add_argument("--relstderr", type=float,
                              help="relative standard error to test convergence, default 0.05",
                              default=0.05)
    chain_parser.add_argument("--numnegk", type=int, help="number of negative ks to use", default=0)
    chain_parser.add_argument("--lownegk", type=float,
                              help="lowest value of negative k's to use, default -2.5",
                              default=-2.5)
    chain_parser.add_argument("--experimental", action='store_true',
                              help="read experimental data format",
                              default=False)
    chain_parser.add_argument("--nocell", action='store_true',
                              help="don't use cell lists, default: False",
                              default=False)
    chain_parser.add_argument("--delraw", action='store_true',
                              help="Delete raw timeseries textfiles after parallel "
                              "tempering and only use the HDF5 format.", default=False)
    chain_parser.add_argument("--explore-dir", type=str,
                              help="Start of the directory name for the basinvolume data. "
                                   "Default: 'explore_bv_jammed_packing'",
                              default='explore_bv_jammed_packing')
    chain_parser.add_argument('-p', "--packings-dir", type=str,
                              help="Directory containing the jammed packings. "
                                   "Default: 'jammed_packings'",
                              default='jammed_packings')
    chain_parser.add_argument("--threads", type=int,
                              help="Number of OpenMP threads to use. Default: 1",
                              default=1)
    chain_parser.add_argument("--pt-workers", type=int,
                              help="Number of workers to use for parallel tempering. "
                                   "PT handshake is used instead of the job queue if "
                                   "PT_WORKERS = PT_RUNNERS. Default: 4",
                              default=4)
    chain_parser.add_argument("--pt-runners", type=int,
                              help="Number of runners (replica) to use for "
                                   "parallel tempering. Default: 16",
                              default=16)
    chain_parser.add_argument("--pt-sleep-seconds", type=float,
                              help="Waiting time between MPI probes for the "
                                   "parallel tempering job queue master. "
                                   "Default: 0.0001 (100us)",
                              default=0.0001)
    chain_parser.add_argument("--pt-exchange-scheme", type=str,
                              help="Exchange scheme used in parallel tempering. "
                                   "Options: 'NEIGHBOR_EXCHANGE', 'INDEPENDENCE_SAMPLING'. "
                                   "Default: 'NEIGHBOR_EXCHANGE'",
                              default='NEIGHBOR_EXCHANGE')
    chain_parser.add_argument("--pt-nocollectminima", action='store_true',
                              help="Don't collect a database of minima.",
                              default=False)
    chain_parser.add_argument("--sort", action='store_true',
                              help="Sort the atoms before running PT. This "
                                   "improves performance, especially in combination "
                                   "with multithreading. Default: False",
                              default=False)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.INFO)
    logging.info(args)

    if args.batch_system.upper() in BatchSystem.__members__:
        batch_system = BatchSystem[args.batch_system.upper()]
    else:
        raise ValueError("Unknown batch system: {}".format(args.batch_system))

    if args.minimizer.upper() in Minimizer.__members__:
        minimizer = Minimizer[args.minimizer.upper()]
    else:
        raise ValueError("Unknown minimizer: {}".format(args.minimizer))

    if args.pt_exchange_scheme.upper() in ExchangeScheme.__members__:
        pt_exchange_scheme = ExchangeScheme[args.pt_exchange_scheme.upper()]
    else:
        raise ValueError("Unknown exchange scheme: {}".format(args.pt_exchange_scheme))

    if args.sort:
        if args.nocell:
            logging.warning("Sorting without cell lists does not do anything.")
        packings_dir = os.path.join(args.workdir, args.packings_dir)
        packing_config = read_jammed_packing_config(os.path.join(packings_dir, 'jammed_packing0.config'))
        if (not packing_config['sorted']
            or (packing_config['sorted_nsubdoms'] != args.threads
                and packing_config['pot_kwargs']['balance_omp'])):
            # Only sort when packings have not yet been sorted
            logging.info("Sorting jammed packings")
            unsorted_dir = os.path.join(os.path.dirname(packings_dir), 'jammed_unsorted')
            shutil.move(packings_dir, unsorted_dir)
            os.environ['OMP_NUM_THREADS'] = str(args.threads)
            sorter = HS_Generate_Jammed_Packing(target_packing_frac=packing_config['packing_frac'],
                                                tol=1e30, maxstep_factor=packing_config['maxstep_factor'],
                                                use_cell_lists=not args.nocell,
                                                show=False, interaction=Interaction.HS_WCA,
                                                minimizer=minimizer, packings_dir=unsorted_dir,
                                                outdir=packings_dir, sort_atoms=True,
                                                import_jammed=True, check_packing=False)
            sorter.run()

    submit_bv = SubmitBV(args.ndim, batch_system=batch_system,
                         workdir=args.workdir, job_label=args.job_label,
                         nojmin=args.nojmin, nojmax=args.nojmax, nodays=args.nodays,
                         experimental=args.experimental, minimizer=minimizer,
                         record_steps_timeseries=args.rsts,
                         kmax_start=args.kmax_start, mintotniter=args.mintotniter,
                         maxtotniter=args.maxtotniter,
                         relstderr=args.relstderr, numnegk=args.numnegk,
                         lownegk=args.lownegk, nocell=args.nocell, delraw=args.delraw,
                         explore_dir=args.explore_dir,
                         structures_dir=args.packings_dir,
                         cores_per_node=args.cores_per_node, nthreads=args.threads,
                         pt_workers=args.pt_workers, pt_runners=args.pt_runners,
                         pt_exchange_scheme=pt_exchange_scheme,
                         pt_sleep_seconds=args.pt_sleep_seconds,
                         pt_nocollectminima=args.pt_nocollectminima)

    if args.mode == 'chain':
        submit_bv.submit_chain_calculations(args.k_queue_type, args.k_walltime_hours,
                                            args.pt_queue_type, args.pt_walltime_hours,
                                            args.path_to_script)
    else:
        assert (not ((args.kmin is True or args.kmax is True) and args.pt is True))
        if args.kmin:
            submit_bv.submit_kmin_calculations(args.queue_type, args.walltime_hours,
                                               args.path_to_script, args.force)
        if args.kmax:
            submit_bv.submit_kmax_calculations(args.queue_type, args.walltime_hours,
                                               args.path_to_script, args.force)
        if args.pt:
            submit_bv.submit_pt_calculations(args.queue_type, args.walltime_hours,
                                             args.path_to_script, args.force)
        if args.mbar:
            submit_bv.submit_innersphere_dos_calculations(args.queue_type, args.walltime_hours,
                                                          args.path_to_script, args.force)
