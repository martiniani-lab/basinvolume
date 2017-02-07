from __future__ import division
import os
import re
import ConfigParser
import numpy as np
import argparse
from basinvolume.cluster_manager import BuildPBSScript
from basinvolume.utils import trymakedir, check_kmax_reasonable
import shutil
import shlex
import subprocess


def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


class BVSubmitPBS(object):
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
    def __init__(self, ndim, workdir=None, job_label='32_70_88_2D', explore_dir='explore_bv_jammed_packing',
                 kmax_config='findk_jammed_packing',
                 kmin_config='kmin_jammed_packing', innersphere_dos_config='innersphere_jammed_packing',
                 pt_config='explore_jammed_packing',
                 packing_naming='jammed_packing', structures_dir='jammed_packings', nojmin=0, nojmax=1e6, nodays=False,
                 experimental=False, minimizer='fire', record_steps_timeseries=False, kmax_start=500, mintotniter=5e5,
                 maxtotniter=2e6, relstderr=0.05, numnegk=0, lownegk=-2.5, pot_opt_str='hs_wca', nocell=False, delraw=False):
        if not workdir:
            workdir = os.getcwd()
        if not os.path.isabs(workdir):
            workdir = os.path.abspath(workdir)
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
        self.pot_opt_str = pot_opt_str
        self.nocell = nocell
        self.delraw = delraw
        self.pt_output_files = ["exchanges", "rem_permutations", "temperatures"]
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

    def _remove_pbs_output(self, explore_dir_path, output_signature):
        p = subprocess.call(
            shlex.split("find {} -maxdepth 1 -type f -name \"{}\" -exec rm -vf '{{}}' \;".format(explore_dir_path,
                                                                                                 output_signature)))
        if p != 0:
            raise Exception("removing pbs output file failed")

    def _remove_innersphere_dos_old_data(self, explore_dir_path, config_fname,
                                         output_signature="bv*innersphere_dos*.o*",
                                         rmdata="inner_sphere.timeseries"):
        try:
            os.remove(os.path.join(explore_dir_path, rmdata))
            print "removed {}".format(os.path.join(explore_dir_path, rmdata))
        except OSError:
            pass  # nothing to remove
        # remove config file and pbs output
        self._remove_pbs_output(explore_dir_path, output_signature)
        self._remove_pbs_output(explore_dir_path, config_fname + "*.config")

    def _remove_pt_old_data(self, explore_dir_path, config_fname, output_signature="bv*pt*.o*", pt=False):
        for root, dirs, files in os.walk(explore_dir_path):
            for dir in dirs:
                if dir.isdigit():
                    print "removing ", os.path.join(root, dir)
                    shutil.rmtree(os.path.join(root, dir))
            for file in files:
                if file in self.pt_output_files:
                    print "removing ", os.path.join(root, file)
                    os.remove(os.path.join(root, file))
        # remove config file and pbs output
        self._remove_pbs_output(explore_dir_path, output_signature)
        self._remove_pbs_output(explore_dir_path, config_fname + "*.config")

    def _get_findk_command(self, noj, path_to_script, script='bv_find_kmin.py', record_steps_timeseries=False):
        """
        this function returns the correct command line
        """
        packing = self.packing_naming + noj + self.ext
        findk_script = os.path.join(path_to_script, script)
        command = ('python {0} {1} -p ${{PBS_O_WORKDIR}}/jammed_packings '
                   '--opt-pot {2}').format(findk_script, packing, self.pot_opt_str)
        command += " --minimizer {}".format(self.minimizer)
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
        command = ('python {0} {1} -p ${{PBS_O_WORKDIR}}/jammed_packings '
                   '--opt-pot {2}').format(innersphere_dos_script, packing, self.pot_opt_str)
        command += " --minimizer {}".format(self.minimizer)
        if self.nocell:
            command += " --nocell"
        return command

    def submit_kmin_calculations(self, queue_type, nodes, cores, walltime, path_to_script, force):
        """
        launch kmin calculations manually if they have not been launched yet
        (this method only checks that the config file is not ready or present,
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *pbs object of type BuildPBSScript
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
                                #########remove old pbs output#######
                                self._remove_pbs_output(explore_dir, "bv_{}_kmin{}.o*".format(self.label, noj))
                                #####################################
                                if not os.path.isabs(path_to_script):
                                    path_to_script = os.path.abspath(path_to_script)
                                command = self._get_findk_command(noj, path_to_script, script='bv_find_kmin.py',
                                                                  record_steps_timeseries=self.record_steps_timeseries)
                                pbs = BuildPBSScript(queue_type, nodes, cores, walltime, command, outdir=path,
                                                     nodays=self.nodays)
                                pbs.submit_PBS('bv_kmin' + noj + '.sh', 'bv_' + self.label + '_kmin' + noj)
                            else:
                                pass

    def submit_kmax_calculations(self, queue_type, nodes, cores, walltime, path_to_script, force):
        """
        launch kmax calculations manually if they have not been launched yet
        (this method only checks that the config file is not ready or present,
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *pbs object of type BuildPBSScript
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
                                #########remove old pbs output#######
                                self._remove_pbs_output(explore_dir, "bv_{}_kmax{}.o*".format(self.label, noj))
                                #####################################
                                if not os.path.isabs(path_to_script):
                                    path_to_script = os.path.abspath(path_to_script)
                                command = self._get_findk_command(noj, path_to_script, script='bv_find_kmax.py')
                                pbs = BuildPBSScript(queue_type, nodes, cores, walltime, command, outdir=path,
                                                     nodays=self.nodays)
                                pbs.submit_PBS('bv_kmax' + noj + '.sh', 'bv_' + self.label + '_kmax' + noj)
                            else:
                                pass

    def submit_innersphere_dos_calculations(self, queue_type, nodes, cores, walltime, path_to_script, force):
        """
        launch innersphere_dos bv_innersphere_dos calculations manually if they have not been launched yet
        (this method only checks that the config file is not ready or present,
        hence this method should only be used when there are no calculations running,
        as the calculation might have already been launched and it is in the queue)
        *pbs object of type BuildPBSScript
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
                                pbs = BuildPBSScript(queue_type, nodes, cores, walltime, command, outdir=path,
                                                     nodays=self.nodays)
                                pbs.submit_PBS('bv_innersphere_dos' + noj + '.sh',
                                               'bv_' + self.label + '_innersphere_dos' + noj)
                            else:
                                pass

    def _get_pt_command(self, noj, path_to_script, ncores, script='bv_parallel_tempering.py'):
        """
        this function returns the correct command line for the parallel tempering calculation
        """
        packing = self.packing_naming + noj + self.ext
        explore_dir = self.explore_dir + noj
        pt_script = os.path.join(path_to_script, script)
        if ncores % 2 == 0:
            command = ('-n {0} python {1} {2} ${{PBS_O_WORKDIR}}/{3} '
                       '--mintotniter {4} --maxtotniter {5} --relstderr {6} '
                       '--opt-pot {7}').format(ncores - 1, pt_script, packing, explore_dir, self.mintotniter,
                                               self.maxtotniter, self.relstderr, self.pot_opt_str)
        else:
            command = ('python {0} {1} ${{PBS_O_WORKDIR}}/{2} '
                       '--mintotniter {3} --maxtotniter {4} --relstderr {5} '
                       '--opt-pot {6}').format(pt_script, packing, explore_dir, self.mintotniter,
                                               self.maxtotniter, self.relstderr, self.pot_opt_str)
        if self.nocell:
            command += " --nocell"
        command += " --minimizer {}".format(self.minimizer)
        if self.numnegk > 0:
            command += " --numnegk {0} --lownegk {1}".format(self.numnegk, self.lownegk)
        if self.delraw > 0:
            command += "--delraw"
        return command

    def submit_pt_calculations(self, queue_type, nodes, cores, walltime, path_to_script, force):
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
                                    command = self._get_pt_command(noj, path_to_script, nodes * cores)
                                    pbs = BuildPBSScript(queue_type, nodes, cores, walltime, command, outdir=path,
                                                         nodays=self.nodays)
                                    pbs.submit_PBS('bv_pt' + noj + '.sh', 'bv_' + self.label + '_pt' + noj)
                            else:
                                pass

    def submit_chain_calculations(self, k_queue_type, k_nodes, k_cores, k_walltime,
                                  pt_queue_type, pt_nodes, pt_cores, pt_walltime, path_to_script):
        """
        launch a chain of calculations. The strategy is to create all 3 bash files at the start and
        then qsub them in the following order: kmin -> kmax -> pt.
        This function assumes that the path_to_script is the same for all 3 executables
        *pbs object of type BuildPBSScript
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
                            kmin_path = os.path.join(path, self.kmin_config + noj + '.config')
                            pt_path = os.path.join(path, self.pt_config + noj + '.config')
                            innersphere_dos_path = os.path.join(path, self.innersphere_dos_config + noj + '.config')

                            if not self._check_innersphere_dos_config_file_ready(innersphere_dos_path):
                                #########remove old innersphere data#######
                                self._remove_innersphere_dos_old_data(explore_dir, self.innersphere_dos_config + noj,
                                                                      output_signature="bv_{}_innersphere_dos{}.o*".format(
                                                                          self.label, noj))
                                #####################################
                                if not os.path.isabs(path_to_script):
                                    path_to_script = os.path.abspath(path_to_script)
                                # kmax_fname = 'bv_kmax'+noj+'.sh' #unused
                                kmin_fname = 'bv_kmin' + noj + '.sh'
                                pt_fname = 'bv_pt' + noj + '.sh'
                                innersphere_dos_fname = 'bv_innersphere_dos' + noj + '.sh'
                                innersphere_dos_command = self._get_innersphere_dos_command(noj, path_to_script,
                                                                                            script='bv_innersphere_dos.py')
                                pbs = BuildPBSScript(k_queue_type, k_nodes, k_cores, k_walltime,
                                                     innersphere_dos_command, outdir=path, nodays=self.nodays)
                                if self._check_pt_config_file_ready(pt_path):
                                    pbs.submit_PBS('bv_innersphere_dos' + noj + '.sh',
                                                   'bv_' + self.label + '_innersphere_dos' + noj)
                                else:
                                    pbs.writePBSscript(innersphere_dos_fname,
                                                       'bv_' + self.label + '_innersphere_dos' + noj)
                                    #########remove old pt data#######
                                    self._remove_pt_old_data(explore_dir, self.pt_config + noj,
                                                             output_signature="bv_{}_pt{}.o*".format(self.label, noj))
                                    ##################################
                                    kmax_ready = self._check_kmax_config_file_ready(kmax_path)
                                    kmin_ready = self._check_kmin_config_file_ready(kmin_path)
                                    # prepare PT command
                                    pt_command = self._get_pt_command(noj, path_to_script, pt_nodes * pt_cores)
                                    pt_command += ' && qsub ${{PBS_O_WORKDIR}}/{}'.format(innersphere_dos_fname)
                                    pbs = BuildPBSScript(pt_queue_type, pt_nodes, pt_cores, pt_walltime, pt_command,
                                                         outdir=path, nodays=self.nodays)
                                    # if kmax is either not terminated or is reasonable then continue
                                    if check_kmax_reasonable(kmax_path):
                                        # if kmin and kmax terminated
                                        if kmax_ready and kmin_ready:
                                            pbs.submit_PBS('bv_pt' + noj + '.sh', 'bv_' + self.label + '_pt' + noj)
                                        else:
                                            pbs.writePBSscript(pt_fname, 'bv_' + self.label + '_pt' + noj)
                                            if not kmin_ready:
                                                #########remove old pbs output#######
                                                self._remove_pbs_output(explore_dir,
                                                                        "bv_{}_kmin{}.o*".format(self.label, noj))
                                                #####################################
                                                kmin_command = self._get_findk_command(noj, path_to_script,
                                                                                       script='bv_find_kmin.py',
                                                                                       record_steps_timeseries=self.record_steps_timeseries)
                                                kmin_command += ' && qsub ${{PBS_O_WORKDIR}}/{}'.format(pt_fname)
                                                pbs = BuildPBSScript(k_queue_type, k_nodes, k_cores, k_walltime,
                                                                     kmin_command, outdir=path, nodays=self.nodays)
                                                if kmax_ready:
                                                    pbs.submit_PBS('bv_kmin' + noj + '.sh',
                                                                   'bv_' + self.label + '_kmin' + noj)
                                                else:
                                                    #########remove old pbs output#######
                                                    self._remove_pbs_output(explore_dir,
                                                                            "bv_{}_kmax{}.o*".format(self.label, noj))
                                                    #####################################
                                                    pbs.writePBSscript(kmin_fname, 'bv_' + self.label + '_kmin' + noj)
                                                    kmax_command = self._get_findk_command(noj, path_to_script,
                                                                                           script='bv_find_kmax.py')
                                                    kmax_command += ' && qsub ${{PBS_O_WORKDIR}}/{}'.format(kmin_fname)
                                                    pbs = BuildPBSScript(k_queue_type, k_nodes, k_cores, k_walltime,
                                                                         kmax_command, outdir=path, nodays=self.nodays)
                                                    pbs.submit_PBS('bv_kmax' + noj + '.sh',
                                                                   'bv_' + self.label + '_kmax' + noj)
                                            else:
                                                kmax_command = self._get_findk_command(noj, path_to_script,
                                                                                       script='bv_find_kmax.py')
                                                kmax_command += ' && qsub ${{PBS_O_WORKDIR}}/{}'.format(pt_fname)
                                                pbs = BuildPBSScript(k_queue_type, k_nodes, k_cores, k_walltime,
                                                                     kmax_command, outdir=path, nodays=self.nodays)
                                                pbs.submit_PBS('bv_kmax' + noj + '.sh',
                                                               'bv_' + self.label + '_kmax' + noj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="perform parallel tempering for basin volume method")
    subparsers = parser.add_subparsers(help="Choose a subparser: single to submit jobs individually, "
                                            "chain to submit a bv jobs chain", dest='mode')
    single_parser = subparsers.add_parser('single', help='submit individual jobs')
    chain_parser = subparsers.add_parser('chain', help='submit a bv chain')

    single_parser.add_argument("ndim", type=int, help="dimensionality")
    single_parser.add_argument("workdir", type=str,
                               help="working directory (folder containing the packings and jammed_packings subfolders)")
    single_parser.add_argument("path_to_script", type=str, help="path to the file to execute")
    single_parser.add_argument("job_label", type=str, help="suggested: Nn_Pp_Pp_nD: 32_70_80_2D")
    single_parser.add_argument("queue_type", type=str, help="queue type")
    single_parser.add_argument("nodes", type=int, help="number of nodes to use")
    single_parser.add_argument("cores", type=int, help="number of processors per node to use")
    single_parser.add_argument("walltime_hours", type=float, help="wall-time in hours")
    single_parser.add_argument("--kmin", action='store_true', help="compute kmin", default=False)
    single_parser.add_argument("--kmax", action='store_true', help="compute kmax", default=False)
    single_parser.add_argument("--mbar", action='store_true', help="compute innersphere dos", default=False)
    single_parser.add_argument("--pt", action='store_true', help="perform parallel tempering", default=False)
    single_parser.add_argument("--nojmin", type=int,
                               help="number of minimum job ID to submit (to selectively submit a range of jobs)",
                               default=0)
    single_parser.add_argument("--nojmax", type=int,
                               help="number of maximum job ID to submit (to selectively submit a range of jobs)",
                               default=1e6)
    single_parser.add_argument("--nodays", action='store_true', help="don't use days in walltime format", default=False)
    single_parser.add_argument("--experimental", action='store_true', help="read experimental data format",
                               default=False)
    single_parser.add_argument("--minimizer", type=str, help="Energy minimization algorithm "
                               "used for quenching. Options: 'cg', 'fire', 'lbfgs'. "
                               "Default: 'fire'", default='fire')
    single_parser.add_argument("--rsts", action='store_true',
                               help="record steps timeseries for diffusion studies, default: False", default=False)
    single_parser.add_argument("--kmax_start", type=float, help="starting value for kmax calculation", default=500)
    single_parser.add_argument("--mintotniter", type=float, help="minimum number of energy evaluation per replica, "
                               "before checking for convergence default: 5e5. This sets a lower bound", default=5e5)
    single_parser.add_argument("--maxtotniter", type=float, help="maximum number of energy evaluation per replica, "
                               "This sets an upper bound default: 2e6", default=2e6)
    single_parser.add_argument("--relstderr", type=float, help="relative standard error to test convergence, "
                               "default 0.05", default=0.05)
    single_parser.add_argument("--numnegk", type=int, help="number of negative ks to use", default=0)
    single_parser.add_argument("--lownegk", type=float, help="lowest value of negative k's to use, default -2.5",
                               default=-2.5)
    single_parser.add_argument("--force", action='store_true', help="force run", default=False)
    single_parser.add_argument("--nocell", action='store_true', help="don't use cell lists, default: False",
                               default=False)
    single_parser.add_argument("--delraw", action='store_true',
                               help="Delete raw timeseries textfiles after parallel "
                               "tempering and only use the HDF5 format.", default=False)

    chain_parser.add_argument("ndim", type=int, help="dimensionality")
    chain_parser.add_argument("workdir", type=str,
                              help="working directory (folder containing the packings and jammed_packings subfolders)")
    chain_parser.add_argument("path_to_script", type=str, help="path to the file to execute")
    chain_parser.add_argument("job_label", type=str, help="suggested: Nn_Pp_Pp_nD: 32_70_80_2D")
    chain_parser.add_argument("k_queue_type", type=str, help="queue type")
    chain_parser.add_argument("k_nodes", type=int, help="number of nodes to use")
    chain_parser.add_argument("k_cores", type=int, help="number of processors per node to use")
    chain_parser.add_argument("k_walltime_hours", type=float, help="wall-time in hours")
    chain_parser.add_argument("pt_queue_type", type=str, help="queue type")
    chain_parser.add_argument("pt_nodes", type=int, help="number of nodes to use")
    chain_parser.add_argument("pt_cores", type=int, help="number of processors per node to use")
    chain_parser.add_argument("pt_walltime_hours", type=float, help="wall-time in hours")
    chain_parser.add_argument("--nojmin", type=int,
                              help="number of minimum job ID to submit (to selectively submit a range of jobs)",
                              default=0)
    chain_parser.add_argument("--nojmax", type=int,
                              help="number of maximum job ID to submit (to selectively submit a range of jobs)",
                              default=1e6)
    chain_parser.add_argument("--nodays", action='store_true', help="don't use days in walltime format", default=False)
    chain_parser.add_argument("--minimizer", type=str, help="Energy minimization algorithm "
                              "used for quenching. Options: 'cg', 'fire', 'lbfgs'. "
                              "Default: 'fire'", default='fire')
    chain_parser.add_argument("--rsts", action='store_true',
                              help="record steps timeseries for diffusion studies, default: False", default=False)
    chain_parser.add_argument("--kmax_start", type=float, help="starting value for kmax calculation", default=500)
    chain_parser.add_argument("--mintotniter", type=float, help="minimum number of energy evaluation per replica, "
                              "before checking for convergence default: 5e5. This sets a lower bound", default=5e5)
    chain_parser.add_argument("--maxtotniter", type=float, help="maximum number of energy evaluation per replica, "
                              "This sets an upper bound default: 2e6", default=2e6)
    chain_parser.add_argument("--relstderr", type=float,
                              help="relative standard error to test convergence, default 0.05", default=0.05)
    chain_parser.add_argument("--numnegk", type=int, help="number of negative ks to use", default=0)
    chain_parser.add_argument("--lownegk", type=float, help="lowest value of negative k's to use, default -2.5",
                              default=-2.5)
    chain_parser.add_argument("--experimental", action='store_true', help="read experimental data format",
                              default=False)
    chain_parser.add_argument("--nocell", action='store_true', help="don't use cell lists, default: False",
                              default=False)
    chain_parser.add_argument("--delraw", action='store_true',
                              help="Delete raw timeseries textfiles after parallel "
                              "tempering and only use the HDF5 format.", default=False)

    args = parser.parse_args()
    print args

    if args.minimizer.lower() not in ['cg', 'fire', 'lbfgs']:
        raise NotImplementedError("Undefined minimizer: {}".format(args.minimizer))

    bvpbs = BVSubmitPBS(args.ndim, workdir=args.workdir, job_label=args.job_label, nojmin=args.nojmin,
                        nojmax=args.nojmax, nodays=args.nodays, experimental=args.experimental, minimizer=args.minimizer,
                        record_steps_timeseries=args.rsts, kmax_start=args.kmax_start, mintotniter=args.mintotniter,
                        maxtotniter=args.maxtotniter,
                        relstderr=args.relstderr, numnegk=args.numnegk, lownegk=args.lownegk, nocell=args.nocell, delraw=args.delraw)

    if args.mode == 'chain':
        bvpbs.submit_chain_calculations(args.k_queue_type, args.k_nodes, args.k_cores,
                                        args.k_walltime_hours, args.pt_queue_type, args.pt_nodes,
                                        args.pt_cores, args.pt_walltime_hours, args.path_to_script)
    else:
        assert (not ((args.kmin is True or args.kmax is True) and args.pt is True))
        if args.kmin:
            bvpbs.submit_kmin_calculations(args.queue_type, args.nodes, args.cores, args.walltime_hours,
                                           args.path_to_script, args.force)
        if args.kmax:
            bvpbs.submit_kmax_calculations(args.queue_type, args.nodes, args.cores, args.walltime_hours,
                                           args.path_to_script, args.force)
        if args.pt:
            bvpbs.submit_pt_calculations(args.queue_type, args.nodes, args.cores, args.walltime_hours,
                                         args.path_to_script, args.force)
        if args.mbar:
            bvpbs.submit_innersphere_dos_calculations(args.queue_type, args.nodes, args.cores, args.walltime_hours,
                                                      args.path_to_script, args.force)
