"""
THIS IS OBSOLETE.
But it has some useful code in it which should be migrated to the newer implementations.
"""

from __future__ import division
import numpy as np
import os
import re
import glob
from basinvolume.utils import trymakedir
from basinvolume.utils import to_string, read_txt, write_csv_xy, import_pt_time_series
import ConfigParser
from basinvolume.post_processing import F_Basin_From_MC_Data
from basinvolume.post_processing import F_Basin_From_MC_Data_Free_COM
from basinvolume.post_processing import Gauss_Lobatto_abscissas
from basinvolume.post_processing import VolumeSanityCheck
from basinvolume.spheres.generate_jammed_packing import read_jammed_packing_config
import traceback
import argparse
from itertools import cycle
try:
    import pylab as plt
    from matplotlib import rc
except ImportError as err:
    print err

#######################SET LATEX OPTIONS###################
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
#rc('text.latex',preamble=r'\usepackage{times}')
plt.rcParams.update({'font.size': 18})
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams.update({'figure.autolayout': True})
##########################################################
####SET COLOUR MAP######
def get_color_cycle():
    cm = plt.get_cmap('Set2')
    color_cycle=cycle([cm(1. * i / 7) for i in xrange(7)])
    return color_cycle
########################

class _collect_u2_vs_k(object):
    """
    this is a class that implements _collect_u2_vs_k class
    *ts_skip number of points skipped when printing time series (every ts_skip)
    """
    def __call__(self, ts_skip=5000, fname='jammed_packing0', base_dir='analysis',
                 explore_dir='explore_bv_', packings_dir='packings', jammed_packings_dir='jammed_packings',
                 plot_ts_integrand_data=True, frozen=False, show=False, plot_only=False, verbose=False):

        self.fname = fname
        if not os.path.isabs(jammed_packings_dir):
            jammed_packings_dir = os.path.join(os.getcwd(),jammed_packings_dir)
            packings_dir = os.path.join(os.getcwd(), packings_dir)
        self.jammed_packings_dir = jammed_packings_dir
        self.packings_dir = packings_dir
        if not os.path.isabs(explore_dir):
            explore_dir = os.path.join(os.getcwd(),explore_dir+fname)
        self.explore_dir = explore_dir
        self.base_directory = self.explore_dir + '/' + base_dir
        self.frozen = frozen
        n = int(re.findall(r'\d+', self.fname)[0])
        self.packing_configpath = os.path.join(packings_dir, 'packing{}.config'.format(n))
        assert os.path.isfile(self.packing_configpath)
        self.jammed_packing_configpath = os.path.join(jammed_packings_dir, '{}.config'.format(self.fname))
        assert os.path.isfile(self.jammed_packing_configpath)
        self.findk_configpath = os.path.join(self.explore_dir, 'findk_' + fname + '.config')
        assert os.path.isfile(self.findk_configpath)
        self.kmin_configpath = os.path.join(self.explore_dir, 'kmin_' + fname + '.config')
        assert os.path.isfile(self.kmin_configpath)
        self.pt_configpath = os.path.join(self.explore_dir, 'explore_' + fname + '.config')
        assert os.path.isfile(self.pt_configpath)
        self.verbose = verbose
        if self.verbose:
            print("self.packing_configpath", self.packing_configpath)
            print("self.jammed_packing_configpath", self.jammed_packing_configpath)
            print("self.findk_configpath", self.findk_configpath)
            print("self.kmin_configpath", self.kmin_configpath)

        self.ts_skip = ts_skip
        self.plot_ts_integrand_data = plot_ts_integrand_data
        self.show = show
        self.plot_only = plot_only
        self._import_config_files()
        self.run()

    def run(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        """
        Full volume computation, assuming that PT data is available
        """
        try:
            if self.plot_only:
                self._plot_data()
            else:
                self._import_ks()
                self._import_u2_reverse()
                self._remove_negative_k()
                self._print_u2_vs_k()
                self._compute_hs_fluid_volume() # Maybe we can move this to the entropy computation part,
                self._compute_volume()          # there is no reason to also compute the accessible volume at this point.
                self._plot_data()
        except Exception as err:
            print("Exception: ", err)
        """
        Print basin volumes for further processing
        """
        self._print_volumes()

    def _import_config_files(self):
        imp_packing = read_jammed_packing_config(str(self.jammed_packing_configpath), self.frozen)
        self.nparticles = imp_packing['nparticles']
        self.packing_frac = imp_packing['packing_frac']
        self.bdim = imp_packing['bdim']
        self.ndim = imp_packing['ndim']
        self.boxv = imp_packing['boxv'].copy()
        self.vcavity = imp_packing['vcavity']
        self.sca = imp_packing['sca']
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.findk_configpath))
        self.kmax = configf.getfloat('FINDK', 'kmax')
        self.prob_kmax = configf.getfloat('FINDK', 'prob')
        self.displ_k_max = configf.getfloat('FINDK', 'displ_k_max')
        self.var_displ_k_max = configf.getfloat('FINDK', 'var_displ_k_max')
        self.kmax_iteration = configf.getfloat("FINDK_MCRUNNER_STATUS", "iteration")
        self.std_error_kmax = np.sqrt(self.var_displ_k_max / (self.prob_kmax * self.kmax_iteration))
        configf.read(str(self.kmin_configpath))
        self.kmin = configf.getfloat('KMIN_MCRUNNER', 'k')
        self.displ_k_min = configf.getfloat('KMIN', 'displ_k_min')
        self.var_displ_k_min = configf.getfloat('KMIN', 'var_displ_k_min')
        configf.read(str(self.pt_configpath))
        self.adjustf_niter = configf.getfloat('MCRUNNER', 'adjustf_niter')

    def _import_ks(self):
        """
        must run before import u2
        """
        karray = []
        path = os.path.join(self.explore_dir, 'temperatures')
        f = open(path, "r")
        while True:
            k = f.readline()
            if not k: break
            karray.extend([float(k)])
        #prepend kmax
        karray.insert(0, self.kmax)
        self.karray = np.array(karray[::-1], dtype='d')

    def _import_u2_reverse(self):
        n = len(self.karray)-1
        self.u2_array = [0 for _ in xrange(n)]
        self.var_array = [0 for _ in xrange(n)]
        self.std_error_array = [0 for _ in xrange(n)]
        for subdir, dirs, files in os.walk(self.explore_dir):
            for dir in dirs:
                if dir.isdigit():
                    path = os.path.join(self.explore_dir, dir + '/hist_mean')
                    fileHandle = open (path, "r")
                    lineList = fileHandle.readlines()
                    fileHandle.close()
                    niter, u2, var, std_err = lineList[-1].split()
                    self.u2_array[int(dir)] = u2
                    self.var_array[int(dir)] = var
                    self.std_error_array[int(dir)] = std_err
        #prepend u2 kmax
        self.u2_array.insert(0, self.displ_k_max)
        self.var_array.insert(0, self.var_displ_k_max)
        self.std_error_array.insert(0, self.std_error_kmax)
        self.u2_array = np.array(self.u2_array[::-1], dtype='d')
        self.var_array = np.array(self.var_array[::-1], dtype='d')
        self.std_error_array = np.array(self.std_error_array[::-1], dtype='d')

    def _remove_negative_k(self):
        try:
            k0_idx = next(idx for idx, value in enumerate(self.karray) if value == 0)
        except Exception:
            k0_idx = -1
        if k0_idx >= 0:
            self.karray = self.karray[k0_idx:]
            assert all(k >= 0 for k in self.karray), "karray not all positive"
            self.u2_array = self.u2_array[k0_idx:]
            self.var_array = self.var_array[k0_idx:]
            self.std_error_array = self.std_error_array[k0_idx:]

    def _import_time_series(self):
        self.timeseries = import_pt_time_series(self.explore_dir, self.adjustf_niter,
                                                max_series_size=0, ncores=4,
                                                crop_adjustf_niter=False, del_raw=True)

    def _import_steps_time_series_diffusion(self, eqtime=int(2e5)):
        import re
        timeseries = []
        series_order = []
        path = os.path.join(self.explore_dir, "diffusion")
        file_list = glob.glob(path + '/StepsTimeSeries*')
        file_list = sorted(file_list, key = lambda x: int(x.split(".")[1]))
        series = []
        for series_path in file_list:
            fname = str(os.path.split(series_path)[-1].split())
            digits = map(int, re.findall(r'\d+', fname))
            series_order.append(digits[-1])
            timeseries.append(read_txt(series_path))
        Y = series_order
        X = np.array(timeseries)
        step_timeseries = np.array([x for (y, x) in sorted(zip(Y, X))])
        step_timeseries_order =  np.sort(series_order)
        step_timeseries_mean_path = []
        step_timeseries_mean_path_std = []
        step_timeseries_mean_eucdist = []
        step_timeseries_mean_eucdist_std = []
        for i,n in enumerate(step_timeseries_order[1:]):
            mean_arr = []
            nsubs = step_timeseries[0][eqtime:].size // n
            for j in xrange(nsubs):
                mean_arr.append(np.sum(step_timeseries[0][eqtime+j*n:eqtime+(j+1)*n]))
            mean, stdev = np.mean(np.array(mean_arr)), np.std(np.array(mean_arr))
            step_timeseries_mean_path.append(mean)
            step_timeseries_mean_path_std.append(stdev/np.sqrt(len(mean_arr)))
            step_timeseries_mean_eucdist.append(np.mean(step_timeseries[i+1][eqtime//n:]))
            step_timeseries_mean_eucdist_std.append(np.std(step_timeseries[i+1][eqtime//n:])/np.sqrt(len(step_timeseries[i+1])))
        return step_timeseries_mean_path, step_timeseries_mean_path_std, step_timeseries_mean_eucdist, step_timeseries_mean_eucdist_std

    def _print_u2_vs_k(self):
        """writes <u2> and variance vs """
        dname = 'u2_vs_k'
        fname = '{}/{}'.format(self.base_directory,dname)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#{:>15}\t{:>15}\n'.format('<u2>', 'var(<u2>)'))
        for i in xrange(len(self.u2_array)):
            f.write('{:>15.15e}\t{:>15.15e}\n'.format(self.u2_array[i], self.var_array[i]))
        f.close()

    def _compute_volume(self):
        """
        numerical volume obtained by integrating over the PT data
        Note that to function get_free_energy_F0, we need to pass the array of squared standard errors of the data points to get the correct error bars.
        This was not done previously, so the naming in the subsequent function calls can be confusing, suggesting that we are actually passing the array of variances of the displ2 points.
        """

        #sqared_std_errors = self.var_array # This line is just to illustrate how the code worked before.
        sqared_std_errors = self.std_error_array ** 2

        self.F0, self.sigF0, self.farray, self.sigfarray = F_Basin_From_MC_Data(self.bdim, self.nparticles, self.karray,\
                                                                                self.u2_array, self.vcavity,\
                                                                                self.prob_kmax, displ_k_min_trafo=self.displ_k_min).get_free_energy_F0(sqared_std_errors)

        self.F0unc, self.sigF0unc, self.farrayunc, self.sigfarrayunc= F_Basin_From_MC_Data_Free_COM(self.bdim, self.nparticles, self.karray,\
                                                                                self.u2_array, self.vcavity,\
                                                                                self.prob_kmax, displ_k_min_trafo=self.displ_k_min).get_free_energy_F0(sqared_std_errors)
        self.tarray = Gauss_Lobatto_abscissas(len(self.u2_array))()

        self.unit_box_F0 = self.F0 + self.nparticles * np.log(self.vcavity)
        self.unit_box_F0unc = self.F0unc + self.nparticles * np.log(self.vcavity)
        print 'unit_box_F0 {} unit_box_F0unc {}'.format(self.unit_box_F0, self.unit_box_F0unc)

    def _compute_hs_fluid_volume(self, numerical_moments=False):
        volume_sanity_check = VolumeSanityCheck(self.packing_configpath, numerical_moments=numerical_moments)
        self.F0_acc = volume_sanity_check.F0_acc
        self.ideal_gas_F_acc = - self.nparticles*np.log(self.vcavity)

    def _plot_diffusion(self):
        x, dx, y, dy = self._import_steps_time_series_diffusion()
        x, dx, y, dy = np.array(x), np.array(dx), np.array(y), np.array(dy)
        pol = np.poly1d(np.polyfit(np.log(x)[:2], np.log(y)[:2], 1,  w=(y/dy)[:2])) #[5:-1]
        w = np.polyfit(np.log(x)[:2], np.log(y)[:2], 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(np.log(x), np.log(y), fmt='o', xerr=dx/x, yerr=dy/y)
        ax.plot(np.log(x), pol(np.log(x)), '-', label='a={} b={}'.format(w[0],w[1]))
        ax.set_ylabel(r'$\log(\Delta r)$')
        ax.set_xlabel(r'$\log (\Delta s)$')
        #plt.yscale('log')
        #plt.xscale('log')
        ax.legend(frameon=False, loc=1)
        plt.savefig(os.path.join(self.base_directory, 'diffusion_logr_vs_logt.pdf'))
        write_csv_xy(np.log(x), np.log(y), xerr=dx/x, yerr=dy/y, fit=pol(np.log(x)),
                     fname=os.path.join(self.base_directory, 'diffusion_logr_vs_logt.csv'))
        if self.show:
            plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(np.log(x), np.log(y)-0.5*np.log(x), fmt='o', xerr=dx/x+dy/y, yerr=dy/y)
        ax.set_ylabel(r'$\log(\Delta r) - \frac{1}{2}\log(\Delta s)$')
        ax.set_xlabel(r'$\log (\Delta s)$')
        ax.legend(frameon=False, loc=1)
        plt.savefig(os.path.join(self.base_directory, 'diffusion_red_logr_vs_logt.pdf'))
        write_csv_xy(np.log(x), np.log(y)-0.5*np.log(x), xerr=dx/x+dy/y, yerr=dy/y,
                     fname=os.path.join(self.base_directory, 'diffusion_red_logr_vs_logt.csv'))
        if self.show:
            plt.show()

    def _plot_data(self):
        if self.plot_ts_integrand_data is False:
            return
        lines = ["-", "--", "-."]
        linecycler = cycle(lines)

        #try to plot cumulative sum of steps_timeseries
        try:
            self._plot_diffusion()
        except Exception, e:
            print e
            print('_collect_u2_vs_k diffusion: %s' % (traceback.format_exc()))

        cont_karray = np.linspace(self.kmin, self.kmax, 100)
        u2_array_app = (cont_karray + (self.nparticles * self.bdim) / self.displ_k_min) / (self.nparticles * self.bdim)
        u2_array_app = 1.0 / u2_array_app

        if True:
            #timeseries
            color_cycle = get_color_cycle()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            self._import_time_series()
            for i,series in enumerate(self.timeseries):
                ax.plot(series[::self.ts_skip], ls=next(linecycler), color=color_cycle.next(), linewidth=1.8, label=str(i))
            ax.set_ylabel(r'$|{\bf r} - {\bf r}_0|$', fontsize=18)
            ax.set_xlabel('steps/{}'.format(self.ts_skip), fontsize=18)
            ax.set_xlim((0,150))
            #plt.yscale('log')
            #plt.xscale('log')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], frameon=False, loc='best', prop={'size':18}, numpoints=1, scatterpoints=1,
                      markerscale=1, columnspacing=0.25, labelspacing=0.25, handletextpad=0.1, handlelength=1)
            plt.savefig(self.base_directory + '/time_series.pdf')
            if self.show:
                plt.show()

        #mean square displacement and integrand in inset
        color_cycle = get_color_cycle()
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        (line, caps, _) = ax2.errorbar(self.karray, self.u2_array, yerr=np.sqrt(self.var_array), marker='o', ms=12, linestyle='',
                     color=color_cycle.next(), clip_on=False, zorder=100, capsize=5, elinewidth=2)
        for cap in caps:
            cap.set_zorder(100)
        line.set_zorder(100)

        ax2.plot(cont_karray, u2_array_app, '--', linewidth=2, color=color_cycle.next())
        ax2.set_xlabel(r'$k$')
        ax2.set_ylabel(r'$\langle |\mathbf{r} - \mathbf{r}_0|^2\rangle_k $')
        ax2.set_ylim(bottom=0)
        #plt.xscale('symlog')
        #plt.yscale('log')
        write_csv_xy(self.karray, self.u2_array, yerr=np.sqrt(self.var_array),
                     fname=os.path.join(self.base_directory, 'u2_vs_k.csv'))
        #inset
        ax3 = fig2.add_axes([0.45,0.42,0.4,0.4], alpha=0.5)
        ax3.errorbar(self.tarray, self.farray, yerr=self.sigfarray, marker='o', color=color_cycle.next(), ms=12, clip_on=False, zorder=100)
        ax3.set_xlabel(r'$t$', fontsize=18)
        ax3.locator_params(axis = 'x', nbins = 4)
        ax3.locator_params(axis = 'y', nbins = 4)
        ax3.tick_params(axis='both', which='major', labelsize=18)
        #ax3.set_ylabel('integrand')
        write_csv_xy(self.tarray, self.farray, yerr=self.sigfarray,
                     fname=os.path.join(self.base_directory, 'integrand.csv'))
        plt.savefig(self.base_directory + '/u2_vs_k.pdf')
        if self.show:
            plt.show()


    def _print_volumes(self):
        dname = 'volume_data'
        fname = '{}/{}'.format(self.base_directory,dname)
        f = open(fname, 'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        def _to_file(name, value):
            f.write((name + ": {}\n").format(to_string(value)))
        f.write('[VOLUME_HS_FLUID]\n')
        if hasattr(self, "F0_acc"):
            _to_file("F0_acc", self.F0_acc)
            _to_file("F0_ideal_gas", self.ideal_gas_F_acc)
        f.write('[VOLUME_FULL_PT]\n')
        if hasattr(self, "F0"):
            _to_file("F0", self.F0)
            _to_file("sigF0", self.sigF0)
            _to_file("unit_box_F0", self.unit_box_F0)
        f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="analyze PT data from thermodynamic integration")
    #parser.add_argument("nparticles", type=int, help="number of particles")
    parser.add_argument("-f","--fname", type=str, help="specify packing to analyze",default=None)
    parser.add_argument("-d","--fdir", type=str, help="directory containing file, if not absolute path by default: fdir+fname", default='explore_bv_')
    parser.add_argument("-w","--workdir", type=str, help="directory containing PT data (all) must be absolute, default chwdir", default=os.getcwd())
    parser.add_argument("--frozen", action='store_true', help="has frozen atoms, default: False", default=False)
    parser.add_argument("--plotonly", action='store_true', help="plot only, default: False", default=False)
    parser.add_argument("--show", action='store_true', help="show plots, default: False", default=False)
    args = parser.parse_args()
    print args

    fname = args.fname
    fdir = args.fdir
    wdir = args.workdir
    assert(os.path.isabs(wdir))

    if not os.path.isabs(fdir):
        fdir = os.path.join(wdir,fdir + fname)

    sim = _collect_u2_vs_k()

    if (fname != None):
        sim(fname=fname, explore_dir=fdir, frozen=args.frozen, plot_only=args.plotonly, show=args.show, verbose=True)
    else :
        for subdir, dirs, files in os.walk(wdir):
            for dir in dirs:
                if dir is not 'packings' and dir is not 'jammed_packings' and dir is not 'analysis':
                    path = os.path.join(wdir, dir)
                    sim(explore_dir=path, frozen=args.frozen)
