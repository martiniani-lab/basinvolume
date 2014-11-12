from __future__ import division
import numpy as np
import abc
import os
import glob
from pele.potentials import Harmonic
from basinvolume.spheres import Findk_MCrunner
from basinvolume.utils import trymakedir, read_xyzdr, read_xydr, to_string, read_txt
import ConfigParser
from basinvolume.post_processing import F_Basin_From_MC_Data, F_Basin_From_MC_Data_Free_COM, Gauss_Lobatto_abscissas, F_Basin_From_MC_Data__get_free_energy_F0_approx_kmax_displ0
import argparse
from itertools import cycle
try:
    import pylab as plt
except ImportError as err:
    print err
    
class _collect_u2_vs_k(object):
    """
    this is a class that implements _collect_u2_vs_k class 
    *ts_skip number of points skipped when printing time series (every ts_skip)
    """
        
    def __call__(self, ts_skip=5000, fname='explore_bv_jammed_packing0', base_dir='analysis', 
                 explore_dir='explore_bv_', packings_dir='jammed_packings', plot_ts_integrand_data = True):
               
        self.fname = fname
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir
        if not os.path.isabs(explore_dir):
            explore_dir = os.path.join(os.getcwd(),explore_dir+fname)
        self.explore_dir = explore_dir
        self.base_directory = self.explore_dir + '/' + base_dir
        
        self.packing_configpath = os.path.join(packings_dir,'jammed_packings.config')
        self.findk_configpath = os.path.join(self.explore_dir,'findk_'+fname+'.config')  
        self.kmin_configpath = os.path.join(self.explore_dir,'kmin_'+fname+'.config')
        
        self.ts_skip = ts_skip
        self.plot_ts_integrand_data = plot_ts_integrand_data
        self._import_config_files()
        self.run()
    
    def run(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        """
        Full volume computation, assuming that PT data is available
        """
        try:
            self._import_ks()
            self._import_u2_reverse()
            self._print_u2_vs_k()
            self._compute_volume()
            self._import_time_series()
            self._plot_data()
        except IOError as err:
            print err
        """
        Volume compuation based on ingregral approximation with kmax and displ_k0 
        """
        try:
            self._compute_approx_volume()
            self._compute_PTu2k0_approx_volume()
        except IOError as err:
            print err
        """
        Print basin volumes for further processing
        """
        self._print_volumes()
    
    def _import_config_files(self):
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.packing_configpath))
        self.nparticles = configf.getint('JAMMED_PACKING','nparticles')
        self.bdim = configf.getint('JAMMED_PACKING','boxdim')
        assert self.bdim==2 or self.bdim==3, "bdim={} not implemented".format(self.bdim)
        self.ndim = self.nparticles * self.bdim
        boxv = configf.get('JAMMED_PACKING','boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        self.imp_packing_frac = configf.getfloat('JAMMED_PACKING','packing_fraction')
        self.sca = configf.getfloat('JAMMED_PACKING','sca')
        configf.read(str(self.findk_configpath))
        self.kmax = configf.getfloat('FINDK','kmax')
        self.prob_kmax = configf.getfloat('FINDK','prob')
        self.displ_k_max = configf.getfloat('FINDK','displ_k_max')
        self.var_displ_k_max = configf.getfloat('FINDK','var_displ_k_max')
        configf.read(str(self.kmin_configpath))
        self.kmin = configf.getfloat('KMIN_MCRUNNER','k')
        self.displ_k_min = configf.getfloat('KMIN','displ_k_min')
        self.var_displ_k_min = configf.getfloat('KMIN','var_displ_k_min')
    
    def _import_ks(self):
        """
        must run before import u2
        """
        karray = [] 
        path = os.path.join(self.explore_dir,'temperatures')
        f = open(path, "r")
        while True:
            k = f.readline()
            if not k: break
            karray.extend([float(k)])
        #prepend kmax
        karray.insert(0,self.kmax)
        self.karray = np.array(karray[::-1],dtype='d')

    def _import_u2_reverse(self):
        n = len(self.karray)-1
        self.u2_array = [0 for _ in xrange(n)]
        self.var_array = [0 for _ in xrange(n)] 
        for subdir, dirs, files in os.walk(self.explore_dir):
            for dir in dirs:
                if dir.isdigit():
                    path = os.path.join(self.explore_dir,dir+'/hist_mean')
                    fileHandle = open (path,"r")
                    lineList = fileHandle.readlines()
                    fileHandle.close()
                    niter, u2, var, std_err = lineList[-1].split()
                    self.u2_array[int(dir)] = u2
                    self.var_array[int(dir)] = var
        #prepend u2 kmax
        self.u2_array.insert(0,self.displ_k_max)
        self.var_array.insert(0,self.var_displ_k_max)
        self.u2_array = np.array(self.u2_array[::-1],dtype='d')
        self.var_array = np.array(self.var_array[::-1],dtype='d')
        
    def _import_time_series(self):
        timeseries = []
        series_order = []
        for subdir, dirs, files in os.walk(self.explore_dir):
            for dir in dirs:
                if dir.isdigit():
                    path = os.path.join(self.explore_dir,dir)
                    file_list = glob.glob(path + '/TimeSeries*')
                    file_list = sorted(file_list, key = lambda x: int(x.split(".")[1]))
                    series_order.append(int(dir))
                    series = []
                    for series_path in file_list:
                        series.extend(read_txt(series_path))
                    timeseries.append(series)
        X = np.array(timeseries)
        Y = series_order
        self.timeseries = np.array([x for (y,x) in sorted(zip(Y,X))])
                    
    def _print_u2_vs_k(self):
        """writes <u2> and variance vs """
        dname = 'u2_vs_k'
        fname = '{}/{}'.format(self.base_directory,dname)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#{:>15}\t{:>15}\n'.format('<u2>','var(<u2>)'))
        for i in xrange(len(self.u2_array)):
            f.write('{:>15.15e}\t{:>15.15e}\n'.format(self.u2_array[i],self.var_array[i])) 
        f.close()
        
    def _compute_volume(self):
        """
        numerical volume obtained by integrating over the PT data
        """
        
        self.F0, self.sigF0, self.farray, self.sigfarray = F_Basin_From_MC_Data(self.bdim, self.nparticles, self.karray,\
                                                                                self.u2_array, np.prod(self.boxv),\
                                                                                self.prob_kmax, displ_k_min_trafo=self.displ_k_min).get_free_energy_F0(self.var_array)
        
        self.F0unc, self.sigF0unc, self.farrayunc, self.sigfarrayunc= F_Basin_From_MC_Data_Free_COM(self.bdim, self.nparticles, self.karray,\
                                                                                self.u2_array, np.prod(self.boxv),\
                                                                                self.prob_kmax, displ_k_min_trafo=self.displ_k_min).get_free_energy_F0(self.var_array)
        self.tarray = Gauss_Lobatto_abscissas(len(self.u2_array))()
        self.unit_box_F0 = self.F0 + self.nparticles * np.log(np.prod(self.boxv))
        self.unit_box_F0unc = self.F0unc + self.nparticles * np.log(np.prod(self.boxv))
        print 'unit_box_F0 {} unit_box_F0unc {}'.format(self.unit_box_F0, self.unit_box_F0unc)
        
    def _compute_approx_volume(self):
        """
        numerical volume obtained by approximating from kmax, and displ_k0
        """
        self.displ_k_min_error = np.sqrt(self.var_displ_k_min)
        self.F0_approx, self.F0_approx_error = F_Basin_From_MC_Data__get_free_energy_F0_approx_kmax_displ0(self.displ_k_min, self.displ_k_min_error, self.kmax, np.prod(self.boxv), self.nparticles, self.bdim, self.prob_kmax)
        self.unit_box_F0_approx = self.F0_approx + self.nparticles * np.log(np.prod(self.boxv))
        print 'unit_box_F0_approx {}'.format(self.unit_box_F0_approx)
        
    def _compute_PTu2k0_approx_volume(self):
        """
        numerical volume obtained by approximating from kmax, and displ_k0, but using the displ_k0 as obtained from PT runs
        """
        self.PTu2k0 = self.u2_array[0]
        self.PTu2k0_error = np.sqrt(self.var_array[0])
        if np.amax(self.u2_array) > self.u2_array[0]:
            raise Exception("_compute_PTu2k0_approx_volume: displacement-squared array is messed up")
        self.F0_approx_PTu2k0, self.F0_approx_PTu2k0_error = F_Basin_From_MC_Data__get_free_energy_F0_approx_kmax_displ0(self.PTu2k0, self.PTu2k0_error, self.kmax, np.prod(self.boxv), self.nparticles, self.bdim, self.prob_kmax)
        self.unit_box_F0_approx_PTu2k0 = self.F0_approx_PTu2k0 + self.nparticles * np.log(np.prod(self.boxv))
        print "unit_box_F0_approx_PTu2k0 {}".format(self.unit_box_F0_approx_PTu2k0)

    def _plot_data(self):
        if self.plot_ts_integrand_data is False:
            return
        lines = ["-","--","-."]
        linecycler = cycle(lines)
        
        cont_karray = np.linspace(self.kmin, self.kmax, 100)
        u2_array_app = (cont_karray + (self.nparticles*self.bdim)/self.displ_k_min) / (self.nparticles*self.bdim)
        u2_array_app = 1.0/u2_array_app
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #timeseries
        for i,series in enumerate(self.timeseries):
            ax.plot(series[::self.ts_skip],ls=next(linecycler),linewidth=1,label=str(i))
        #plt.yscale('symlog')
        ax.legend(frameon=False,loc=1)
        plt.savefig(self.base_directory+'/time_series.eps')
        plt.show()
        #integrand
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(self.tarray,self.farray,yerr=self.sigfarray)
        ax.set_xlabel('t')
        ax.set_ylabel('integrand')
        plt.savefig(self.base_directory+'/integrand.eps')
        plt.show()
        #plt.figure()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cont_karray,u2_array_app,'-')
        ax.errorbar(self.karray,self.u2_array,yerr=np.sqrt(self.var_array),marker='s',linestyle='')
        ax.set_xlabel('k')
        ax.set_ylabel('<u2>')
        ax.set_ylim(bottom=0)
        #plt.xscale('symlog')
        #plt.yscale('log')
        plt.savefig(self.base_directory+'/u2_vs_k.eps') 
        plt.show()
        
    def _print_volumes(self):
        dname = 'volume_data'
        fname = '{}/{}'.format(self.base_directory,dname)
        f = open(fname,'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        def _to_file(name, value):
            f.write((name + ": {}\n").format(to_string(value)))
        f.write('[VOLUME_APPROXIMATED]\n')
        if hasattr(self, "F0_approx"):
            _to_file("F0_approx", self.F0_approx)
            _to_file("F0_approx_error", self.F0_approx_error)
            _to_file("unit_box_F0_approx", self.unit_box_F0_approx)
        f.write('[VOLUME_FULL_PT]\n')
        if hasattr(self, "F0"):
            _to_file("F0", self.F0)
            _to_file("sigF0", self.sigF0)
            _to_file("unit_box_F0", self.unit_box_F0)
        f.write('[VOLUME_PTU2_APPROXIMATED]\n')
        if hasattr(self, "F0_approx_PTu2k0"):
            _to_file("F0_approx_PTu2k0", self.F0_approx_PTu2k0)
            _to_file("F0_approx_PTu2k0_error", self.F0_approx_PTu2k0_error)
            _to_file("unit_box_F0_approx_PTu2k0", self.unit_box_F0_approx_PTu2k0)
        f.close()
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="analyze PT data from thermodynamic integration")
    #parser.add_argument("nparticles", type=int, help="number of particles")
    parser.add_argument("-f","--fname", type=str, help="specify packing to analyze",default=None)
    parser.add_argument("-d","--fdir", type=str, help="directory containing file, if not absolute path by default: fdir+fname",default='explore_bv_')
    parser.add_argument("-w","--workdir", type=str, help="directory containing PT data (all) must be absolute, default chwdir",default=os.getcwd())
    args = parser.parse_args()
    print args
    
    fname = args.fname
    fdir = args.fdir
    wdir = args.workdir
    assert(os.path.isabs(wdir))
    
    if not os.path.isabs(fdir):
        fdir = os.path.join(wdir,fdir+fname)
    
    sim = _collect_u2_vs_k()
    
    if (fname != None):
        sim(fname=fname,explore_dir=fdir)
    else :
        for subdir, dirs, files in os.walk(wdir):
            for dir in dirs:
                if dir is not 'packings' and dir is not 'jammed_packings' and dir is not 'analysis':
                    path = os.path.join(wdir,dir)
                    sim(explore_dir=path)
                    
            
    
