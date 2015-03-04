from __future__ import division
import numpy as np
import abc
import os
import glob
from basinvolume.utils import trymakedir, read_xyzdr, read_xydr
from basinvolume.utils import to_string, read_txt, volume_nball, surface_nball
import ConfigParser
from basinvolume.post_processing import F_Basin_From_MC_Data
from basinvolume.post_processing import F_Basin_From_MC_Data_Free_COM
from basinvolume.post_processing import Gauss_Lobatto_abscissas
from basinvolume.post_processing import F_Basin_From_MC_Data__get_free_energy_F0_approx_kmax_displ0
from pymbar.timeseries import detectEquilibration_binary_search
import bisect
import argparse
from itertools import cycle
try:
    import pylab as plt
    from histogram_reweighting.wham_potential import WhamPotential
    from histogram_reweighting import wham_utils
    from scipy.integrate import simps
except ImportError as err:
    print err
    
class _wham_compute_dos(object):
    """
    this is a class that implements _wham_compute_dos class 
    """
        
    def __call__(self, fname='jammed_packing0', nbins=500, base_dir='analysis',
                 explore_dir='explore_bv_', packings_dir='jammed_packings', plot_data=True,
                 frozen=False, show=False, verbose=True):
        
        self.fname = fname
        self.nbins = nbins
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(),packings_dir)
        self.packings_dir = packings_dir
        if not os.path.isabs(explore_dir):
            explore_dir = os.path.join(os.getcwd(),explore_dir+fname)
        self.explore_dir = explore_dir
        self.base_directory = self.explore_dir + '/' + base_dir
        self.frozen = frozen
        if not frozen:
            self.packing_configpath = os.path.join(packings_dir, 'jammed_packings.config')
        else:
            self.packing_configpath = os.path.join(packings_dir, fname + '.config')
        self.pt_configpath = os.path.join(self.explore_dir, 'explore_' + fname + '.config')
        self.findk_configpath = os.path.join(self.explore_dir, 'findk_' + fname + '.config')
        self.innersphere_configpath = os.path.join(self.explore_dir, 'innersphere_' + fname + '.config')
        
        self.plot_data = plot_data
        self.show = show
        self.verbose = verbose
        self._import_config_files()
        self.run()
    
    def run(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        """
        Full volume computation, assuming that PT data is available
        """
        #try:
        print "importing k array"
        self._import_ks()
        print "importing time series"
        self._import_pt_time_series()
        print "detecting equilibration point"
        self._find_eqtime()
        print "building histograms"
        self._build_histogram()
        print "unbiasing histograms"
        self._unbias_histogram()
        print "running wham"
        self._wham_dos()
        print "computing volume"
        self._compute_volume()
        print "plotting data"
        self._plot_data()
        #except Exception as err:
        #    print err
        """
        Print basin volumes for further processing
        """
        self._print_volumes()
    
    def _import_config_files(self):
        configf = ConfigParser.ConfigParser()
        configf.read(str(self.packing_configpath))
        self.nparticles = configf.getint('JAMMED_PACKING', 'nparticles')
        self.bdim = configf.getint('JAMMED_PACKING', 'boxdim')
        assert self.bdim==2 or self.bdim==3, "bdim={} not implemented".format(self.bdim)
        self.ndim = self.nparticles * self.bdim
        boxv = configf.get('JAMMED_PACKING', 'boxv')
        self.boxv = np.array([float(x) for x in boxv.split()])
        self.imp_packing_frac = configf.getfloat('JAMMED_PACKING', 'packing_fraction')
        self.sca = configf.getfloat('JAMMED_PACKING', 'sca')
        if self.frozen:
            self.vcavity = configf.getfloat('JAMMED_PACKING', 'vcavity')
        else:
            self.vcavity = np.prod(self.boxv)
        configf.read(str(self.pt_configpath))
        self.adjustf_niter = configf.getfloat('MCRUNNER', 'adjustf_niter')
        configf.read(str(self.findk_configpath))
        self.kmax = configf.getfloat('FINDK', 'kmax')
        self.prob_kmax = configf.getfloat('FINDK', 'prob')
        self.displ_k_max = 0.1 #configf.getfloat('FINDK', 'displ_k_max')
        self.var_displ_k_max = configf.getfloat('FINDK', 'var_displ_k_max') #DEBUG
        configf.read(str(self.innersphere_configpath))
        self.k_innersphere = configf.getfloat('INNERSPHERE_MCRUNNER', 'k')
        self.ndof = (self.nparticles-1)*self.bdim
        
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
        #prepend k innersphere
        karray.insert(0, self.k_innersphere)
        self.karray = np.array(karray)
        
    def _import_pt_time_series(self):
        timeseries = []
        series_order = []
        for subdir, dirs, files in os.walk(self.explore_dir):
            for dir in dirs:
                if dir.isdigit():
                    path = os.path.join(self.explore_dir, dir)
                    file_list = glob.glob(path + '/TimeSeries*')
                    file_list = sorted(file_list, key = lambda x: int(x.split(".")[1]))
                    series_order.append(int(dir))
                    series = []
                    for series_path in file_list:
                        series.extend(read_txt(series_path))
                    timeseries.append(series)
        X = np.array(timeseries)
        Y = series_order
        self.timeseries = np.array([x for (y, x) in sorted(zip(Y, X))])
    
    def _find_eqtime(self, full=False):
        if full:
            eq_time = 0
            for i,ts in enumerate(self.timeseries):
                max_eq_time = ts.size // 2
                time = detectEquilibration_binary_search(ts, bs_nodes=10)[0]
                time = np.amin([max_eq_time, time]) #this should avoid detecting artifacts near the end of the series
                new_eq_time = np.amax([time, self.adjustf_niter]) #guarantees that eq_time is larger than the mcrunner adapted number of steps
                if self.verbose:
                    print "eq_time{}: {}".format(i, new_eq_time)
                #gather values, find largest, then broadcast it
                if new_eq_time > eq_time:
                    eq_time = new_eq_time
        else:
            eq_time = self.adjustf_niter
        self.eq_time = int(eq_time)
    
    def _build_histogram(self):
        ts_sphere = np.genfromtxt(os.path.join(self.explore_dir,"inner_sphere.timeseries"))
        ts_sphere = np.trim_zeros(ts_sphere)                #remove trailing 0s
        self.timeseries = self.timeseries[:,self.eq_time:]  #remove equilibration region from timeseries
        hist_visits = []
        bin_edges = np.linspace(np.amin(np.append(self.timeseries,ts_sphere)), np.amax(np.append(self.timeseries,ts_sphere)), self.nbins+1)
        hist = np.histogram(ts_sphere, bin_edges)[0]
        hist_visits.append(hist)
        for timeseries in self.timeseries:
            hist = np.histogram(timeseries, bin_edges)[0]
            hist_visits.append(hist)
        
        self.hist_visits = np.array(hist_visits)
        self.bin_edges = bin_edges + (bin_edges[1]-bin_edges[0])/2
        
    def _unbias_histogram(self):
        hist_unbiased = np.outer(0.5*self.karray[1:], self.bin_edges[:-1]**2)
        hist_unbiased = np.vstack((((24-1)*3-1)*np.log(self.bin_edges[:-1])+0.5*self.karray[0]*self.bin_edges[:-1]**2, hist_unbiased))
        self.hist_unbiased = hist_unbiased
        assert self.hist_visits.shape == self.hist_unbiased.shape
        assert self.hist_visits.shape[0] == self.karray.size
    
    def _wham_dos(self):
        
        hist_visits, hist_unbiased, karray, bin_edges = self.hist_visits, self.hist_unbiased, self.karray, self.bin_edges
        nreps, nbins = hist_visits.shape
        
        whampot = WhamPotential(hist_visits, hist_unbiased)
        if False:
            X = np.random.rand( nreps + nbins )
        else:
            # estimate an initial guess for the offsets and density of states
            # so the minimizer converges more rapidly
            offsets_estimate, log_dos_estimate = wham_utils.estimate_dos(hist_visits, hist_unbiased)
            X = np.concatenate((offsets_estimate, log_dos_estimate))
        
        E0, grad = whampot.getEnergyGradient(X)
        rms0 = np.linalg.norm(grad) / np.sqrt(grad.size)
        
        try:
            from pele.optimize import lbfgs_cpp as quench
            if True:
                print "minimizing with pele lbfgs"
            ret = quench(X, whampot, tol=1e-5, maxstep=1e4, nsteps=10000, iprint=0)
        except ImportError:
            from wham_utils import lbfgs_scipy
            if True:
                print "minimizing with scipy lbfgs"
            ret = lbfgs_scipy(X, whampot, tol=1e-5, nsteps=10000)
        
        if self.verbose:
            print "chi^2 went from %g (rms %g) to %g (rms %g) in %d iterations" % (
                E0, rms0, ret.energy, ret.rms, ret.nfev)
        
        X = ret.coords
        self.logn_E = np.array(X[nreps:])
        self.w_i_final = np.array(X[:nreps])
        
    
    def _compute_volume(self):
        """
        numerical volume obtained by integrating over the PT data
        """
        
        logn_E = self.logn_E - np.amax(self.logn_E)
        self.dos = np.exp(logn_E)
        dx = self.bin_edges[1] - self.bin_edges[0]
        assert (dx - (self.bin_edges[-1] - self.bin_edges[-2])) < 1e-14
        
        bin_edges = self.bin_edges
        s = np.ceil(self.displ_k_max / dx)
        self.rmin = (s-1)*dx + bin_edges[0]
        vmin = volume_nball(self.rmin, self.ndof)
        A = vmin / simps(self.dos[:s], dx=dx)
        Vol = A*simps(self.dos, dx=dx) + volume_nball(bin_edges[0], self.ndof) #add the contribution from 0 to the first bin
        if self.verbose:
            print "A: {} Vol: {}".format(A, Vol)
            print "F0: {}".format(-np.log(Vol) - np.log(self.vcavity))
        
        #print "const simpson", simps(np.ones(10), dx=dx) - dx*9
    
        #currently don't have an estimate for the error
        self.F0, self.sigF0 = -np.log(Vol) - np.log(self.vcavity), 0
        self.F0unc, self.sigF0unc = -np.log(Vol), 0
        
        self.unit_box_F0 = self.F0 + self.nparticles * np.log(self.vcavity)
        self.unit_box_F0unc = self.F0unc + self.nparticles * np.log(self.vcavity)
        
        if self.verbose:
            print 'unit_box_F0 {} unit_box_F0unc {}'.format(self.unit_box_F0, self.unit_box_F0unc)

    def _plot_data(self):
        if self.plot_data is False:
            return
        lines = ["-", "--", "-."]
        linecycler = cycle(lines)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.bin_edges[:-1], self.hist_visits.transpose(), linewidth=2)
        plt.savefig(self.base_directory + '/histograms.eps')
        if self.show:
            plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in xrange(len(self.karray)):
            y = np.log(self.hist_visits[i,:]) + self.hist_unbiased[i,:] + self.w_i_final[i]
            ax.plot(self.bin_edges[:-1], y, linewidth=2, label=str(i))
        ax.set_xlabel(r'$\Delta r$')
        ax.set_ylabel('WHAM')
        plt.savefig(self.base_directory + '/wham.eps')
        if self.show:
            plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.bin_edges[:-1], self.logn_E, label=r'$\log(n_E)$')
        #dx = self.bin_edges[1] - self.bin_edges[0]
        rg = self.logn_E - (self.ndof-1)*np.log(self.bin_edges[:-1])
        ax.plot(self.bin_edges[:-1], rg, label=r'$\log(g(r)/r^{N-1})$')
        ax.set_xlabel(r'$\Delta r$')
        ax.legend(frameon=False, loc="best")
        plt.ylim((np.amin(rg),1.1*np.amax(rg)))
        plt.savefig(self.base_directory + '/log_dos.eps')
        if self.show:
            plt.show()
        
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.plot(1./self.bin_edges[:-1], self.logn_E, label=r'$\log(n_E)$')
#        ax.plot(1./self.bin_edges[:-1], rg, label=r'$\log(g(r)/r^{N-1})$')
#        ax.set_xlabel(r'1/$\Delta r$')
#        ax.legend(frameon=False, loc="best")
#        plt.xlim((0,1))
#        plt.savefig(self.base_directory + '/rec_log_dos.eps')
#        if self.show:
#            plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.bin_edges[:-1], self.dos)
        ax.set_xlabel(r'$\Delta r$')
        ax.set_ylabel('DOS')
        plt.savefig(self.base_directory + '/dos.eps')
        if self.show:
            plt.show()
        
    def _print_volumes(self):
        dname = 'dos_volume_data'
        fname = '{}/{}'.format(self.base_directory,dname)
        f = open(fname, 'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        def _to_file(name, value):
            f.write((name + ": {}\n").format(to_string(value)))
        f.write('[VOLUME_DOS]\n')
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
    args = parser.parse_args()
    print args
    
    fname = args.fname
    fdir = args.fdir
    wdir = args.workdir
    assert(os.path.isabs(wdir))
    
    if not os.path.isabs(fdir):
        fdir = os.path.join(wdir,fdir + fname)
    
    sim = _wham_compute_dos()
    
    if (fname != None):
        sim(fname=fname, explore_dir=fdir, frozen=args.frozen, show=False)
    else :
        for subdir, dirs, files in os.walk(wdir):
            for dir in dirs:
                if dir is not 'packings' and dir is not 'jammed_packings' and dir is not 'analysis':
                    path = os.path.join(wdir, dir)
                    sim(explore_dir=path, frozen=args.frozen)
                    
            
    
