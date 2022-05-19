from __future__ import division
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import zip
from builtins import str
from builtins import next
from builtins import range
from builtins import object
import numpy as np
import os
import re
import argparse
import time
import logging
from itertools import cycle
try:
    import matplotlib
    matplotlib.use('Agg') # matplotlib.use('Agg', warn=False)
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from joblib import Parallel, delayed
except ImportError as err:
    print(err)
from scipy.integrate import simps
from basinvolume.utils import trymakedir
from basinvolume.utils import to_string, log_volume_nball, surface_nball, write_csv_xy, import_pt_time_series
from basinvolume.post_processing import VolumeSanityCheck
from configparser import ConfigParser
# import configparser
from pymbar.timeseries import detectEquilibration_binary_search, subsampleCorrelatedData, statisticalInefficiency_fft
from pymbar.mbar import MBAR
from basinvolume.experiment_2d.cross_validation_bandwidth_selection import get_bandwidth_estimate, get_pdf
from basinvolume.spheres import read_jammed_packing_config
#######################SET LATEX OPTIONS###################
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
#rc('text.latex',preamble=r'\usepackage{times}')
plt.rcParams.update({'font.size': 28})
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['figure.figsize'] = 10, 7.7
##########################################################
####SET COLOUR MAP######
def get_color_cycle(ncol=7):
    cm = plt.get_cmap('Set2')
    color_cycle=cycle([cm(1. * i / ncol) for i in range(ncol)])
    return color_cycle
########################
#####################LINE STYLE CYCLER####################
lines = ["-","--","-.", ":", "_"]
linecycler = cycle(lines)
##########################################################

def dos_from_offsets(visits, log_dos_all, offsets, nodata_value=0.):
    log_dos_all = log_dos_all + offsets[:,np.newaxis]

    ldos = np.sum(log_dos_all * visits, axis=0)
    norm = visits.sum(0)
    ldos = np.where(norm > 0, ldos / norm, nodata_value)
    return ldos

def get_kde_hist(timeseries, bin_edges, kernel="gaussian", bw=0.02, method="cross_validation", skip=1):
    if kernel == "gaussian":
        if method == "cross_validation":
            skip = max(1, int(len(timeseries) / 1e5))
        bw = get_bandwidth_estimate(np.array(timeseries[::skip]), kernel="gaussian", method=method)
        logging.info("bandwidth {}".format(bw))
        #bw *= 3
    hist = get_pdf(timeseries, bin_edges, bandwidth=bw, kernel=kernel)
    area = simps(hist, bin_edges)
    logging.info("kde pdf area {}".format(area))
#    normalise again by hand
#    hist /= area
#    area = simps(hist, bin_edges)
#    logging.info("kde pdf area {}".format(area))
    return hist

def find_eqtime(ts):
    max_eq_time = ts.size // 2
    time = detectEquilibration_binary_search(ts, bs_nodes=30)[0]
    time = np.amin([max_eq_time, time]) #this should avoid detecting artifacts near the end of the series
    return int(time)

#def run_parallel(func, *args):


class mbar_compute_dos(object):
    """
    this is a class that implements _mbar_compute_dos class
    """
    def __init__(self, nbins=1000, bootstrap=False, kde=True, plot_dos_data=True, ncores=7):
        self.nbins = np.power(2, int(np.log2(nbins) + 0.5)) + 1#approximate to nearest power of 2 plus 1 (for rhomb integration)
        self.kde = kde
        self.bootstrap = bootstrap
        self.plot_dos_data = plot_dos_data
        self.ncores = ncores

    def __call__(self, fname='jammed_packing0', base_dir='analysis',
                 explore_dir='explore_bv_jammed_packing', packings_dir='packings',
                 jammed_packings_dir='jammed_packings', frozen=False, show=False,
                 verbose=True):
        self.fname = fname
        if not os.path.isabs(packings_dir):
            packings_dir = os.path.join(os.getcwd(), packings_dir)
        self.packings_dir = packings_dir
        if not os.path.isabs(jammed_packings_dir):
            jammed_packings_dir = os.path.join(os.getcwd(),jammed_packings_dir)
        self.jammed_packings_dir = jammed_packings_dir
        if not os.path.isabs(explore_dir):
            number = int(re.findall('\d+', self.fname)[0])
            explore_dir = os.path.join(os.getcwd(),explore_dir+number)
        self.explore_dir = explore_dir
        self.base_directory = os.path.join(self.explore_dir, base_dir)
        self.frozen = frozen
        n = int(re.findall(r'\d+', self.fname)[0])
        self.packing_configpath = os.path.join(packings_dir, 'packing{}.config'.format(n))
        assert os.path.isfile(self.packing_configpath)
        self.jammed_packing_configpath = os.path.join(jammed_packings_dir, '{}.config'.format(self.fname))
        assert os.path.isfile(self.jammed_packing_configpath)
        self.pt_configpath = os.path.join(self.explore_dir, 'explore_' + fname + '.config')
        assert os.path.isfile(self.pt_configpath)
        self.findk_configpath = os.path.join(self.explore_dir, 'findk_' + fname + '.config')
        assert os.path.isfile(self.findk_configpath)
        self.kmin_configpath = os.path.join(self.explore_dir, 'kmin_' + fname + '.config')
        assert os.path.isfile(self.kmin_configpath)
        self.innersphere_configpath = os.path.join(self.explore_dir, 'innersphere_' + fname + '.config')
        assert os.path.isfile(self.innersphere_configpath)

        self.show = show
        self.verbose = verbose
        self._import_config_files()
        if not self.bootstrap:
            self.run()
        else:
            self.run_bs()

    def run(self):
        """
        Full volume computation, assuming that PT data is available
        """
        base_directory = self.base_directory
        logging.info("Analysing {}".format(self.explore_dir))
        trymakedir(base_directory)
        logging.info("importing k array")
        self._import_ks()
        logging.info("importing time series")
        self._import_pt_time_series()
        logging.info("subtracting equilibration point")
        self._subtract_eqtime()
        logging.info("importing innersphere time series")
        self._import_ts_sphere()
        logging.info("subsampling time series")
        self._build_flat_timeseries()
        logging.info("building mbar")
        self._build_mbar()
        logging.info("mbar computing volume")
        self._mbar_compute_volume()
        self._compute_hs_fluid_volume()
        self._print_volumes()
        if self.plot_dos_data:
            logging.info("plotting data")
            self._build_histogram(kde=self.kde)
            self._compute_dos()
            self._plot_dos_data()
            #self._pmf()

    def run_bs(self, nr_subsamples=10):
        base_directory = self.base_directory
        logging.info("Analysing {}".format(self.explore_dir))
        trymakedir(base_directory)
        #first compute the volume using the full set of data
        self._import_ks()
        logging.info("importing time series")
        self._import_pt_time_series()
        logging.info("subtracting equilibration point")
        self._subtract_eqtime()
        logging.info("importing innersphere time series")
        self._import_ts_sphere()
        logging.info("subsampling time series")
        self._build_flat_timeseries()
        logging.info("building mbar")
        self._build_mbar()
        logging.info("mbar computing volume")
        self._mbar_compute_volume()
        self._compute_hs_fluid_volume()
        self._print_volumes()
        logging.info("plotting data all")
        if self.plot_dos_data:
            self._build_histogram(kde=self.kde)
            self._compute_dos()
            #now bootstrap timeseries to compute error bars on dos
            #the timeseries after find_eqtime has already discared the burn out region
            full_flat_timeseries = np.copy(self.flat_timeseries)
            self.logn_E_subs = self.logn_E.copy()
            initial_f_k = np.array(self.mbar.f_k)
            for iter in range(nr_subsamples):
                logging.info("subsampling - iteration {}".format(iter))
                j = 0
                for n_k in self.N_k:
                    idx = np.random.randint(0, n_k, size=n_k)
                    ts = full_flat_timeseries[j:j+n_k]
                    self.flat_timeseries[j:j+n_k] = ts[idx]
                    j+=n_k
                start = time.time()
                #self._build_mbar(verbose=False, maxiter=1000, reltol=1.0e-7, initial_f_k=initial_f_k, subsampling=16)
                self._build_mbar(verbose=False, initial_f_k=initial_f_k)
                logging.info("t: {}".format(time.time() - start))
                #compute the weights, skip the volume calculation
                Deltaf_ij, dDeltaf_ij, Theta_ij = self.mbar.getFreeEnergyDifferences(return_theta=True)
                self.w_i_final = -Deltaf_ij[0]
                #now build histogram and compute dos
                self._build_histogram(compute_binedges=False, kde=self.kde)
                self._compute_dos()
                self.logn_E_subs = np.vstack((self.logn_E_subs, self.logn_E))
            #plot data
            self._plot_dos_data()

    def _import_config_files(self):
        imp_packing = read_jammed_packing_config(str(self.jammed_packing_configpath))
        self.nparticles = imp_packing['nparticles']
        self.bdim = imp_packing['bdim']
        assert self.bdim==2 or self.bdim==3, "bdim={} not implemented".format(self.bdim)
        self.ndim = imp_packing['ndim']
        self.boxv = imp_packing['boxv'].copy()
        self.packing_frac = imp_packing['packing_frac']
        self.vcavity = imp_packing['vcavity']
        self.sca = imp_packing['sca']
        configf = ConfigParser() # configf = configparser.ConfigParser()
        configf.read(str(self.pt_configpath))
        self.adjustf_niter = configf.getint('MCRUNNER', 'adjustf_niter')
        configf.read(str(self.findk_configpath))
        self.kmax = configf.getfloat('FINDK', 'kmax')
        self.prob_kmax = configf.getfloat('FINDK', 'prob')
        configf.read(str(self.innersphere_configpath))
        self.k_innersphere = configf.getfloat('INNERSPHERE_MCRUNNER', 'k')
        self.ndof = (self.nparticles-1)*self.bdim
        self.ref_radius = configf.getfloat('INNERSPHERE_BALLPICK_MCRUNNER_STATUS', 'stepsize')
        self.ref_acceptance = configf.getfloat('INNERSPHERE_BALLPICK_MCRUNNER_STATUS', 'acc_frac')

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
        self.k0_index = np.where(self.karray==0.)[0][0]

    def _import_pt_time_series(self):
        self.timeseries = import_pt_time_series(self.explore_dir, self.adjustf_niter,
                                                max_series_size=int(1e6), ncores=self.ncores,
                                                crop_adjustf_niter=True, del_raw=False)

    def _subtract_eqtime(self):
        #remove equilibration region from pt timeseries
        results = Parallel(n_jobs=max(1,self.ncores))(delayed(find_eqtime)(timeseries) for timeseries in self.timeseries)
        logging.info("eq_times: {}".format(results))
        eq_time = int(np.amax(results))
        self.timeseries = self.timeseries[:,eq_time:]

    def _build_u_kn(self, flat_timeseries):
        K, N = self.karray.size, flat_timeseries.size
        u_kn = np.empty((K, N))

        for i in range(K):
            if i == 0:
                u_kn[i] = (self.ndof-1)*np.log(flat_timeseries)+0.5*self.karray[i]*flat_timeseries**2
            else:
                u_kn[i] = 0.5 * self.karray[i] * flat_timeseries**2
        assert self.karray.size == u_kn.shape[0]
        assert N == u_kn.shape[1]
        return u_kn

    def _subsample_timeseries(self, ts_sphere, timeseries):
        """
        returns a flatten timeseries of the uncorrelated data
        """
        K = self.karray.size
        g = np.ones(K)
        N_k = np.zeros(K, dtype='i')
        flat_ts = np.empty(0)

        #deal with ts separately
        g[0] = statisticalInefficiency_fft(ts_sphere)
        indices = np.array(subsampleCorrelatedData(ts_sphere, g=g[0])) # indices of uncorrelated samples
        N_k[0] = len(indices) # number of uncorrelated samples
        flat_ts = np.append(flat_ts, ts_sphere[indices])
        #now loop through pt timeseries
        print(K)
        print(len(timeseries))
        for i in range(K-1):  #subsample the energies
            j = i+1
            print("i = {0}, j = {1}\n".format(i,j))
            g[j] = statisticalInefficiency_fft(timeseries[i])
            indices = np.array(subsampleCorrelatedData(timeseries[i], g=g[j])) # indices of uncorrelated samples
            N_k[j] = len(indices) # number of uncorrelated samples
            flat_ts = np.append(flat_ts, timeseries[i,indices])
        return flat_ts, N_k, g

    def _import_ts_sphere(self):
        self.ts_sphere = np.genfromtxt(os.path.join(self.explore_dir,"inner_sphere.timeseries"))
        self.ts_sphere = np.trim_zeros(self.ts_sphere)

    def _build_flat_timeseries(self):
        self.flat_timeseries, self.N_k, g = self._subsample_timeseries(self.ts_sphere, self.timeseries)

    def _build_mbar(self, verbose=True, initial_f_k=None, maxiter=10000, reltol=1.0e-7): #subsampling=6 no longer supported
        self.u_kn = self._build_u_kn(self.flat_timeseries)
        self.mbar = MBAR(self.u_kn, self.N_k, maximum_iterations=maxiter, relative_tolerance=reltol,
                         initial_f_k=initial_f_k, initialize='BAR', verbose=verbose) # subsampling=subsampling no longer supported

    def _mbar_compute_volume(self):
        Deltaf_ij, dDeltaf_ij, Theta_ij = self.mbar.getFreeEnergyDifferences(return_theta=True) #Default is now false for return_theta
        self.w_i_final = -Deltaf_ij[0] #the free energy differences are nothing but the log weights that one would compute from wham
        # logging.info("effective sample number {}".format(self.mbar.computeEffectiveSampleNumber()))

        rmin = self.ref_radius # 1./np.sqrt(self.kmax) #we choose rmin to be 1/sqrt(k_max)
        logging.info("kmax {}".format(self.kmax))
        logging.info("rmin {}".format(rmin))
        logging.info("ref acceptance {}".format(self.ref_acceptance))
        logvmin = log_volume_nball(rmin, self.ndof)
        Fmin = -logvmin - np.log(self.ref_acceptance)

        u_lk = np.copy(self.u_kn[self.k0_index])
        r = self.flat_timeseries
        LARGE = 1e70
        u_lk = np.where(r < rmin, u_lk, LARGE)
        u_lk = np.reshape(u_lk, (1, u_lk.size))
        u_lk = np.vstack((u_lk, self.u_kn[self.k0_index])) #measure free energy difference between k=0 and kw
        Deltaf_ij, dDeltaf_ij = self.mbar.computePerturbedFreeEnergies(u_lk)
        #vol = Deltaf_ij[1,0]
        self.F0, self.sigF0 = (Fmin - Deltaf_ij[1,0]) - np.log(self.vcavity), dDeltaf_ij[1,0]
        self.F0unc, self.sigF0unc = (Fmin - Deltaf_ij[1,0]), dDeltaf_ij[1,0]

        self.unit_box_F0 = self.F0 + self.nparticles * np.log(self.vcavity)
        self.unit_box_F0unc = self.F0unc + self.nparticles * np.log(self.vcavity)

        if self.verbose:
            logging.info('F0 {} F0unc {} +/- {}'.format(self.F0, self.F0unc, self.sigF0))
            logging.info('unit_box_F0 {} unit_box_F0unc {} +/- {}'.format(self.unit_box_F0, self.unit_box_F0unc, self.sigF0))

    def _compute_hs_fluid_volume(self, numerical_moments=False):
        volume_sanity_check = VolumeSanityCheck(self.packing_configpath,
                                                numerical_moments=numerical_moments)
        self.F0_acc = volume_sanity_check.F0_acc
        self.ideal_gas_F_acc = - self.nparticles*np.log(self.vcavity)

    def _build_histogram(self, compute_binedges=True, kde=False):
        """
        set compute bin_edges to false when subsampling so that all subsamples have the same number of bins over the same range
        """
        if not compute_binedges:
            assert self.bootstrap
        if compute_binedges:
            bin_edges = np.linspace(np.amin(np.append(self.timeseries, self.ts_sphere)),
                                    np.amax(np.append(self.timeseries, self.ts_sphere)), self.nbins+1)
        else:
            bin_edges = self.bin_edges - (self.bin_edges[1]-self.bin_edges[0])/2

        if kde:
            hist_visits = self._build_histogram_kde(bin_edges)
        else:
            hist_visits = self._build_histogram_simple(bin_edges)

        self.hist_visits = np.array(hist_visits)
        self.bin_edges = bin_edges + (bin_edges[1]-bin_edges[0])/2 #shift bin edges by bin/2
        self._unbias_histogram()

    def _build_histogram_simple(self, bin_edges):
        hist_visits = []
        hist = np.histogram(self.ts_sphere, bin_edges, density=True)[0]
        hist_visits.append(hist)
        results = Parallel(n_jobs=self.ncores)(delayed(np.histogram)(timeseries, bin_edges, density=True) for timeseries in self.timeseries)
        for hist in results:
            hist_visits.append(hist[0])
        return hist_visits

    def _build_histogram_kde(self, bin_edges):
        hist_visits = []
        #hist = np.histogram(self.ts_sphere, bin_edges, normed=True)[0]
        kde_bin_edges = np.array(bin_edges[:-1])
        kde_bin_edges += (kde_bin_edges[1]-kde_bin_edges[0])/2
        hist = get_kde_hist(self.ts_sphere, kde_bin_edges, kernel="epanechnikov", bw=0.001)
        hist_visits.append(hist)
        results = Parallel(n_jobs=max(1,self.ncores))(delayed(get_kde_hist)(timeseries, kde_bin_edges) for timeseries in self.timeseries)
        logging.info(np.shape(results))
        for hist in results:
            hist_visits.append(hist)
        logging.info(np.shape(hist_visits))
        return hist_visits

    def _unbias_histogram(self):
        hist_unbiased = np.outer(0.5*self.karray[1:], self.bin_edges[:-1]**2)
        hist_unbiased = np.vstack(((self.ndof-1)*np.log(self.bin_edges[:-1])+0.5*self.karray[0]*self.bin_edges[:-1]**2, hist_unbiased))
        self.hist_unbiased = hist_unbiased
        assert self.hist_visits.shape == self.hist_unbiased.shape
        assert self.hist_visits.shape[0] == self.karray.size

    def _compute_dos(self):
        """
        compute the dos from the log weights obtained by mbar
        """

        hist_visits, hist_unbiased, karray, bin_edges = self.hist_visits, self.hist_unbiased, self.karray, self.bin_edges
        nreps, nbins = hist_visits.shape
        SMALL = 0.
        log_dos = np.where(self.hist_visits==0, SMALL, np.log(hist_visits) + hist_unbiased) #sends back a warning
        ldos = dos_from_offsets(self.hist_visits, log_dos, self.w_i_final)
        self.logn_E = np.array(ldos)

    def _plot_dos_data(self):
        self._plot_raw()
        if self.bootstrap:
            self._plot_dos_bs()
        else:
            self._plot_dos()
        plt.close('all')

    def _plot_raw(self):
        lines = ["-", "--", "-."]
        linecycler = cycle(lines)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        color_cycle = get_color_cycle()
        skip = max(1, int(len(self.timeseries[0])/1e3))
        for i,series in enumerate(self.timeseries):
            ax.plot(series[series>0].flatten()[::skip], ls=next(linecycler),
                    color=next(color_cycle), linewidth=1.8, label=str(i))
        ax.set_ylabel(r'$r$', fontsize=28)
        ax.set_xlabel('steps/{}'.format(skip), fontsize=28)
        ax.set_xlim((0,150))
        #plt.yscale('log')
        #plt.xscale('log')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], frameon=False, loc='best', prop={'size':18}, numpoints=1, scatterpoints=1,
                  markerscale=1, columnspacing=0.25, labelspacing=0.25, handletextpad=0.1, handlelength=1)
        plt.savefig(self.base_directory + '/time_series.pdf')
        if self.show:
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        color_cycle = get_color_cycle(ncol=32)
        herr = []
        for i,hist in enumerate(self.hist_visits):
            var = (1-hist/np.sum(hist))*hist/np.sum(hist)
            herr.append(np.sqrt(var)/(np.amax(self.bin_edges)-np.amin(self.bin_edges)))
        herr = np.array(herr)
        for i, (hist, err) in enumerate(zip(self.hist_visits, herr)):
            ax.errorbar(self.bin_edges[:-1], hist, linewidth=2, color=next(color_cycle),
                        label='{:.1f}'.format(self.karray[i])) #yerr=err
        ax.legend(frameon=False, loc="best", prop={'size':17}, numpoints=1, scatterpoints=1,
                  markerscale=1, columnspacing=0.25, labelspacing=0.25, handletextpad=0.1, handlelength=1, ncol=2)
        ax.set_xlabel(r'$r$', fontsize=28)
        plt.savefig(self.base_directory + '/histograms.eps')
        if self.show:
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(self.karray)):
            SMALL = 0.
            y = np.where(self.hist_visits[i, :]==0, SMALL, np.log(self.hist_visits[i,:]) + self.hist_unbiased[i,:] + self.w_i_final[i]) #sends back a warning
            # y = np.log(self.hist_visits[i,:]) + self.hist_unbiased[i,:] + self.w_i_final[i]
            ax.plot(self.bin_edges[:-1], y, linewidth=2, label=str(i))
        ax.set_xlabel(r'$\Delta r$')
        ax.set_ylabel('logDOS')
        plt.savefig(self.base_directory + '/raw_log_dos.eps')
        if self.show:
            plt.show()

        #plot of the variance of the histograms as a function of k
        fig = plt.figure()
        ax = fig.add_subplot(111)
        var = []
        for ts in self.timeseries:
            var.append(np.var(ts))
        var = np.array(var)
        ax.plot(self.karray[1:], var)
        ax.set_xlabel(r'k')
        ax.set_ylabel('$var(r)$')
        plt.xlim((self.karray[-1],self.karray[1]))
        plt.savefig(self.base_directory + '/hist_var_k.eps')
        write_csv_xy(self.karray[1:], var, fname=os.path.join(self.base_directory, 'hist_var_k.csv'))
        if self.show:
            plt.show()

    def _plot_dos(self):
        lines = ["-", "--", "-."]
        linecycler = cycle(lines)

        #bin_edges=self.bin_edges[:-1]

        finindx = np.where(np.isfinite(self.logn_E))[0]
        logn_E = np.array([self.logn_E[i] for i in finindx])
        bin_edges = np.array([self.bin_edges[i] for i in finindx])

        logn_E = np.array(logn_E - np.amax(logn_E))
        dos = np.exp(logn_E)
        dos /= simps(dos, bin_edges)
        logn_E = np.log(dos)

        color_cycle = get_color_cycle()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(bin_edges, logn_E, label=r'$\log(\xi(r))$', color=next(color_cycle), linewidth=2)
        #dx = self.bin_edges[1] - self.bin_edges[0]
        rg = np.array(logn_E)
        rg -= (self.ndof-1)*np.log(bin_edges)
        rg -= np.mean(rg[:3])
        # logging.info("mean(rg[:3]) = {}".format(np.mean([rg[finindx[0]],rg[finindx[1]],rg[finindx[2]]])))
        assert(abs(np.mean(rg[:3])) < 1e-8)
        ax.plot(bin_edges, rg, label=r'$\log(\xi(r)/r^{N-1})$', color=next(color_cycle), linewidth=2)
        ax.set_xlabel(r'$\Delta r$')
        ax.legend(frameon=False, loc="best")
        plt.ylim((min(np.amin(rg), np.amin(logn_E)), max(np.amax(rg),np.amax(logn_E)) ))
        plt.savefig(self.base_directory + '/log_dos.eps')
        write_csv_xy(bin_edges, logn_E, fname=os.path.join(self.base_directory, 'log_gr.csv'))
        write_csv_xy(bin_edges, rg, fname=os.path.join(self.base_directory, 'log_gr_ratio.csv'))
        if self.show:
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(bin_edges, np.exp(rg)) #-np.amax(rg)
        ax.set_xlabel(r'$\Delta r$')
        ax.set_ylabel(r'$\xi(r)/r^{N-1}$')
        plt.savefig(self.base_directory + '/ratio_g.eps')
        write_csv_xy(bin_edges, np.exp(rg), fname=os.path.join(self.base_directory, 'gr_ratio.csv')) # -np.amax(rg)
        if self.show:
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(bin_edges, rg) # -np.amax(rg)
        ax.set_xlabel(r'$\Delta r$')
        ax.set_ylabel(r'$\log(\xi(r)/r^{N-1})$')
        ax.set_xscale('log')
        plt.savefig(self.base_directory + '/ratio_g_loglog.eps')
        if self.show:
            plt.show()

#        corey's S_n^gamma function
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        A = surface_nball(1.,self.ndof)
#        Vrat = np.exp(self.F0 - self.nparticles*self.bdim*np.log(self.vcavity))
#        ax.plot(bin_edges, np.exp(rg-np.amax(rg)) * (bin_edges**self.ndof)
#        ax.set_xlabel(r'$\Delta r$')
#        ax.set_ylabel(r'$S_n^{\Gamma}$')
#        #ax.set_yscale('log')
#        #ax.set_xscale('log')
#        #plt.savefig(self.base_directory + '/wbp.eps')
#        if self.show:
#            plt.show()

        color_cycle = get_color_cycle()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(bin_edges, dos, color=next(color_cycle), linewidth=2)
        ax.set_xlabel(r'$\Delta r$')
        ax.set_ylabel('DOS')
        plt.savefig(self.base_directory + '/dos.eps')
        write_csv_xy(bin_edges, dos, fname=os.path.join(self.base_directory, 'dos.csv'))
        if self.show:
            plt.show()

    def _plot_dos_bs(self, alpha=0.05):
        lines = ["-", "--", "-."]
        linecycler = cycle(lines)

        dos_subs = np.empty(self.logn_E_subs.shape)
        for i, logn_E in enumerate(self.logn_E_subs):
            #logn_E -= np.amax(logn_E)
            logn_E -= np.median(logn_E)
            self.logn_E_subs[i] = logn_E
            dos_subs[i] = np.exp(logn_E)

        dos = np.mean(dos_subs,axis=0)
        nsamples = dos_subs.shape[0]
        low_dos =  np.sort(dos_subs, axis=0)[(alpha/2.0)*nsamples,:]
        high_dos =  np.sort(dos_subs, axis=0)[(1-alpha/2.0)*nsamples,:]

        #biased estimate of the mean np.mean(self.logn_E_subs,axis=0)
        #unbiased estimate of the mean
        logn_E = np.log(dos)
        nsamples = self.logn_E_subs.shape[0]
        #these are the unbiased estimates of the error because log is a monotonic convex function
        #and the we pick the 2.5 and 97.5 percentiles to have 95% intervals of confidence
        low_logn_E =  np.sort(self.logn_E_subs, axis=0)[(alpha/2.0)*nsamples,:]
        high_logn_E =  np.sort(self.logn_E_subs, axis=0)[(1-alpha/2.0)*nsamples,:]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.bin_edges[:-1], logn_E, label=r'$\log(g(r))$', color='b')
        ax.plot(self.bin_edges[:-1], low_logn_E, color='r')
        ax.plot(self.bin_edges[:-1], high_logn_E, color='r')
        #dx = self.bin_edges[1] - self.bin_edges[0]
        rg = logn_E - (self.ndof-1)*np.log(self.bin_edges[:-1])
        low_rg = low_logn_E - (self.ndof-1)*np.log(self.bin_edges[:-1])
        high_rg = high_logn_E - (self.ndof-1)*np.log(self.bin_edges[:-1])
        ax.plot(self.bin_edges[:-1], rg, label=r'$\log(g(r)/r^{N-1})$', color='g')
        ax.plot(self.bin_edges[:-1], low_rg, color='r')
        ax.plot(self.bin_edges[:-1], high_rg, color='r')
        ax.set_xlabel(r'$\Delta r$')
        ax.legend(frameon=False, loc="best")
        plt.ylim((np.amin(rg),1.1*np.amax(rg)))
        plt.savefig(self.base_directory + '/log_dos_bs.eps')
        #write_csv_xy(self.bin_edges[:-1], logn_E, fname=os.path.join(self.base_directory, 'log_gr_bs.csv'))
        #write_csv_xy(self.bin_edges[:-1], rg, fname=os.path.join(self.base_directory, 'log_gr_ratio_bs.csv'))
        if self.show:
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.bin_edges[:-1], np.exp(rg-np.amax(rg)))
        ax.plot(self.bin_edges[:-1], np.exp(low_rg-np.amax(rg)), color='r')
        ax.plot(self.bin_edges[:-1], np.exp(high_rg-np.amax(rg)), color='r')
        ax.set_xlabel(r'$\Delta r$')
        ax.set_ylabel(r'$g(r)/r^{N-1}$')
        plt.savefig(self.base_directory + '/ratio_g_bs.eps')
        #write_csv_xy(self.bin_edges[:-1], np.exp(rg-np.amax(rg)), fname=os.path.join(self.base_directory, 'gr_ratio.csv'))
        if self.show:
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.bin_edges[:-1], np.exp(rg-np.amax(rg)))
        ax.plot(self.bin_edges[:-1], np.exp(low_rg-np.amax(rg)), color='r')
        ax.plot(self.bin_edges[:-1], np.exp(high_rg-np.amax(rg)), color='r')
        ax.set_xlabel(r'$\Delta r$')
        ax.set_ylabel(r'$\log(g(r)/r^{N-1})$')
        ax.set_xscale('log')
        plt.savefig(self.base_directory + '/ratio_g_loglog_bs.eps')
        if self.show:
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.bin_edges[:-1], dos)
        ax.plot(self.bin_edges[:-1], low_dos, color='r')
        ax.plot(self.bin_edges[:-1], high_dos, color='r')
        ax.set_xlabel(r'$\Delta r$')
        ax.set_ylabel('DOS')
        plt.savefig(self.base_directory + '/dos_bs.eps')
        #write_csv_xy(self.bin_edges[:-1], dos, fname=os.path.join(self.base_directory, 'dos.csv'))
        if self.show:
            plt.show()

    def _print_volumes(self):
        dname = 'mbar_volume_data'
        fname = '{}/{}'.format(self.base_directory,dname)
        f = open(fname, 'w')
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        def _to_file(name, value):
            f.write((name + ": {}\n").format(to_string(value)))
        f.write('[VOLUME_HS_FLUID]\n')
        if hasattr(self, "F0_acc"):
            _to_file("F0_acc", self.F0_acc)
            _to_file("F0_ideal_gas", self.ideal_gas_F_acc)
        f.write('[VOLUME_MBAR]\n')
        if hasattr(self, "F0"):
            _to_file("F0", self.F0)
            _to_file("sigF0", self.sigF0)
            _to_file("unit_box_F0", self.unit_box_F0)
        f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="analyze PT data from thermodynamic integration")
    #parser.add_argument("nparticles", type=int, help="number of particles")
    parser.add_argument("-f","--fname", type=str, help="specify packing to analyze",default=None)
    parser.add_argument("-d","--fdir", type=str, help="Beginning of explore directory containing the results. If not absolute path: fdir+fname", default='explore_bv_')
    parser.add_argument("-w","--workdir", type=str, help="directory containing PT data (all) must be absolute, default chwdir", default=os.getcwd())
    parser.add_argument("--frozen", action='store_true', help="has frozen atoms, default: False", default=False)
    parser.add_argument("--show", action='store_true', help="show plots, default: False", default=False)
    parser.add_argument("--bootstrap", action='store_true', help="run bootstrap (slow!), default: False", default=False)
    parser.add_argument("--kde", action='store_true', help="use kernel density estimate, default: False", default=False)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging.INFO)
    logging.info(args)

    fname = args.fname
    fdir = args.fdir
    wdir = args.workdir
    assert(os.path.isabs(wdir))

    sim = mbar_compute_dos(bootstrap=args.bootstrap, kde=args.kde, plot_dos_data=True)
    if (fname != None):
        if not os.path.isabs(fdir):
            fdir = os.path.join(wdir, fdir + fname)
        sim(fname=fname, explore_dir=fdir, frozen=args.frozen, show=args.show)
    else:
        for dirpath, dirs, files in os.walk(wdir):
            for dir in dirs:
                if dir.startswith(fdir):
                    path = os.path.join(dirpath, dir)
                    sim(fname=dir[len(fdir):], explore_dir=path, frozen=args.frozen)
