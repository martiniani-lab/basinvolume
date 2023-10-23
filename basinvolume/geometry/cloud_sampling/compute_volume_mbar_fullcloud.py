from __future__ import division
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import zip
from builtins import range
import numpy as np
from basinvolume.mbar_spheres import mbar_compute_dos
from basinvolume.mbar_spheres.mbar_compute_volume import find_eqtime
import os
import configparser
import argparse
from basinvolume.utils import import_pt_cloud_drops_time_series_raw
from basinvolume.mbar_spheres.mbar_compute_volume import dos_from_offsets, get_kde_hist
from pymbar.timeseries import detectEquilibration_binary_search, subsampleCorrelatedData, statisticalInefficiency_fft
from joblib import Parallel, delayed
from itertools import chain
from compiler.ast import flatten

class hyperelem_mbar_compute_dos(mbar_compute_dos):
    """
    this is a class that implements _mbar_compute_dos class 
    """
    def __init__(self, nbins=1000, bootstrap=False, kde=True, plot_dos_data=True, ncores=7):
        super(hyperelem_mbar_compute_dos, self).__init__(nbins=nbins, bootstrap=bootstrap,
                                                   kde=kde, plot_dos_data=plot_dos_data, 
                                                   ncores=ncores)
    
    def __call__(self, explore_dir, base_dir='analysis', show=False, verbose=True):
        assert 'oracle' in explore_dir and 'hyper' in explore_dir
        if not os.path.isabs(explore_dir):
            self.explore_dir = os.path.join(os.getcwd(), explore_dir)
        else:
            self.explore_dir = explore_dir
        self.base_directory = os.path.join(self.explore_dir, base_dir)

        base_name = os.path.basename(os.path.normpath(self.explore_dir))
        dname = str(base_name).replace('explore_bv_', '')
        print(dname)
        self.pt_configpath = os.path.join(self.explore_dir, 'explore_' + dname + '.config')
        assert os.path.isfile(self.pt_configpath)
        self.findk_configpath = os.path.join(self.explore_dir,'findk_'+dname+'.config')
        assert os.path.isfile(self.findk_configpath)
        self.kmin_configpath = os.path.join(self.explore_dir,'kmin_'+dname+'.config')
        assert os.path.isfile(self.kmin_configpath)
        self.innersphere_configpath = os.path.join(self.explore_dir, 'innersphere_' + dname + '.config')
        assert os.path.isfile(self.innersphere_configpath)

        self.show = show
        self.verbose = verbose
        self._import_config_files()
        if not self.bootstrap:
            self.run()
        else:
            self.run_bs()
    
    def _import_config_files(self):
        configf = configparser.ConfigParser()
        configf.read(str(self.pt_configpath))
        self.report_steps = configf.getfloat('MCRUNNER', 'report_steps')
        configf.read(str(self.findk_configpath))
        self.kmax = configf.getfloat('FINDK', 'kmax')
        self.prob_kmax = configf.getfloat('FINDK', 'prob') 
        configf.read(str(self.innersphere_configpath))
        self.k_innersphere = configf.getfloat('INNERSPHERE_MCRUNNER', 'k')
        self.ndof = configf.getfloat('INNERSPHERE_HYPERELEM', 'ndof')
        self.geometry = configf.get('INNERSPHERE_HYPERELEM', 'geometry')
        geom_params = configf.get('INNERSPHERE_HYPERELEM', 'geom_params')
        self.geom_params = np.array([float(x) for x in geom_params.split()])
        self.nparticles = 1
        self.vcavity = 1
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
        # karray.insert(0, self.k_innersphere)
        self.karray = np.array(karray)
        self.k0_index = np.where(self.karray==0.)[0][0]

    def _import_pt_time_series(self):
        #import cloud drops timeseries

        self.timeseries = import_pt_cloud_drops_time_series_raw(self.explore_dir,
                                                                max_series_size = int(1e4),
                                                                ncores=self.ncores)

    def _subtract_eqtime(self):
        #remove equilibration region from pt timeseries
        #in principle we should do this on the backbone distance array that we do not record at the moment
        eq_time_min = int(self.timeseries.shape[1] * 0.1) # remove at least 10% of points
        flattened_ts = np.array([x.flatten()[x.flatten() > 0] for x in self.timeseries])
        results = Parallel(n_jobs=max(1,self.ncores))(delayed(find_eqtime)(timeseries) for timeseries in flattened_ts)
        results *= np.array([min(self.timeseries.shape[2],
                                 len(x.flatten())/len(x.flatten()[x.flatten() > 0]))
                             for x in self.timeseries])
        print("eq_times: ", results)
        eq_time = max(eq_time_min, int(np.amax(results)) // self.timeseries.shape[2])
        print("cloud eq_time_min: ", eq_time_min)
        print("cloud eq time: ", eq_time)
        self.timeseries = self.timeseries[:,eq_time:,:]

    def _subsample_timeseries(self, ts_sphere, timeseries):
        """
        returns a flatten timeseries of the uncorrelated data
        """
        self.subsample_indices = []
        K = self.karray.size
        g = np.ones(K)
        N_k = np.zeros(K, dtype='i')
        flat_ts = []

        # # deal with ts separately
        # g[0] = statisticalInefficiency_fft(ts_sphere)
        # indices = np.array(subsampleCorrelatedData(ts_sphere, g=g[0]))  # indices of uncorrelated samples
        # tss = ts_sphere[indices]
        # tss = tss[:tss.size - tss.size % self.timeseries.shape[2]]
        # flat_ts.append(tss.tolist())
        # self.tss = np.array(tss.tolist())
        # N_k[0] = self.tss.size #len(indices)  # number of uncorrelated samples
        # now loop through pt timeseries
        for i in range(K):  # subsample the energies
            print(timeseries[i].shape)
            j = i
            x = timeseries[i].flatten()
            si = statisticalInefficiency_fft(x[x>0]) # should we do this on the backbone?
            si *= min(timeseries.shape[2], x.size/x[x>0].size) #rescale by the fraction of zero values
            print("statistical inefficiency: ",si)
            g[j] = max(1, int(si // float(self.timeseries.shape[2])))
            indices = np.array(subsampleCorrelatedData(np.ones(timeseries[i].shape[0]), g=g[j]))  # indices of uncorrelated samples
            print(len(indices))
            self.subsample_indices.append(indices)
            ts = timeseries[i][indices].flatten()
            ts = ts[ts>0]
            N_k[j] = ts.size  # number of uncorrelated samples > 0
            flat_ts.append(ts.tolist())
        print("flat_ts shape", np.shape(flat_ts))
        return np.array(flatten(flat_ts)), N_k, g

    def _compute_ow(self):
        # return the weights for each sampled on cloud i drop j:
        # o_j^(i)*w_j^(i) / sum_k^ndrops o_k^(i)*w_k^(i)
        timeseries = np.array(self.timeseries)
        karray = self.karray
        assert timeseries.shape[0] == karray.size
        ow = (timeseries > 0) * np.exp(-0.5 * karray[:, np.newaxis, np.newaxis] * timeseries**2)
        norm = ow.sum(axis=2).reshape(ow.shape[0], ow.shape[1], 1).repeat(ow.shape[2], 2)
        # self.ow = np.array(ow / norm)
        self.ow = ow/norm
        # return (1./norm) * (timeseries > 0)

    def _compute_ow_k(self, k):
        # return the weights for each sampled on cloud i drop j:
        # o_j^(i)*w_j^(i) / sum_k^ndrops o_k^(i)*w_k^(i)
        timeseries = np.array(self.timeseries)
        karray = np.ones(self.karray.shape) * k
        assert timeseries.shape[0] == karray.size
        # ow1 = (timeseries > 0) * np.exp(-0.5 * self.karray[:, np.newaxis, np.newaxis] * timeseries**2)
        ow = (timeseries > 0) * np.exp(-0.5 * karray[:, np.newaxis, np.newaxis] * timeseries**2)
        norm = ow.sum(axis=2).reshape(ow.shape[0], ow.shape[1], 1).repeat(ow.shape[2], 2)
        tmp, tmp2 = [], []
        for i, (x, xx) in enumerate(zip(norm,ow)):
            x = x[self.subsample_indices[i]].flatten()
            xx = xx[self.subsample_indices[i]].flatten()
            tmp.extend(x[xx > 0].tolist())
            tmp2.extend(xx[xx > 0].tolist())
        return np.log(tmp) #+ 0.5*k*self.flat_timeseries**2


    def _build_u_kn(self, flat_timeseries):
        K, N = self.karray.size, flat_timeseries.size
        u_kn = np.empty((K, N))
        # subsample ow so that it matched the entries in flat_timeseries
        # and remove all 0 entries
        def _flatten_ow(ow):
            tmp = []
            for i, x in enumerate(ow):
                x = x[self.subsample_indices[i]].flatten()
                tmp.extend(x[x>0].tolist())
            return np.array(tmp)

        self._compute_ow()
        ow = np.array(self.ow)
        ow = _flatten_ow(ow)
        # ow = np.where(ow>0, np.log(ow), -1e70)

        for i in range(K):
            # if i == 0:
            #     u_kn[i] = (self.ndof - 1) * np.log(flat_timeseries) + \
            #               0.5 * self.karray[i] * flat_timeseries ** 2
            # else:
            # u_kn[i] = np.zeros(u_kn[i].size)  # 0.5 * self.karray[i] * flat_timeseries**2
            # x = self.tss**2
            # x = np.exp(-0.5 * self.karray[i] * x.reshape((-1,self.timeseries.shape[2]))).sum(axis=1)
            # x = x.reshape((x.size, 1)).repeat(self.timeseries.shape[2], 1)
            # x = np.array([xx.flatten() for xx in x]).flatten()
            # u_kn[i][:self.tss.size] = -np.log(x)
            ow = self._compute_ow_k(self.karray[i])
            u_kn[i] = ow
        assert self.karray.size == u_kn.shape[0]
        assert N == u_kn.shape[1]
        return u_kn

    def _compute_hs_fluid_volume(self, numerical_moments=False):
        self.F0_acc = 0
        self.ideal_gas_F_acc = 0

    def _build_histogram(self, compute_binedges=True, kde=False):
        """
        set compute bin_edges to false when subsampling so that all subsamples have the same number of bins over the same range
        """
        if not compute_binedges:
            assert self.bootstrap
        if compute_binedges:
            bin_edges = np.linspace(np.amin(np.append(self.timeseries, self.ts_sphere)),
                                    np.amax(np.append(self.timeseries, self.ts_sphere)), self.nbins + 1)
        else:
            bin_edges = self.bin_edges - (self.bin_edges[1] - self.bin_edges[0]) / 2

        if kde:
            hist_visits = self._build_histogram_kde(bin_edges)
        else:
            hist_visits = self._build_histogram_simple(bin_edges)

        self.hist_visits = np.array(hist_visits)
        self.bin_edges = bin_edges + (bin_edges[1] - bin_edges[0]) / 2  # shift bin edges by bin/2
        self._unbias_histogram()

    def _build_histogram_simple(self, bin_edges):
        hist_visits = []
        hist = np.histogram(self.ts_sphere, bin_edges, normed=True)[0]
        hist_visits.append(hist)
        results = Parallel(n_jobs=self.ncores)(
            delayed(np.histogram)(timeseries[timeseries>0], bin_edges, normed=True,
                                  weights=ow[timeseries>0]) for timeseries, ow in zip(self.timeseries, self.ow))
        for hist in results:
            hist_visits.append(hist[0])
        return hist_visits

    def _build_histogram_kde(self, bin_edges):
        hist_visits = []
        # hist = np.histogram(self.ts_sphere, bin_edges, normed=True)[0]
        kde_bin_edges = np.array(bin_edges[:-1])
        kde_bin_edges += (kde_bin_edges[1] - kde_bin_edges[0]) / 2
        hist = get_kde_hist(self.ts_sphere, kde_bin_edges, kernel="epanechnikov", bw=0.001)
        hist_visits.append(hist)
        results = Parallel(n_jobs=max(1, self.ncores))(
            delayed(get_kde_hist)(timeseries[timeseries>0], kde_bin_edges, weights=ow[timeseries>0])
            for timeseries, ow in zip(self.timeseries, self.ow))
        print(np.shape(results))
        for hist in results:
            hist_visits.append(hist)
        print(np.shape(hist_visits))
        return hist_visits

    def _unbias_histogram(self):
        hist_unbiased = np.outer(0.5 * self.karray[1:], self.bin_edges[:-1] ** 2)
        hist_unbiased = np.vstack((
                                  (self.ndof - 1) * np.log(self.bin_edges[:-1])
                                  + 0.5 * self.karray[0] * self.bin_edges[:-1] ** 2,
                                  hist_unbiased))
        self.hist_unbiased = hist_unbiased
        assert self.hist_visits.shape == self.hist_unbiased.shape
        assert self.hist_visits.shape[0] == self.karray.size

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="analyze PT data from thermodynamic integration")
    parser.add_argument("explore_dir", type=str, help="explore_dir")
    parser.add_argument("--show", action='store_true', help="show plots, default: False", default=False)
    parser.add_argument("--bootstrap", action='store_true', help="run bootstrap (slow!), default: False", default=False)
    parser.add_argument("--kde", action='store_true', help="use kernel density estimate, default: False", default=False)
    parser.add_argument("-j", "--ncores", type=int, help="number of cores", default=7)
    args = parser.parse_args()
    print(args)
    
    sim = hyperelem_mbar_compute_dos(bootstrap=args.bootstrap, kde=args.kde, plot_dos_data=True, ncores=args.ncores)
    
    sim(args.explore_dir, show=args.show)
#    else :
#        for subdir, dirs, files in os.walk(wdir):
#            for dir in dirs:
#                if dir is not 'packings' and dir is not 'jammed_packings' and dir is not 'analysis':
#                    path = os.path.join(wdir, dir)
#                    sim(explore_dir=path, frozen=args.frozen)
        