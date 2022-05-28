from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
from basinvolume.mbar_spheres import mbar_compute_dos
from basinvolume.mbar_spheres.mbar_compute_volume import find_eqtime
import os
import ConfigParser
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
        print dname
        self.pt_configpath = os.path.join(self.explore_dir, 'explore_' + dname + '.config')
        assert os.path.isfile(self.pt_configpath)
        self.findk_configpath = os.path.join(self.explore_dir, 'findk_' + dname + '.config')
        assert os.path.isfile(self.findk_configpath)
        self.kmin_configpath = os.path.join(self.explore_dir, 'kmin_' + dname + '.config')
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
        configf = ConfigParser.ConfigParser()
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

    def _import_pt_time_series(self):
        # import cloud drops timeseries

        self.cloud_timeseries = import_pt_cloud_drops_time_series_raw(self.explore_dir,
                                                                      max_series_size=int(1e4),
                                                                      ncores=self.ncores)
        self._compute_ow(self.cloud_timeseries)
        rn = np.random.uniform(size=self.ow.shape)
        timeseries = [x.flatten()[r.flatten()<ow.flatten()]
                      for (r, ow, x) in zip(rn, self.ow, self.cloud_timeseries)]
        length = np.amin([len(x) for x in timeseries])
        self.timeseries = np.array([x[:length] for x in timeseries])


    def _compute_ow(self, cloud_timeseries):
        # return the weights for each sampled on cloud i drop j:
        # o_j^(i)*w_j^(i) / sum_k^ndrops o_k^(i)*w_k^(i)
        timeseries = np.array(cloud_timeseries)
        karray = self.karray[1:]
        assert timeseries.shape[0] == karray.size
        ow = (timeseries > 0) * np.exp(-0.5 * karray[:, np.newaxis, np.newaxis] * timeseries ** 2)
        rosenbluth = ow.sum(axis=2).reshape(ow.shape[0], ow.shape[1], 1).repeat(ow.shape[2], 2)
        self.ow = ow / rosenbluth

    def _subtract_eqtime(self):
        #remove equilibration region from pt timeseries
        results = Parallel(n_jobs=max(1,self.ncores))(delayed(find_eqtime)(timeseries) for timeseries in self.timeseries)
        print "eq_times: ", results
        eq_time = int(np.amax(results))
        self.timeseries = self.timeseries[:,eq_time:]

    def _subsample_timeseries(self, ts_sphere, timeseries):
        """
        returns a flatten timeseries of the uncorrelated data
        """
        K = self.karray.size
        g = np.ones(K)
        N_k = np.zeros(K, dtype='i')
        flat_ts = np.empty(0)

        # deal with ts separately
        g[0] = statisticalInefficiency_fft(ts_sphere)
        indices = np.array(subsampleCorrelatedData(ts_sphere, g=g[0]))  # indices of uncorrelated samples
        N_k[0] = len(indices)  # number of uncorrelated samples
        flat_ts = np.append(flat_ts, ts_sphere[indices])
        # now loop through pt timeseries
        for i in xrange(K - 1):  # subsample the energies
            j = i + 1
            g[j] = statisticalInefficiency_fft(timeseries[i])
            indices = np.array(subsampleCorrelatedData(timeseries[i], g=g[j]))  # indices of uncorrelated samples
            N_k[j] = len(indices)  # number of uncorrelated samples
            flat_ts = np.append(flat_ts, timeseries[i, indices])
        return flat_ts, N_k, g

    def _build_u_kn(self, flat_timeseries):
        K, N = self.karray.size, flat_timeseries.size
        u_kn = np.empty((K, N))

        for i in xrange(K):
            if i == 0:
                u_kn[i] = (self.ndof - 1) * np.log(flat_timeseries) \
                          + 0.5 * self.karray[i] * flat_timeseries ** 2
            else:
                u_kn[i] = 0.5 * self.karray[i] * flat_timeseries ** 2
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
            timeseries = np.array([x.flatten() for x in self.cloud_timeseries])
            timeseries = timeseries[timeseries>0]
            bin_edges = np.linspace(np.amin(np.append(timeseries, self.ts_sphere)),
                                    np.amax(np.append(timeseries, self.ts_sphere)), self.nbins + 1)
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
            delayed(np.histogram)(timeseries[timeseries > 0], bin_edges,
                                  normed=True, weights=ow[timeseries > 0])
            for timeseries, ow in zip(self.cloud_timeseries, self.ow))
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
            delayed(get_kde_hist)(timeseries[timeseries > 0], kde_bin_edges, weights=ow[timeseries > 0])
            for timeseries, ow in zip(self.cloud_timeseries, self.ow))
        print np.shape(results)
        for hist in results:
            hist_visits.append(hist)
        print np.shape(hist_visits)
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
    print args

    sim = hyperelem_mbar_compute_dos(bootstrap=args.bootstrap, kde=args.kde, plot_dos_data=True, ncores=args.ncores)

    sim(args.explore_dir, show=args.show)
    #    else :
    #        for subdir, dirs, files in os.walk(wdir):
    #            for dir in dirs:
    #                if dir is not 'packings' and dir is not 'jammed_packings' and dir is not 'analysis':
    #                    path = os.path.join(wdir, dir)
    #                    sim(explore_dir=path, frozen=args.frozen)