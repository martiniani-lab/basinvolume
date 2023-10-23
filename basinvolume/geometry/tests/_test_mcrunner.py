from __future__ import division
from __future__ import print_function
from builtins import object
import numpy as np
from basinvolume.geometry.point_sampling import HyperElemMCrunner
from basinvolume.geometry.cloud_sampling import HyperElemOracleMCrunner
from pele.potentials import Harmonic
from mcpele.monte_carlo import NullPotential
from basinvolume.utils import isClose

def get_disk_mean_r2(radius):
    return radius ** 2 / 2

def get_disk_exp_mean_r2(u, d):
    return (24 * d ** 4 + 24 * d ** 3 * u + 12 * d ** 2 * u ** 2 + 4 * d * u ** 3 + u ** 4) / (4 * d ** 2 + 4 * d * u + 2 * u ** 2)

def get_square_mean_r2(side_length):
    return side_length ** 2 / 6

def get_square_exp_mean_r2(u, d):
    return 2 * (6 * d ** 3 + 6 * d ** 2 * u + 3 * d * u ** 2 + u ** 3) / (3 * (d + u))

def get_r2_mean(geometry, params):
    if geometry=='cube':
        return get_square_mean_r2(params[0])
    if geometry=='sphere':
        return get_disk_mean_r2(params[0])
    if geometry=='cube_exp_decay':
        return get_square_exp_mean_r2(params[0], params[1])
    if geometry=='sphere_exp_decay':
        return get_disk_exp_mean_r2(params[0], params[1])

class TestMCrunnerStochastic(object):
    def __init__(self, ndim, geometry, geom_params, stepsize, cloud_radius, nr_cloud_points, k):
        self.niter = int(1e5)
        self.pt_eq_niter = self.niter / 10
        self.origin = np.zeros(ndim)
        self.full_coords = np.array(self.origin)
        self.potential =  Harmonic(self.origin, k, bdim=ndim, com=False)
        self.mcrunner = HyperElemOracleMCrunner(self.potential, self.full_coords, 1, stepsize, self.niter, self.origin,
                                                geometry=geometry, geom_params=geom_params, pt_eq_niter=self.pt_eq_niter,
                                                cloud_radius=cloud_radius, nr_cloud_points=nr_cloud_points, k=k)
    def run(self):
        """return mean and variance of the random walk"""
        self.mcrunner.run()
        drops_ts = self.mcrunner.get_timeseries()
        cloud = self.mcrunner.cloud_test.get_old_cloud()
        print(cloud.x)
        print(cloud.oracle)
        print(cloud.bias)
        print("cloud 1 ",cloud[0])
        cloud[0] = (np.ones(2) * 2, True, 10)
        self.mcrunner.cloud_test.set_cloud(cloud)
        cloud2 = self.mcrunner.cloud_test.get_old_cloud()
        print("cloud 2 ", cloud2[0])
        print("cloud 2.bias ", cloud2.bias)
        cloud2.bias = np.ones(cloud2.size)*10
        self.mcrunner.cloud_test.set_cloud_bias(cloud2)
        cloud3 = self.mcrunner.cloud_test.get_old_cloud()
        print("cloud3.bias: ", cloud3.bias)
        # print "len(cloud): ", len(cloud)
        # print "cloud.size: ", cloud.size
        # print "cloud.ndof: ", cloud.ndof
        # for i, c in enumerate(cloud):
        #     print cloud[i]
        #     print c
        #     print "\n"
        return drops_ts, self.mcrunner.get_displ2_kmin()

class TestMCrunnerDeterministic(object):
    def __init__(self, ndim, geometry, geom_params, stepsize, k):
        self.niter = int(1e6)
        self.report_steps = self.niter / 10
        self.origin = np.zeros(ndim)
        self.full_coords = np.array(self.origin)
        self.potential =  Harmonic(self.origin, k, bdim=ndim, com=False)
        self.mcrunner = HyperElemMCrunner(self.potential, self.full_coords, 1, stepsize, self.niter, self.origin,
                                          geometry=geometry, geom_params=geom_params, k=k, record_histogram=True,
                                          pt_eq_niter=0, report_steps=self.report_steps)
    def run(self):
        """return mean and variance of the random walk"""
        self.mcrunner.run()
        # print self.mcrunner.get_status()
        drops_ts = self.mcrunner.get_timeseries()
        return drops_ts, self.mcrunner.get_displ2_kmin()

def get_drops_ts_r2_mean(drops_ts, k):
    mean = 0.
    for x in drops_ts:
        x *= x
        mean += np.sum(x*np.exp(-0.5*x*k))/np.count_nonzero(x)
    return mean / drops_ts.shape[0]

if __name__=="__main__":
    ndim = 2
    stepsize= 0.5
    cloud_radius = 0.25
    nr_cloud_points = 20
    k=0. # the test is only meaningful for k=0 (exact result for k != 0 not known here)
    rel_tol = 1e-2
    geom_list = [['cube', [1.]], ['sphere', [1.]], ['cube_exp_decay', [1., 0.1]], ['sphere_exp_decay', [1., 0.1]]]
    for geom in geom_list:
        geometry = geom[0]
        geom_params = geom[1]
        test = TestMCrunnerStochastic(ndim, geometry, geom_params, stepsize, cloud_radius, nr_cloud_points, k)
        drops_ts, (r2mean, r2var) = test.run()
        r2mean_ts = get_drops_ts_r2_mean(drops_ts, k)
        r2mean_exact = get_r2_mean(geometry, geom_params)
        print(test.mcrunner.get_status())
        print("geometry: {}".format(geometry))
        print("num_r2mean: {} exact_r2mean: {}".format(r2mean, r2mean_exact))
        print("are close : {} (rel_tol={})".format(isClose(r2mean, r2mean_exact, rel_tol=rel_tol), rel_tol))
        print("ts_r2mean: {} exact_r2mean: {}".format(r2mean_ts, r2mean_exact))
        print("are close : {} (rel_tol={})".format(isClose(r2mean_ts, r2mean_exact, rel_tol=rel_tol), rel_tol))
    for geom in geom_list:
        geometry = geom[0]
        geom_params = geom[1]
        test = TestMCrunnerDeterministic(ndim, geometry, geom_params, stepsize, k)
        drops_ts, (r2mean, r2var) = test.run()
        r2mean_ts = get_drops_ts_r2_mean(drops_ts, k)
        r2mean_exact = get_r2_mean(geometry, geom_params)
        print("geometry: {}".format(geometry))
        print("num_r2mean: {} exact_r2mean: {}".format(r2mean, r2mean_exact))
        print("are close : {} (rel_tol={})".format(isClose(r2mean, r2mean_exact, rel_tol=rel_tol), rel_tol))
        print("ts_r2mean: {} exact_r2mean: {}".format(r2mean_ts, r2mean_exact))
        print("are close : {} (rel_tol={})".format(isClose(r2mean_ts, r2mean_exact, rel_tol=rel_tol), rel_tol))








