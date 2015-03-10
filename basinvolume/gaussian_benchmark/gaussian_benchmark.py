from __future__ import division
import numpy as np

class GaussianBenchmark(object):
    def __init__(self, means=None, cov=None, minimum_index=0):
        self.means = means
        self.cov = cov
        self.minimum_index = minimum_index
        #
        if not self.means:
            raise Exception("GaussianBenchmark: illegal input: means")
        if not self.cov:
            raise Exception("GaussianBenchmark: illegal input: cov")
    def set_up_potential(self):
        print("set up potential")
        self.pot_optimizer = SumGaussianPot(self.means, self.cov)
    def find_kmax(self):
        print("find kmax")
    def run_kmin(self):
        print("run kmin")
    def run_PT(self):
        print("run PT")
    def compute_volume(self):
        print("compute volume")

if __name__ == "__main__":
    means = [
    [-0.66188835 -4.90248303]
    [-2.50068746  1.00984605]
    [ 2.79156189  3.46309925]
    [ 4.7283517  -1.92263871]
    [-7.2936999  -2.33272689]
    [-5.75772061  6.56907688]
    [ 1.26304236  8.80647431]
    [ 7.82900305  2.89542514]
    [ 5.35866288 -7.17580499]
    [-5.16164399 -7.13075119]        
    ]
    cov = [
    [ 2.89579414  2.89579414]
    [ 3.27805735  3.27805735]
    [ 2.36765046  2.36765046]
    [ 4.02608698  4.02608698]
    [ 1.87897462  1.87897462]
    [ 3.45861227  3.45861227]
    [ 4.63020449  4.63020449]
    [ 5.88827858  5.88827858]
    [ 1.97514666  1.97514666]
    [ 1.62236091  1.62236091]
    ]
    bm = GaussianBenchmark(means=means, cov=cov, minimum_index=0)
    bm.set_up_potential()
    bm.find_kmax()
    bm.run_kmin()
    bm.run_PT()
    bm.compute_volume()
