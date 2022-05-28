from __future__ import division
import numpy as np
import os
import ConfigParser
from basinvolume.utils import import_pt_cloud_drops_time_series_raw, compute_mean_var_cloud_ts
from sympy.mpmath import gammainc as sympy_gammainc
from sympy.mpmath import gamma as sympy_gamma

def gammainc(z, a, regularized=False):
    return np.complex(sympy_gammainc(z,a=a,regularized=regularized))

def gamma(z):
    return np.complex(sympy_gamma(z))

def hypersphere_rmsd(k, R, d):
    """
    returns expectation value for r
    """
    if k == 0.:
        return d*R/(d+1)
    elif k > 0:
        rmsd = np.sqrt(2/np.complex(k,0))*(gamma((d+1)/2)-gammainc((d+1)/2,k*R**2/2))/\
               (gamma(d/2)-gammainc(d/2,k*R**2/2))
    else:
        rmsd = np.complex(0,1)**d * np.sqrt(2) * np.power(-np.complex(k,0),d/2) * \
               np.power(np.complex(k,0),-(1+d)/2) * \
               (gamma((d+1)/2)-gammainc((d+1)/2,k*R**2/2))/\
               (gamma(d/2)-gammainc(d/2,k*R**2/2))
    return np.real(rmsd)

class _testMSD(object):
    def __init__(self, explore_dir, base_dir='analysis', ncores=2):
        self.ncores = ncores
        assert 'hyper' in explore_dir
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

        self._import_config_files()
        self._import_ks()
        self._import_pt_time_series()

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
        self.karray = np.array(karray)
        self.k0_index = np.where(self.karray == 0.)[0][0]

    def _import_pt_time_series(self):
        self.timeseries = import_pt_cloud_drops_time_series_raw(self.explore_dir, max_series_size=int(1e5), ncores=self.ncores)
        print self.timeseries.shape

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

    def compute_msd(self):
        num_mean, num_var = compute_mean_var_cloud_ts(self.timeseries, self.karray)
        print num_var
        exact_msd = np.array([hypersphere_rmsd(k, self.geom_params[0], self.ndof) for k in self.karray])
        return num_mean, exact_msd

if __name__== "__main__":
    test = _testMSD("explore_bv_oracle_hypersphere_n2_r1.0", ncores=2)
    num_msd, exact_msd = test.compute_msd()
    print num_msd
    print exact_msd
    print "num-exact:", num_msd-exact_msd
