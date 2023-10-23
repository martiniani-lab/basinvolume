from __future__ import division
from __future__ import print_function

from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
import time
import warnings
import os
import configparser
import numpy as np

from pele.potentials import Harmonic

from basinvolume.geometry.point_sampling import HyperElemMCrunner
from basinvolume.spheres import _configure_mcrunner
from basinvolume.utils import *


class _hyperelem_bv_mcrunner(_configure_mcrunner):
    """
    """
    
    def __init__(self, rank, nprocs):
        self.rank = rank
        self.nprocs = nprocs
    
    def __call__(self, base_dir, k=1.0, stepsize=1e-3, niter=2e4, hmin=0, 
                 hmax=100, hbinsize=1, acceptance=0.2, adjustf=0.9, report_steps=5e3, adjustf_navg=100,
                 pt_eq_niter=0, ts_niter=None, ts_freq=1, single=False, seeds=None, 
                 record_trajectory=False, record_trajectory_npoints=1e4,
                 record_histogram=False,verbose=False):
                
        self.temperature=1.0
        self._set_paths(base_dir)
        self._import_packing_config_files()
        self.coords = np.zeros(self.ndof)
        hbinsize = self._get_histogram_bin(k)
                
        #set parameters
        #self.mc_params = dict(k=k, temperature=temperature, )
        kwargs = dict(geom_params=self.geom_params, geometry=self.geometry,
                      k=k, acceptance=acceptance, adjustf=adjustf,
                      report_steps=report_steps, adjustf_navg = adjustf_navg,
                      pt_eq_niter=pt_eq_niter, ts_niter=ts_niter, ts_freq=ts_freq,
                      hmin=hmin, hmax=hmax, hbinsize=hbinsize,
                      record_trajectory=record_trajectory, record_trajectory_npoints=record_trajectory_npoints,
                      seeds=seeds, single=single, record_histogram=record_histogram)

        self.mc_params = dict(temperature=self.temperature, niter=niter, stepsize=stepsize)
        self.mc_params.update(kwargs)
        if seeds is None:
            warnings.warn("seeds not passed")
        
        self._initialise()
        
        #construct mcrunner
        #self.coords is origin, set initial configuration and origin to be the same
        #harmonic potential with fixed centre of mass
        potential = Harmonic(self.coords, k, bdim=self.ndof, com=False)
        mcrunner = HyperElemMCrunner(potential, self.coords, self.temperature, stepsize, niter,
                                     self.coords, **kwargs)
        return mcrunner

    def _set_paths(self, base_dir):
        """
        set base_directory, packings_directory and configpaths, configfile
        """
        assert 'hyper' in base_dir and 'explore_bv' in base_dir
        if not os.path.isabs(base_dir):
            base_directory = os.path.join(os.getcwd(), base_dir)
            assert (os.path.exists(base_directory))
        else:
            base_directory = base_dir
        self.base_directory = base_directory

        base_name = os.path.basename(os.path.normpath(self.base_directory))
        dname = str(base_name).replace('explore_bv_', '')
        print("dname: ", dname)
        self.findk_configpath = os.path.join(self.base_directory, 'findk_' + dname + '.config')
        self.kmin_configpath = os.path.join(self.base_directory, 'kmin_' + dname + '.config')
        self.configfile = '{}/explore_{}.config'.format(self.base_directory, dname)
    
    def _get_histogram_bin(self, k):
        """automatically estimate size of histogram"""
        hmax = self.displ_k_min*k #self.displ_k_max*self.kmax
        hbinsize= hmax * 0.0001 
        return hbinsize
    
    def _initialise(self):
        """initialisation function"""
        #change directory only at the end of initialise
        self._print_initialise()
        os.chdir(self.base_directory)
        
    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        if self.rank == 0:
            self._print_parameters()
    
    def _write_sim_params(self, f):
        """
        write simulation parameters
        """    
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Explore_Jammed_Packings wrapper class input parameters\n')
        f.write('[BV_HYPERELEM]\n')
        f.write('ndof: {}\n'.format(self.ndof))
        f.write('geometry: {}\n'.format(self.geometry))
        f.write('geom_params: ')
        for val in self.geom_params:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        f.write('[MCRUNNER]\n')
        for key, value in list(self.mc_params.items()) :
            f.write('{}: {}\n'.format(key,value))
        f.write('[STATUS]\n')
        for i in range(self.nprocs):
            f.write('success_rank{}: {}\n'.format(str(i), "False"))
    
    def _import_packing_config_files(self):
        configf = configparser.ConfigParser()
        configf.read(str(self.findk_configpath))
        kmax_ndof = configf.getfloat('FINDK_HYPERELEM','ndof')
        kmax_geometry = configf.get('FINDK_HYPERELEM', 'geometry')
        geom_params = configf.get('FINDK_HYPERELEM', 'geom_params')
        kmax_geom_params = np.array([float(x) for x in geom_params.split()])
        self.kmax = configf.getfloat('FINDK','kmax')
        self.prob_kmax = configf.getfloat('FINDK','prob')
        self.displ_k_max = configf.getfloat('FINDK','displ_k_max')
        self.var_displ_k_max = configf.getfloat('FINDK','var_displ_k_max')
        configf.read(str(self.kmin_configpath))
        self.displ_k_min = configf.getfloat('KMIN','displ_k_min')
        self.var_displ_k_min = configf.getfloat('KMIN','var_displ_k_min')
        kmin_ndof = configf.getfloat('KMIN_HYPERELEM','ndof')
        kmin_geometry = configf.get('KMIN_HYPERELEM', 'geometry')
        geom_params = configf.get('KMIN_HYPERELEM', 'geom_params')
        kmin_geom_params = np.array([float(x) for x in geom_params.split()])
        assert kmax_ndof == kmin_ndof
        assert np.all(kmax_geom_params == kmin_geom_params)
        assert kmax_geometry == kmin_geometry
        self.ndof = kmin_ndof
        self.geom_params = kmin_geom_params
        self.geometry = kmin_geometry
    
    def print_success_all(self, success):
        """
        print whether calculation has completed successfully
        """
        assert(hasattr(self, 'configfile'))
        if self.rank == 0:
            configf = configparser.ConfigParser()
            configf.read(str(self.configfile))
            for i in range(self.nprocs):
                configf.set('STATUS', 'success_rank{}'.format(str(i)), success)
            configf.write(open(str(self.configfile),'w'))        
    
if __name__ == "__main__":
        
    #first 7 primary pseudo perfect numbers
    pppn = [2,6,42,1806,47058,2214502422,52495396602]
    seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    
    sim = _hyperelem_bv_mcrunner(0, 1)
    mcrunner = sim('explore_bv_oracle_hypersphere_n2_r1.0', seeds=seeds, verbose=True, niter=1e6)
    print('simulation started')
    start=time.time()
    mcrunner.run()
    end=time.time()
    print('time elapsed', end-start)
    status = mcrunner.get_status()
    print(status)
    
        
                
            
              
                
                
                
