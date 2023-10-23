from __future__ import division
from __future__ import print_function

from future import standard_library
standard_library.install_aliases()
from builtins import str
import time
import warnings
import os
import configparser
import numpy as np

from pele.potentials import Harmonic

from basinvolume.geometry.cloud_sampling import HyperElemOracleMCrunner
from basinvolume.spheres import _configure_mcrunner
from basinvolume.geometry.point_sampling.utils import _append_geom_params
from basinvolume.utils import *


class _oracle_hyperelem_kmin_mcrunner(_configure_mcrunner):
    """
    """
    def __init__(self, ndof, geometry="cube", geom_params=[1.],
                 cloud_radius=1., nr_cloud_points=10,
                 k=0.0, stepsize=5e-1, niter=5e4, acceptance=0.2,
                 adjustf=0.9, report_steps=0, adjustf_navg = 100, hmin=0,
                 hmax=0.01, hbinsize=0.0005, single=True,
                 seeds=None, verbose=False, workspace=None):
                
        self.temperature=1.0
        self.ndof = ndof
        self.geometry=geometry
        self.geom_params = geom_params
        self.cloud_radius = cloud_radius
        self.nr_cloud_points = nr_cloud_points
        self.coords = np.zeros(self.ndof)
        if workspace is None:
            self.workspace = os.getcwd()
        else:
            self.workspace = os.path.abspath(workspace)
        
        self._set_paths()
        
        #self.mc_params = dict(k=k, temperature=temperature, )    
        kwargs = dict(geometry=self.geometry, geom_params=self.geom_params, k=k,
                      cloud_radius=self.cloud_radius, nr_cloud_points=self.nr_cloud_points,
                      acceptance=acceptance, adjustf=adjustf,
                      report_steps=report_steps, adjustf_navg = adjustf_navg,
                      hmin=hmin, hmax=hmax, hbinsize=hbinsize,
                      seeds=seeds, single=single)

        self.mc_params = dict(temperature=self.temperature, niter=niter, stepsize=stepsize)
        self.mc_params.update(kwargs)
        if seeds is None:
            warnings.warn("seeds not passed")
                
        #construct mcrunner
        #self.coords is origin, set initial configuration and origin to be the same
        potential = Harmonic(self.coords, k, bdim=self.ndof, com=False)
        self.mcrunner = HyperElemOracleMCrunner(potential, self.coords, self.temperature, stepsize, niter,
                                                self.coords, **kwargs)
        
        self._initialise()
        
    def run(self):
        try:
            self.mcrunner.run()
            self.displ_k_min, self.var_displ_k_min = self.mcrunner.get_displ2_kmin()
            self._print_results()
            self._print_success(True)
        except:
            view_traceback()
            self._print_success(False)

    def _set_paths(self):
        dname = 'oracle_hyper{}_n'.format(self.geometry)+str(self.ndof)
        dname = _append_geom_params(dname, self.geometry, self.geom_params)
        self.base_directory = os.path.join(self.workspace,'explore_bv_'+dname)
        configfile = 'kmin_' + dname
        self.configfile = '{}/{}.config'.format(self.base_directory,configfile)
    
    def _initialise(self):
        self._print_initialise()
         
    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        self._print_parameters()
    
    def _write_sim_params(self, f):
        """
        write simulation parameters
        """
        f.write('#AUTOMATICALLY GENERATED FILE - DO NOT MODIFY BY HAND\n')
        f.write('#Explore_Jammed_Packings wrapper class input parameters\n')
        f.write('[KMIN_HYPERELEM]\n')
        f.write('ndof: {}\n'.format(self.ndof))
        f.write('geometry: {}\n'.format(self.geometry))
        f.write('geom_params: ')
        for val in self.geom_params:
            f.write('{:.16f} '.format(val))
        f.write('\n')
        f.write('[KMIN_ORACLE]\n')
        f.write('cloud_radius: {}\n'.format(self.cloud_radius))
        f.write('nr_cloud_points: {}\n'.format(self.nr_cloud_points))
        f.write('[KMIN_MCRUNNER]\n')
        for key, value in list(self.mc_params.items()) :
            f.write('{}: {}\n'.format(key,value))
    
    def _print_results_once(self, fname):
        """
        note that self.displ_k_min *= 1.5 to account for the limited computation time, 
        this is just an approximation 
        """
        f = open(fname,'a')
        f.write('[KMIN_MCRUNNER_STATUS]\n')
        status = self.mcrunner.get_status()
        for key, value in list(status.items()) :
            f.write('{}: {}\n'.format(key,value))
        f.write('[KMIN]\n')
        f.write('displ_k_min: {:.16f}\n'.format(self.displ_k_min))
        f.write('var_displ_k_min: {:.16f}\n'.format(self.var_displ_k_min))
        f.close()
    
    def _print_results(self):
        assert(hasattr(self, 'configfile'))
        self._print_results_once(self.configfile)
    
    def _print_success_once(self, success, fname):
        """
        print whether calculation has completed successfully
        this method is overloaded her to check whether this is a 
        diffusion only calculations 
        """
        f = open(fname, 'a')
        f.write('[STATUS]\n')
        f.write('success: {}\n'.format(str(success)))
        f.close()
    
    def _print_success(self, success):
        assert(hasattr(self, 'configfile'))
        self._print_success_once(success, self.configfile)
    
    def _import_packing_config_files(self):
        """pure virtual, must overload"""
        pass
    
if __name__ == "__main__":
    
    #pppn = [2,6,42,1806,47058,2214502422,52495396602]
    #seeds = dict(seed_takestep=pppn[0], seed_metropolis=pppn[1])
    i32max = np.iinfo(np.int32).max
    seeds = dict(seed_takestep=np.random.randint(i32max),seed_metropolis=np.random.randint(i32max),
                 seed_cloud=np.random.randint(i32max), seed_record_drop_r=np.random.randint(i32max),
                 seed_oracle=np.random.randint(i32max))
    ndof = 2
    cloud_radius = 0.25
    nr_cloud_points = 10
    sim = _oracle_hyperelem_kmin_mcrunner(ndof, geometry="sphere", geom_params=[1.],
                                          niter=1e6, k=0, cloud_radius=cloud_radius,
                                          nr_cloud_points=nr_cloud_points, seeds=seeds, single=True,
                                          verbose=True, hmax=15, hbinsize=0.001, stepsize=cloud_radius*2)
    #record_steps_timeseries=True, record_steps_timeseries_every=[int(np.ceil(1.5**n)) for n in xrange(22)],)
    print('simulation started')
    start=time.time()
    sim.run()
    end=time.time()
    print('time elapsed', end-start)
    status = sim.mcrunner.get_status()
    print(status)
    print('d kmin: ',sim.displ_k_min)
    print('var: ',sim.var_displ_k_min)

              
                
                
                
